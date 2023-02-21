import importlib

import paddle
import numpy as np

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont

import math
from typing import Union, List, Optional, Tuple
def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = paddle.to_tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, paddle.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x,paddle.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=True):
    total_params = sum(p.numel() for p in model.parameters()).item()
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


@paddle.no_grad()
def make_grid(
    tensor: Union[paddle.Tensor, List[paddle.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
) -> paddle.Tensor:
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (paddle.is_tensor(tensor) or
            (isinstance(tensor, list) and all(paddle.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = paddle.concat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = paddle.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = paddle.full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in np.arange(ymaps):
        for x in np.arange(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pypaddle.org/docs/stable/tensors.html#paddle.Tensor.copy_
            grid[:,x * width + padding:x * width + padding + width - padding, y * height + padding:y * height + padding + height - padding] = tensor[k]
            k = k + 1
    return grid

# TODO
AdamWwithEMAandWings = None
# class AdamWwithEMAandWings(optim.Optimizer):
#     # credit to https://gist.github.com/crowsonkb/65f7265353f403714fce3b2595e0b298
#     def __init__(self, params, lr=1.e-3, betas=(0.9, 0.999), eps=1.e-8,  # TODO: check hyperparameters before using
#                  weight_decay=1.e-2, amsgrad=False, ema_decay=0.9999,   # ema decay to match previous code
#                  ema_power=1., param_names=()):
#         """AdamW that saves EMA versions of the parameters."""
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {}".format(eps))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
#         if not 0.0 <= weight_decay:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
#         if not 0.0 <= ema_decay <= 1.0:
#             raise ValueError("Invalid ema_decay value: {}".format(ema_decay))
#         defaults = dict(lr=lr, betas=betas, eps=eps,
#                         weight_decay=weight_decay, amsgrad=amsgrad, ema_decay=ema_decay,
#                         ema_power=ema_power, param_names=param_names)
#         super().__init__(params, defaults)

#     def __setstate__(self, state):
#         super().__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault('amsgrad', False)

#     @paddle.no_grad()
#     def step(self, closure=None):
#         """Performs a single optimization step.
#         Args:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             with paddle.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             params_with_grad = []
#             grads = []
#             exp_avgs = []
#             exp_avg_sqs = []
#             ema_params_with_grad = []
#             state_sums = []
#             max_exp_avg_sqs = []
#             state_steps = []
#             amsgrad = group['amsgrad']
#             beta1, beta2 = group['betas']
#             ema_decay = group['ema_decay']
#             ema_power = group['ema_power']

#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 params_with_grad.append(p)
#                 if p.grad.is_sparse:
#                     raise RuntimeError('AdamW does not support sparse gradients')
#                 grads.append(p.grad)

#                 state = self.state[p]

#                 # State initialization
#                 if len(state) == 0:
#                     state['step'] = 0
#                     # Exponential moving average of gradient values
#                     state['exp_avg'] = paddle.zeros_like(p, memory_format=paddle.preserve_format)
#                     # Exponential moving average of squared gradient values
#                     state['exp_avg_sq'] = paddle.zeros_like(p, memory_format=paddle.preserve_format)
#                     if amsgrad:
#                         # Maintains max of all exp. moving avg. of sq. grad. values
#                         state['max_exp_avg_sq'] = paddle.zeros_like(p, memory_format=paddle.preserve_format)
#                     # Exponential moving average of parameter values
#                     state['param_exp_avg'] = p.detach().float().clone()

#                 exp_avgs.append(state['exp_avg'])
#                 exp_avg_sqs.append(state['exp_avg_sq'])
#                 ema_params_with_grad.append(state['param_exp_avg'])

#                 if amsgrad:
#                     max_exp_avg_sqs.append(state['max_exp_avg_sq'])

#                 # update the steps for each param group update
#                 state['step'] += 1
#                 # record the step after step update
#                 state_steps.append(state['step'])

#             optim._functional.adamw(params_with_grad,
#                     grads,
#                     exp_avgs,
#                     exp_avg_sqs,
#                     max_exp_avg_sqs,
#                     state_steps,
#                     amsgrad=amsgrad,
#                     beta1=beta1,
#                     beta2=beta2,
#                     lr=group['lr'],
#                     weight_decay=group['weight_decay'],
#                     eps=group['eps'],
#                     maximize=False)

#             cur_ema_decay = min(ema_decay, 1 - state['step'] ** -ema_power)
#             for param, ema_param in zip(params_with_grad, ema_params_with_grad):
#                 ema_param.mul_(cur_ema_decay).add_(param.float(), alpha=1 - cur_ema_decay)

#         return loss