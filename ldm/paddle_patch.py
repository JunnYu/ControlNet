import builtins, logging, paddle, contextlib, time
import paddle.nn as nn
import paddle.nn.functional as F
from fastcore.all import patch_to, TypeDispatch
import numpy as np

logger = logging.getLogger()
nn.Module = nn.Layer
nn.ModuleList = nn.LayerList
nn.ModuleDict = nn.LayerDict
nn.Conv1d = nn.Conv1D
nn.Conv3d = nn.Conv3D
nn.SiLU = nn.Silu
nn.BatchNorm1d = nn.BatchNorm1D
nn.BatchNorm2d = nn.BatchNorm2D
nn.BatchNorm3d = nn.BatchNorm3D
nn.ConvTranspose1d = nn.Conv1DTranspose
nn.ConvTranspose2d = nn.Conv2DTranspose
nn.ConvTranspose3d = nn.Conv3DTranspose
nn.AvgPool1d = nn.AvgPool1D
nn.AvgPool2d = nn.AvgPool2D
nn.AvgPool3d = nn.AvgPool3D
nn.MaxPool1d = nn.MaxPool1D
nn.MaxPool2d = nn.MaxPool2D
nn.MaxPool3d = nn.MaxPool3D
nn.Dropout1d = nn.Dropout
nn.Dropout2d = nn.Dropout2D
nn.Dropout3d = nn.Dropout2D
nn.MaxUnpool1d = nn.MaxUnPool1D
nn.MaxUnpool2d = nn.MaxUnPool2D
nn.MaxUnpool3d = nn.MaxUnPool3D
paddle.eq = paddle.equal



def load_dict(checkpoint_file, return_numpy=False, map_location: str = "cpu"):
    """
    Reads a Paddle checkpoint file, returning properly formatted errors if they arise.
    """
    try:
        if map_location == "cpu":
            raw_device = paddle.get_device()
            paddle.set_device("cpu")
            state_dict = paddle.load(checkpoint_file, return_numpy=return_numpy)
            paddle.set_device(raw_device)
        else:
            state_dict = paddle.load(checkpoint_file, return_numpy=return_numpy)
        return state_dict
    except:
        pass
 
paddle.load = load_dict

class Cuda:
    pass

paddle.cuda = Cuda
paddle.cuda.is_available = paddle.is_compiled_with_cuda
paddle.to_tensor = paddle.to_tensor

class LinearPT(nn.Layer):
    def __init__(self, in_features, out_features, bias=True, bias_attr=None, name=None):
        super().__init__()
        self._dtype = self._helper.get_default_dtype()
        self._bias_attr = bias if bias_attr is None else bias_attr
        self.weight = self.create_parameter(shape=[out_features, in_features],
                                            dtype=self._dtype,
                                            is_bias=False)
        self.bias = self.create_parameter(shape=[out_features],
                                          attr=self._bias_attr,
                                          dtype=self._dtype,
                                          is_bias=True)
        self.name = name

    def forward(self, input):
        out = F.linear(x=input,
                       weight=self.weight.t(),
                       bias=self.bias,
                       name=self.name)
        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'in_features={}, out_features={}, dtype={}{}'.format(
            self.weight.shape[0], self.weight.shape[1], self._dtype, name_str)

class Conv2D(nn.Conv2D):
    def __init__(self, *args, **kwargs):
        bias = kwargs.pop("bias", None)
        if bias is not None:
            kwargs["bias_attr"] = bias
        super().__init__(*args, **kwargs)

class GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("affine", None)
        eps = kwargs.pop("eps", None)
        if eps is not None:
            kwargs["epsilon"] = eps
        super().__init__(*args, **kwargs)

nn.Conv2d = Conv2D
nn.GroupNorm = GroupNorm
nn.LinearPT = LinearPT

from paddle.fluid.framework import Parameter as PPParameter


class ParameterDict(nn.ParameterList):

    def __init__(self, parameters_dict=None):
        super(ParameterDict, self).__init__()
        if parameters_dict is not None:
            for name, param in parameters_dict.items():
                assert isinstance(param, PPParameter)
                self.add_parameter(name, param)

    def update(self, parameters_dict):
        for name, param in parameters_dict.items():
            assert isinstance(param, PPParameter)
            self.add_parameter(name, param)
        return self


nn.ParameterDict = ParameterDict


def cast(x:paddle.Tensor, dtype=paddle.float32):
    if x.dtype != dtype:
        return x.cast(dtype)
    else:
        return x

# attr
paddle.long = paddle.int64
paddle.int = paddle.int32
paddle.double = paddle.float64
paddle.half = paddle.float16

# other
paddle.Tensor.mul_ = paddle.Tensor.scale_
paddle.Tensor.half = lambda x: cast(x, paddle.float16)
paddle.Tensor.float = lambda x: cast(x, paddle.float32)
paddle.Tensor.double = lambda x: cast(x, paddle.float64)
paddle.Tensor.int = lambda x: cast(x, paddle.int32)
paddle.Tensor.long = lambda x: cast(x, paddle.int64)
paddle.Tensor.bool = lambda x: cast(x, paddle.bool)
paddle.Tensor.device = paddle.Tensor.place
paddle.Tensor.add = lambda x, y: x + y
paddle.Tensor.sub = lambda x, y: x - y
paddle.Tensor.mul = lambda x, y: x * y
paddle.Tensor.div = lambda x, y: x / y
setattr(paddle.Tensor, "data", property(lambda x: x))
paddle.Tensor.clamp = paddle.clip
paddle.Tensor.data_ptr = lambda x: x.value().get_tensor()._ptr()
paddle.clamp = paddle.clip
paddle.view = paddle.reshape
paddle.permute = paddle.transpose


def convert_tuple(x):
    if isinstance(x, int):
        return (x, )
    elif isinstance(x, float):
        return (x, )
    return x

from ppdiffusers.ppnlp_patch_utils import get_rng_state_tracker

def randn_pt(*shape: builtins.int, dtype=None, name=None, **kwargs):
    shape = convert_tuple(shape)
    generator = kwargs.get("generator", None)
    with get_rng_state_tracker().rng_state(generator):
        return paddle.randn(shape, dtype=dtype, name=name)


paddle.randn = TypeDispatch([randn_pt, paddle.randn])
paddle.randn_like = lambda x, dtype=None, generator=None: paddle.randn(
    *x.shape, dtype=dtype, generator=generator)

raw_expand = paddle.expand
def expand_pt(x, *shape: builtins.int, name=None):
    shape = convert_tuple(shape)
    return raw_expand(x, shape=shape, name=name)
    
paddle.expand = TypeDispatch([expand_pt, raw_expand])
paddle.Tensor.expand = paddle.expand 


raw_repeat = paddle.tile
def repeat_pt(x, *repeat_times: builtins.int, name=None):
    repeat_times = convert_tuple(repeat_times)
    return raw_repeat(x, repeat_times=repeat_times, name=name)
    
paddle.repeat = TypeDispatch([repeat_pt, raw_repeat])
paddle.Tensor.repeat = paddle.repeat


def rand_pt(*shape: builtins.int, dtype=None, name=None, **kwargs):
    shape = convert_tuple(shape)
    generator = kwargs.get("generator", None)
    with get_rng_state_tracker().rng_state(generator):
        return paddle.rand(shape, dtype=dtype, name=name)


paddle.rand = TypeDispatch([rand_pt, paddle.rand])


def zeros_pt(*shape: builtins.int, dtype=None, name=None):
    shape = convert_tuple(shape)
    return paddle.zeros(shape, dtype=dtype, name=name)


paddle.zeros = TypeDispatch([zeros_pt, paddle.zeros])

def ones_pt(*shape: builtins.int, dtype=None, name=None):
    shape = convert_tuple(shape)
    return paddle.ones(shape, dtype=dtype, name=name)

raw_full = paddle.full
def full_pt(shape, fill_value, dtype=None, name=None, **kwargs):
    return raw_full(shape, fill_value, dtype, name)

paddle.full = full_pt

paddle.ones = TypeDispatch([ones_pt, paddle.ones])

raw_transpose = paddle.transpose

def permute_pt(x, *perm: builtins.int, name=None):
    return paddle.transpose(x, perm=perm, name=name)


paddle.Tensor.permute = TypeDispatch([permute_pt, raw_transpose])


def transpose_pt(x, dim0: builtins.int, dim1: builtins.int, name=None):
    perm_list = list(range(x.ndim))
    perm_list[dim1], perm_list[dim0] = perm_list[dim0], perm_list[dim1]
    return raw_transpose(x, perm=perm_list, name=name)


# 这个transpose有点费事，底层用的transpose，不如自己写一个。
# def transpose_pt(x, dim0: builtins.int, dim1: builtins.int, name=None):
#     return paddle.moveaxis(x, [dim0, dim1], [dim1, dim0], name=name)
# 这个既支持原版paddle.transpose又支持pypaddle的transpose。
paddle.Tensor.transpose = TypeDispatch([transpose_pt, raw_transpose])


def reshape_pt(x, *shape: builtins.int, name=None):
    shape = convert_tuple(shape)
    return paddle.reshape(x, shape=shape, name=name)


paddle.Tensor.reshape = TypeDispatch([reshape_pt, paddle.reshape])
paddle.Tensor.view = paddle.Tensor.reshape

## concat 兼容dim， 其他算子兼容工作量太大
raw_concat = paddle.concat

paddle.sigmoid = F.sigmoid

def concat_pt(x, axis=0, name=None, dim=None):
    axis = dim if dim is not None else axis
    return raw_concat(x, axis=axis, name=name)


paddle.concat = concat_pt
paddle.cat = concat_pt

## softmax 兼容dim， 其他算子兼容工作量太大
raw_softmax = F.softmax


def softmax_pt(x, axis=-1, dtype=None, name: str = None, dim=None):
    axis = dim if dim is not None else axis
    return raw_softmax(x, axis=axis, dtype=dtype, name=name)

raw_chunk = paddle.chunk

def chunk_pt(x, chunks, axis=0, name:str=None, dim=None):
    axis = dim if dim is not None else axis
    return raw_chunk(x, chunks=chunks, axis=axis, name=name)

paddle.chunk = chunk_pt
paddle.Tensor.chunk = chunk_pt 


def softmax_pt(x, axis=-1, dtype=None, name: str = None, dim=None):
    axis = dim if dim is not None else axis
    return raw_softmax(x, axis=axis, dtype=dtype, name=name)

paddle.softmax = softmax_pt
paddle.Tensor.softmax = softmax_pt
F.softmax = softmax_pt

# 这个看情况，有的paddle算子的shape判断的时候默认用的是list，可能会有错，最好别改。。。。。。。。
# paddle.Tensor.shape_beifen = paddle.Tensor.shape
# setattr(paddle.Tensor, "shape", property(lambda x: tuple(x.shape_beifen)))


def size_pt(self, i=None):
    if i is None:
        return self.shape
    return self.shape[i]


paddle.Tensor.size = size_pt


def narrow_pt(self, axis, start, length):
    return paddle.slice(self, input, [axis], [start], [start + length])


paddle.Tensor.narrow = narrow_pt


def to_pt(self, dtype=paddle.float32, **kwargs):
    if paddle.is_tensor(dtype):
        dtype = dtype.dtype
    try:
        self = cast(self, dtype)
    except:
        pass
    return self

@patch_to(paddle.Tensor)
def type(self, dtype=paddle.float32, **kwargs):
    if paddle.is_tensor(dtype):
        dtype = dtype.dtype
    try:
        self = cast(self, dtype)
    except:
        pass
    return self

@patch_to(paddle.Tensor)
def contiguous(self):
    return self


paddle.Tensor.to = to_pt

paddle.Tensor.copy_beifen_ = paddle.Tensor.copy_
paddle.Tensor.copy_ = lambda self, x, blocking=True: self.copy_beifen_(
    x, blocking)
########################################################## Tensor


@patch_to(nn.Layer)
def half(self:nn.Layer):
    self.to(dtype="float16")
    return self


@patch_to(nn.Layer)
def float(self:nn.Layer):
    self.to(dtype="float32")
    return self


@patch_to(nn.Layer)
def double(self:nn.Layer):
    self.to(dtype="float64")
    return self


@patch_to(nn)
def Parameter(data: paddle.Tensor, requires_grad=True):
    tensor = paddle.create_parameter(
        data.shape,
        dtype=data.dtype,
        default_initializer=nn.initializer.Assign(data))
    if not requires_grad:
        tensor.stop_gradient = True
    return tensor


# scatter
def scatter_paddle(tensor, axis, index, src):
    assert axis == 0 or axis == 1
    assert tensor.ndim == index.ndim == src.ndim == 2
    index = index.cast("int64")
    i, j = index.shape
    grid_x, grid_y = paddle.meshgrid(paddle.arange(i), paddle.arange(j))
    if axis == 0:
        index = paddle.stack([index.flatten(), grid_y.flatten()], axis=1)
    else:
        index = paddle.stack([grid_x.flatten(), index.flatten()], axis=1)
    updates_index = paddle.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    updates = paddle.gather_nd(src, index=updates_index)
    return paddle.scatter_nd_add(tensor, index, updates)


paddle.scatter_paddle = scatter_paddle
paddle.Tensor.scatter_paddle = scatter_paddle


def gather_torch(tensor, dim, index):
    return paddle.take_along_axis(tensor, index, dim)

@patch_to(nn.Layer)
def eval(self):
    # Layer-level setting
    self.training = False
    for layer in self.sublayers():
        layer.training = False
    return self

paddle.gather_torch = gather_torch
paddle.Tensor.gather_torch = gather_torch


@patch_to(nn.Layer)
def load_state_dict(self: nn.Layer, state_dict: dict, use_structured_name=True, strict=True):
    orig = self.state_dict()
    orig_keys = set([k for k in orig.keys()])
    loaded_keys = set([k for k in state_dict.keys()])

    missing_keys = list(orig_keys - loaded_keys)
    unexpected_keys = list(loaded_keys - orig_keys)
    print(f"missing_keys: {missing_keys}")
    print(f"unexpected_keys: {unexpected_keys}")
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        raise ValueError("state_dict donot match the orignial state_dict!")
    return self.load_dict(state_dict, use_structured_name=use_structured_name)


# device
def get_parameter_device(parameter: nn.Layer):
    # try:
    #     return next(parameter.named_parameters())[1].place
    # except StopIteration:
    return paddle.get_device()


def get_parameter_dtype(parameter: nn.Layer):
    try:
        return next(parameter.named_parameters())[1].dtype
    except StopIteration:
        return paddle.get_default_dtype()


@patch_to(nn.Layer, as_prop=True)
def device(self):
    return get_parameter_device(self)

@patch_to(nn.Layer)
def cuda(self):
    return self

@patch_to(nn.Layer)
def cpu(self):
    return self

@patch_to(nn.Layer, as_prop=True)
def dtype(self):
    return get_parameter_dtype(self)


@patch_to(nn.Layer)
def modules(self: nn.Layer):
    return self.sublayers(include_self=True)


# @patch_to(nn.Layer)
# def layers(self: nn.Layer):
#     return self.sublayers(include_self=True)


# @patch_to(nn.Layer)
# def named_layers(self: nn.Layer):
#     return self.named_sublayers(include_self=True)


@patch_to(nn.Layer)
def named_modules(self: nn.Layer):
    return self.named_sublayers(include_self=True)


@patch_to(nn.Layer)
def stop_gradient_(self: nn.Layer, value=True):
    for paramater in self.parameters():
        paramater.stop_gradient = value


@patch_to(nn.Layer)
def requires_grad_(self: nn.Layer, value=True):
    for paramater in self.parameters():
        paramater.stop_gradient = not value
    return self


@patch_to(paddle.Tensor)
def requires_grad_(self, value=True):
    self.stop_gradient = not value
    return self


@patch_to(paddle.Tensor)
def stop_gradient_(self, value=False):
    self.stop_gradient = value
    return self


@patch_to(paddle.Tensor, as_prop=True)
def requires_grad(self):
    return not self.stop_gradient


def finfo(dtype):
    if dtype == paddle.float32:
        return np.finfo(np.float32)
    if dtype == paddle.float16:
        return np.finfo(np.float16)
    if dtype == paddle.float64:
        return np.finfo(np.float64)

paddle.finfo = finfo