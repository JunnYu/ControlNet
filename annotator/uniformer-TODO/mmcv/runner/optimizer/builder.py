# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect

import paddle

from ...utils import Registry, build_from_cfg

OPTIMIZERS = Registry('optimizer')
OPTIMIZER_BUILDERS = Registry('optimizer builder')


def register_paddle_optimizers():
    paddle_optimizers = []
    for module_name in dir(paddle.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(paddle.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  paddle.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            paddle_optimizers.append(module_name)
    return paddle_optimizers


paddle_OPTIMIZERS = register_paddle_optimizers()


def build_optimizer_constructor(cfg):
    return build_from_cfg(cfg, OPTIMIZER_BUILDERS)


def build_optimizer(model, cfg):
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    optimizer = optim_constructor(model)
    return optimizer
