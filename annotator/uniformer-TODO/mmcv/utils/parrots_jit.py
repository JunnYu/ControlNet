# Copyright (c) OpenMMLab. All rights reserved.
import os

from .parrots_wrapper import paddle_VERSION

parrots_jit_option = os.getenv('PARROTS_JIT_OPTION')

if paddle_VERSION == 'parrots' and parrots_jit_option == 'ON':
    from parrots.jit import pat as jit
else:

    def jit(func=None,
            check_input=None,
            full_shape=True,
            derivate=False,
            coderize=False,
            optimize=False):

        def wrapper(func):

            def wrapper_inner(*args, **kargs):
                return func(*args, **kargs)

            return wrapper_inner

        if func is None:
            return wrapper
        else:
            return func


if paddle_VERSION == 'parrots':
    from parrots.utils.tester import skip_no_elena
else:

    def skip_no_elena(func):

        def wrapper(*args, **kargs):
            return func(*args, **kargs)

        return wrapper
