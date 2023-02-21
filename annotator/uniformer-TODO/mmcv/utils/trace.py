import warnings

import paddle

from annotator.uniformer.mmcv.utils import digit_version


def is_jit_tracing() -> bool:
    if (paddle.__version__ != 'parrots'
            and digit_version(paddle.__version__) >= digit_version('1.6.0')):
        on_trace = paddle.jit.is_tracing()
        # In Pypaddle 1.6, paddle.jit.is_tracing has a bug.
        # Refers to https://github.com/pypaddle/pypaddle/issues/42448
        if isinstance(on_trace, bool):
            return on_trace
        else:
            return paddle._C._is_tracing()
    else:
        warnings.warn(
            'paddle.jit.is_tracing is only supported after v1.6.0. '
            'Therefore is_tracing returns False automatically. Please '
            'set on_trace manually if you are using trace.', UserWarning)
        return False
