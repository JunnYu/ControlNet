
import io
import pickle
from functools import lru_cache
from typing import Union
from zipfile import ZipFile
import paddle

import numpy as np

# patch_bf16 safe tensors
import safetensors.numpy

np.bfloat16 = np.uint16
safetensors.numpy._TYPES.update({"BF16": np.uint16})


def _maybe_decode_ascii(bytes_str: Union[bytes, str]) -> str:
    # When using encoding='bytes' in Py3, some **internal** keys stored as
    # strings in Py2 are loaded as bytes. This function decodes them with
    # ascii encoding, one that Py3 uses by default.
    #
    # NOTE: This should only be used on internal keys (e.g., `typename` and
    #       `location` in `persistent_load` below!
    if isinstance(bytes_str, bytes):
        return bytes_str.decode("ascii")
    return bytes_str


@lru_cache(maxsize=None)
def _storage_type_to_dtype_to_map():
    """convert storage type to numpy dtype"""
    return {
        "DoubleStorage": np.double,
        "FloatStorage": np.float32,
        "HalfStorage": np.half,
        "LongStorage": np.int64,
        "IntStorage": np.int32,
        "ShortStorage": np.int16,
        "CharStorage": np.int8,
        "ByteStorage": np.uint8,
        "BoolStorage": np.bool_,
        "ComplexDoubleStorage": np.cdouble,
        "ComplexFloatStorage": np.cfloat,
        "BFloat16Storage": np.uint16,
    }


class StorageType:
    """Temp Class for Storage Type"""

    def __init__(self, name):
        self.dtype = _storage_type_to_dtype_to_map()[name]

    def __str__(self):
        return f"StorageType(dtype={self.dtype})"


def _element_size(dtype: str) -> int:
    """
    Returns the element size for a dtype, in bytes
    """
    if dtype in [np.float16, np.float32, np.float64]:
        return np.finfo(dtype).bits >> 3
    elif dtype == np.bool_:
        return 1
    else:
        return np.iinfo(dtype).bits >> 3


class UnpicklerWrapperStage(pickle.Unpickler):
    def find_class(self, mod_name, name):
        if type(name) is str and "Storage" in name:
            try:
                return StorageType(name)
            except KeyError:
                pass

        # pure torch tensor builder
        if mod_name == "torch._utils":
            return _rebuild_tensor_stage

        # pytorch_lightning tensor builder
        if "pytorch_lightning" in mod_name:
            return dumpy
        return super().find_class(mod_name, name)


def _rebuild_tensor_stage(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    # if a tensor has shape [M, N] and stride is [1, N], it's column-wise / fortran-style
    # if a tensor has shape [M, N] and stride is [M, 1], it's row-wise / C-style
    # defautls to C-style
    if stride is not None and len(stride) > 1 and stride[0] == 1 and stride[1] > 1:
        order = "F"
    else:
        order = "C"

    return storage.reshape(size, order=order)


def dumpy(*args, **kwarsg):
    return None


def load_torch(path: str, **pickle_load_args):
    try:
        pickle_load_args.update({"encoding": "utf-8"})
        torch_zip = ZipFile(path, "r")
        loaded_storages = {}

        def load_tensor(dtype, numel, key, location):
            name = f"archive/data/{key}"
            typed_storage = np.frombuffer(torch_zip.open(name).read()[:numel], dtype=dtype)
            return typed_storage

        def persistent_load(saved_id):
            assert isinstance(saved_id, tuple)
            typename = _maybe_decode_ascii(saved_id[0])
            data = saved_id[1:]

            assert (
                typename == "storage"
            ), f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
            storage_type, key, location, numel = data
            dtype = storage_type.dtype

            if key in loaded_storages:
                typed_storage = loaded_storages[key]
            else:
                nbytes = numel * _element_size(dtype)
                typed_storage = load_tensor(dtype, nbytes, key, _maybe_decode_ascii(location))
                loaded_storages[key] = typed_storage

            return typed_storage

        data_iostream = torch_zip.open("archive/data.pkl").read()
        unpickler_stage = UnpicklerWrapperStage(io.BytesIO(data_iostream), **pickle_load_args)
        unpickler_stage.persistent_load = persistent_load
        result = unpickler_stage.load()
        torch_zip.close()
    except:
        import torch
        result = {k: v.float().numpy() for k, v in torch.load(path, map_location="cpu").items()}
    return result

def load(ckpt):
    sd = None
    if "pt" in ckpt or "ckpt" in ckpt or "bin" in ckpt:
        pl_sd = load_torch(ckpt)
    elif "safetensors" in ckpt:
        try: 
            from safetensors.torch import load_file
            pl_sd = load_file(ckpt, device="cpu")
        except:
            from safetensors.numpy import load_file
            pl_sd = load_file(ckpt)   
    else:
        pl_sd = paddle.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd.get("state_dict", pl_sd)
    
    return sd
import paddle

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    state_dict = load(ckpt_path)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model
