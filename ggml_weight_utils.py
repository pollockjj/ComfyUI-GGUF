import torch
from .dequant import dequantize_tensor

try:
    from .ggml_tensor import GGMLTensor
except ImportError:
    GGMLTensor = None

def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item

def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
    if tensor is None:
        return None
    patch_list = []
    device = tensor.device
    for function, patches, key in getattr(tensor, "patches", []):
        patch_list += move_patch_to_device(patches, device)
    weight = dequantize_tensor(tensor, dtype, dequant_dtype)
    if GGMLTensor is not None and isinstance(weight, GGMLTensor):
        weight.__class__ = torch.Tensor
    if patch_list:
        if patch_dtype is None:
            weight = function(patch_list, weight, key)
        else:
            computed_patch_dtype = dtype if patch_dtype == "target" else patch_dtype
            weight = function(patch_list, weight, key, computed_patch_dtype)
    return weight
