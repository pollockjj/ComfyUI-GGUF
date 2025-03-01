import torch
import time
import weakref

from .dequant import dequantize_tensor

try:
    from .ggml_tensor import GGMLTensor
except ImportError:
    GGMLTensor = None


cached_tensors = {}
cached_tensor = None
patch_cache = {}

def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item

def retrieve_cached_patch(patches_item, device, key):
    cache_key = tuple(key) if isinstance(key, (list, tuple)) else key
    if cache_key in patch_cache:
        return patch_cache[cache_key]
    patch = move_patch_to_device(patches_item, device)
    patch_cache[cache_key] = patch
    return patch

def compute_size(item):
    if isinstance(item, torch.Tensor):
        return item.numel() * item.element_size()
    elif isinstance(item, (list, tuple)):
        return sum(compute_size(x) for x in item)
    else:
        return 0

def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
    global cached_tensors, dequant_cache, cached_tensor

    if tensor is None:
        return None

    if cached_tensor is None:
        cache_final_weight = True
    else:
        cache_final_weight = False

    ggml_tensor_ptr = tensor.data_ptr()
    
    if ggml_tensor_ptr in cached_tensors:
        weight = cached_tensors[ggml_tensor_ptr]['final_tensor']().clone()
        print(f"HIT")
        return weight

    patch_list = []
    device = tensor.device
    patches_data = getattr(tensor, "patches", [])
    for function, patches_item, key in patches_data:
        patch_result = retrieve_cached_patch(patches_item, device, key)
        patch_list += patch_result

    weight = dequantize_tensor(tensor, dtype, dequant_dtype)
    if GGMLTensor is not None and isinstance(weight, GGMLTensor):
        weight.__class__ = torch.Tensor
    if patch_list:
        if patch_dtype is None:
            weight = function(patch_list, weight, key)
        else:
            computed_patch_dtype = dtype if patch_dtype == "target" else patch_dtype
            weight = function(patch_list, weight, key, computed_patch_dtype)
            
    if cache_final_weight:
        cached_tensor = weight.clone()
        cached_tensors[ggml_tensor_ptr] = {'final_tensor': weakref.ref(cached_tensor)}
        print(ggml_tensor_ptr, "CACHED")
    
    return weight