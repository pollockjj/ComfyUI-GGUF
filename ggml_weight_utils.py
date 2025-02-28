import torch
import time
import weakref

from .dequant import dequantize_tensor

try:
    from .ggml_tensor import GGMLTensor
except ImportError:
    GGMLTensor = None


ggml_tensor_pointers = {}
dequant_cache = None
final_cache = None
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
    global ggml_tensor_pointers, dequant_cache, final_cache

    if tensor is None:
        return None

    ggml_layer_pointer = tensor.data_ptr()

    # Register the tensor's original pointer with placeholders for the dequantized and patched tensor references.
    if ggml_layer_pointer not in ggml_tensor_pointers:
        ggml_tensor_pointers[ggml_layer_pointer] = {
            'dequant_tensor': None,
            'final_tensor': None
        }
    
    patch_list = []
    device = tensor.device
    patches_data = getattr(tensor, "patches", [])
    for function, patches_item, key in patches_data:
        patch_result = retrieve_cached_patch(patches_item, device, key)
        patch_list += patch_result
    
    
    
    if dequant_cache is None:
        weight = dequantize_tensor(tensor, dtype, dequant_dtype)
        dequant_cache = weight.clone()
        for pointer in ggml_tensor_pointers:
            if pointer == ggml_layer_pointer:
                ggml_tensor_pointers[ggml_layer_pointer]['dequant_tensor'] = weakref.ref(dequant_cache)
                print(f"DEQUANT_CACHED")
                break
    for pointer in ggml_tensor_pointers:
        if pointer == ggml_layer_pointer:
            if ggml_tensor_pointers[ggml_layer_pointer]['dequant_tensor'] is not None:
                weight = ggml_tensor_pointers[ggml_layer_pointer]['dequant_tensor']().clone()
                print(f"HIT DEQUANT")
                break
            else:
                weight = dequantize_tensor(tensor, dtype, dequant_dtype)
                break       
    if GGMLTensor is not None and isinstance(weight, GGMLTensor):
        weight.__class__ = torch.Tensor
    if patch_list:
        if final_cache is None:
            if patch_dtype is None:
                weight = function(patch_list, weight, key)
            else:
                computed_patch_dtype = dtype if patch_dtype == "target" else patch_dtype
                weight = function(patch_list, weight, key, computed_patch_dtype)
            final_cache = weight.clone()
            print(f"CLONED")
            for pointer in ggml_tensor_pointers:
                if pointer == ggml_layer_pointer:
                    ggml_tensor_pointers[ggml_layer_pointer]['final_tensor'] = weakref.ref(final_cache)
                    print(f"FINAL_CACHED")
                    break
        for pointer in ggml_tensor_pointers:
            if pointer == ggml_layer_pointer:
                if ggml_tensor_pointers[ggml_layer_pointer]['final_tensor'] is not None:
                    weight = ggml_tensor_pointers[ggml_layer_pointer]['final_tensor']().clone()
                    print(f"HIT FINAL")
                else:
                    if patch_dtype is None:
                        weight = function(patch_list, weight, key)
                    else:
                        computed_patch_dtype = dtype if patch_dtype == "target" else patch_dtype
                        weight = function(patch_list, weight, key, computed_patch_dtype)
                    break       



    return weight