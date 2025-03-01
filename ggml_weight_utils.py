import torch
import time
import weakref
from .dequant import dequantize_tensor

try:
    from .ggml_tensor import GGMLTensor
except ImportError:
    GGMLTensor = None

patch_cache = {}

cached_tensors = {}
cached_tensor_list = []  # Global list for strong references

# Create and initialize CUDA streams for cuda:0 and cuda:1
cuda0_stream = torch.cuda.Stream(device="cuda:0")
cuda1_stream = torch.cuda.Stream(device="cuda:1")

@profile
def move_patch_to_device(item, device):
    # Select the appropriate stream based on the target device
    if "cuda:0" in str(device):
        stream = cuda0_stream
    elif "cuda:1" in str(device):
        stream = cuda1_stream
    else:
        stream = None

    if isinstance(item, torch.Tensor):
        if stream is not None:
            with torch.cuda.stream(stream):
                return item.to(device, non_blocking=True)
        else:
            return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item

@profile
def retrieve_cached_patch(patches_item, device, key):
    cache_key = tuple(key) if isinstance(key, (list, tuple)) else key
    if cache_key in patch_cache:
        return patch_cache[cache_key]
    patch = move_patch_to_device(patches_item, device)
    patch_cache[cache_key] = patch
    return patch

@profile
def compute_size(item):
    if isinstance(item, torch.Tensor):
        return item.numel() * item.element_size()
    elif isinstance(item, (list, tuple)):
        return sum(compute_size(x) for x in item)
    else:
        return 0

@profile
def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
    global cached_tensors, cached_tensor_list

    if tensor is None:
        return None

    if len(cached_tensor_list) < 250:
        cache_final_weight = True
    else:
        cache_final_weight = False

    ggml_tensor_ptr = tensor.data_ptr()
    
    if ggml_tensor_ptr in cached_tensors:
        weight = cached_tensors[ggml_tensor_ptr]['final_tensor']().clone()
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
        # Use the cuda:1 stream to move the cached tensor to cuda:1
        with torch.cuda.stream(cuda1_stream):
            cached_clone = weight.clone().to("cuda:1", non_blocking=True)
        cached_tensor_list.append(cached_clone)
        cached_tensors[ggml_tensor_ptr] = {'final_tensor': weakref.ref(cached_clone)}
    
    return weight
