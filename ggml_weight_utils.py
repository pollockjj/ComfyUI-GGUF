import torch
import time
import weakref
from .dequant import dequantize_tensor

try:
    from .ggml_tensor import GGMLTensor
except ImportError:
    GGMLTensor = None

patch_cache = {}

cached_tensor_map = {}
cached_tensors = []  # Global list for strong references
prev_ggml_tensor_ptr = None # Global variable to track previous tensor key
level_one_tensor = None

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
    global cached_tensor_map, cached_tensors, prev_ggml_tensor_ptr, level_one_tensor

    if tensor is None:
        return None

    if len(cached_tensors) < 250:
        cache_final_weight = True
    else:
        cache_final_weight = False

    ggml_tensor_ptr = tensor.data_ptr()

    if ggml_tensor_ptr in cached_tensor_map: ## I either data in a level 1 (onboard) cache or a level 2 (other VRAM) cache
        if level_one_tensor is not None:
            weight = level_one_tensor
            #print(f"DEBUG: Fetched final weight for GGML {ggml_tensor_ptr} from Level 1 Cache.")
        else:    
            weight = cached_tensor_map[ggml_tensor_ptr]['level_two_cache_location']().clone()
            #print(f"DEBUG: Fetched final weight for GGML {ggml_tensor_ptr} from Level 2 Cache.")
            cached_tensor_map[prev_ggml_tensor_ptr]['level_one_prefetch']=cached_tensor_map[ggml_tensor_ptr]['level_two_cache_location']
            #print(f"DEBUG: Final weight for GGML {ggml_tensor_ptr} written to {prev_ggml_tensor_ptr}'s level_one_prefetch.")
            prev_ggml_tensor_ptr = ggml_tensor_ptr  # initialize and/or update global variable
        
        if  cached_tensor_map[ggml_tensor_ptr]['level_one_prefetch'] is not None:
            with torch.cuda.stream(cuda1_stream):
                new_level_one = (cached_tensor_map[ggml_tensor_ptr]['level_one_prefetch']().clone().to("cuda:0", non_blocking=True))
            level_one_tensor = new_level_one
            #print(f"DEBUG: Updated level_one_tensor with final weight for next GGML.")
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
        cloned_final_weight = weight.clone().to("cuda:1", non_blocking=False)
        cached_tensors.append(cloned_final_weight)
        #print(f"DEBUG: Cloned final weight for tensor ptr {ggml_tensor_ptr} and stored in Level 2 Cache on cuda:1.")
        cached_tensor_map[ggml_tensor_ptr] = {'level_two_cache_location': weakref.ref(cloned_final_weight),'level_one_prefetch':None}
        prev_ggml_tensor_ptr = ggml_tensor_ptr  # initialize and/or update global variable

    return weight