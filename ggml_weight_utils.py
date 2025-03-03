import torch
import time
import weakref
from .dequant import dequantize_tensor
import os

try:
    from .ggml_tensor import GGMLTensor
except ImportError:
    GGMLTensor = None

# Check for baseline mode
BASELINE_MODE = True  # Always use baseline for now (disable all caching)

# Patch cache
patch_cache = {}

# Only initialize these if not in baseline mode
if not BASELINE_MODE:
    cached_tensor_map = {}
    cached_tensors = []  # Global list for strong references for level 2 (cuda:1)
    level_zero_tensors = []  # Global list for strong references for level 0 (cuda:0)
    prev_ggml_tensor_ptr = None  # Global variable to track previous tensor key
    level_one_tensor = None  # Prefetch slot

    # Create and initialize CUDA streams for cuda:0 and cuda:1
    cuda0_stream = torch.cuda.Stream(device="cuda:0")
    cuda1_stream = torch.cuda.Stream(device="cuda:1")


def move_patch_to_device(item, device):
    # In baseline mode, just do a simple transfer
    if BASELINE_MODE:
        if isinstance(item, torch.Tensor):
            return item.to(device)
        elif isinstance(item, tuple):
            return tuple(move_patch_to_device(x, device) for x in item)
        elif isinstance(item, list):
            return [move_patch_to_device(x, device) for x in item]
        else:
            return item
    
    # Full implementation with streams
    else:
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


@profile
def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
    if tensor is None:
        return None

    # In baseline mode, just process normally without multi-level caching
    # but STILL use the patch caching from retrieve_cached_patch
    if BASELINE_MODE:
        # Process tensor patches
        patch_list = []
        device = tensor.device
        patches_data = getattr(tensor, "patches", [])
        for function, patches_item, key in patches_data:
            patch_result = retrieve_cached_patch(patches_item, device, key)
            patch_list += patch_result

        # Dequantize tensor
        weight = dequantize_tensor(tensor, dtype, dequant_dtype)
        if GGMLTensor is not None and isinstance(weight, GGMLTensor):
            weight.__class__ = torch.Tensor
        
        # Apply patches if needed
        if patch_list:
            if patch_dtype is None:
                weight = function(patch_list, weight, key)
            else:
                computed_patch_dtype = dtype if patch_dtype == "target" else patch_dtype
                weight = function(patch_list, weight, key, computed_patch_dtype)
        
        return weight
    
    # Full implementation with multi-level caching
    else:
        global cached_tensor_map, cached_tensors, level_zero_tensors, prev_ggml_tensor_ptr, level_one_tensor

        # Get the unique identifier for this tensor
        ggml_tensor_ptr = tensor.data_ptr()

        # First check if we have this tensor in any cache level
        if ggml_tensor_ptr in cached_tensor_map:
            # Check for level 0 cache first (direct access on cuda:0)
            if 'level_zero_cache_location' in cached_tensor_map[ggml_tensor_ptr]:
                # Direct reference from level 0 cache
                return cached_tensor_map[ggml_tensor_ptr]['level_zero_cache_location']
            
            # Then check level 1 prefetch cache
            if level_one_tensor is not None:
                weight = level_one_tensor
                #print(f"DEBUG: Fetched from Level 1 Cache (prefetch).")
            else:
                # Fallback to level 2 cache (cuda:1)
                with torch.cuda.stream(cuda1_stream):
                    weight = cached_tensor_map[ggml_tensor_ptr]['level_two_cache_location']().clone()
                #print(f"DEBUG: Fetched from Level 2 Cache (cuda:1).")
                
                # Set up prefetching for next access
                if prev_ggml_tensor_ptr in cached_tensor_map:
                    cached_tensor_map[prev_ggml_tensor_ptr]['level_one_prefetch'] = cached_tensor_map[ggml_tensor_ptr]['level_two_cache_location']
                
                prev_ggml_tensor_ptr = ggml_tensor_ptr
            
            # Prefetch the next tensor if available
            if 'level_one_prefetch' in cached_tensor_map[ggml_tensor_ptr] and cached_tensor_map[ggml_tensor_ptr]['level_one_prefetch'] is not None:
                with torch.cuda.stream(cuda1_stream):
                    new_level_one = cached_tensor_map[ggml_tensor_ptr]['level_one_prefetch']().clone().to("cuda:0", non_blocking=True)
                level_one_tensor = new_level_one
                #print(f"DEBUG: Updated prefetch cache with next tensor.")
            
            return weight

        # Not in cache, process normally
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

        # Determine caching level - simple rule: 1/5 of tensors go to level 0 (cuda:0), 4/5 to level 2 (cuda:1)
        # Using tensor pointer % 5 == 0 as a deterministic rule
        if ggml_tensor_ptr % 5 == 0:
            # Level 0 cache - store directly on cuda:0 with hard reference
            with torch.cuda.stream(cuda0_stream):
                level0_tensor = weight.clone().to("cuda:0", non_blocking=True)
            
            # Store strong reference
            level_zero_tensors.append(level0_tensor)
            
            # Create cache entry if needed
            if ggml_tensor_ptr not in cached_tensor_map:
                cached_tensor_map[ggml_tensor_ptr] = {}
            
            # Store direct reference
            cached_tensor_map[ggml_tensor_ptr]['level_zero_cache_location'] = level0_tensor
            
            # Set up for prefetching
            prev_ggml_tensor_ptr = ggml_tensor_ptr
            
            return level0_tensor
        else:
            # Level 2 cache - store on cuda:1 with prefetching to level 1
            with torch.cuda.stream(cuda1_stream):
                level2_tensor = weight.clone().to("cuda:1", non_blocking=True)
            
            # Store strong reference
            cached_tensors.append(level2_tensor)
            
            # Create or update cache entry
            if ggml_tensor_ptr not in cached_tensor_map:
                cached_tensor_map[ggml_tensor_ptr] = {}
            
            # Store weakref
            cached_tensor_map[ggml_tensor_ptr]['level_two_cache_location'] = weakref.ref(level2_tensor)
            cached_tensor_map[ggml_tensor_ptr]['level_one_prefetch'] = None
            
            # Set up for prefetching
            prev_ggml_tensor_ptr = ggml_tensor_ptr
            
            return weight