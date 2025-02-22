import torch
import time
from .dequant import dequantize_tensor

try:
    from .ggml_tensor import GGMLTensor
except ImportError:
    GGMLTensor = None

global_op_counter = 0
TOTAL_GLOBAL = 8512
TOTAL_GROUP = 304

cumulative_data = {}
table_printed = False

def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item

def compute_size(item):
    if isinstance(item, torch.Tensor):
        return item.numel() * item.element_size()
    elif isinstance(item, (list, tuple)):
        return sum(compute_size(x) for x in item)
    else:
        return 0

def update_cumulative_data(local_idx, key_str, elapsed_ms, size_mb):
    if local_idx not in cumulative_data:
        cumulative_data[local_idx] = {
            "key": key_str,
            "cumulative_time_ms": elapsed_ms,
            "count": 1,
            "size_mb": size_mb
        }
    else:
        cumulative_data[local_idx]["cumulative_time_ms"] += elapsed_ms
        cumulative_data[local_idx]["count"] += 1

def maybe_print_table():
    global table_printed
    if table_printed or global_op_counter < TOTAL_GLOBAL:
        return
    header = f"{'local_idx':<10s}{'key':<60s}{'cumulative_time_ms':>20s}{'count':>10s}{'size_mb':>10s}"
    print("\n=== BEGIN 304-LINE TABLE ===")
    print(header)
    print("-" * len(header))
    for i in range(TOTAL_GROUP):
        data = cumulative_data.get(i, None)
        if data is not None:
            print(f"{i:<10d}{data['key']:<60s}{data['cumulative_time_ms']:>20.3f}{data['count']:>10d}{data['size_mb']:>10.3f}")
        else:
            print(f"{i:<10d}{'N/A':<60s}")
    print("=== END 304-LINE TABLE ===\n")
    table_printed = True

def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
    global global_op_counter
    if tensor is None:
        return None
    patch_list = []
    device = tensor.device
    patches_data = getattr(tensor, "patches", [])
    for function, patches_item, key in patches_data:
        if isinstance(key, (list, tuple)):
            start = time.perf_counter()
            patch_result = move_patch_to_device(patches_item, device)
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            size_mb = compute_size(patch_result) / (1024 * 1024)
            for sub_key in key:
                local_idx = global_op_counter % TOTAL_GROUP
                update_cumulative_data(local_idx, str(sub_key), elapsed_ms, size_mb)
                global_op_counter += 1
            patch_list += patch_result
        else:
            local_idx = global_op_counter % TOTAL_GROUP
            start = time.perf_counter()
            patch_result = move_patch_to_device(patches_item, device)
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            size_mb = compute_size(patch_result) / (1024 * 1024)
            update_cumulative_data(local_idx, str(key), elapsed_ms, size_mb)
            global_op_counter += 1
            patch_list += patch_result
    if global_op_counter >= TOTAL_GLOBAL:
        maybe_print_table()
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
