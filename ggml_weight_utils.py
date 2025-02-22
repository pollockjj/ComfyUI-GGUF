import torch
import time
from .dequant import dequantize_tensor

try:
    from .ggml_tensor import GGMLTensor
except ImportError:
    GGMLTensor = None

# Global counters and constants
global_op_counter = 0
TOTAL_GLOBAL = 8512  # Total operations
TOTAL_GROUP = 304    # Unique patch indices (0 to 303)

# Dictionary to accumulate measurements per unique patch index.
# Each entry stores:
#   "key": patch key (string)
#   "cumulative_time_ms": sum of measured times (in ms)
#   "count": number of measurements (should be 28)
#   "size_mb": patch size in MB (recorded once)
cumulative_data = {}
table_printed = False  # Ensures the table is printed only once

def move_patch_to_device(item, device):
    """
    Recursively moves a patch (or collection of patches) to the specified device.
    """
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item

def compute_size(item):
    """
    Recursively computes the size in bytes of a tensor or collection of tensors.
    """
    if isinstance(item, torch.Tensor):
        return item.numel() * item.element_size()
    elif isinstance(item, (list, tuple)):
        return sum(compute_size(x) for x in item)
    else:
        return 0

def update_cumulative_data(local_idx, key_str, elapsed_ms, size_mb):
    """
    Updates cumulative_data for the given local index.
    Adds elapsed_ms to the cumulative time and increments the count.
    For local index 0:
       - On the first measurement, print "Caching this layer" and then the running total.
       - On subsequent measurements, print "Applying local cache; " followed by the running total.
    For local index 1, simply print the running total.
    """
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

    if local_idx == 0:
        running_total = cumulative_data[0]["cumulative_time_ms"]
        if cumulative_data[0]["count"] == 1:
            print("Caching this layer")
        else:
            print("Applying local cache; ", end="")
        print(f"Running total for 0/303: {running_total:.3f} ms")
    elif local_idx == 1:
        running_total = cumulative_data[1]["cumulative_time_ms"]
        print(f"Running total for 1/303: {running_total:.3f} ms")

def maybe_print_table():
    """
    Once global_op_counter reaches TOTAL_GLOBAL, prints a single neatly aligned
    304-line cumulative table.
    """
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
    """
    Dequantizes the tensor and applies patches.
    
    For each patch operation, measures the time (in ms) to move the patch to device
    and computes its size (in MB). Updates cumulative_data based on the local index (global_op_counter % TOTAL_GROUP).
    
    Since there are 304 unique patches processed 28 times (8512 operations total), the final table
    will have 304 cumulative entries. Running totals for local indices 0 and 1 are printed as specified.
    """
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
            elapsed_ms = (end - start) * 1000  # in ms
            size_mb = compute_size(patch_result) / (1024 * 1024)  # in MB
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
            elapsed_ms = (end - start) * 1000  # in ms
            size_mb = compute_size(patch_result) / (1024 * 1024)  # in MB
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
