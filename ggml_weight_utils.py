import torch
from .dequant import dequantize_tensor

try:
    from .ggml_tensor import GGMLTensor
except ImportError:
    GGMLTensor = None

global_op_counter = 0
TOTAL_GLOBAL = 8512
TOTAL_GROUP = 304

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
    global global_op_counter
    if tensor is None:
        return None
    patch_list = []
    device = tensor.device
    patches_data = getattr(tensor, "patches", [])
    for function, patches_item, key in patches_data:
        if isinstance(key, (list, tuple)):
            for sub_key in key:
                local_count = global_op_counter % TOTAL_GROUP
                print(f"Operation {global_op_counter}/{TOTAL_GLOBAL}, {local_count}/{TOTAL_GROUP}: key: {sub_key}")
                global_op_counter += 1
        else:
            local_count = global_op_counter % TOTAL_GROUP
            print(f"Operation {global_op_counter}/{TOTAL_GLOBAL}, {local_count}/{TOTAL_GROUP}: key: {key}")
            global_op_counter += 1
        patch_list += move_patch_to_device(patches_item, device)
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
