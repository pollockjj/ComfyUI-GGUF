"""
    This file contains monkey patching functions to integrate our optimized lora.py with ComfyUI.
    Copyright (C) 2024 ComfyUI-GGUF Contributors

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import logging
import importlib

# Store original functions to restore if needed
_original_functions = {}

def apply_monkey_patches():
    """Apply monkey patches to ComfyUI's original lora.py to use our optimized functions"""
    global _original_functions
    
    try:
        # Import our custom lora module
        from . import gguf_lora
        import comfy.lora
        
        # Store original functions for potential restoration
        _original_functions = {
            "calculate_weight": comfy.lora.calculate_weight,
            "load_lora": comfy.lora.load_lora,
            "model_lora_keys_clip": comfy.lora.model_lora_keys_clip,
            "model_lora_keys_unet": comfy.lora.model_lora_keys_unet,
        }
        
        # Replace with our implementations
        comfy.lora.calculate_weight = gguf_lora.calculate_weight
        comfy.lora.load_lora = gguf_lora.load_lora
        comfy.lora.model_lora_keys_clip = gguf_lora.model_lora_keys_clip
        comfy.lora.model_lora_keys_unet = gguf_lora.model_lora_keys_unet
        
        logging.info("Successfully applied ComfyUI-GGUF monkey patches to ComfyUI's LoRA functions")
        return True
    except Exception as e:
        logging.error(f"Failed to apply ComfyUI-GGUF monkey patches: {e}")
        return False

def restore_original_functions():
    """Restore original ComfyUI LoRA functions"""
    global _original_functions
    
    if not _original_functions:
        logging.warning("No original functions stored, cannot restore")
        return False
    
    try:
        import comfy.lora
        
        # Restore original functions
        for func_name, func in _original_functions.items():
            setattr(comfy.lora, func_name, func)
        
        logging.info("Restored original ComfyUI LoRA functions")
        return True
    except Exception as e:
        logging.error(f"Failed to restore original functions: {e}")
        return False