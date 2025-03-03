# only import if running as a custom node
try:
    import comfy.utils
except ImportError:
    pass
else:
    import logging
    from .nodes import NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
    
    # Import and apply monkey patches
    try:
        from .gguf_lora_patch import apply_monkey_patches
        
        # For initial development and testing, let's keep monkey patching disabled by default
        # We'll enable it after testing that everything works with the unmodified version
        ENABLE_MONKEY_PATCHING = False
        
        if ENABLE_MONKEY_PATCHING:
            success = apply_monkey_patches()
            if success:
                logging.info("ComfyUI-GGUF: Successfully applied LoRA monkey patches")
            else:
                logging.warning("ComfyUI-GGUF: Failed to apply LoRA monkey patches")
        else:
            logging.info("ComfyUI-GGUF: LoRA monkey patching is currently disabled")
            
    except Exception as e:
        logging.error(f"ComfyUI-GGUF: Error during initialization: {e}")