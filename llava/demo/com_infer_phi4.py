import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch
import transformers
import tokenizers
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM, LlavaPhiConfig
from PIL import Image
import pickle
import argparse
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
from utils import find_all_linear_names, add_special_tokens_and_resize_model, load_weights, expand2square

def infer():
    """
    Inference script with support for INT8 quantization to reduce memory usage.
    
    Usage examples:
    # Standard FP16 inference
    python com_infer_phi4.py --dtype FP16 --model_name_or_path "model/phi-4" ...
    
    # INT8 quantized inference (reduces memory by ~50%)
    python com_infer_phi4.py --dtype INT8 --model_name_or_path "model/phi-4" ...
    
    Note: INT8 requires bitsandbytes library: pip install bitsandbytes
    """
    parser = argparse.ArgumentParser(description='HealthGPT inference with optional INT8 quantization')
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/Phi-3-mini-4k-instruct')
    parser.add_argument('--dtype', type=str, default='FP32', choices=['FP32', 'FP16', 'BF16', 'INT8'],
                        help='Model data type: FP32, FP16, BF16, or INT8 (quantization, reduces memory by ~50%%)')
    parser.add_argument('--attn_implementation', type=str, default=None)
    parser.add_argument('--hlora_r', type=int, default=16)
    parser.add_argument('--hlora_alpha', type=int, default=32)
    parser.add_argument('--hlora_dropout', type=float, default=0.0)
    parser.add_argument('--hlora_nums', type=int, default=4)
    parser.add_argument('--vq_idx_nums', type=int, default=1024)
    parser.add_argument('--instruct_template', type=str, default='phi3_instruct')
    parser.add_argument('--vit_path', type=str, default='openai/clip-vit-large-patch14-336')
    parser.add_argument('--hlora_path', type=str, default=None)
    parser.add_argument('--fusion_layer_path', type=str, default=None)
    parser.add_argument('--question', type=str, default=None)
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    
    

    args = parser.parse_args()

    # Determine model dtype and quantization config
    use_int8 = args.dtype == 'INT8'
    if use_int8:
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
            # Explicitly import to ensure PEFT can detect it
            # Try to get version, but don't fail if not available
            try:
                version = getattr(bnb, '__version__', 'unknown')
                print(f"bitsandbytes version: {version}")
            except:
                print("bitsandbytes loaded successfully")
        except ImportError as e:
            raise ImportError("bitsandbytes is required for INT8 quantization. Install it with: pip install bitsandbytes") from e
        
        # Configure int8 quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        model_dtype = torch.float16  # Base dtype for int8 quantized models
        print("Using INT8 quantization to reduce memory usage")
    else:
        quantization_config = None
        model_dtype = torch.float32 if args.dtype == 'FP32' else (torch.float16 if args.dtype == 'FP16' else torch.bfloat16)
        print(f"Using {args.dtype} precision")

    # Use low_cpu_mem_usage to avoid loading entire model into CPU memory first
    # This significantly reduces memory usage during model loading
    # Try to load directly to GPU, fallback to CPU if device_map not supported
    load_kwargs = {
        'pretrained_model_name_or_path': args.model_name_or_path,
        'attn_implementation': args.attn_implementation,
        'low_cpu_mem_usage': True,
    }
    
    if use_int8:
        # For int8, quantization_config handles device placement
        load_kwargs['quantization_config'] = quantization_config
        load_kwargs['device_map'] = "auto"  # Let bitsandbytes handle device placement
        load_kwargs['torch_dtype'] = model_dtype
    else:
        load_kwargs['torch_dtype'] = model_dtype
        load_kwargs['device_map'] = "cuda:0"  # Load directly to GPU
    
    try:
        model = LlavaPhiForCausalLM.from_pretrained(**load_kwargs)
        if use_int8:
            print("Model loaded with INT8 quantization")
        else:
            print("Model loaded directly to GPU using device_map")
    except (TypeError, ValueError) as e:
        # Fallback if device_map is not supported (for non-int8 models)
        if use_int8:
            raise RuntimeError(f"Failed to load model with INT8 quantization: {e}")
        
        print(f"device_map not supported, loading to CPU first: {e}")
        load_kwargs.pop('device_map', None)
        model = LlavaPhiForCausalLM.from_pretrained(**load_kwargs)
        # Move to GPU immediately after loading to free CPU memory
        print("Moving model to GPU...")
        model = model.to(device='cuda', dtype=model_dtype)
        # Clear CPU cache after moving to GPU
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Apply LoRA after model is on GPU to avoid CPU memory accumulation
    # For INT8 models, ensure bitsandbytes is imported before applying LoRA
    if use_int8:
        try:
            import bitsandbytes as bnb
            # Force import to ensure PEFT can detect it
            _ = bnb.nn.Linear8bitLt
            print("bitsandbytes modules loaded successfully for LoRA")
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not verify bitsandbytes for LoRA: {e}")
    
    from llava.peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=args.hlora_r,
        lora_alpha=args.hlora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=args.hlora_dropout,
        bias='none',
        task_type="CAUSAL_LM",
        lora_nums=args.hlora_nums,
    )
    model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )
    num_new_tokens = add_special_tokens_and_resize_model(tokenizer, model, args.vq_idx_nums)
    print(f"Number of new tokens added for unified task: {num_new_tokens}")

    from utils import com_vision_args
    com_vision_args.model_name_or_path = args.model_name_or_path
    com_vision_args.vision_tower = args.vit_path
    com_vision_args.version = args.instruct_template

    model.get_model().initialize_vision_modules(model_args=com_vision_args)
    
    # Ensure model is on GPU with correct dtype (should already be there from above)
    # For int8 quantized models, don't try to move or change dtype as it's handled by bitsandbytes
    if not use_int8:
        try:
            first_param = next(model.parameters())
            if first_param.device.type != 'cuda':
                print("Warning: Model not on GPU, moving now...")
                model = model.to(device='cuda', dtype=model_dtype)
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            else:
                # Just ensure dtype is correct
                model = model.to(dtype=model_dtype)
        except StopIteration:
            pass
    
    # Ensure vision tower is on GPU with correct dtype
    # For int8 models, vision tower should use float16 to avoid issues
    vision_dtype = torch.float16 if use_int8 else model_dtype
    if hasattr(model.get_vision_tower(), 'to'):
        try:
            model.get_vision_tower().to(device='cuda', dtype=vision_dtype)
        except Exception as e:
            print(f"Warning: Could not move vision tower to GPU: {e}")
            # Try without device specification for int8 models
            if use_int8:
                model.get_vision_tower().to(dtype=vision_dtype)
    
    # Ensure mm_projector is on GPU with correct dtype
    # This is critical for INT8 models where device_map="auto" may leave mm_projector on CPU
    if hasattr(model.get_model(), 'mm_projector') and model.get_model().mm_projector is not None:
        try:
            model.get_model().mm_projector.to(device='cuda', dtype=vision_dtype)
            print("mm_projector moved to GPU successfully")
        except Exception as e:
            print(f"Warning: Could not move mm_projector to GPU: {e}")
            # Try without device specification for int8 models
            if use_int8:
                try:
                    model.get_model().mm_projector.to(dtype=vision_dtype)
                except Exception as e2:
                    print(f"Warning: Could not change mm_projector dtype: {e2}")

    # Load weights - now they will be loaded directly to GPU since model is already on GPU
    model = load_weights(model, args.hlora_path)
    model.eval()
    # Clear GPU cache after loading weights
    torch.cuda.empty_cache()

    question = args.question
    img_path = args.img_path

    if img_path:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question
    else:
        qs = question
    conv = conversation_lib.conv_templates[args.instruct_template].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda().unsqueeze_(0)
    if img_path:
        image = Image.open(img_path).convert('RGB')
        image = expand2square(image, tuple(int(x*255) for x in model.get_vision_tower().image_processor.image_mean))
        # Process image and move to GPU immediately to avoid CPU memory accumulation
        # For int8 models, use float16 for image tensor
        img_dtype = torch.float16 if use_int8 else model_dtype
        image_tensor = model.get_vision_tower().image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze_(0).to(device='cuda', dtype=img_dtype)
    with torch.inference_mode():
        output_ids = model.base_model.model.generate(
        input_ids,
        images=image_tensor if img_path else None,
        image_sizes=image.size if img_path else None,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True)
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f'Q: {question}')
    print(f'HealthGPT: {response}')


if __name__ == "__main__":

    infer()