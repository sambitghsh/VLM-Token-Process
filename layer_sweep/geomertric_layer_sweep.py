#!/usr/bin/env python3
"""
Generic CLI script for running geometric layer sweep analysis on vision-language models.
"""

import argparse
import os
import sys
import json
import csv
import datetime
import base64
from io import BytesIO
from typing import List, Dict

import torch
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForImageTextToText
)
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np

# Try to import sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("⚠️  sentence-transformers not available. Semantic similarity will be skipped.")
    print("   Install with: pip install sentence-transformers")


# Global variables
IMG_TOKEN_INDEX = None
SEMANTIC_MODEL = None


def get_semantic_model():
    """Lazy load semantic similarity model."""
    global SEMANTIC_MODEL
    if SEMANTIC_MODEL is None and SENTENCE_TRANSFORMER_AVAILABLE:
        print("Loading semantic similarity model...")
        SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return SEMANTIC_MODEL


def compute_semantic_similarity(text1, text2):
    """Compute semantic similarity between two texts using sentence transformers."""
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        return None
    
    model = get_semantic_model()
    if model is None:
        return None
    
    try:
        embeddings = model.encode([text1, text2], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return similarity
    except Exception as e:
        print(f"⚠️  Error computing semantic similarity: {e}")
        return None


def compute_exact_match(text1, text2):
    """Compute exact match (1.0 if identical, 0.0 otherwise)."""
    return 1.0 if text1.strip() == text2.strip() else 0.0


    # ============================================================
    #  Model Setup
    # ============================================================

def setup_model(model_id, bnb_4bit=True, cache_dir=None, device=None):
        """
        Setup model and processor with configurable cache directory and device.
        
        Parameters
        ----------
        model_id : str
            HuggingFace model identifier
        bnb_4bit : bool
            Whether to use 4-bit quantization
        cache_dir : str, optional
            Directory to cache downloaded models
        device : str, optional
            Device to load model on (e.g., 'cuda:0', 'cpu')
        
        Returns
        -------
        model, processor : tuple
            Loaded model and processor
        """
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        ) if bnb_4bit else None

        model_kwargs = {
            'dtype': torch.float16,
            'low_cpu_mem_usage': True,
            'quantization_config': bnb_config,
            'device_map': device
        }
        
        if cache_dir:
            model_kwargs['cache_dir'] = cache_dir

        if 'llava' in model_id.lower():
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                **model_kwargs
            )
        elif 'qwen' in model_id.lower():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                **model_kwargs
            )
        elif 'opengvlab' in model_id.lower():
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                **model_kwargs
            )
        else:
            # Generic fallback
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                **model_kwargs
            )
        
        processor_kwargs = {}
        if cache_dir:
            processor_kwargs['cache_dir'] = cache_dir
            
        processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
        return model, processor


    # ============================================================
    #  Metrics + Projections
    # ============================================================

def cosine_sim(a, b, dim=-1):
    return F.cosine_similarity(a, b, dim=dim)
def kl_divergence_from_logits(p_logits, q_logits):
    p_logprob = F.log_softmax(p_logits, dim=-1)
    q_logprob = F.log_softmax(q_logits, dim=-1)
    p_prob = p_logprob.exp()
    kl = (p_prob * (p_logprob - q_logprob)).sum(dim=-1).mean().item()
    return kl
def js_divergence_from_logits(p_logits, q_logits, eps=1e-8):
    """Numerically stable JS divergence between two logit distributions."""
    p_prob = F.softmax(p_logits, dim=-1)
    q_prob = F.softmax(q_logits, dim=-1)
    # Avoid zeros
    p_prob = p_prob.clamp(min=eps)
    q_prob = q_prob.clamp(min=eps)
    m = 0.5 * (p_prob + q_prob)
    m = m.clamp(min=eps)
    kl_pm = (p_prob * (p_prob.log() - m.log())).sum(dim=-1)
    kl_qm = (q_prob * (q_prob.log() - m.log())).sum(dim=-1)
    js = 0.5 * (kl_pm + kl_qm)
    return js.mean().item()
def logit_lens_logits(hidden_states, model):
    lm_head = model.lm_head
    return hidden_states @ lm_head.weight.T
# ============================================================
#  Dynamic Hook Creator
# ============================================================
def create_dynamic_hooks(
    source_idx,
    target_idx,
    img_indices,
    alpha=1.0,
    replace_fraction=1.0,
    every_n=1,
    reorder_text_first=False
):
    """
    Dynamic hook — supports either hidden-state replacement or reordering.
    """
    hooks = {"source_hidden_states": None, "captured": False}
    def source_hook(module, input, output):
        if hooks["captured"]:
            return
        hs = output[0] if isinstance(output, tuple) else output
        hooks["source_hidden_states"] = hs[img_indices].detach().clone()
        hooks["captured"] = True
    def target_hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        if reorder_text_first:
            # Reorder: text tokens first, image tokens after
            return output
        else:
            # Token replacement logic
            if hooks["source_hidden_states"] is None:
                return output
            batch_idx, seq_idx = img_indices
            src_emb = hooks["source_hidden_states"]
            total = len(batch_idx)
            take_n = max(1, int(total * replace_fraction))
            sel_idxs = list(range(0, total, every_n))[:take_n]
            seq_len = hs.shape[1]
            max_pos = seq_idx.max().item()
            if seq_len <= max_pos:
                return output
            for i in sel_idxs:
                b = batch_idx[i].item()
                s = seq_idx[i].item()
                if s >= seq_len:
                    continue
                orig = hs[b, s, :]
                src = src_emb[i]
                hs[b, s, :] = (1.0 - alpha) * orig + alpha * src
        return (hs,) + output[1:] if is_tuple else hs
    return source_hook, target_hook, hooks
# ============================================================
#  Capture Hidden States
# ============================================================
def capture_hidden_states(model, inputs, layers: List[int]) -> Dict[str, torch.Tensor]:
    captured, handles = {}, []
    def hook_fn(name):
        def fn(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[name] = hs.detach().clone()
        return fn
    for idx in layers:
        layer = model.model.language_model.layers[idx]
        handles.append(layer.register_forward_hook(hook_fn(f"layer_{idx}")))
    with torch.no_grad():
        _ = model(**inputs)
    for h in handles:
        h.remove()
    return captured
# ============================================================
#  Token Selection Utilities
# ============================================================
def find_token_indices(inputs, processor, token_type="image"):
    """
    Return indices (batch_idx, seq_idx) for tokens to perturb.
    """
    input_ids = inputs.input_ids
    image_token_id = IMG_TOKEN_INDEX
    if token_type == "image":
        mask = (input_ids == image_token_id)
    elif token_type == "text":
        mask = (input_ids != image_token_id)
    elif isinstance(token_type, (list, tuple, torch.Tensor)):
        token_type = torch.tensor(token_type, device=input_ids.device)
        mask = torch.isin(input_ids, token_type)
    else:
        raise ValueError(f"Unsupported token_type: {token_type}")
    return torch.where(mask)
# ============================================================
#  Image Loading Utilities
# ============================================================
def load_blink_image(image_data):
    """
    Load image from BLINK-style data (base64 encoded dict or file path).
    """
    from PIL import Image
    from io import BytesIO
    if isinstance(image_data, dict):
        # BLINK format: {"bytes": b"...", "path": "..."}
        if "bytes" in image_data and image_data["bytes"]:
            try:
                # Handle bytes object directly (not base64)
                img_bytes = image_data["bytes"]
                if isinstance(img_bytes, str):
                    # If it's a string representation, it's an error
                    raise ValueError("Bytes should be bytes object, not string")
                img_io = BytesIO(img_bytes)
                img_io.seek(0)
                return Image.open(img_io).convert("RGB")
            except Exception as e:
                # If bytes fail, try path
                if "path" in image_data and image_data["path"]:
                    return Image.open(image_data["path"]).convert("RGB")
                raise e
        elif "path" in image_data and image_data["path"]:
            return Image.open(image_data["path"]).convert("RGB")
    elif isinstance(image_data, str):
        # Direct file path
        return Image.open(image_data).convert("RGB")
    
    raise ValueError(f"Unsupported image data format: {type(image_data)}")
# ============================================================
#  Model Input Preparation
# ============================================================
def prepare_inputs_for_model(model, processor, image, prompt, device=None):
    """
    Universal model input adapter.
    Handles model-specific preprocessing logic.
    """
    device = device or getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")
    model_name = getattr(model.config, "_name_or_path", "").lower()
    if "llava" in model_name:
        # LLaVA-family models
        prompt = f"USER:<image>{prompt}\nASSISTANT:"
        return processor(text=prompt, images=image, return_tensors="pt").to(device)
    elif "blip" in model_name:
        # BLIP / InstructBLIP-style models
        return processor(images=image, text=prompt, return_tensors="pt").to(device)
    
    elif "qwen" in model_name:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        return processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(device)
    elif "opengvlab" in model_name or "internvl" in model_name:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        return processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(device)
    else:
        # Default fallback
        try:
            return processor(text=prompt, images=image, return_tensors="pt").to(device)
        except Exception:
            raise NotImplementedError(
                f"Unknown model type: cannot automatically prepare inputs for {model_name}"
            )
# ============================================================
#  Exhaustive Layer Sweep
# ============================================================
def run_exhaustive_layer_sweep(
    model,
    processor,
    inputs,
    token_type="image",
    alpha_values=(1.0,),
    step=1,
    min_gap=8,
    max_gap=None,
    far_only=False,
    save_dir="results_exhaustive"
):
    """
    Systematically explore all (source, target) layer combinations.
    """
    num_layers = len(model.model.language_model.layers)
    os.makedirs(save_dir, exist_ok=True)
    token_indices = find_token_indices(inputs, processor, token_type=token_type)
    print(f"🔍 Found {len(token_indices[0])} {token_type} tokens.")
    # Compute all valid (source, target) pairs with constraints
    layer_pairs = []
    for s in range(0, num_layers, step):
        for t in range(0, num_layers, step):
            if s >= t:
                continue
            gap = t - s
            if gap < min_gap:
                continue
            if max_gap and gap > max_gap:
                continue
            layer_pairs.append((s, t, gap))
    # Optionally, select only farthest 20% pairs
    if far_only and layer_pairs:
        sorted_pairs = sorted(layer_pairs, key=lambda x: x[2], reverse=True)
        keep_n = max(1, len(sorted_pairs) // 5)
        layer_pairs = sorted_pairs[:keep_n]
        print(f"🧠 Using only farthest {keep_n} layer pairs (top 20%)")
    print(f"🎯 Total valid (source→target) pairs: {len(layer_pairs)}")
    all_records = []
    out_json = os.path.join(save_dir, f"exhaustive_{token_type}_gap{min_gap}.json")
    out_csv = os.path.join(save_dir, f"exhaustive_{token_type}_gap{min_gap}.csv")
    
    for s, t, gap in layer_pairs:
        for a in alpha_values:
            print(f"\n=== Source {s} → Target {t} (gap={gap}) | α={a} | tokens={token_type} ===")
            try:
                source_hook, target_hook, _ = create_dynamic_hooks(
                    s, t, token_indices, alpha=a, reorder_text_first=False
                )
                src_layer = model.model.language_model.layers[s]
                tgt_layer = model.model.language_model.layers[t]
                src_handle = src_layer.register_forward_hook(source_hook)
                tgt_handle = tgt_layer.register_forward_hook(target_hook)
                # Capture baseline vs hooked
                baseline_hidden = capture_hidden_states(model, inputs, [t])
                hooked_hidden = capture_hidden_states(model, inputs, [t])
                src_handle.remove()
                tgt_handle.remove()
                # Metrics
                h_base = baseline_hidden[f"layer_{t}"][:, -1, :]
                h_hook = hooked_hidden[f"layer_{t}"][:, -1, :]
                cos_sim = F.cosine_similarity(h_base, h_hook, dim=-1).mean().item()
                base_logits = logit_lens_logits(h_base, model)
                hook_logits = logit_lens_logits(h_hook, model)
                kl = kl_divergence_from_logits(base_logits, hook_logits)
                js = js_divergence_from_logits(base_logits, hook_logits)
                # Generate baseline & hooked outputs
                with torch.no_grad():
                    baseline_out = model.generate(**inputs, max_new_tokens=50)
                baseline_text = processor.tokenizer.decode(baseline_out[0], skip_special_tokens=True).lower().split('assistant')[-1].replace(':', '').replace('(', '').replace(')', '').strip()
                src_handle = src_layer.register_forward_hook(source_hook)
                tgt_handle = tgt_layer.register_forward_hook(target_hook)
                with torch.no_grad():
                    hooked_out = model.generate(**inputs, max_new_tokens=50)
                src_handle.remove()
                tgt_handle.remove()
                hooked_text = processor.tokenizer.decode(hooked_out[0], skip_special_tokens=True).lower().split('assistant')[-1].replace(':', '').replace('(', '').replace(')', '').strip()
                changed = baseline_text.strip() != hooked_text.strip()
                
                # Compute additional similarity metrics
                exact_match = compute_exact_match(baseline_text, hooked_text)
                semantic_sim = compute_semantic_similarity(baseline_text, hooked_text)
                record = {
                    "source": s,
                    "target": t,
                    "gap": gap,
                    "alpha": a,
                    "token_type": token_type,
                    "cosine_hidden": cos_sim,
                    "exact_match": exact_match,
                    "semantic_similarity": semantic_sim,
                    "kl": kl,
                    "js": js,
                    "changed": changed,
                    "baseline_text": baseline_text,
                    "hooked_text": hooked_text
                }
                all_records.append(record)
                
                # Print metrics
                print(f"   Cosine (hidden): {cos_sim:.4f}")
                print(f"   Exact match: {exact_match:.4f}")
                if semantic_sim is not None:
                    print(f"   Semantic sim: {semantic_sim:.4f}")
                print(f"   KL: {kl:.4e}, JS: {js:.4e}")
                print(f"   Changed: {changed}")
                
                # Save incrementally after each result (both JSON and CSV)
                with open(out_json, 'w') as f:
                    json.dump(all_records, f, indent=2)
                
                df = pd.DataFrame(all_records)
                df.to_csv(out_csv, index=False)
                print(f"💾 Saved progress: {len(all_records)} results to {out_json} and {out_csv}")
            except Exception as e:
                print(f"⚠️ Skipped ({s},{t}) due to error: {e}")
                continue
    # Final save
    with open(out_json, 'w') as f:
        json.dump(all_records, f, indent=2)
    
    df = pd.DataFrame(all_records)
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Completed! Saved {len(all_records)} results to:")
    print(f"   JSON: {out_json}")
    print(f"   CSV:  {out_csv}")
    return df
# ============================================================
#  CSV Processing Pipeline
# ============================================================
def run_exhaustive_for_csv(
    model,
    processor,
    df,
    image_col="image",
    question_col="question",
    out_root="results_exhaustive_csv",
    token_types=("image", "text"),
    alpha_values=(1.0,),
    step=4,
    prep_fn=None,
):
    """
    Generic CSV pipeline to run exhaustive layer perturbation experiments.
    """
    if prep_fn is None:
        prep_fn = prepare_inputs_for_model
    
    os.makedirs(out_root, exist_ok=True)
    master_records = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Exhaustive CSV Sweep"):
        # Load image
        raw_img = row[image_col]
        prompt = "Describe the image."
        try:
            # Parse string representation of dict if needed
            if isinstance(raw_img, str):
                # Try to parse as dict literal
                try:
                    import ast
                    raw_img = ast.literal_eval(raw_img)
                except (ValueError, SyntaxError):
                    # It's a file path
                    pass
            
            # Try BLINK-style decode first, fallback to direct path
            if isinstance(raw_img, dict):
                image = load_blink_image(raw_img)
            elif isinstance(raw_img, str):
                image = Image.open(raw_img).convert("RGB")
            else:
                image = Image.open(raw_img).convert("RGB")
        except Exception as e:
            print(f"⚠️ Skipping row {idx} due to image error: {e}")
            continue
        # Prepare model inputs
        try:
            inputs = prep_fn(model, processor, image, prompt)
        except Exception as e:
            print(f"⚠️ Skipping row {idx} (prep_fn failed): {e}")
            import traceback
            traceback.print_exc()
            continue
        # Per-row results folder
        row_dir = os.path.join(out_root, f"example_{idx:04d}")
        os.makedirs(row_dir, exist_ok=True)
        print(f"\n🧩 Running exhaustive sweeps for example {idx} | Prompt: {prompt[:60]}...")
        for token_type in token_types:
            try:
                df_result = run_exhaustive_layer_sweep(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    token_type=token_type,
                    alpha_values=alpha_values,
                    step=step,
                    min_gap=10,
                    save_dir=row_dir
                )
                df_result["example_id"] = idx
                df_result["token_type"] = token_type
                df_result["prompt"] = prompt
                master_records.append(df_result)
            except Exception as e:
                print(f"❌ Error on token_type={token_type} for row {idx}: {e}")
    # Merge & save master CSV
    if master_records:
        df_master = pd.concat(master_records, ignore_index=True)
        out_csv = os.path.join(out_root, "exhaustive_master.csv")
        df_master.to_csv(out_csv, index=False)
        print(f"\n✅ All results saved → {out_csv}")
        return df_master
    else:
        print("⚠️ No valid runs completed.")
        return pd.DataFrame()
# ============================================================
#  CLI Argument Parser
# ============================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run geometric layer sweep analysis on vision-language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--model_id',
        type=str,
        required=True,
        help='HuggingFace model identifier (e.g., llava-hf/llava-1.5-7b-hf)'
    )
    
    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path to CSV file containing image data and questions'
    )
    
    parser.add_argument(
        '--out_root',
        type=str,
        required=True,
        help='Root directory for saving results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='Directory to cache downloaded models (optional)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to run model on (e.g., cuda:0, cuda:1, cpu)'
    )
    
    parser.add_argument(
        '--image_col',
        type=str,
        default='image_1',
        help='Column name in CSV containing image data'
    )
    
    parser.add_argument(
        '--question_col',
        type=str,
        default='question',
        help='Column name in CSV containing questions/prompts'
    )
    
    parser.add_argument(
        '--token_types',
        type=str,
        nargs='+',
        default=['image', 'text'],
        help='Token types to analyze (space-separated)'
    )
    
    parser.add_argument(
        '--alpha_values',
        type=float,
        nargs='+',
        default=[1.0],
        help='Alpha values for perturbation strength (space-separated)'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        default=2,
        help='Layer step size for source/target sweep'
    )
    
    parser.add_argument(
        '--bnb_4bit',
        action='store_true',
        default=True,
        help='Use 4-bit quantization (default: True)'
    )
    
    parser.add_argument(
        '--no_bnb_4bit',
        action='store_false',
        dest='bnb_4bit',
        help='Disable 4-bit quantization'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of examples to process (for testing)'
    )
    
    parser.add_argument(
        '--img_token_index',
        type=int,
        default=None,
        help='Image token index for the model (auto-detected if not provided)'
    )
    
    return parser.parse_args()
# ============================================================
#  Main Execution
# ============================================================
def main():
    """Main execution function."""
    global IMG_TOKEN_INDEX
    
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv_path):
        print(f"❌ Error: CSV file not found: {args.csv_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.out_root, exist_ok=True)
    
    print("=" * 80)
    print("🚀 Geometric Layer Sweep Analysis")
    print("=" * 80)
    print(f"Model ID:      {args.model_id}")
    print(f"CSV Path:      {args.csv_path}")
    print(f"Output Root:   {args.out_root}")
    print(f"Cache Dir:     {args.cache_dir or 'Default'}")
    print(f"Device:        {args.device}")
    print(f"Token Types:   {', '.join(args.token_types)}")
    print(f"Alpha Values:  {args.alpha_values}")
    print(f"Step Size:     {args.step}")
    print(f"4-bit Quant:   {args.bnb_4bit}")
    print("=" * 80)
    
    # Load dataset
    print(f"\n📊 Loading dataset from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    print(f"✅ Loaded {len(df)} examples")
    
    # Apply limit if specified
    if args.limit:
        df = df.head(args.limit)
        print(f"⚠️  Limited to {args.limit} examples for testing")
    
    # Load model and processor
    print(f"\n🤖 Loading model: {args.model_id}...")
    try:
        model, processor = setup_model(
            model_id=args.model_id,
            bnb_4bit=args.bnb_4bit,
            cache_dir=args.cache_dir,
            device=args.device
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Set image token index
    if args.img_token_index:
        IMG_TOKEN_INDEX = args.img_token_index
    else:
        # Auto-detect based on model
        if 'llava' in args.model_id.lower():
            IMG_TOKEN_INDEX = 32000
        elif 'qwen' in args.model_id.lower():
            IMG_TOKEN_INDEX = 151655
        elif 'opengvlab' in args.model_id.lower():
            IMG_TOKEN_INDEX = 151667
        else:
            print("⚠️  Warning: Could not auto-detect image token index. Using 32000 as default.")
            IMG_TOKEN_INDEX = 32000
    
    print(f"📌 Using image token index: {IMG_TOKEN_INDEX}")
    
    # Run exhaustive layer sweep analysis
    print(f"\n🔬 Running exhaustive layer sweep analysis...")
    try:
        master_results = run_exhaustive_for_csv(
            model=model,
            processor=processor,
            df=df,
            image_col=args.image_col,
            question_col=args.question_col,
            out_root=args.out_root,
            token_types=tuple(args.token_types),
            alpha_values=tuple(args.alpha_values),
            step=args.step,
            prep_fn=prepare_inputs_for_model,
        )
        
        print("\n" + "=" * 80)
        print("✅ Analysis complete!")
        print(f"📁 Results saved to: {args.out_root}")
        print("=" * 80)
        
        # Print summary statistics if results available
        if not master_results.empty:
            print(f"\n📈 Summary:")
            print(f"   Total experiments: {len(master_results)}")
            print(f"   Examples processed: {master_results['example_id'].nunique()}")
            print(f"   Token types: {master_results['token_type'].unique().tolist()}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == "__main__":
    main()
# Made with Bob
