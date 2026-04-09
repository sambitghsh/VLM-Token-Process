#!/usr/bin/env python3
import json
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from transformers import AutoProcessor
import time
from qwen_2_5_vlm.modelling_qwen25 import Qwen2_5_VLForConditionalGeneration
from llava_1_5_vlm.modelling_llava import LlavaForConditionalGeneration
import torch, gc
torch.cuda.empty_cache()
gc.collect()


# ------------------------------------------------
# Model loading
# ------------------------------------------------
def load_model(
    model_id: str | None,
    model_path: str | None,
    prune_after_layer: int | None,
    device: str | None,
    cache_dir: str | None
):
    """
    model_path:
        None           → base HF model
        path/to/model  → finetuned model directory (LoRA adapter)
    prune_after_layer:
        None → no pruning
        int  → enable pruning
    """

    # Always load processor from base model
    processor = AutoProcessor.from_pretrained(model_id)


    if model_path is None:
        print("🔹 Loading base model")
        if "Qwen" in model_id:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=device,
                cache_dir = cache_dir
            )
        elif "llava" in model_id:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=device,
                cache_dir = cache_dir
            )

        from pdb import set_trace as st
        print(model)
    else:
        print(f"🔹 Loading finetuned model from {model_path}")
        # Load base model first
        if "Qwen" in model_id:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=device,
                cache_dir = cache_dir

            )
        elif "llava" in model_id:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=device,
                cache_dir = cache_dir
            )

        # Then load LoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge adapter weights into base model

    model.eval()

    # --------------------------------------------
    # Enable / disable pruning
    # --------------------------------------------
    if prune_after_layer is not None:
        image_token_id = model.config.image_token_id
        print(f"✂️  Pruning enabled after layer {prune_after_layer}")
        model.set_pruning_params(
            prune_after_layer=prune_after_layer,
            prune_token_id=image_token_id,
        )
    else:
        print("✅ No pruning (full model)")
        model.set_pruning_params(None, None)

    return model, processor

# ------------------------------------------------
# Input preparation
# ------------------------------------------------
def prepare_inputs(processor, image_path: str, question: str, device):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    return inputs


# ------------------------------------------------
# Generation
# ------------------------------------------------
@torch.no_grad()
def generate_answer(
    model,
    processor,
    inputs,
    num_return_sequences=5,
    temperature=0.7,
    top_p=0.9,
):
    results = {}

    # ---------- GREEDY ----------
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()

    greedy_outputs = model.generate(
        **inputs,
        max_new_tokens=1000,
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
        use_cache=True,
    )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    greedy_time = time.perf_counter() - t0

    prompt_len = inputs["input_ids"].shape[1]
    greedy_ids = greedy_outputs[:, prompt_len:]

    greedy_text = processor.batch_decode(
        greedy_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip().split('<eos>')[0].replace('<eos>', '').strip()

    results["greedy"] = {
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 1000,
        },
        "outputs": [greedy_text],
        "time_seconds": greedy_time,
    }

    # ---------- SAMPLING ----------
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()

    sampled_outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        use_cache=True,
    )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    sampling_time = time.perf_counter() - t0

    sampled_ids = sampled_outputs[:, prompt_len:]

    sampled_texts = processor.batch_decode(
        sampled_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    sampled_texts = [t.strip().split('<eos>')[0].replace('<eos>', '').strip() for t in sampled_texts]

    results["sampling"] = {
        "decoding": {
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "num_return_sequences": num_return_sequences,
            "max_new_tokens": 1000,
        },
        "outputs": sampled_texts,
        "time_seconds": sampling_time,
    }

    # ---------- TOTAL ----------
    results["total_time_seconds"] = greedy_time + sampling_time

    return results





def run_inference(
    model_id: str,
    test_json: str,
    output_json: str,
    model_path: str | None,
    prune_after_layer: int | None,
    device: str | None,
    cache_dir: str | None
):
    # -------- Load data once --------
    with open(test_json, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # ============================================================
    # PASS 1: BASE MODEL (PRUNED)
    # ============================================================
    print("\n==============================")
    print("🚀 Running BASE pruned model")
    print("==============================")

    # def load_model(
    # model_id: str | None,
    # model_path: str | None,
    # prune_after_layer: int | None,
    # device: str | None,
 

    base_model, processor = load_model(
        model_id,
        model_path=None,
        prune_after_layer=prune_after_layer,
        device=device,
        cache_dir=cache_dir
    )
    device = next(base_model.parameters()).device

    base_answers = []

    for sample in tqdm(data, desc="Base model inference"):
        inputs = prepare_inputs(
            processor,
            sample["image"],
            sample["question"],
            device,
        )
        answer = generate_answer(base_model, processor, inputs)
        base_answers.append(answer)

    # -------- Free GPU memory --------
    del base_model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ============================================================
    # PASS 2: FINETUNED MODEL (PRUNED)
    # ============================================================
    print("\n==============================")
    print("🚀 Running FINETUNED pruned model")
    print("==============================")

    ft_model, _ = load_model(
        model_id=model_id,
        model_path=model_path,
        prune_after_layer=prune_after_layer,
        device=device,
        cache_dir=cache_dir
    )
    device = next(ft_model.parameters()).device

    ft_answers = []

    import time
    import csv

    start_time = time.time()

    for sample in tqdm(data, desc="Finetuned model inference"):
        inputs = prepare_inputs(
            processor,
            sample["image"],
            sample["question"],
            device,
        )
        answer = generate_answer(ft_model, processor, inputs)
        ft_answers.append(answer)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    elapsed = end_time - start_time

    log_path = Path(output_json).parent / "inference_time.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["prune_after_layer", "seconds", "minutes"])
        writer.writerow([prune_after_layer, elapsed, elapsed / 60])


    del ft_model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    
    # ============================================================
    # MERGE RESULTS
    # ============================================================
    results = []

    for idx, sample in enumerate(data):
        results.append(
            {
                "idx": idx,
                "image": sample["image"],
                "question": sample["question"],
                "ground_truth": sample["answer"],
                "base_pruned_answer": base_answers[idx],
                "finetuned_pruned_answer": ft_answers[idx],
                "prune_after_layer": prune_after_layer,
                "model_path": model_path or "base",
            }
        )

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved results to {output_json}")


import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-VL inference with optional pruning and LoRA finetuning"
    )

    # ----------------------------
    # Data paths
    # ----------------------------
    parser.add_argument(
        "--test_json",
        type=str,
        required=True,
        help="Path to test JSONL file"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to save output JSON"
    )

    # ----------------------------
    # Model configuration
    # ----------------------------
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HF model ID for base model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to finetuned LoRA adapter (optional)"
    )

    # ----------------------------
    # Pruning
    # ----------------------------
    parser.add_argument(
        "--prune_after_layer",
        type=int,
        default=None,
        help="Enable pruning after this layer index (optional)"
    )

    # ----------------------------
    # Runtime
    # ----------------------------
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device or device_map (e.g. cuda, cuda:0, auto)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HF cache directory (optional)"
    )

    return parser.parse_args()


def main():
    args = parse_args()


    run_inference(
        model_id=args.model_id,
        test_json=args.test_json,
        output_json=args.output_json,
        model_path=args.model_path,
        prune_after_layer=args.prune_after_layer,
        device=args.device,
        cache_dir=args.cache_dir
    )


if __name__ == "__main__":
    main()

# # ------------------------------------------------
# # Main evaluation loop
# # ------------------------------------------------
# def run_inference(
#     test_json: str,
#     output_json: str,
#     model_path: str | None,
#     prune_after_layer: int | None,
# ):
#     model, processor = load_model(model_path, prune_after_layer)
#     device = next(model.parameters()).device

#     with open(test_json, "r") as f:
#         data = json.load(f)

#     results = []

#     for idx, sample in enumerate(tqdm(data, desc="Running inference")):
#         image_path = sample["image"]
#         question = sample["question"]
#         ground_truth = sample["answer"]

#         inputs = prepare_inputs(processor, image_path, question, device)
#         answer = generate_answer(model, processor, inputs)

#         results.append(
#             {
#                 "idx": idx,
#                 "image": image_path,
#                 "question": question,
#                 "answer": answer,
#                 "ground_truth": ground_truth,
#                 "prune_after_layer": prune_after_layer,
#                 "model_path": model_path or "base",
#             }
#         )

#     Path(output_json).parent.mkdir(parents=True, exist_ok=True)
#     with open(output_json, "w") as f:
#         json.dump(results, f, indent=2)

#     print(f"✅ Saved results to {output_json}")


# # ------------------------------------------------
# # Entry point
# # ------------------------------------------------
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--test_json", required=True)
#     parser.add_argument("--output_json", required=True)

#     parser.add_argument(
#         "--model_path",
#         type=str,
#         default=None,
#         help="Path to finetuned model (omit for base model)",
#     )
#     parser.add_argument(
#         "--prune_after_layer",
#         type=int,
#         default=5,
#         help="Layer index to prune visual tokens after",
#     )

#     args = parser.parse_args()
#     # print("=="*100)
#     # print(args.prune_after_layer)
#     # print("=="*100)
#     run_inference(
#         test_json=args.test_json,
#         output_json=args.output_json,
#         model_path=args.model_path,
#         prune_after_layer=args.prune_after_layer,
#     )
