#!/usr/bin/env python3
import json
import torch
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoProcessor, BitsAndBytesConfig
from qwen_2_5_vlm.modelling_qwen25 import Qwen2_5_VLForConditionalGeneration
from llava_1_5_vlm.modelling_llava import LlavaForConditionalGeneration



# ------------------------------------------------
# Input preparation
# ------------------------------------------------
def prepare_inputs(processor, image_path: Path, question: str, device):
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
# Generation (timed)
# ------------------------------------------------
@torch.no_grad()
def generate_answer(model, processor, inputs, max_new_tokens):
    start = time.perf_counter()

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return text, elapsed


# ------------------------------------------------
# Inference over one split (STREAMING)
# ------------------------------------------------
def run_split_streaming(
    model,
    processor,
    split_name: str,
    dataset_root: Path,
    output_dir: Path,
    device,
    max_new_tokens: int,
):
    split_dir = dataset_root / split_name
    images_dir = split_dir / "images"
    data_json = split_dir / "data.json"

    with open(data_json, "r") as f:
        data = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split_name}.jsonl"

    split_start = time.perf_counter()
    num_samples = 0

    with open(out_path, "w") as out_f:
        for sample in tqdm(data, desc=f"Inferencing {split_name}"):
            image_path = images_dir / sample["image_id"]
            question = sample["question"]

            inputs = prepare_inputs(processor, image_path, question, device)
            answer, runtime = generate_answer(model, processor, inputs, max_new_tokens)
            image_path_abs = (images_dir / sample["image_id"]).resolve()
            record = {
                "index": sample["index"],
                "image": str(image_path_abs),  
                "question": question,
                "answer": answer,
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            num_samples += 1

    split_runtime = time.perf_counter() - split_start
    return split_runtime




# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_root", default="inference_outputs")
    parser.add_argument("--max_new_tokens", type=int, default=128)

    args = parser.parse_args()
    device = torch.device(args.device)

    # ------------------------------------------------
    # HF processor
    # ------------------------------------------------
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
    )

    # ------------------------------------------------
    # BitsAndBytes 4-bit config
    # ------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # ------------------------------------------------
    # Model loading
    # ------------------------------------------------
    if "llava" in args.model_name:
        model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": device},
        cache_dir=args.cache_dir,
    )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map={"": device},
            cache_dir=args.cache_dir,
        )

    model.set_pruning_params(prune_after_layer=None, prune_token_id=None)
    model.eval()

    dataset_root = Path(args.dataset_root)
    dataset_name = dataset_root.name
    model_name_clean = args.model_name.replace("/", "_")

    model_out_dir = (
        Path(args.output_root)
        / dataset_name
        / model_name_clean
    )

    for split in ["train" ,"validation", "test"]:
        run_split_streaming(
            model=model,
            processor=processor,
            split_name=split,
            dataset_root=dataset_root,
            output_dir=model_out_dir,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )

    print(f"\n✅ Inference complete")
    print(f"📁 Saved to: {model_out_dir}")


if __name__ == "__main__":
    main()
