import sys
import gc
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import csv
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from dataset import LlavaVisionSFTDataset
from collator import QwenVLDataCollator, LlavaDataCollator
from qwen_2_5_vlm.modelling_qwen25 import Qwen2_5_VLForConditionalGeneration
from llava_1_5_vlm.modelling_llava import LlavaForConditionalGeneration



# -----------------------------------
# Vision freeze helper
# -----------------------------------
def freeze_vision(model, is_llava=False):
    for name, p in model.named_parameters():
        if is_llava:
            if (
                name.startswith("model.vision_tower")
                or name.startswith("model.multi_modal_projector")
            ):
                p.requires_grad = False
        else:
            if (
                name.startswith("model.vision_tower")
                or name.startswith("model.vision_projector")
                or name.startswith("model.visual")
            ):
                p.requires_grad = False
            elif name.startswith("model.multi_modal_projector"):
                p.requires_grad = False


# -----------------------------------
# Training for a single prune layer
# -----------------------------------
def train_with_prune_layer(args, prune_layer: int):
    print(f"\n{'=' * 80}")
    print(f"Starting training with PRUNE_LAYER={prune_layer}")
    print(f"{'=' * 80}\n")

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        cache_dir=args.model_cache_dir,
    )

    # device_index = int(args.device.split(":")[-1])
    # torch.cuda.set_device(device_index)

    # -----------------------------------
    # 4-bit loading (QLoRA-style)
    # # -----------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    if "llava" in args.model_id:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id,
            cache_dir=args.model_cache_dir,
            quantization_config=bnb_config,
            device_map=args.device,
        )
    
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id,
            cache_dir=args.model_cache_dir,
            quantization_config=bnb_config,
            device_map=args.device,
        )

    # -----------------------------------
    # Enable visual-token pruning
    # -----------------------------------
    is_llava = "llava" in args.model_id
    if hasattr(model, "set_pruning_params") and getattr(model.config, "image_token_id", None) is not None:
        model.set_pruning_params(
            prune_after_layer=prune_layer,
            prune_token_id=model.config.image_token_id,
        )

    # -----------------------------------
    # Freeze vision tower
    # -----------------------------------
    print(model)
    freeze_vision(model, is_llava=is_llava)
    model.enable_input_require_grads()

    # -----------------------------------
    # LoRA (LM only)
    # -----------------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    # -----------------------------------
    # Data
    # -----------------------------------
    train_dataset = LlavaVisionSFTDataset(args.train_json, processor)
    val_dataset = LlavaVisionSFTDataset(args.val_json, processor)
    if "llava" in args.model_id:
        collator = LlavaDataCollator(processor)
    else:
        collator = QwenVLDataCollator(processor)
    

    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    # -----------------------------------
    # Training args
    # -----------------------------------
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/llava-vl-prune_layer_{prune_layer}",
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=True,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        local_rank=-1,
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # -----------------------------
    # Timing start (GPU-accurate)
    # -----------------------------
    torch.cuda.synchronize()
    start_time = time.time()

    trainer.train()

    # -----------------------------
    # Timing end (GPU-accurate)
    # -----------------------------
    torch.cuda.synchronize()
    end_time = time.time()

    elapsed = end_time - start_time

    print(
        f"[PRUNE_LAYER={prune_layer}] "
        f"Training time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)"
    )

    # -----------------------------
    # Persist timing to CSV
    # -----------------------------
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    log_path = Path(args.output_dir) / "training_time.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["prune_layer", "seconds", "minutes"])
        writer.writerow([prune_layer, elapsed, elapsed / 60])

    trainer.save_model()

    print(f"\nCompleted PRUNE_LAYER={prune_layer}")
    print(f"Saved to {training_args.output_dir}")

# Argparse + sweep driver
# -----------------------------------
def main():
    parser = argparse.ArgumentParser()

    # Model / paths
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model_cache_dir", default=None)
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--val_json", required=True)
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--device", default="cuda:2")

    # Pruning sweep
    parser.add_argument("--prune_start", type=int, default=0)
    parser.add_argument("--prune_end", type=int, default=30)
    parser.add_argument("--prune_step", type=int, default=1)

    # Training
    parser.add_argument("--epochs", type=int, default=3) 
    parser.add_argument("--lr", type=float, default=2e-4) 
    parser.add_argument("--train_bs", type=int, default=1) 
    parser.add_argument("--eval_bs", type=int, default=1) 
    parser.add_argument("--grad_accum", type=int, default=8) 
    parser.add_argument("--eval_steps", type=int, default=500) 
    parser.add_argument("--save_steps", type=int, default=500) 
    parser.add_argument("--logging_steps", type=int, default=10) 
    
    # LoRA 
    parser.add_argument("--lora_r", type=int, default=16) 
    parser.add_argument("--lora_alpha", type=int, default=32) 
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    prune_layers = list(range(args.prune_start, args.prune_end, args.prune_step))
    print(f"\nRunning prune-layer sweep: {prune_layers}\n")

    for i, prune_layer in enumerate(prune_layers, 1):
        print(f"\n>>> [{i}/{len(prune_layers)}] PRUNE_LAYER={prune_layer}")
        try:
            train_with_prune_layer(args, prune_layer)
        except Exception as e:
            print(f"\n❌ Failed PRUNE_LAYER={prune_layer}")
            print(e)
        finally:
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
