import os
import math
import warnings
import random
import json
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from dadapy.data import Data as ID_DATA
import matrix_itl as itl
import os
from PIL import Image
from io import BytesIO
import ast
import argparse
from pdb import set_trace 

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = ""
os.environ["HF_HUB_CACHE"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""
os.environ["HF_HUB_DISABLE_XET"] = "1"


from transformers import AutoModel, AutoProcessor, Qwen3VLForConditionalGeneration, LlavaForConditionalGeneration, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText
from transformers import Mistral3ForConditionalGeneration, FineGrainedFP8Config, Gemma3ForConditionalGeneration, AutoModelForCausalLM
#from internvl_utils import load_image

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Setup ---
random.seed(42)
warnings.filterwarnings("ignore")
sns.set()

def load_model(model_name, device, cache_dir):
    MODEL_ID = model_name
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    if "Qwen3-VL" in MODEL_ID:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
                cache_dir = cache_dir,
                device_map=device
            )

    if "Qwen2-VL" in MODEL_ID:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
                cache_dir = cache_dir,
                device_map=device
            )
    elif "Qwen2.5-VL" in MODEL_ID:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
                cache_dir = cache_dir,
                device_map=device
            )

    elif "InternVL" in MODEL_ID:
        model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
                cache_dir = cache_dir,
                device_map=device,
                trust_remote_code=True
            )

    elif "Phi" in MODEL_ID:
        model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
                cache_dir = cache_dir,
                device_map=device,
                trust_remote_code=True
            )

    elif "gemma" in MODEL_ID:
        model = Gemma3ForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
                cache_dir = cache_dir,
                device_map=device,
                token = ""
            )
    
    elif "Pixtral" in MODEL_ID:
        model = Mistral3ForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
                cache_dir = cache_dir,
                device_map=device,
                trust_remote_code=True
            )
        
    elif "llava" in MODEL_ID:
        model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
                cache_dir = cache_dir,
                device_map=device
            )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir = cache_dir, token = "", trust_remote_code=True)
    return model, processor



# --- Utility functions ---
def normalize(R):
    with torch.no_grad():
        mean = R.mean(dim=0)
        R = R - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R / norms
    return R

def input_hidden_states_fetch(model, inputs):
    with torch.no_grad():
        forward_out = model(**inputs, output_hidden_states=True)
        hidden_states = [h.cpu() for h in forward_out.hidden_states]
    return hidden_states

def entropy_normalization(entropy, normalization, N, D):
    assert normalization in ['maxEntropy', 'logN', 'logD', 'logNlogD', 'raw', 'length']
    if normalization == 'maxEntropy':
        #print(N, D)
        entropy /= min(math.log(N), math.log(D))
    elif normalization == 'logN':
        entropy /= math.log(N)
    elif normalization == 'logD':
        entropy /= math.log(D)
    elif normalization == 'logNlogD':
        entropy /= (math.log(N) * math.log(D))
    elif normalization == 'raw':
        pass
    elif normalization == 'length':
        entropy = N
    return entropy

def compute_entropy(hidden_states, alpha=1, normalizations=['maxEntropy']):
    L, N, D = hidden_states.shape
    if N > D:
        cov = torch.matmul(hidden_states.transpose(1, 2), hidden_states)
    else:
        cov = torch.matmul(hidden_states, hidden_states.transpose(1, 2))
    cov = torch.clamp(cov, min=0)
    entropies = []
    for layer_cov in cov:
        try:
            layer_cov = layer_cov.double() / torch.trace(layer_cov.double())
            entropies.append(itl.matrixAlphaEntropy(layer_cov, alpha=alpha).item())
        except Exception:
            entropies.append(np.nan)
    return {norm: [entropy_normalization(x, norm, N, D) for x in entropies] for norm in normalizations}

def compute_intrinsic_dimension(hidden_states, nn=2, skip_first=True):
    intrinsic_dimensions = []
    normalized_intrinsic_dimensions = []

    for layer_num, layer in enumerate(hidden_states):
        # if skip_first and layer_num == 0:
        #     pass  # intentionally kept as pass

        layer = layer.detach().float().cpu().numpy()
        layer = layer.reshape(layer.shape[0], -1)
        data = ID_DATA(layer)
        id_est, id_error, id_distance = data.compute_id_2NN()

        intrinsic_dimensions.append(id_est)
        normalized_intrinsic_dimensions.append(id_est / math.log(layer.shape[0]))
    return {'raw': intrinsic_dimensions, 'logN': normalized_intrinsic_dimensions}

def compute_curvature(hidden_states, k=1):
    L, N, D = hidden_states.shape

    def calculate_paired_curvature(a, b):
        dotproduct = torch.abs(a.T @ b)
        norm_a = torch.norm(a)
        norm_b = torch.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0
        argument = torch.clamp(dotproduct / (norm_a * norm_b), min=-1, max=1)
        curvature = torch.arccos(argument)
        return curvature.item()

    def calculate_layer_average_k_curvature(layer_p):
        summation, counter = 0, 0
        for k in range(1, layer_p.shape[0]-1):
            v_k = layer_p[k].unsqueeze(1) - layer_p[k-1].unsqueeze(1)
            v_kplusone = layer_p[k+1].unsqueeze(1) - layer_p[k].unsqueeze(1)
            curvature = calculate_paired_curvature(v_kplusone, v_k)
            summation += curvature
            counter += 1
        return summation / counter if counter > 0 else 0

    curvatures = [calculate_layer_average_k_curvature(layer.double()) for layer in hidden_states]
    return {'raw': curvatures, 'logD': [x / math.log(D) for x in curvatures]}
        
def safe_id(hidden_states):
    try:
        out = compute_intrinsic_dimension(hidden_states)
        val = out['raw'][0]
        if np.isnan(val) or np.isinf(val):
            return np.nan
        return val
    except Exception:
        return np.nan
    
def get_text_vision_special_indices(input_ids, tokenizer, image_token_id):
    # if isinstance(input_ids, torch.Tensor):
    #     input_ids = input_ids.tolist()

    # Get special token IDs
    special_ids = set(tokenizer.all_special_ids)

    text_indices = []
    vision_indices = []
    special_indices = []

    # print(input_ids)
    # print(image_token_id)
    # set_trace()
    vision_indices = torch.where(input_ids[0] == image_token_id)[0].detach().cpu()
    #vision_indices = torch.where(input_ids[0].eq(image_token_id))[0].detach().cpu()

    for pos, tok_id in enumerate(input_ids[0]):

        if tok_id in special_ids:
            special_indices.append(pos)
        else:
            if tok_id != image_token_id:
                text_indices.append(pos)

    return text_indices, vision_indices, special_indices

def compute_geometry_metrics(model, processor, inputs):
    input_hidden_states = input_hidden_states_fetch(model, inputs)
    layers = range(len(input_hidden_states))

    image_curv, text_curv = [], []
    image_ent, text_ent = [], []
    image_id, text_id = [], []
    text_patch, image_patch, _ = get_text_vision_special_indices(inputs.input_ids, processor.tokenizer, model.config.image_token_id)

    for hidden_state in input_hidden_states:
        image_hidden_states = hidden_state[0][image_patch]
        text_hidden_states  = hidden_state[0][text_patch]

        norm_image = normalize(image_hidden_states)
        norm_text  = normalize(text_hidden_states)

        image_curv.append(compute_curvature(norm_image.unsqueeze(0))['raw'][0])
        text_curv.append(compute_curvature(norm_text.unsqueeze(0))['raw'][0])

        image_ent.append(compute_entropy(norm_image.unsqueeze(0))['maxEntropy'][0])
        text_ent.append(compute_entropy(norm_text.unsqueeze(0))['maxEntropy'][0])

        image_id.append(safe_id(norm_image.unsqueeze(0)))
        text_id.append(safe_id(norm_text.unsqueeze(0)))

    return {
        "layers": list(layers),
        "image": {"curv": image_curv, "ent": image_ent, "id": image_id},
        "text":  {"curv": text_curv,  "ent": text_ent,  "id": text_id}
    }

# --- Plotting functions ---
def plot_geometry_metrics(metrics, save_path="geometry_metrics_over_layers.png"):
    layers = metrics["layers"]
    image_curv, text_curv = metrics["image"]["curv"], metrics["text"]["curv"]
    image_ent, text_ent   = metrics["image"]["ent"],  metrics["text"]["ent"]
    image_id, text_id     = metrics["image"]["id"],   metrics["text"]["id"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(15,4), sharex=True)
    axs[0].plot(layers, image_curv, marker='o', label="image")
    axs[0].plot(layers, text_curv, marker='x', label="text")
    axs[0].set_title("Curvature")

    axs[1].plot(layers, image_ent, marker='o', label="image")
    axs[1].plot(layers, text_ent, marker='x', label="text")
    axs[1].set_title("Matrix Entropy")

    axs[2].plot(layers, image_id, marker='o', label="image")
    axs[2].plot(layers, text_id, marker='x', label="text")
    axs[2].set_title("Intrinsic Dimension")

    for ax in axs:
        ax.set_xlabel("Layer")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved geometry metrics plot → {save_path}")

def plot_dual_radar(metrics, save_dir="radar_plots"):
    layers = metrics["layers"]
    image_curv, text_curv = metrics["image"]["curv"], metrics["text"]["curv"]
    image_ent, text_ent   = metrics["image"]["ent"],  metrics["text"]["ent"]
    image_id, text_id     = metrics["image"]["id"],   metrics["text"]["id"]

    all_metrics = np.column_stack([image_curv, text_curv, image_ent, text_ent, image_id, text_id])
    scaler = MinMaxScaler()
    all_norm = scaler.fit_transform(np.nan_to_num(all_metrics, nan=0.0))

    os.makedirs(save_dir, exist_ok=True)
    labels = ["Curvature", "Entropy", "Intrinsic Dim"]

    for layer_idx in range(len(layers)):
        metrics_image_norm = [all_norm[layer_idx,0], all_norm[layer_idx,2], all_norm[layer_idx,4]]
        metrics_text_norm  = [all_norm[layer_idx,1], all_norm[layer_idx,3], all_norm[layer_idx,5]]

        num_vars = len(labels)
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
        metrics_image_norm += metrics_image_norm[:1]
        metrics_text_norm  += metrics_text_norm[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        ax.plot(angles, metrics_image_norm, marker='o', label="Image")
        ax.fill(angles, metrics_image_norm, alpha=0.25)

        ax.plot(angles, metrics_text_norm, marker='x', label="Text")
        ax.fill(angles, metrics_text_norm, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title(f"Normalized geometry profile at Layer {layer_idx}", size=14)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

        fname = os.path.join(save_dir, f"radar_layer_{layer_idx}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close(fig)

    print(f"Radar plots saved in: {save_dir}")


def load_blink_image(raw_img):
    """
    Convert BLINK image field (stringified dict or dict with bytes) into a PIL Image.
    """
    if raw_img is None:
        return None
    
    # Case 1: raw_img is a string that looks like a dict
    if isinstance(raw_img, str):
        try:
            raw_img = ast.literal_eval(raw_img)
        except Exception as e:
            raise ValueError(f"Failed to parse image string: {e}")
    
    # Case 2: raw_img is already a dict with bytes
    if isinstance(raw_img, dict) and "bytes" in raw_img:
        return Image.open(BytesIO(raw_img["bytes"]))
    
    raise TypeError(f"Unsupported image format: {type(raw_img)}")

    
def main(model_name, device, save_dir, dataset, cache_dir):
    model, processor = load_model(model_name, device, cache_dir)
    model.eval()
    df = pd.read_csv(dataset)

    results, correct_metrics, incorrect_metrics = run_evaluation(
        model=model,
        processor=processor,
        df=df,
        device=device,
        save_dir=save_dir,
        model_name=model_name
    )
    file_name =  f"{model_name.split('/')[-1].replace('.','_').replace('-','_')}.json"
    save_results(results, f"{save_dir}/{file_name}")


def build_conversation(model_name, prompt_text):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

def prepare_inputs(model_name, processor, prompt_text, conversation, image, device):
    text_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    return inputs.to(device)


@torch.no_grad()
def generate_response(model, processor, inputs, max_new_tokens=50):
    #     pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
    # generation_config = dict(max_new_tokens=1024, do_sample=True)
    # question = '<image>\nPlease describe the image shortly.'
    # response = model.chat(tokenizer, pixel_values, question, generation_config)
    # print(f'User: {question}\nAssistant: {response}')

    gen_out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    return processor.decode(
        gen_out[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

def normalize_answer(text: str):
    return (
        text.replace("(", "")
            .replace(")", "")
            .strip()
            .lower()
    )


def is_correct_answer(gt: str, response: str):
    pred = normalize_answer(response.split("assistant")[-1])
    gt = normalize_answer(gt)
    return gt == pred


def process_example(
    row,
    model,
    processor,
    device,
    save_dir,
    idx,
    model_name
):
    prompt_text = "Describe the image in one line."
    image = load_blink_image(row["image_1"])
    answer = row["answer"]

    conversation = build_conversation(model_name, prompt_text)
    inputs = prepare_inputs(model_name, processor, prompt_text, conversation, image, device)

    response = generate_response(model, processor, inputs)
    correct = is_correct_answer(answer, response)

    metrics = compute_geometry_metrics(model, processor, inputs)

    plot_geometry_metrics(
        metrics,
        save_path=f"{save_dir}/geometry/geometry_metrics_{idx}.png"
    )
    # plot_dual_radar(
    #     metrics,
    #     save_dir=f"{save_dir}/radar_plots/radar_plots_{idx}"
    # )

    return {
        "image_path": row.get("idx", idx),
        "question": row.get("question", ""),
        "ground_truth": answer,
        "model_response": response,
        "is_correct": correct,
        "metrics": metrics,
    }, correct, metrics


def run_evaluation(
    model,
    processor,
    df,
    device,
    save_dir,
    model_name
):
    results = []
    all_metrics_correct = []
    all_metrics_incorrect = []

    for i, row in df.iterrows():
        # if i >= 2:
        #     break

        result, correct, metrics = process_example(
            row=row,
            model=model,
            processor=processor,
            device=device,
            save_dir=save_dir,
            idx=i,
            model_name=model_name
        )

        results.append(result)
        #(all_metrics_correct if correct else all_metrics_incorrect).append(metrics)

        # print("prompt, ground truth ------------")
        # print("Describe the image.", row["answer"])
        # print("response--------------")
        print(result["model_response"].split("assistant")[-1].strip())
        # print(correct)
        # print("=" * 50)

    return results, all_metrics_correct, all_metrics_incorrect


def save_results(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"All responses saved to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VLM evaluation with geometry metrics"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default = "Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model name or path (e.g. llava-hf/llava-1.5-7b-hf)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default = "blink_counting_val.csv",
        help="Path to CSV dataset (e.g. blink_counting_val.csv)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device (default: cuda:0)"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Directory to save outputs"
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Directory to save huggingface cache"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(
        model_name=args.model_name,
        device=args.device,
        save_dir=args.save_dir,
        dataset=args.dataset,
        cache_dir=args.cache_dir
    )



  

