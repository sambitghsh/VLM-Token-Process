import json
from PIL import Image
from torch.utils.data import Dataset
import os
class QwenVisionSFTDataset(Dataset):
    def __init__(self, json_path, processor):
        with open(json_path, "r") as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        image = Image.open(ex["image"]).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": ex["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ex["answer"]}],
            },
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {
            "image": image,
            "text": text,
            "question": ex["question"]
        }




class LlavaVisionSFTDataset(Dataset):
    def __init__(self, data_source, processor, prompt_template=None):
        """
        Args:
            data_source: Either a path to JSON file or a HuggingFace dataset identifier
            processor: The processor for the model
            prompt_template: Custom prompt template. Default: "USER: <image>\n{question}\nASSISTANT: {answer}"
                            Use {question} and {answer} as placeholders.
        """
        self.processor = processor
        
        # Set default prompt template
        if prompt_template is None:
            prompt_template = "USER: <image>\n{question}\nASSISTANT: {answer}<eos>"
        self.prompt_template = prompt_template
        
        # Load data from HuggingFace or local JSON/JSONL
        if os.path.exists(data_source):
            with open(data_source, "r") as f:
                if data_source.endswith(".jsonl"):
                    self.data = [json.loads(line) for line in f if line.strip()]
                else:
                    self.data = json.load(f)
        else:
            # HuggingFace dataset
            try:
                from datasets import load_dataset
                # Assume format: "dataset_name" or "dataset_name/config_name"
                if "/" in data_source and not data_source.startswith("/"):
                    parts = data_source.split("/", 1)
                    dataset_name = parts[0]
                    config_name = parts[1] if len(parts) > 1 else None
                    if config_name:
                        self.data = load_dataset(dataset_name, config_name, split="train")
                    else:
                        self.data = load_dataset(dataset_name, split="train")
                else:
                    self.data = load_dataset(data_source, split="train")
                # Convert to list for compatibility
                self.data = list(self.data)
            except Exception as e:
                raise ValueError(f"Failed to load dataset from {data_source}. Error: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        # Handle both dict and dataset row formats
        if isinstance(ex, dict):
            image_path = ex.get("image", ex.get("image_path", None))
            question = ex.get("question", ex.get("text", ""))
            answer = ex.get("answer", ex.get("response", ""))
        else:
            # HuggingFace dataset row
            image_path = getattr(ex, "image", None) or getattr(ex, "image_path", None)
            question = getattr(ex, "question", "") or getattr(ex, "text", "")
            answer = getattr(ex, "answer", "") or getattr(ex, "response", "")

        # Load image - handle both paths and PIL Images
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Image.Image):
            image = image_path.convert("RGB")
        else:
            raise ValueError(f"Unexpected image type: {type(image_path)}")

        # Format prompt using template
        prompt = self.prompt_template.format(question=question, answer=answer)

        return {
            "image": image,
            "text": prompt,
        }

