import torch
import os

class LlavaDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, batch):
        images = []
        full_texts = []
        prompt_texts = []

        for sample in batch:
            text = sample["text"]
            image = sample["image"]

            if "ASSISTANT:" not in text:
                raise ValueError("Missing 'ASSISTANT:' in training sample")

            # ---- split BEFORE tokenization (CRITICAL) ----
            prompt, answer = text.split("ASSISTANT:", 1)
            prompt = prompt + "ASSISTANT:"

            images.append(image)
            full_texts.append(prompt + answer)
            prompt_texts.append(prompt)

        # ---- tokenize full prompt + answer (processor expands <image> to image tokens) ----
        model_inputs = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()

        # ---- get prompt length per sample using processor so <image> is expanded same way ----
        prompt_lengths = []
        for i, (prompt_text, image) in enumerate(zip(prompt_texts, images)):
            out = self.processor(
                text=[prompt_text],
                images=[image],
                padding=False,
                add_special_tokens=True,
                return_tensors="pt",
            )
            prompt_lengths.append(out["input_ids"].shape[1])

        # ---- mask prompt tokens ----
        for i in range(len(labels)):
            labels[i, : prompt_lengths[i]] = -100

        # ---- mask padding ----
        labels[input_ids == self.pad_token_id] = -100
        target_ids = labels[labels != -100]
        decoded_target = self.processor.tokenizer.decode(target_ids)
        print(decoded_target)
        #from pdb import set_trace as st
        #st()
        model_inputs["labels"] = labels
        return model_inputs

class QwenVLDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        
        # Identify the tokens that signal the start of the assistant's response
        # For Qwen2-VL, this is typically '<|im_start|>assistant\n'
        self.assistant_prefix = "<|im_start|>assistant\n"

    def __call__(self, batch):
        images = [b["image"] for b in batch]
        texts = [b["text"] for b in batch]

        # 1. Process full text with images (this expands <image> into many tokens)
        model_inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()

        # 2. Masking logic
        for i in range(len(labels)):
            # Convert IDs back to tokens to find the split point accurately
            # or search for the encoded assistant prefix IDs
            token_ids = input_ids[i].tolist()
            
            # Find the index where the assistant response starts
            # We search for the start of the assistant role
            prefix_ids = self.tokenizer.encode(self.assistant_prefix, add_special_tokens=False)
            
            # Simple sub-list search
            start_idx = None
            for j in range(len(token_ids) - len(prefix_ids)):
                if token_ids[j : j + len(prefix_ids)] == prefix_ids:
                    # We want to mask everything UP TO the end of the prefix
                    start_idx = j + len(prefix_ids)
                    break
            
            if start_idx is not None:
                labels[i, :start_idx] = -100
            else:
                # Fallback: if not found, mask nothing or handle error
                # Usually occurs if the chat template doesn't match
                print(f"Warning: Assistant prefix not found in sample {i}")

        # 3. Mask padding
        labels[input_ids == self.pad_token_id] = -100
        
        # Filter out -100 to see what the model actually "sees" as a target
        target_ids = labels[labels != -100]
        decoded_target = self.processor.tokenizer.decode(target_ids)
        #print(decoded_target)
        from pdb import set_trace as st
        #st()

        model_inputs["labels"] = labels
        return model_inputs

