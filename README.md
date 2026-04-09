# Do Vision Language Models Need to Process Image Tokens?

VLMs over-process images: visual tokens stabilize early, are depth-redundant, but still critical for multi-step reasoning tasks

### Key Insights
- Visual tokens stabilize early in VLMs, unlike text tokens which evolve across depth  
- Deeper layers add limited new information to visual representations  
- Visual depth is task-dependent: shallow is enough for classification, deep is needed for generation  
- Image tokens shape reasoning paths more than final outputs  
- Fine-tuning can recover truncated models, but not without sufficient visual depth  

## Project Structure
- **`ground_truth_generation.py`** - Generates ground truth outputs from VLMs (Qwen2.5-VL, LLaVA) for evaluation and analysis

### Layer Sweep
- **`layer_sweep/geomertric_layer_sweep.py`** - Performs geometric layer sweep analysis to identify optimal pruning layers using semantic similarity metrics

### Metric Calculation
- **`metric_calculation/uncovering_layers_vlm.py`** - Computes geometry metrics for VLM layers: matrix entropy, curvature, and intrinsic dimension (separately for image and text tokens)

### Models
- **`models/llava_1_5_vlm/modelling_llava.py`** - Modified LLaVA model with visual token pruning after specified layers
- **`models/qwen_2_5_vlm/modelling_qwen25.py`** - Qwen2.5-VL model with visual token pruning capabilities

### Training
- **`train/collator.py`** - Data collators for batching and processing VLM training samples with proper label masking
- **`train/dataset.py`** - Dataset classes for loading and formatting vision-language training data (Qwen, LLaVA)
- **`train/inference.py`** - Inference utilities for running trained/pruned VLM models with custom configurations
- **`train/train.py`** - Training script with LoRA fine-tuning support and vision encoder freezing options
