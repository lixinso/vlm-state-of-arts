# Molmo (Allen Institute for AI)

Molmo is a family of open-source Vision Language Models developed by the Allen Institute for AI (AI2), demonstrating that truly open-source VLMs can compete with proprietary models.

## Overview

Molmo achieves remarkable performance across its model family, with the 72B model outperforming proprietary models like Gemini 1.5 Pro and Claude 3.5 Sonnet on academic benchmarks. Even the smaller 7B and 1B models rival GPT-4V in several tasks.

## Key Differentiators

1. **Fully Open**: Model weights, training data, and code all openly released
2. **Novel Data Collection**: Human-annotated dense descriptions instead of synthetic data
3. **Pointing Capability**: Can point to specific image locations
4. **Strong Small Models**: Competitive performance even at 1B parameters

## Architecture

### Components

| Component | Molmo-72B | Molmo-7B | Molmo-1B |
|-----------|-----------|----------|----------|
| Vision Encoder | ViT | ViT | ViT |
| Language Model | LLaMA-72B | LLaMA-7B | OLMo-1B |
| Connector | Molmo Connector | Molmo Connector | Molmo Connector |
| Context Length | 128K | 128K | 32K |

### Architecture Diagram

```
                    ┌─────────────────────────┐
                    │      Input Image        │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Vision Transformer    │
                    │       (ViT)             │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Molmo Connector       │
                    │   (Custom Projection)   │
                    └───────────┬─────────────┘
                                │
        Text Input ────────────►├
                                │
                    ┌───────────▼─────────────┐
                    │   Language Model        │
                    │   (LLaMA / OLMo)        │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  Text + Point Response  │
                    └─────────────────────────┘
```

## Benchmark Performance

### Comparison with Proprietary Models

| Benchmark | Molmo-72B | GPT-4V | Gemini 1.5 Pro | Claude 3.5 Sonnet |
|-----------|-----------|--------|----------------|-------------------|
| AI2D | 94.8 | 89.4 | 90.2 | 88.1 |
| ChartQA | 87.5 | 85.7 | 86.1 | 84.6 |
| DocVQA | 92.1 | 92.8 | 90.3 | 89.5 |
| InfoVQA | 76.2 | 75.1 | 73.8 | 71.2 |
| RealWorldQA | 68.7 | 68.9 | 67.5 | 66.3 |

### Model Size Comparison

| Benchmark | Molmo-72B | Molmo-7B | Molmo-1B |
|-----------|-----------|----------|----------|
| AI2D | 94.8 | 93.2 | 85.6 |
| VQAv2 | 85.7 | 83.4 | 76.2 |
| TextVQA | 79.3 | 76.8 | 68.1 |

## Unique Features

### 1. Pointing Capability

Molmo can point to specific locations in images:

```
User: Where is the red car?
Molmo: The red car is at [x=0.73, y=0.45] in the image.
```

This enables:
- Precise object localization
- Spatial reasoning verification
- Interactive image exploration

### 2. Dense Captioning Data

Unlike models trained on synthetic data, Molmo uses human-annotated descriptions:

```
Traditional Data:
  "A dog in a park"

Molmo Training Data:
  "A golden retriever with wet fur running through grass,
   its tongue out, approaching the camera from the left side
   of the frame, with tall oak trees visible in the background..."
```

### 3. PixMo Dataset

AI2 created PixMo, a new dataset for training Molmo:
- Human-annotated dense descriptions
- Pointing annotations
- Diverse image sources
- High-quality multimodal conversations

## Training Approach

### Data Collection

1. **Dense Description**: Humans write detailed, exhaustive image descriptions
2. **Pointing Tasks**: Annotators mark specific locations in images
3. **Conversation Data**: Multi-turn dialogues about images
4. **Quality Control**: Multiple rounds of verification

### Training Stages

```
Stage 1: Vision-Language Alignment
├── Train connector on image-caption pairs
└── Freeze vision encoder and LLM

Stage 2: Instruction Tuning
├── Train on PixMo dataset
├── Include pointing tasks
└── Fine-tune full model or LoRA
```

## Model Variants

| Model | Parameters | Best For |
|-------|------------|----------|
| Molmo-72B | 72B | Maximum accuracy |
| Molmo-7B-D | 7B | Balance of speed/accuracy |
| Molmo-7B-O | 7B | OLMo backbone variant |
| Molmo-1B | 1B | Edge deployment |

## Usage

### Installation

```bash
pip install molmo
```

### Basic Usage

```python
from molmo import Molmo

model = Molmo.from_pretrained("allenai/Molmo-72B-0924")

# Basic VQA
response = model.generate(
    image="path/to/image.jpg",
    prompt="Describe this image in detail."
)
print(response)

# Pointing
response = model.generate(
    image="path/to/image.jpg",
    prompt="Point to the person wearing a red shirt."
)
print(response)  # Includes coordinates
```

### Hugging Face

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True
)

inputs = processor(
    text="Describe this image.",
    images=image,
    return_tensors="pt"
)
output = model.generate(**inputs)
```

## Deployment Options

### vLLM

```bash
vllm serve allenai/Molmo-72B-0924 \
    --tensor-parallel-size 4 \
    --trust-remote-code
```

### Ollama

```bash
ollama run molmo:7b
```

## Comparison with Other Open Models

| Feature | Molmo | LLaVA | InternVL | Qwen-VL |
|---------|-------|-------|----------|---------|
| Pointing | ✓ | ✗ | ✗ | ✓ |
| Fully Open Data | ✓ | Partial | ✗ | ✗ |
| 1B Model | ✓ | ✗ | ✗ | ✓ |
| Video | ✗ | ✓ | ✓ | ✓ |

## Resources

- [Project Page](https://molmo.allenai.org/)
- [GitHub Repository](https://github.com/allenai/molmo)
- [Hugging Face Models](https://huggingface.co/allenai/Molmo-72B-0924)
- [PixMo Dataset](https://huggingface.co/datasets/allenai/pixmo)
- [Technical Report](https://arxiv.org/abs/2409.17146)

## Citation

```bibtex
@article{deitke2024molmo,
  title={Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models},
  author={Deitke, Matt and others},
  journal={arXiv preprint arXiv:2409.17146},
  year={2024}
}
```
