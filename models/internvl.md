# InternVL Series

InternVL is a family of open-source Vision Language Models developed by Shanghai AI Lab, representing the current state-of-the-art in open-source multimodal AI.

## Overview

InternVL3-78B has achieved the highest MMMU score (72.2) among open-source models, demonstrating performance competitive with leading proprietary models like GPT-4o and Gemini.

## Architecture

### Components

| Component | InternVL3-78B |
|-----------|---------------|
| Vision Encoder | InternViT-6B-448px-V2_5 |
| Language Model | Qwen2.5-72B |
| Total Parameters | 78.41B |
| Context Length | 256K tokens |

### InternViT Vision Encoder

InternViT is a specialized vision transformer designed specifically for VLM tasks:
- Pre-trained on large-scale image-text pairs
- Supports high-resolution images (448px+)
- Optimized for feature extraction that aligns well with language model spaces

### Architecture Diagram

```
                    ┌─────────────────────────┐
                    │      Input Image        │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  InternViT-6B-448px-V2_5│
                    │    (Vision Encoder)     │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │    MLP Projection       │
                    │  (Vision-Text Align)    │
                    └───────────┬─────────────┘
                                │
        Text Input ────────────►├
                                │
                    ┌───────────▼─────────────┐
                    │      Qwen2.5-72B        │
                    │   (Language Model)      │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │     Text Response       │
                    └─────────────────────────┘
```

## Model Variants

| Model | Vision | LLM | Parameters | MMMU |
|-------|--------|-----|------------|------|
| InternVL3-78B | InternViT-6B | Qwen2.5-72B | 78B | 72.2 |
| InternVL3-38B | InternViT-6B | Qwen2.5-32B | 38B | ~68 |
| InternVL2.5-78B | InternViT-6B | InternLM2-76B | 78B | 70.1 |
| InternVL2-26B | InternViT-6B | InternLM2-20B | 26B | 58.2 |

## Benchmark Performance

| Benchmark | InternVL3-78B | GPT-4o | Gemini 1.5 Pro |
|-----------|---------------|--------|----------------|
| MMMU | 72.2 | ~69 | ~67 |
| MMBench-EN | 84.4 | 83.4 | 81.3 |
| MMStar | 70.4 | 68.1 | 66.2 |
| RealWorldQA | 70.0 | 68.9 | 67.5 |

## Key Features

1. **Dynamic Resolution**: Supports various image resolutions without fixed cropping
2. **Multi-Image Input**: Can process multiple images in a single conversation
3. **Long Context**: Supports up to 256K token context windows
4. **Multilingual**: Strong performance across multiple languages

## Training Approach

InternVL uses a multi-stage training process:

1. **Stage 1: Vision Encoder Pre-training**
   - Large-scale image-text contrastive learning
   - Builds strong visual representations

2. **Stage 2: Vision-Language Alignment**
   - Bridges vision encoder with language model
   - Uses curated image-caption datasets

3. **Stage 3: Instruction Tuning**
   - Fine-tunes on instruction-following data
   - Includes visual question answering, OCR, reasoning tasks

## Usage

### Hugging Face

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(
    "OpenGVLab/InternVL3-78B",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "OpenGVLab/InternVL3-78B",
    trust_remote_code=True
)
```

### vLLM Deployment

```bash
vllm serve OpenGVLab/InternVL3-78B \
    --tensor-parallel-size 4 \
    --trust-remote-code
```

## Resources

- [GitHub Repository](https://github.com/OpenGVLab/InternVL)
- [Hugging Face Model](https://huggingface.co/OpenGVLab/InternVL3-78B)
- [Technical Report](https://arxiv.org/abs/2404.16821)

## Citation

```bibtex
@article{chen2024internvl,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
```
