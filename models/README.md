# Vision Language Model Architectures

This directory contains detailed documentation of state-of-the-art Vision Language Model architectures.

## Model Categories

### Proprietary Models
- [GPT-4V / GPT-4o](./gpt4v.md) - OpenAI's multimodal models
- [Gemini](./gemini.md) - Google's multimodal AI family
- [Claude Vision](./claude.md) - Anthropic's vision-capable models

### Open-Source Models
- [InternVL](./internvl.md) - Shanghai AI Lab's open-source VLM
- [Qwen-VL](./qwen-vl.md) - Alibaba's vision-language series
- [LLaVA](./llava.md) - Visual instruction tuning approach
- [Molmo](./molmo.md) - Allen Institute's open VLM family
- [Pixtral](./pixtral.md) - Mistral's multimodal model
- [Gemma](./gemma.md) - Google's open-source lightweight models

## Architecture Comparison

| Model | Vision Encoder | Language Model | Projection | Context Length |
|-------|---------------|----------------|------------|----------------|
| GPT-4o | Proprietary | GPT-4 | Proprietary | 128K |
| Gemini 2.5 Pro | Proprietary | Gemini | Proprietary | 1M+ |
| InternVL3-78B | InternViT-6B | Qwen2.5-72B | MLP | 256K |
| Qwen2.5-VL-72B | ViT (Dynamic) | Qwen2.5-72B | Cross-Attention | 128K |
| Molmo-72B | ViT | LLaMA-72B | Connector | 128K |
| LLaVA-OneVision | SigLIP-400M | Qwen2.5 | MLP | 32K |
| Pixtral-12B | Custom ViT | Mistral | - | 128K |

## Common Architecture Patterns

### Vision Encoder Types

1. **ViT (Vision Transformer)**: Standard transformer-based image encoding
2. **SigLIP**: Google's efficient vision encoder with improved training
3. **InternViT**: Specialized vision encoder optimized for VLM tasks
4. **CLIP**: Contrastive language-image pre-training encoder

### Projection Methods

1. **MLP (Multi-Layer Perceptron)**: Simple linear projection layers
2. **Q-Former**: Querying transformer for cross-modal alignment
3. **Perceiver Resampler**: Attention-based dimension reduction
4. **Cross-Attention**: Direct attention between vision and text features

### Language Model Backbones

1. **LLaMA/LLaMA 2/LLaMA 3**: Meta's open-source LLM family
2. **Qwen/Qwen2/Qwen2.5**: Alibaba's multilingual LLM series
3. **Mistral**: Efficient open-source LLM
4. **Proprietary**: GPT-4, Gemini, Claude backends
