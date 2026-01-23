# LLaVA (Large Language and Vision Assistant)

LLaVA is a pioneering approach to visual instruction tuning that connects vision encoders with large language models through simple but effective projection methods.

## Overview

LLaVA demonstrated that combining a pre-trained vision encoder with a language model via a simple MLP projection can achieve strong multimodal performance. The approach has inspired numerous follow-up works and remains a popular baseline architecture.

## Architecture

### Core Components

| Component | LLaVA-OneVision |
|-----------|-----------------|
| Vision Encoder | SigLIP-400M |
| Language Model | Qwen2.5-0.5B-Instruct |
| Projection | MLP (2-layer) |
| Context Length | 32K tokens |

### Architecture Diagram

```
                    ┌─────────────────────────┐
                    │      Input Image        │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   SigLIP-400M Vision    │
                    │       Encoder           │
                    └───────────┬─────────────┘
                                │
                         Visual Features
                                │
                    ┌───────────▼─────────────┐
                    │   2-Layer MLP           │
                    │   (Projection)          │
                    └───────────┬─────────────┘
                                │
                         Projected Features
                                │
        Text Tokens ───────────►├◄──── [IMG] tokens
                                │
                    ┌───────────▼─────────────┐
                    │   Qwen2.5 Language      │
                    │       Model             │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │     Text Response       │
                    └─────────────────────────┘
```

## Evolution of LLaVA

### LLaVA 1.0 (2023)
- First version demonstrating visual instruction tuning
- CLIP ViT-L/14 vision encoder
- Vicuna-7B/13B language model
- Simple linear projection

### LLaVA 1.5 (2023)
- Improved training data
- MLP projection instead of linear
- CLIP ViT-L/14 @ 336px
- Better benchmark performance

### LLaVA-NeXT (2024)
- Higher resolution support
- AnyRes technique for dynamic resolution
- Multiple LLM backbones supported

### LLaVA-OneVision (2024-2025)
- SigLIP vision encoder
- Qwen2.5 backbone
- State-of-the-art efficiency
- Video understanding support

## Model Variants

| Model | Vision | LLM | Parameters |
|-------|--------|-----|------------|
| LLaVA-OneVision-0.5B | SigLIP | Qwen2.5-0.5B | ~1B |
| LLaVA-OneVision-7B | SigLIP | Qwen2.5-7B | ~8B |
| LLaVA-OneVision-72B | SigLIP | Qwen2.5-72B | ~78B |
| LLaVA-NeXT-34B | CLIP | Yi-34B | ~35B |
| LLaVA-1.5-13B | CLIP | Vicuna-13B | ~14B |

## Training Pipeline

LLaVA uses a two-stage training approach:

### Stage 1: Pre-training (Feature Alignment)

```
Objective: Align visual features with text embedding space
Data: Image-caption pairs (595K from CC3M)
Trainable: Only the MLP projection layer
Frozen: Vision encoder + Language model
```

### Stage 2: Fine-tuning (Visual Instruction Tuning)

```
Objective: Follow multimodal instructions
Data: LLaVA-Instruct (158K conversations)
Trainable: MLP projection + Language model
Frozen: Vision encoder
```

## Benchmark Performance

| Benchmark | LLaVA-OV-7B | LLaVA-1.5-13B | GPT-4V |
|-----------|-------------|---------------|--------|
| VQAv2 | 79.3 | 80.0 | 77.2 |
| GQA | 62.1 | 63.3 | - |
| TextVQA | 61.3 | 61.3 | 78.0 |
| POPE | 87.2 | 85.9 | - |
| MMBench | 67.1 | 67.7 | 75.1 |

## Key Innovations

### 1. Visual Instruction Tuning
LLaVA introduced the concept of generating instruction-following data for vision-language:
- Use GPT-4 to generate conversations about images
- Include diverse tasks: description, reasoning, conversation
- Create high-quality training data efficiently

### 2. Simple Architecture
Demonstrated that complex fusion mechanisms aren't necessary:
- Simple MLP projection works well
- Enables easy experimentation with different components
- Facilitates community adoption and extension

### 3. Data Generation Pipeline
```
Image → [Captions + Bounding Boxes] → GPT-4 → Conversations
                    ↑
         COCO annotations provide context
```

## Usage

### Hugging Face

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-34b-hf"
)

# Process image and text
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"}
        ]
    }
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(prompt, image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### LLaVA-OneVision

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path="lmms-lab/llava-onevision-qwen2-7b-ov",
    model_base=None,
    model_name="llava_qwen"
)
```

## Extending LLaVA

LLaVA's simple architecture makes it easy to extend:

### Swap Vision Encoder
```python
# Use different vision encoder
vision_tower = SigLIPVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
```

### Swap Language Model
```python
# Use different LLM
language_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

### Add New Modalities
```python
# Add audio encoder for audio-vision-language
audio_encoder = WhisperModel.from_pretrained("openai/whisper-large-v3")
```

## Resources

- [LLaVA Project Page](https://llava-vl.github.io/)
- [GitHub Repository](https://github.com/haotian-liu/LLaVA)
- [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/)
- [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [Hugging Face Collection](https://huggingface.co/collections/llava-hf/llava-next-65f75c4afac77fd37dbbe6cf)

## Citation

```bibtex
@article{liu2023llava,
  title={Visual Instruction Tuning},
  author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
  journal={NeurIPS},
  year={2023}
}

@article{liu2024llavanext,
  title={LLaVA-NeXT: Improved reasoning, OCR, and world knowledge},
  author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Li, Bo and Zhang, Yuanhan and Shen, Sheng and Lee, Yong Jae},
  year={2024}
}
```
