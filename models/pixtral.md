# Pixtral (Mistral AI)

Pixtral is Mistral AI's multimodal model, designed to handle images at native resolution with a large context window.

## Overview

Pixtral 12B significantly outperforms other open-source multimodal models in instruction following while maintaining strong visual understanding capabilities.

## Key Features

### 1. Native Resolution Processing
Pixtral processes images at their original resolution without forced resizing:
- Handles varied aspect ratios naturally
- Preserves image detail
- No normalization artifacts

### 2. Multi-Image Support
- Process multiple images in single request
- Compare and contrast images
- Handle image sequences

### 3. Large Context Window
- 128,000 token context
- Supports detailed image analysis
- Long document processing

## Model Specifications

| Specification | Value |
|--------------|-------|
| Parameters | 12B |
| Context Length | 128K tokens |
| Vision Encoder | Custom ViT |
| Base LLM | Mistral architecture |
| License | Apache 2.0 |

## Architecture

```
                    ┌─────────────────────────┐
                    │   Input Image(s)        │
                    │   (Native Resolution)   │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Custom Vision         │
                    │   Transformer           │
                    │   (Variable patches)    │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Projection Layer      │
                    └───────────┬─────────────┘
                                │
        Text Input ────────────►├
                                │
                    ┌───────────▼─────────────┐
                    │   Mistral LLM           │
                    │   (12B parameters)      │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │     Text Response       │
                    └─────────────────────────┘
```

## Benchmark Performance

| Benchmark | Pixtral 12B | Qwen2-VL 7B | LLaVA-OV 7B |
|-----------|-------------|-------------|-------------|
| Instruction Following | Best | Good | Good |
| General VQA | Strong | Strong | Strong |
| OCR | Good | Better | Good |
| Document Understanding | Good | Better | Good |

## Strengths

1. **Instruction Following**: Superior at following complex instructions
2. **Multi-Image**: Handles multiple images well
3. **Native Resolution**: No forced resizing
4. **Efficient Size**: Strong performance at 12B
5. **Open Source**: Apache 2.0 license

## Limitations

1. **OCR**: Not as strong as Qwen-VL for OCR tasks
2. **No Video**: Image-only, no video support
3. **Smaller Community**: Less ecosystem than LLaVA

## Usage

### vLLM

```bash
vllm serve mistralai/Pixtral-12B-2409 \
    --tokenizer_mode "mistral" \
    --limit_mm_per_prompt 'image=4'
```

### Mistral API

```python
from mistralai import Mistral
import base64

client = Mistral(api_key="YOUR_API_KEY")

# Encode image
with open("image.jpg", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode('utf-8')

response = client.chat.complete(
    model="pixtral-12b-2409",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                },
                {
                    "type": "text",
                    "text": "Describe this image in detail."
                }
            ]
        }
    ]
)
```

### Transformers

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained(
    "mistralai/Pixtral-12B-2409",
    torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained("mistralai/Pixtral-12B-2409")

inputs = processor(
    text="Describe this image.",
    images=image,
    return_tensors="pt"
)
output = model.generate(**inputs)
```

### Multiple Images

```python
response = client.chat.complete(
    model="pixtral-12b-2409",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img1}"},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img2}"},
                {"type": "text", "text": "Compare these two images."}
            ]
        }
    ]
)
```

## Comparison with Similar Models

| Feature | Pixtral 12B | Qwen2-VL 7B | LLaVA-OV 7B |
|---------|-------------|-------------|-------------|
| Parameters | 12B | 8B | 8B |
| Context | 128K | 32K | 32K |
| Native Res | Yes | Yes | Partial |
| Multi-Image | Yes | Yes | Yes |
| Video | No | Yes | Yes |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 |

## Hardware Requirements

| Configuration | VRAM Required |
|--------------|---------------|
| FP16 | ~24GB |
| INT8 | ~12GB |
| INT4 | ~6GB |

## Best For

- Multi-image understanding tasks
- Complex instruction following
- Native resolution image analysis
- General-purpose vision-language tasks

## Resources

- [Mistral AI](https://mistral.ai/)
- [Hugging Face Model](https://huggingface.co/mistralai/Pixtral-12B-2409)
- [Documentation](https://docs.mistral.ai/)
- [Blog Post](https://mistral.ai/news/pixtral-12b/)

## Citation

```bibtex
@misc{pixtral2024,
  title={Pixtral 12B},
  author={Mistral AI},
  year={2024},
  url={https://mistral.ai/news/pixtral-12b/}
}
```
