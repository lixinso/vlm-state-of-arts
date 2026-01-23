# GPT-4V / GPT-4o (OpenAI)

GPT-4V and GPT-4o represent OpenAI's vision-capable large language models, setting early benchmarks for proprietary multimodal AI.

## Overview

GPT-4V (Vision) was OpenAI's first publicly available vision-language model, later succeeded by GPT-4o ("o" for "omni") which offers improved performance and native multimodal capabilities.

## Model Evolution

| Model | Release | Key Features |
|-------|---------|--------------|
| GPT-4V | 2023 | First GPT-4 with vision |
| GPT-4 Turbo Vision | 2024 | Improved, faster GPT-4V |
| GPT-4o | 2024 | Native multimodal, faster |
| GPT-4o mini | 2024 | Cost-effective variant |

## GPT-4o Architecture

### Key Properties
- **Native Multimodal**: Single model handles text, image, audio
- **Context Length**: 128K tokens
- **Training**: End-to-end multimodal training
- **Speed**: 2x faster than GPT-4 Turbo

### Capabilities
1. Image understanding and analysis
2. Document/chart comprehension
3. OCR and text extraction
4. Visual reasoning
5. Real-time audio/video (limited)

## Benchmark Performance

| Benchmark | GPT-4o | GPT-4V | Notes |
|-----------|--------|--------|-------|
| MMMU | ~69 | ~56 | College-level reasoning |
| VQAv2 | 77.2 | 75.5 | Visual Q&A |
| TextVQA | 78.0 | 75.3 | Scene text reading |
| ChartQA | 85.7 | 78.5 | Chart understanding |
| DocVQA | 92.8 | 88.4 | Document understanding |

## Key Strengths

### 1. Scientific Reasoning
GPT-4o excels at understanding scientific diagrams, equations, and technical content.

### 2. General Knowledge
Strong world knowledge integration with visual understanding.

### 3. Instruction Following
Excellent at following complex, multi-step visual instructions.

### 4. Code from Images
Can generate code from screenshots, diagrams, and wireframes.

## Limitations

### Known Weaknesses
- **Counting**: Struggles with precise object counting
- **Spatial Reasoning**: Limited spatial relationship understanding
- **Hallucination**: May describe objects not present
- **Mathematical Graphs**: Can misinterpret function plots

### Studies and Findings
Research shows GPT-4o can fail at simple tasks like digit recognition while performing well on complex reasoning, suggesting capability gaps in fundamental perception.

## Usage

### OpenAI API

```python
from openai import OpenAI
import base64

client = OpenAI()

# From URL
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"}
                }
            ]
        }
    ],
    max_tokens=300
)

# From base64
with open("image.jpg", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode('utf-8')

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)
```

### Multiple Images

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two images:"},
                {"type": "image_url", "image_url": {"url": image1_url}},
                {"type": "image_url", "image_url": {"url": image2_url}}
            ]
        }
    ]
)
```

### Detail Levels

```python
# Low detail (faster, cheaper)
{"type": "image_url", "image_url": {"url": url, "detail": "low"}}

# High detail (more accurate)
{"type": "image_url", "image_url": {"url": url, "detail": "high"}}

# Auto (model decides)
{"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
```

## Pricing

| Model | Input (1M tokens) | Output (1M tokens) | Image |
|-------|-------------------|--------------------| ------|
| GPT-4o | $2.50 | $10.00 | ~$0.00425/image |
| GPT-4o mini | $0.15 | $0.60 | ~$0.001275/image |

*Image pricing depends on resolution and detail level*

## Image Processing

### Supported Formats
- PNG, JPEG, GIF, WebP

### Size Limits
- Max 20MB per image
- Up to ~50 images per request

### Resolution Handling
- **Low detail**: 512x512 fixed
- **High detail**: Up to 2048x2048, tiled processing

## Comparison with Open Models

| Feature | GPT-4o | InternVL3-78B | Qwen2.5-VL-72B |
|---------|--------|---------------|----------------|
| MMMU | ~69 | 72.2 | ~70 |
| Open Weights | No | Yes | Yes |
| Fine-tuning | No* | Yes | Yes |
| Self-hosting | No | Yes | Yes |
| API Cost | $$ | Free** | Free** |

*Fine-tuning available for some GPT-4 variants but not GPT-4o vision
**Self-hosting costs for compute

## Use Cases

### Best For
- Quick prototyping
- Production applications needing reliability
- Scientific/technical document analysis
- Code generation from mockups
- General visual Q&A

### Consider Alternatives When
- Need fine-tuning on specific domains
- Cost-sensitive at scale
- Require self-hosting
- Need maximum accuracy on benchmarks

## Resources

- [OpenAI Vision Guide](https://platform.openai.com/docs/guides/vision)
- [API Reference](https://platform.openai.com/docs/api-reference/chat/create)
- [OpenAI Cookbook](https://cookbook.openai.com/)
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

## Safety and Limitations

OpenAI implements safety measures:
- Refuses to identify real people in photos
- Limited on medical diagnosis
- Avoids harmful content generation
- May decline certain image types

These are design choices for responsible deployment, not capability limitations.
