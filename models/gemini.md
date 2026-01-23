# Gemini (Google)

Gemini is Google's family of multimodal AI models, currently leading many VLM benchmarks with its flagship Gemini 2.5 Pro model.

## Overview

Gemini represents Google's state-of-the-art in multimodal AI, capable of understanding text, images, video, and audio. The model family includes various sizes optimized for different use cases.

## Model Family

| Model | Context Length | Strengths | Availability |
|-------|---------------|-----------|--------------|
| Gemini 2.5 Pro | 1M+ tokens | Best overall performance | API |
| Gemini 2.5 Flash | 1M tokens | Fast, cost-effective | API |
| Gemini 1.5 Pro | 2M tokens | Long context | API |
| Gemini 1.5 Flash | 1M tokens | Efficient | API |
| Gemma 3 | - | Open-source | Weights |

## Key Capabilities

### Multimodal Understanding
- **Text**: Advanced language understanding and generation
- **Images**: High-resolution image analysis
- **Video**: Long video understanding (hours of content)
- **Audio**: Speech and sound recognition

### Native Multimodality
Unlike many VLMs that bolt vision onto language models, Gemini is trained natively multimodal from the start.

## Benchmark Performance

| Benchmark | Gemini 2.5 Pro | GPT-4o | Claude 3.5 Sonnet |
|-----------|----------------|--------|-------------------|
| MMMU | ~71 | ~69 | ~68 |
| MMBench-EN | ~85 | 83.4 | ~82 |
| LMArena | #1 | Top 3 | Top 5 |
| WebDevArena | #1 | - | - |

## Architecture

Gemini's architecture details are proprietary, but key known features include:

### Sparse Mixture of Experts (MoE)
- Efficient scaling through sparse activation
- Only subset of parameters activated per token
- Enables larger model capacity with lower compute

### Long Context
- Supports up to 2 million tokens in context
- Efficient attention mechanisms
- Can process entire books, codebases, or long videos

## Key Features

### 1. Extended Context Window
```
Standard VLMs: 4K - 128K tokens
Gemini 1.5 Pro: 2,000,000 tokens
```

This enables:
- Processing entire codebases
- Analyzing long documents
- Understanding multi-hour videos
- In-context learning with many examples

### 2. Video Understanding
Gemini can analyze video content including:
- Temporal reasoning
- Event detection
- Action recognition
- Multi-scene comprehension

### 3. Multimodal Reasoning
Combines information across modalities:
- Image + text reasoning
- Video + audio analysis
- Document understanding with figures

## Usage

### Google AI Studio
```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-2.5-pro-latest")

# Text + Image
response = model.generate_content([
    "What's in this image?",
    image_data
])

# Video analysis
response = model.generate_content([
    "Summarize this video",
    video_file
])
```

### Vertex AI
```python
from vertexai.generative_models import GenerativeModel, Part

model = GenerativeModel("gemini-2.5-pro")
response = model.generate_content([
    Part.from_image(image_data),
    "Describe this image."
])
```

## Pricing (API)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Gemini 2.5 Pro | $1.25 - $2.50 | $5.00 - $10.00 |
| Gemini 2.5 Flash | $0.075 | $0.30 |
| Gemini 1.5 Pro | $1.25 - $2.50 | $5.00 - $10.00 |

*Prices vary by context length and region*

## Strengths

1. **Best-in-class performance** on many benchmarks
2. **Longest context window** available
3. **Native multimodality** - not a bolt-on solution
4. **Strong reasoning** capabilities
5. **Code understanding** and generation

## Limitations

1. **Proprietary** - no open weights
2. **API only** - requires internet connection
3. **Cost** - can be expensive for high-volume use
4. **Availability** - some regions restricted

## Comparison with Open Models

| Feature | Gemini 2.5 Pro | InternVL3-78B | Qwen2.5-VL-72B |
|---------|----------------|---------------|----------------|
| Open Weights | No | Yes | Yes |
| Context Length | 1M+ | 256K | 128K |
| Fine-tuning | No | Yes | Yes |
| Self-hosting | No | Yes | Yes |
| Cost Control | Limited | Full | Full |

## Resources

- [Google AI Studio](https://ai.google.dev/)
- [Vertex AI](https://cloud.google.com/vertex-ai)
- [Gemini Documentation](https://ai.google.dev/docs)
- [Gemini Technical Report](https://arxiv.org/abs/2312.11805)

## Related: Gemma (Open Source)

For open-source Google models, see the Gemma family:
- Gemma 3 (1B, 4B, 12B, 27B)
- Multimodal capabilities
- Apache 2.0 license
- [Gemma Documentation](./gemma.md)
