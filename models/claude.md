# Claude Vision (Anthropic)

Claude is Anthropic's family of AI assistants with vision capabilities, known for strong reasoning, safety focus, and long context windows.

## Overview

Claude's vision capabilities allow it to understand and analyze images alongside text. The Claude 3.5 family represents the current production-ready vision-capable models.

## Model Family

| Model | Context | Best For | Vision |
|-------|---------|----------|--------|
| Claude 3.5 Sonnet | 200K | Balance of speed/capability | Yes |
| Claude 3.5 Haiku | 200K | Fast, cost-effective | Yes |
| Claude 3 Opus | 200K | Complex reasoning | Yes |

## Key Strengths

### 1. Safety and Reliability
- Trained with Constitutional AI principles
- Honest about uncertainties
- Avoids harmful outputs
- Consistent behavior

### 2. Long Context
- 200K token context window
- Process many images in one request
- Handle large documents

### 3. Reasoning Quality
- Strong analytical capabilities
- Step-by-step problem solving
- Good at complex multi-step tasks

### 4. Instruction Following
- Precise adherence to instructions
- Structured output generation
- Format compliance

## Benchmark Performance

| Benchmark | Claude 3.5 Sonnet | GPT-4o | Notes |
|-----------|-------------------|--------|-------|
| MMMU | ~68 | ~69 | College-level |
| VQAv2 | ~75 | 77.2 | Visual Q&A |
| DocVQA | ~89 | 92.8 | Documents |
| Chart understanding | Strong | Strong | Comparable |

## Vision Capabilities

### Supported Tasks
- Image description and analysis
- Document understanding
- Chart and graph interpretation
- Diagram explanation
- Screenshot analysis
- Multi-image comparison
- Visual reasoning

### Image Requirements
- Supported formats: JPEG, PNG, GIF, WebP
- Max size: 20MB per image
- Recommended: Under 5MB for optimal speed
- Max resolution: 8K x 8K pixels

## Usage

### Anthropic API

```python
import anthropic
import base64

client = anthropic.Anthropic()

# From URL (via base64)
import httpx
image_data = base64.standard_b64encode(
    httpx.get("https://example.com/image.jpg").content
).decode("utf-8")

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": "Describe what you see in this image."
                }
            ]
        }
    ]
)

print(message.content[0].text)
```

### Multiple Images

```python
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image1_data
                    }
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image2_data
                    }
                },
                {
                    "type": "text",
                    "text": "Compare these two images and list the differences."
                }
            ]
        }
    ]
)
```

### Using URL directly

```python
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/image.jpg"
                    }
                },
                {
                    "type": "text",
                    "text": "What's in this image?"
                }
            ]
        }
    ]
)
```

## Pricing

| Model | Input (1M tokens) | Output (1M tokens) |
|-------|-------------------|-------------------|
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Claude 3.5 Haiku | $0.80 | $4.00 |
| Claude 3 Opus | $15.00 | $75.00 |

*Images are converted to tokens based on size*

### Image Token Calculation
```
Approximate tokens per image:
- Small (≤384px): ~280 tokens
- Medium (≤768px): ~1,000 tokens
- Large (≤1500px): ~2,500 tokens
- Very large: Up to ~8,000 tokens
```

## Best Practices

### 1. Image Placement
Place images before text that references them:
```python
content = [
    {"type": "image", "source": {...}},  # Image first
    {"type": "text", "text": "What's in the image above?"}
]
```

### 2. Clear Instructions
Be specific about what you want:
```python
# Good
"List all text visible in this screenshot, organized by section."

# Less effective
"What does this show?"
```

### 3. Structured Output
Request specific formats:
```python
"Analyze this chart and provide output as JSON with keys:
'title', 'data_type', 'key_findings', 'trends'"
```

## Comparison with Open Models

| Feature | Claude 3.5 Sonnet | InternVL3-78B | Molmo-72B |
|---------|-------------------|---------------|-----------|
| Open Weights | No | Yes | Yes |
| Context Length | 200K | 256K | 128K |
| Fine-tuning | No | Yes | Yes |
| Safety Focus | High | Medium | Medium |
| API Availability | Yes | Community | Community |

## Use Cases

### Ideal For
- Document analysis and extraction
- Code review from screenshots
- Technical diagram explanation
- Multi-image comparison
- Structured data extraction
- Quality assurance visual checks

### Consider Alternatives When
- Need custom fine-tuning
- Require self-hosting
- Maximum benchmark performance needed
- Budget constraints at scale

## Computer Use (Beta)

Claude has experimental "computer use" capabilities:
- Can control mouse/keyboard
- Navigate software interfaces
- Automate visual tasks
- Currently in beta/research preview

```python
# Computer use requires specific tool definitions
# See Anthropic documentation for details
```

## Safety Considerations

Claude is designed with safety in mind:
- Refuses to identify real individuals
- Avoids generating harmful content
- Honest about limitations
- Maintains consistent ethical behavior

These aren't bugs but intentional design choices for responsible AI deployment.

## Resources

- [Anthropic Vision Documentation](https://docs.anthropic.com/en/docs/vision)
- [API Reference](https://docs.anthropic.com/en/api)
- [Claude Cookbook](https://github.com/anthropics/anthropic-cookbook)
- [Model Card](https://www.anthropic.com/claude)

## Updates

Anthropic regularly updates Claude models. Check the official documentation for:
- Latest model versions
- New capabilities
- API changes
- Pricing updates
