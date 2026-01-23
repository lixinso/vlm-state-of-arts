# Gemma (Google)

Gemma is Google's family of lightweight, open-source models built on the same research as Gemini. Gemma 3 includes multimodal capabilities.

## Overview

Gemma 3 models are designed for efficiency while maintaining strong performance. They offer open-source alternatives to proprietary models for both text and vision tasks.

## Model Family

| Model | Parameters | Vision | Context | License |
|-------|------------|--------|---------|---------|
| Gemma 3 27B | 27B | Yes | 128K | Gemma License |
| Gemma 3 12B | 12B | Yes | 128K | Gemma License |
| Gemma 3 4B | 4B | Yes | 128K | Gemma License |
| Gemma 3 1B | 1B | Yes | 32K | Gemma License |

## Gemma 3 Vision Capabilities

### Supported Tasks
- Image understanding
- Short video analysis
- Visual question answering
- Image-to-text generation
- Multi-image reasoning

### Key Features
1. **Efficiency**: Strong performance at small model sizes
2. **Multilingual**: Good multilingual support
3. **Long Context**: Up to 128K tokens
4. **Video**: Short video understanding
5. **Open**: Weights available for download

## Architecture

Gemma 3 multimodal uses:
- **Vision Encoder**: SigLIP-based vision encoder
- **Language Model**: Gemma 3 decoder
- **Projection**: Efficient bridging layer

```
                    ┌─────────────────────────┐
                    │   Image / Video Frame   │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   SigLIP Vision         │
                    │   Encoder               │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Projection Layer      │
                    └───────────┬─────────────┘
                                │
        Text Input ────────────►├
                                │
                    ┌───────────▼─────────────┐
                    │   Gemma 3 Decoder       │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │     Text Output         │
                    └─────────────────────────┘
```

## Benchmark Performance

| Model | MMMU | VQAv2 | Notes |
|-------|------|-------|-------|
| Gemma 3 27B | ~58 | ~75 | Best Gemma vision |
| Gemma 3 12B | ~52 | ~70 | Good balance |
| Gemma 3 4B | ~45 | ~65 | Edge-capable |
| Gemma 3 1B | ~35 | ~55 | Mobile-ready |

*Scores are approximate; check official benchmarks*

## Usage

### Transformers

```python
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

model = Gemma3ForConditionalGeneration.from_pretrained(
    "google/gemma-3-27b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("google/gemma-3-27b-it")

# Process image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

inputs = processor(
    text=processor.apply_chat_template(messages, add_generation_prompt=True),
    images=[image],
    return_tensors="pt"
)

output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0]))
```

### Vertex AI

```python
from vertexai.generative_models import GenerativeModel

model = GenerativeModel("gemma-3-27b-it")
response = model.generate_content([image, "What's in this image?"])
```

### Ollama

```bash
ollama run gemma3:27b
```

## Hardware Requirements

| Model | FP16 VRAM | INT8 VRAM | INT4 VRAM |
|-------|-----------|-----------|-----------|
| 27B | ~54GB | ~27GB | ~14GB |
| 12B | ~24GB | ~12GB | ~6GB |
| 4B | ~8GB | ~4GB | ~2GB |
| 1B | ~2GB | ~1GB | ~0.5GB |

## Strengths

1. **Size Efficiency**: Good performance at small sizes
2. **Deployment Flexibility**: From mobile to data center
3. **Google Quality**: Same research base as Gemini
4. **Open Weights**: Download and self-host
5. **Long Context**: 128K for larger models

## Limitations

1. **Not Fully Open**: Gemma license has some restrictions
2. **Performance Gap**: Behind larger open models on benchmarks
3. **Vision is Newer**: Text capabilities more mature
4. **Limited Video**: Only short video support

## Comparison

| Feature | Gemma 3 27B | LLaVA-OV 7B | Qwen2.5-VL 7B |
|---------|-------------|-------------|---------------|
| Parameters | 27B | ~8B | ~8B |
| MMMU | ~58 | ~55 | ~58 |
| Context | 128K | 32K | 32K |
| Video | Short | Yes | Yes |
| License | Gemma | Apache 2.0 | Apache 2.0 |
| Edge Deploy | 1B-4B | No | 3B |

## Use Cases

### Best For
- Edge deployment (1B-4B models)
- Resource-constrained environments
- Applications needing Google-quality models
- Multilingual vision tasks

### Consider Alternatives When
- Maximum accuracy required (use larger models)
- Need long video understanding
- Fully open Apache 2.0 license required
- Need extensive fine-tuning community

## Deployment Options

### Local Inference
- Transformers + PyTorch
- vLLM
- Ollama
- llama.cpp (GGUF format)

### Cloud
- Google Vertex AI
- Google AI Studio
- Any cloud with GPU

### Edge
- TensorFlow Lite (mobile)
- ONNX Runtime
- MediaPipe

## Fine-tuning

Gemma supports various fine-tuning methods:
- Full fine-tuning
- LoRA / QLoRA
- PEFT methods

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)
```

## Resources

- [Google AI - Gemma](https://ai.google.dev/gemma)
- [Hugging Face Models](https://huggingface.co/google/gemma-3-27b-it)
- [Kaggle Models](https://www.kaggle.com/models/google/gemma)
- [Technical Report](https://arxiv.org/abs/2403.08295)
- [Responsible AI Toolkit](https://ai.google.dev/responsible)

## License

Gemma uses the Gemma Terms of Use license:
- Free for most uses
- Some restrictions on harmful applications
- Commercial use generally allowed
- Check license for specific use cases

## Citation

```bibtex
@article{gemma2024,
  title={Gemma: Open Models Based on Gemini Research and Technology},
  author={Gemma Team, Google DeepMind},
  journal={arXiv preprint},
  year={2024}
}
```
