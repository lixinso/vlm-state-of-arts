# Vision Language Models - State of the Art

A comprehensive resource tracking the latest developments in Vision Language Models (VLMs) - AI systems that can understand and reason about both images and text together.

## Overview

Vision Language Models represent a significant advancement in AI, combining computer vision capabilities with large language models to enable multimodal understanding. These models can analyze images, answer questions about visual content, perform OCR, understand documents, and execute complex visual reasoning tasks.

## Repository Structure

```
vlm-state-of-arts/
├── README.md              # This file - overview and quick reference
├── models/                # Detailed documentation of VLM architectures
├── benchmarks/            # Benchmark descriptions and comparisons
├── papers/                # Key research papers and summaries
├── industry-adoption/     # Industry use cases and deployment analysis
└── resources/             # Additional resources, tutorials, and tools
```

## Quick Navigation

- [Top Models (2025-2026)](#top-models-2025-2026)
- [Key Benchmarks](#key-benchmarks)
- [Architecture Overview](#architecture-overview)
- [Current Trends](#current-trends)
- [Industry Adoption](#industry-adoption)

## Top Models (2025-2026)

### Proprietary Models

| Model | Provider | Key Strengths |
|-------|----------|---------------|
| Gemini 2.5 Pro | Google | Leading multimodal performance, 1M+ context |
| GPT-4o | OpenAI | Strong general performance, scientific tasks |
| Claude 3.5 Sonnet | Anthropic | Excellent reasoning, safety-focused |

### Open-Source Models

| Model | Parameters | MMMU Score | Notable Features |
|-------|------------|------------|------------------|
| InternVL3-78B | 78B | 72.2 | SOTA open-source, InternViT + Qwen2.5 |
| Qwen2.5-VL-72B | 72B | ~70 | Native dynamic resolution, hour-long video |
| Molmo-72B | 72B | ~68 | Outperforms GPT-4V on many benchmarks |
| Gemma 3-27B | 27B | - | Lightweight, Google-backed |
| Pixtral-12B | 12B | - | 128K context, multi-image support |
| LLaVA-OneVision | 7B | - | Efficient, SigLIP + Qwen2.5 |

## Key Benchmarks

| Benchmark | Focus Area | Description |
|-----------|------------|-------------|
| **MMMU** | College-level reasoning | 11.5K questions across 6 disciplines |
| **MMBench** | General multimodal | 3,000+ questions, 20 ability dimensions |
| **VQAv2** | Visual Q&A | 1.1M questions on MS-COCO images |
| **TextVQA** | Scene text understanding | Text reading in natural images |
| **DocVQA** | Document understanding | Complex document layouts |
| **RealWorldQA** | Real-world scenarios | Practical visual understanding |
| **MMStar** | Comprehensive evaluation | Multi-domain assessment |

## Architecture Overview

Modern VLMs typically consist of three main components:

1. **Vision Encoder**: Processes images into embeddings (e.g., ViT, SigLIP, InternViT)
2. **Projection Layer**: Aligns visual and text embeddings (MLP, Q-Former, Perceiver)
3. **Language Model**: Generates text responses (LLaMA, Qwen, Mistral)

```
Image → [Vision Encoder] → [Projection] → [LLM] → Text Response
                              ↑
                         Text Input
```

## Current Trends (2025-2026)

### Key Developments

- **Shrinking Gap**: Open-source models now match proprietary performance in many tasks
- **Efficient Models**: Focus on Small VLMs (SVLMs) for resource-constrained environments
- **Inference Scaling**: Performance gains from improved inference rather than just training
- **Native Resolution**: Models handling diverse image sizes without normalization
- **Long Video**: Support for hour-long video understanding
- **GUI Agents**: VLMs specialized for computer use and UI interaction

### Remaining Challenges

- Spatial reasoning and mental model formation
- Object counting and digit recognition
- Hallucination resistance
- Complex mathematical reasoning from images

## Industry Adoption

VLMs are moving from research benchmarks into applied industry workflows. Key adoption areas include:

| Area | Focus |
|------|-------|
| [Cross-industry overview](./industry-adoption/vlm-industry-adoption-overview.md) | Healthcare, retail, robotics, documents, surveillance, manufacturing, accessibility |
| [Autonomous driving](./industry-adoption/autonomous-driving/vlm-autonomous-driving.md) | VLA adoption across robotaxi and EV manufacturers |
| [Robotics and embodied AI](./industry-adoption/robotics/vlm-robotics.md) | VLA architectures, robot control, safety, middleware, edge deployment |

## Getting Started

See individual directories for detailed information:

- [`models/`](./models/) - Detailed architecture documentation
- [`benchmarks/`](./benchmarks/) - Benchmark comparisons and analysis
- [`papers/`](./papers/) - Key research papers
- [`industry-adoption/`](./industry-adoption/) - Industry adoption studies and deployment analysis
- [`resources/`](./resources/) - Tools and tutorials

## Contributing

Contributions welcome! Please see our contributing guidelines for details.

## References

Key resources for staying updated:

- [Hugging Face Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)
- [Papers With Code - VQA](https://paperswithcode.com/task/visual-question-answering)
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
- [Awesome VLM Architectures](https://github.com/gokayfem/awesome-vlm-architectures)

## License

MIT License - see LICENSE file for details.
