# VLM Resources

A collection of tools, tutorials, and resources for working with Vision Language Models.

## Evaluation Tools

### VLMEvalKit
Open-source evaluation toolkit supporting 220+ models and 80+ benchmarks.

- **GitHub**: [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
- **Features**:
  - Unified evaluation interface
  - Support for most VLM benchmarks
  - Easy model integration
  - Batch evaluation

```bash
# Installation
pip install vlmeval

# Run evaluation
python run.py --model InternVL2-26B --data MMBench_DEV_EN
```

### lmms-eval
Language Model Evaluation Harness for multimodal models.

- **GitHub**: [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- **Features**:
  - Standardized evaluation
  - Many benchmark implementations
  - Easy extensibility

---

## Deployment Platforms

### vLLM
High-throughput LLM/VLM serving.

- **GitHub**: [vLLM](https://github.com/vllm-project/vllm)
- **Supported VLMs**: LLaVA, Qwen-VL, InternVL, Phi-Vision

```bash
# Serve VLM
vllm serve llava-hf/llava-v1.6-34b-hf \
    --tensor-parallel-size 2 \
    --max-model-len 32768
```

### Ollama
Local VLM deployment made easy.

- **Website**: [Ollama](https://ollama.ai/)
- **Supported VLMs**: LLaVA, Llama-Vision, Molmo

```bash
# Run VLM locally
ollama run llava:34b
```

### Text Generation Inference (TGI)
Hugging Face's production-ready inference server.

- **GitHub**: [TGI](https://github.com/huggingface/text-generation-inference)
- **Features**: Multi-GPU, quantization, streaming

### SGLang
Fast serving framework for large language and multimodal models.

- **GitHub**: [SGLang](https://github.com/sgl-project/sglang)
- **Features**: RadixAttention, continuous batching

---

## Training Frameworks

### LLaVA Training
Official training code for LLaVA models.

- **GitHub**: [LLaVA](https://github.com/haotian-liu/LLaVA)
- **Features**: Full training pipeline, LoRA support

### InternVL Training
Training code for InternVL models.

- **GitHub**: [InternVL](https://github.com/OpenGVLab/InternVL)
- **Features**: Multi-stage training, data preparation

### Xtuner
Efficient fine-tuning toolkit for VLMs.

- **GitHub**: [Xtuner](https://github.com/InternLM/xtuner)
- **Features**: LoRA, QLoRA, full fine-tuning

```bash
# Fine-tune VLM with LoRA
xtuner train internvl_v2_internlm2_2b_qlora_e1
```

---

## Datasets

### Training Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| LLaVA-Instruct | 158K | Visual instruction tuning data |
| ShareGPT4V | 1.2M | GPT-4V annotated image descriptions |
| PixMo | - | Molmo's dense captioning dataset |
| LAION-5B | 5B | Large-scale image-text pairs |
| DataComp | 12.8B | Curated image-text dataset |

### Benchmark Datasets

| Dataset | Task | Size |
|---------|------|------|
| VQAv2 | VQA | 1.1M questions |
| TextVQA | OCR VQA | 45K questions |
| DocVQA | Document VQA | 50K questions |
| MMMU | Multimodal reasoning | 11.5K questions |
| MMBench | General | 3K questions |

### Dataset Resources
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Papers With Code Datasets](https://paperswithcode.com/datasets)

---

## Leaderboards

### Open VLM Leaderboard
Comprehensive VLM benchmark tracking.

- **Link**: [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)
- **Benchmarks**: MMMU, MMBench, TextVQA, etc.

### MMMU Leaderboard
Official MMMU benchmark rankings.

- **Link**: [MMMU Leaderboard](https://mmmu-benchmark.github.io/#leaderboard)

### Papers With Code
Task-specific leaderboards.

- **VQA**: [VQA Leaderboard](https://paperswithcode.com/task/visual-question-answering)
- **Image Captioning**: [Captioning Leaderboard](https://paperswithcode.com/task/image-captioning)

---

## Model Hubs

### Hugging Face
Largest collection of VLM models.

- **Collections**:
  - [LLaVA Models](https://huggingface.co/collections/llava-hf)
  - [InternVL Models](https://huggingface.co/OpenGVLab)
  - [Qwen-VL Models](https://huggingface.co/Qwen)
  - [Molmo Models](https://huggingface.co/allenai)

### ModelScope
Chinese model hub with VLMs.

- **Link**: [ModelScope](https://modelscope.cn/)
- **Notable Models**: Qwen-VL series

---

## Tutorials and Guides

### Official Documentation
- [LLaVA Documentation](https://github.com/haotian-liu/LLaVA/tree/main/docs)
- [InternVL Documentation](https://internvl.readthedocs.io/)
- [Qwen-VL Documentation](https://qwen.readthedocs.io/)

### Community Tutorials
- [Hugging Face VLM Guide](https://huggingface.co/blog/vlms)
- [LearnOpenCV VLM Evaluation](https://learnopencv.com/vlm-evaluation-metrics/)
- [Voxel51 VLM Blog](https://voxel51.com/blog/)

### Video Tutorials
- Search "Vision Language Model tutorial" on YouTube
- Hugging Face YouTube channel
- Papers explained channels

---

## Development Tools

### Transformers
Hugging Face's main library for VLMs.

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
```

### OpenAI API (for GPT-4V)
```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }]
)
```

### Anthropic API (for Claude Vision)
```python
import anthropic

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
            {"type": "text", "text": "Describe this image."}
        ]
    }]
)
```

---

## Hardware Requirements

### Model Size to GPU Memory Guide

| Model Size | Minimum VRAM | Recommended VRAM |
|------------|--------------|------------------|
| 1-3B | 8GB | 16GB |
| 7B | 16GB | 24GB |
| 13B | 24GB | 32GB |
| 34B | 48GB | 80GB |
| 72B+ | 80GB+ | Multi-GPU |

### Quantization Options

| Method | Memory Reduction | Quality Impact |
|--------|------------------|----------------|
| FP16 | 2x | Minimal |
| INT8 | 4x | Small |
| INT4 | 8x | Moderate |
| GPTQ | 8x | Small |
| AWQ | 8x | Small |

---

## Community

### Discord Servers
- Hugging Face Discord
- LLaVA Discord
- AI/ML Discord communities

### Forums
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [Reddit r/MachineLearning](https://reddit.com/r/MachineLearning)
- [Reddit r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)

### Research Groups
- [OpenGVLab](https://github.com/OpenGVLab) (InternVL)
- [LLaVA-VL](https://github.com/LLaVA-VL)
- [Qwen Team](https://github.com/QwenLM)
- [Allen AI](https://allenai.org/) (Molmo)

---

## Staying Updated

### Newsletters
- [Sebastian Raschka's Magazine](https://magazine.sebastianraschka.com/)
- [The Batch (DeepLearning.AI)](https://www.deeplearning.ai/the-batch/)

### Twitter/X Accounts
- Follow VLM researchers and labs
- Hugging Face (@huggingface)
- AI research labs

### arXiv
- [cs.CV (Computer Vision)](https://arxiv.org/list/cs.CV/recent)
- [cs.CL (Computation and Language)](https://arxiv.org/list/cs.CL/recent)
- [cs.LG (Machine Learning)](https://arxiv.org/list/cs.LG/recent)
