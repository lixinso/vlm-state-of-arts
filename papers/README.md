# Key Papers in Vision Language Models

A curated collection of influential papers in the development of Vision Language Models.

## Foundational Papers

### CLIP (2021)
**Learning Transferable Visual Models From Natural Language Supervision**

- **Authors**: Radford, A., Kim, J.W., Hallacy, C., et al. (OpenAI)
- **Link**: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- **Impact**: Introduced contrastive language-image pre-training, enabling zero-shot image classification
- **Key Contribution**: Showed that training on 400M image-text pairs enables strong transfer learning

```bibtex
@article{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  journal={ICML},
  year={2021}
}
```

### ViT (2020)
**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**

- **Authors**: Dosovitskiy, A., et al. (Google)
- **Link**: [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- **Impact**: Demonstrated that pure transformer architectures work for vision tasks
- **Key Contribution**: Vision Transformer architecture now standard in VLMs

---

## Instruction-Tuned VLMs

### LLaVA (2023)
**Visual Instruction Tuning**

- **Authors**: Liu, H., Li, C., Wu, Q., Lee, Y.J.
- **Link**: [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
- **Impact**: Pioneered visual instruction tuning methodology
- **Key Contribution**: Simple but effective approach to connect vision encoders with LLMs

```bibtex
@article{liu2023llava,
  title={Visual instruction tuning},
  author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
  journal={NeurIPS},
  year={2023}
}
```

### LLaVA-1.5 (2023)
**Improved Baselines with Visual Instruction Tuning**

- **Authors**: Liu, H., et al.
- **Link**: [arXiv:2310.03744](https://arxiv.org/abs/2310.03744)
- **Impact**: Showed that simple improvements significantly boost performance
- **Key Contribution**: MLP projection, higher resolution, better training data

### InstructBLIP (2023)
**Towards General-purpose Vision-Language Models with Instruction Tuning**

- **Authors**: Dai, W., et al. (Salesforce)
- **Link**: [arXiv:2305.06500](https://arxiv.org/abs/2305.06500)
- **Impact**: Q-Former based instruction tuning
- **Key Contribution**: Efficient vision-language alignment

---

## State-of-the-Art Open Models

### InternVL (2024)
**Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks**

- **Authors**: Chen, Z., et al. (Shanghai AI Lab)
- **Link**: [arXiv:2404.16821](https://arxiv.org/abs/2404.16821)
- **Impact**: Achieved SOTA on MMMU among open-source models
- **Key Contribution**: InternViT vision encoder, progressive training

```bibtex
@article{chen2024internvl,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
```

### Qwen-VL (2023-2024)
**Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond**

- **Authors**: Bai, J., et al. (Alibaba)
- **Link**: [arXiv:2308.12966](https://arxiv.org/abs/2308.12966)
- **Impact**: Strong OCR and document understanding
- **Key Contribution**: Native dynamic resolution processing

### Qwen2-VL (2024)
**Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution**

- **Authors**: Wang, P., et al. (Alibaba)
- **Link**: [arXiv:2409.12191](https://arxiv.org/abs/2409.12191)
- **Impact**: Hour-long video understanding
- **Key Contribution**: Improved dynamic resolution, video processing

### Molmo (2024)
**Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models**

- **Authors**: Deitke, M., et al. (Allen Institute for AI)
- **Link**: [arXiv:2409.17146](https://arxiv.org/abs/2409.17146)
- **Impact**: Fully open-source SOTA VLM
- **Key Contribution**: PixMo dataset, pointing capability

```bibtex
@article{deitke2024molmo,
  title={Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models},
  author={Deitke, Matt and others},
  journal={arXiv preprint arXiv:2409.17146},
  year={2024}
}
```

---

## Efficient VLMs

### MiniCPM-V (2024)
**MiniCPM-V: A GPT-4V Level MLLM on Your Phone**

- **Authors**: Yao, Y., et al.
- **Link**: [arXiv:2408.01800](https://arxiv.org/abs/2408.01800)
- **Impact**: Efficient VLM for edge devices
- **Key Contribution**: Strong performance at small scale

### Phi-3-Vision (2024)
**Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone**

- **Authors**: Abdin, M., et al. (Microsoft)
- **Link**: [arXiv:2404.14219](https://arxiv.org/abs/2404.14219)
- **Impact**: Small but capable multimodal model
- **Key Contribution**: High quality data curation

### Small VLM Survey (2025)
**Scaling down, Powering up: A Survey on the Advancements of Small Vision-Language Models**

- **Authors**: Various
- **Link**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S156625352500867X)
- **Impact**: Comprehensive survey of efficient VLMs
- **Key Contribution**: Taxonomy and analysis of small VLMs

---

## Benchmark Papers

### MMMU (2023)
**MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI**

- **Authors**: Yue, X., et al.
- **Link**: [arXiv:2311.16502](https://arxiv.org/abs/2311.16502)
- **Impact**: Standard benchmark for VLM evaluation
- **Key Contribution**: College-level multimodal reasoning

### MMBench (2023)
**MMBench: Is Your Multi-modal Model an All-around Player?**

- **Authors**: Liu, Y., et al.
- **Link**: [arXiv:2307.06281](https://arxiv.org/abs/2307.06281)
- **Impact**: Comprehensive capability evaluation
- **Key Contribution**: 20 ability dimensions

### VQAv2 (2017)
**Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering**

- **Authors**: Goyal, Y., et al.
- **Link**: [arXiv:1612.00837](https://arxiv.org/abs/1612.00837)
- **Impact**: Foundational VQA benchmark
- **Key Contribution**: Balanced answer distributions

---

## Survey Papers

### State of the Art Survey (2025)
**A Survey of State of the Art Large Vision Language Models: Alignment, Benchmark, Evaluations and Challenges**

- **Link**: [arXiv:2501.02189](https://arxiv.org/abs/2501.02189)
- **Impact**: Comprehensive overview of VLM landscape
- **Key Topics**: Alignment methods, benchmarks, challenges

### Comprehensive VLM Survey (2025)
**A comprehensive survey of Vision–Language Models: Pretrained models, fine-tuning, prompt engineering, adapters, and benchmark datasets**

- **Link**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1566253525006955)
- **Impact**: Full coverage of VLM techniques
- **Key Topics**: Training methods, adaptation techniques

### MLLM Survey (2024)
**A Survey on Benchmarks of Multimodal Large Language Models**

- **Link**: [GitHub](https://github.com/swordlidev/Evaluation-Multimodal-LLMs-Survey)
- **Impact**: Benchmark-focused survey
- **Key Topics**: Evaluation methodologies

---

## Architectural Innovations

### SigLIP (2023)
**Sigmoid Loss for Language Image Pre-Training**

- **Authors**: Zhai, X., et al. (Google)
- **Link**: [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
- **Impact**: Improved vision encoder training
- **Key Contribution**: Efficient contrastive learning

### Q-Former / BLIP-2 (2023)
**BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**

- **Authors**: Li, J., et al. (Salesforce)
- **Link**: [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)
- **Impact**: Efficient vision-language bridging
- **Key Contribution**: Querying transformer for modality alignment

---

## Reading Order Recommendations

### For Beginners
1. CLIP - Foundational concepts
2. ViT - Vision transformer basics
3. LLaVA - Simple VLM architecture
4. MMMU - Understanding evaluation

### For Researchers
1. InternVL - Current SOTA techniques
2. Qwen2-VL - Advanced capabilities
3. Molmo - Open research methodology
4. Survey papers - Comprehensive overview

### For Practitioners
1. LLaVA-1.5 - Practical improvements
2. MiniCPM-V - Efficient deployment
3. Benchmark papers - Evaluation guidance
4. Qwen-VL - Production-ready models

---

## Resources

### Paper Collections
- [Papers With Code - VQA](https://paperswithcode.com/task/visual-question-answering)
- [Awesome VLM Architectures](https://github.com/gokayfem/awesome-vlm-architectures)
- [Hugging Face Papers](https://huggingface.co/papers)

### Staying Updated
- [arXiv cs.CV](https://arxiv.org/list/cs.CV/recent)
- [arXiv cs.CL](https://arxiv.org/list/cs.CL/recent)
- [Hugging Face Blog](https://huggingface.co/blog)
