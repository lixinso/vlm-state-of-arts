# Vision Language Models: A Comprehensive Literature Review

## Abstract

Vision Language Models (VLMs) represent a significant advancement in artificial intelligence, enabling systems to understand and reason about both visual and textual information. This literature review surveys the evolution of VLM architectures, training methodologies, evaluation benchmarks, and real-world applications. We analyze key developments from foundational models like CLIP to state-of-the-art systems including GPT-4V, Gemini, and open-source leaders such as InternVL3 and Qwen2.5-VL. The review identifies current challenges including spatial reasoning, hallucination, and efficiency, while highlighting emerging trends in small VLMs, inference-time scaling, and agentic applications.

---

## 1. Introduction

### 1.1 Background

The integration of vision and language understanding has been a long-standing goal in artificial intelligence. Traditional approaches treated these modalities separately, with computer vision systems analyzing images and natural language processing systems handling text. Vision Language Models bridge this gap by creating unified systems capable of multimodal understanding.

### 1.2 Scope and Objectives

This review covers:
- Architectural evolution from early vision-language models to modern VLMs
- Training paradigms including pre-training, alignment, and instruction tuning
- Evaluation methodologies and benchmark landscapes
- Practical applications and deployment considerations
- Current limitations and future research directions

### 1.3 Significance

VLMs have enabled transformative applications including:
- Visual question answering and image captioning
- Document understanding and OCR
- GUI automation and computer use
- Multimodal reasoning and scientific analysis
- Accessible AI systems for visually impaired users

---

## 2. Architectural Evolution

### 2.1 Foundational Approaches

#### 2.1.1 CLIP (2021)

Radford et al. introduced Contrastive Language-Image Pre-training (CLIP), demonstrating that training on 400 million image-text pairs enables strong zero-shot transfer capabilities. CLIP's dual-encoder architecture established the foundation for modern VLM vision encoders.

**Key contributions:**
- Contrastive learning between image and text embeddings
- Zero-shot classification without task-specific training
- Scalable pre-training on web-scraped data

#### 2.1.2 Vision Transformer (ViT)

Dosovitskiy et al. showed that pure transformer architectures, originally designed for NLP, could achieve state-of-the-art results on image classification. ViT's patch-based approach became the standard for VLM vision encoders.

**Architecture:**
```
Image → Patches (16x16) → Linear Projection → Transformer Encoder → Embeddings
```

### 2.2 Modern VLM Architectures

Contemporary VLMs share a common three-component architecture:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Vision Encoder  │ ──► │ Projection Layer │ ──► │ Language Model  │
│ (ViT, SigLIP)   │     │ (MLP, Q-Former)  │     │ (LLaMA, Qwen)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

#### 2.2.1 Vision Encoders

| Encoder | Origin | Key Features |
|---------|--------|--------------|
| CLIP ViT | OpenAI | Contrastive pre-training |
| SigLIP | Google | Sigmoid loss, efficient training |
| InternViT | Shanghai AI Lab | Optimized for VLM alignment |
| EVA-CLIP | BAAI | Enhanced visual representations |

#### 2.2.2 Projection Methods

**MLP Projection (LLaVA approach):**
Simple but effective two-layer MLP for aligning visual features with language model embedding space. Liu et al. demonstrated that complex fusion mechanisms are unnecessary for strong performance.

**Q-Former (BLIP-2 approach):**
Li et al. introduced a querying transformer that uses learnable query tokens to extract relevant information from frozen vision encoders, reducing the number of visual tokens fed to the language model.

**Cross-Attention:**
Used by models like Flamingo, cross-attention layers interleave visual and textual processing, allowing more fine-grained integration of modalities.

#### 2.2.3 Language Model Backbones

Modern VLMs leverage powerful pre-trained LLMs:

| LLM Family | Notable VLMs Using It |
|------------|----------------------|
| LLaMA/LLaMA 2/3 | LLaVA, Molmo |
| Qwen/Qwen2.5 | Qwen-VL, InternVL3 |
| Mistral | Pixtral |
| Vicuna | Early LLaVA versions |
| InternLM | InternVL2 |

### 2.3 Architectural Innovations

#### 2.3.1 Dynamic Resolution Processing

Traditional VLMs resize images to fixed dimensions (e.g., 224×224 or 448×448), losing detail for high-resolution images. Recent models address this:

**Qwen2.5-VL's Native Dynamic Resolution:**
Processes images at their original aspect ratio using adaptive patching, preserving fine details crucial for document understanding and OCR.

**LLaVA-NeXT's AnyRes:**
Divides high-resolution images into tiles, processes each independently, and combines features for comprehensive understanding.

#### 2.3.2 Extended Context Windows

| Model | Context Length |
|-------|---------------|
| GPT-4o | 128K tokens |
| Gemini 1.5 Pro | 2M tokens |
| InternVL3-78B | 256K tokens |
| Qwen2.5-VL-72B | 128K tokens |

Extended context enables processing of long documents, multiple images, and extended video sequences.

---

## 3. Training Methodologies

### 3.1 Pre-training Paradigms

#### 3.1.1 Contrastive Learning

CLIP-style contrastive learning aligns image and text representations in a shared embedding space. Given a batch of image-text pairs, the model learns to maximize similarity between matching pairs while minimizing similarity between non-matching pairs.

**Loss function:**
```
L = -log(exp(sim(I_i, T_i)/τ) / Σ_j exp(sim(I_i, T_j)/τ))
```

#### 3.1.2 Generative Pre-training

Models like Flamingo and later VLMs use next-token prediction on interleaved image-text sequences, learning to generate text conditioned on visual context.

### 3.2 Alignment Training

#### 3.2.1 Feature Alignment

The first stage of VLM training typically aligns visual features with the language model's embedding space:

- **Trainable:** Projection layer only
- **Frozen:** Vision encoder + Language model
- **Data:** Image-caption pairs (e.g., CC3M)
- **Objective:** Minimize distance between projected visual features and text embeddings

#### 3.2.2 Visual Instruction Tuning

Liu et al. pioneered visual instruction tuning with LLaVA, creating training data by prompting GPT-4 to generate conversations about images based on their captions and detected objects.

**Data generation pipeline:**
```
Image + Captions + Bounding Boxes → GPT-4 → Instruction-following conversations
```

### 3.3 Fine-tuning Approaches

#### 3.3.1 Full Fine-tuning

Updates all model parameters on task-specific data. Most effective but computationally expensive.

#### 3.3.2 LoRA (Low-Rank Adaptation)

Hu et al. introduced LoRA, which trains low-rank decomposition matrices alongside frozen pre-trained weights, significantly reducing trainable parameters.

```python
# LoRA adds trainable A and B matrices
W_new = W_frozen + A @ B  # where A ∈ R^(d×r), B ∈ R^(r×k), r << d,k
```

#### 3.3.3 QLoRA

Dettmers et al. combined quantization with LoRA, enabling fine-tuning of large models on consumer hardware by keeping base weights in 4-bit precision.

### 3.4 Data Considerations

#### 3.4.1 Pre-training Data Scale

| Dataset | Scale | Usage |
|---------|-------|-------|
| LAION-5B | 5B image-text pairs | Pre-training |
| DataComp | 12.8B candidates | Pre-training |
| CC3M/CC12M | 3-12M pairs | Alignment |
| WebLI | 10B+ pairs | Proprietary pre-training |

#### 3.4.2 Instruction Tuning Data

| Dataset | Size | Source |
|---------|------|--------|
| LLaVA-Instruct | 158K | GPT-4 generated |
| ShareGPT4V | 1.2M | GPT-4V annotated |
| PixMo | - | Human annotated (Molmo) |

---

## 4. Evaluation and Benchmarks

### 4.1 Benchmark Categories

#### 4.1.1 General Multimodal Understanding

**MMMU (Massive Multi-discipline Multimodal Understanding):**
Yue et al. created MMMU with 11,550 questions from college exams spanning six disciplines, requiring expert-level knowledge and deliberate reasoning.

**Current Performance (as of 2025):**
| Model | MMMU Score |
|-------|------------|
| InternVL3-78B | 72.2 |
| Gemini 2.5 Pro | ~71 |
| Qwen2.5-VL-72B | ~70 |
| GPT-4o | ~69 |

**MMBench:**
Evaluates 20 ability dimensions including perception and reasoning through 3,000+ multiple-choice questions.

#### 4.1.2 Visual Question Answering

**VQAv2:**
The foundational VQA benchmark with 1.1M questions on MS-COCO images, balanced to reduce language biases.

**TextVQA:**
Focuses on reading and reasoning about text in natural images—crucial for practical applications.

**DocVQA:**
Evaluates document understanding including forms, tables, and complex layouts.

#### 4.1.3 Specialized Benchmarks

**ChartQA:** Chart and graph comprehension
**InfoVQA:** Infographic understanding
**AI2D:** Scientific diagram interpretation
**RealWorldQA:** Practical real-world scenarios

#### 4.1.4 Hallucination Evaluation

**POPE (Polling-based Object Probing Evaluation):**
Tests whether models claim to see objects that don't exist, measuring hallucination rates through yes/no questions about object presence.

### 4.2 Emerging Evaluation Challenges

#### 4.2.1 Spatial Reasoning

The MINDCUBE benchmark reveals that current VLMs perform only marginally better than random guessing on spatial mental model formation tasks, highlighting a fundamental gap in spatial reasoning capabilities.

#### 4.2.2 Basic Perception

Meta's research showed that scaling data and model size improves perception but not reasoning, with advanced models failing at simple tasks like digit recognition and object counting.

### 4.3 Evaluation Frameworks

**VLMEvalKit:**
Open-source toolkit supporting 220+ models and 80+ benchmarks, enabling standardized evaluation across the field.

**lmms-eval:**
Extension of the Language Model Evaluation Harness for multimodal models.

---

## 5. State-of-the-Art Models

### 5.1 Proprietary Models

#### 5.1.1 GPT-4V / GPT-4o (OpenAI)

GPT-4V introduced production-ready vision capabilities to ChatGPT, later enhanced with GPT-4o's native multimodal architecture.

**Strengths:** General knowledge, instruction following, scientific reasoning
**Limitations:** Spatial reasoning, counting, closed-source

#### 5.1.2 Gemini (Google)

Gemini models offer native multimodality with industry-leading context lengths (up to 2M tokens in Gemini 1.5 Pro).

**Strengths:** Long context, video understanding, multimodal reasoning
**Limitations:** Proprietary, API-only access

#### 5.1.3 Claude (Anthropic)

Claude's vision capabilities emphasize safety and reliability, with strong performance on document analysis and reasoning tasks.

**Strengths:** Safety focus, instruction following, long context (200K)
**Limitations:** Proprietary, some benchmark gaps

### 5.2 Open-Source Models

#### 5.2.1 InternVL Series

InternVL3-78B achieves the highest MMMU score (72.2) among open-source models, using InternViT-6B for vision and Qwen2.5-72B for language.

**Key innovations:**
- Custom InternViT vision encoder
- Progressive multi-stage training
- Strong multilingual support

#### 5.2.2 Qwen-VL Series

Qwen2.5-VL excels at document understanding with native dynamic resolution and hour-long video support.

**Key innovations:**
- Native dynamic resolution processing
- Superior OCR and document understanding
- GUI agent capabilities (32B variant)

#### 5.2.3 LLaVA Family

LLaVA demonstrated that simple architectures (MLP projection) can achieve strong results, inspiring numerous follow-up works.

**Key innovations:**
- Visual instruction tuning methodology
- Simple, reproducible architecture
- Active research community

#### 5.2.4 Molmo

Allen Institute's Molmo provides fully open weights, training data, and code, achieving competitive performance with unique pointing capabilities.

**Key innovations:**
- Fully open (weights, data, code)
- PixMo dataset with human annotations
- Pointing/localization capability

### 5.3 Efficient Models

The shift toward smaller, efficient VLMs addresses deployment constraints:

| Model | Parameters | Target Use Case |
|-------|------------|-----------------|
| Gemma 3-4B | 4B | Edge devices |
| MiniCPM-V | 3B | Mobile deployment |
| Qwen2.5-VL-3B | 3B | Resource-constrained |
| Molmo-1B | 1B | Embedded systems |

---

## 6. Applications

### 6.1 Document Understanding

VLMs have transformed document processing:
- **Invoice extraction:** Automatic data extraction from invoices
- **Form processing:** Understanding complex form layouts
- **Contract analysis:** Identifying key terms and clauses
- **Receipt OCR:** Expense management automation

Qwen2.5-VL-72B achieves 96.5% on DocVQA, surpassing human-level performance on structured documents.

### 6.2 Visual Question Answering

Consumer and enterprise applications:
- Image search and retrieval
- Product visual search
- Medical image analysis
- Accessibility tools for visually impaired users

### 6.3 GUI Agents and Computer Use

Emerging application area where VLMs control computer interfaces:
- **Claude Computer Use:** Anthropic's experimental GUI automation
- **Qwen2.5-VL-32B:** Optimized for GUI understanding
- **Web agents:** Automated web browsing and form filling

### 6.4 Scientific Applications

- **Chart interpretation:** Automated analysis of scientific figures
- **Diagram understanding:** Technical documentation processing
- **Medical imaging:** Radiology report generation (with appropriate oversight)
- **Research assistance:** Literature figure extraction and summarization

### 6.5 Creative Applications

- **Image captioning:** Accessibility and content management
- **Visual storytelling:** Narrative generation from images
- **Design feedback:** Automated design review
- **Content moderation:** Visual content classification

---

## 7. Challenges and Limitations

### 7.1 Hallucination

VLMs can confidently describe objects or attributes not present in images. This remains a critical challenge for deployment in high-stakes applications.

**Mitigation approaches:**
- RLHF training to reduce hallucination
- Explicit uncertainty quantification
- Retrieval augmentation for factual grounding

### 7.2 Spatial Reasoning

MINDCUBE and similar benchmarks reveal that VLMs struggle with:
- Mental rotation tasks
- Spatial relationship understanding
- Object counting and localization
- 3D scene understanding

### 7.3 Efficiency-Accuracy Trade-offs

Large models (70B+) achieve best benchmark scores but face deployment challenges:
- High computational costs
- Latency constraints
- Memory requirements
- Energy consumption

### 7.4 Data Quality and Bias

Training data issues include:
- Web-scraped data quality variations
- Geographic and cultural biases
- Underrepresentation of certain domains
- Synthetic data limitations

### 7.5 Evaluation Gaps

Current benchmarks may not capture:
- Real-world deployment scenarios
- Long-tail distribution handling
- Robustness to adversarial inputs
- Temporal reasoning in videos

---

## 8. Future Directions

### 8.1 Inference-Time Scaling

Raschka predicts that 2026 will see more progress from inference-time scaling than training improvements, with techniques like:
- Test-time computation scaling
- Chain-of-thought prompting
- Self-consistency methods
- Retrieval augmentation

### 8.2 Small VLMs (SVLMs)

Growing research focus on efficient models for:
- Mobile deployment
- Edge computing
- Real-time applications
- Privacy-preserving on-device inference

### 8.3 Video Understanding

Extending from image to video:
- Temporal reasoning
- Long video comprehension (hours)
- Real-time video analysis
- Action recognition and prediction

### 8.4 Agentic Capabilities

VLMs as autonomous agents:
- GUI automation
- Web navigation
- Robotic control
- Multi-step task planning

### 8.5 Improved Grounding

Reducing hallucination through:
- Better training objectives
- Retrieval-augmented generation
- Uncertainty estimation
- Human feedback integration

---

## 9. Conclusion

Vision Language Models have rapidly evolved from research curiosities to production-ready systems achieving human-level performance on many benchmarks. The field has seen remarkable progress in both proprietary models (GPT-4V, Gemini, Claude) and open-source alternatives (InternVL, Qwen-VL, LLaVA, Molmo).

Key takeaways:
1. **Architecture convergence:** Modern VLMs share similar three-component designs (vision encoder, projection, LLM)
2. **Open-source parity:** Open models now match proprietary performance on many benchmarks
3. **Efficiency focus:** Growing emphasis on small, deployable models
4. **Remaining challenges:** Spatial reasoning, hallucination, and efficiency remain open problems
5. **Emerging applications:** GUI agents and agentic capabilities represent exciting frontiers

The democratization of VLM technology through open-source releases has accelerated research and application development. Future progress will likely come from improved training data quality, inference-time scaling, and better integration of visual and spatial reasoning capabilities.

---

## References

### Foundational Papers

1. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML.

2. Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.

3. Liu, H., et al. (2023). Visual Instruction Tuning. NeurIPS.

4. Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML.

### State-of-the-Art Models

5. Chen, Z., et al. (2024). InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. arXiv:2404.16821.

6. Wang, P., et al. (2024). Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv:2409.12191.

7. Deitke, M., et al. (2024). Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models. arXiv:2409.17146.

8. Liu, H., et al. (2024). LLaVA-NeXT: Improved reasoning, OCR, and world knowledge.

### Benchmarks

9. Yue, X., et al. (2023). MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark. arXiv:2311.16502.

10. Liu, Y., et al. (2023). MMBench: Is Your Multi-modal Model an All-around Player? arXiv:2307.06281.

11. Goyal, Y., et al. (2017). Making the V in VQA Matter. CVPR.

### Surveys

12. A Survey of State of the Art Large Vision Language Models: Alignment, Benchmark, Evaluations and Challenges. arXiv:2501.02189 (2025).

13. Scaling down, Powering up: A Survey on the Advancements of Small Vision-Language Models. ScienceDirect (2025).

### Technical Methods

14. Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

15. Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314.

16. Zhai, X., et al. (2023). Sigmoid Loss for Language Image Pre-Training. arXiv:2303.15343.

---

## Appendix A: Model Comparison Table

| Model | Type | Parameters | MMMU | Context | Video | Open |
|-------|------|------------|------|---------|-------|------|
| Gemini 2.5 Pro | Proprietary | - | ~71 | 1M+ | Yes | No |
| GPT-4o | Proprietary | - | ~69 | 128K | Limited | No |
| Claude 3.5 Sonnet | Proprietary | - | ~68 | 200K | No | No |
| InternVL3-78B | Open | 78B | 72.2 | 256K | Yes | Yes |
| Qwen2.5-VL-72B | Open | 72B | ~70 | 128K | Yes | Yes |
| Molmo-72B | Open | 72B | ~68 | 128K | No | Yes |
| LLaVA-OV-72B | Open | ~78B | ~65 | 32K | Yes | Yes |
| Pixtral-12B | Open | 12B | ~55 | 128K | No | Yes |
| Gemma 3-27B | Open | 27B | ~58 | 128K | Short | Yes |

---

## Appendix B: Benchmark Overview

| Benchmark | Focus | Size | Format |
|-----------|-------|------|--------|
| MMMU | College-level reasoning | 11.5K | Multiple choice |
| MMBench | General abilities | 3K+ | Multiple choice |
| VQAv2 | Visual QA | 1.1M | Open-ended |
| TextVQA | Text in images | 45K | Open-ended |
| DocVQA | Documents | 50K | Open-ended |
| ChartQA | Charts | - | Open-ended |
| POPE | Hallucination | - | Yes/No |
| RealWorldQA | Practical | - | Multiple choice |

---

*Last updated: January 2026*
