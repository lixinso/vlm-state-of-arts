# VLM Benchmarks

This directory contains detailed information about benchmarks used to evaluate Vision Language Models.

## Benchmark Categories

### General Multimodal Understanding
- [MMMU](#mmmu) - College-level multimodal understanding
- [MMBench](#mmbench) - Comprehensive multimodal evaluation
- [MMStar](#mmstar) - Multi-domain assessment

### Visual Question Answering
- [VQAv2](#vqav2) - General visual Q&A
- [TextVQA](#textvqa) - Text in images
- [DocVQA](#docvqa) - Document understanding
- [ChartQA](#chartqa) - Chart comprehension
- [InfoVQA](#infovqa) - Infographic understanding

### Specialized Benchmarks
- [RealWorldQA](#realworldqa) - Practical scenarios
- [AI2D](#ai2d) - Scientific diagrams
- [POPE](#pope) - Hallucination evaluation
- [MINDCUBE](#mindcube) - Spatial reasoning

---

## MMMU

**Massive Multi-discipline Multimodal Understanding**

### Overview
MMMU evaluates multimodal models on advanced tasks requiring college-level subject knowledge and deliberate reasoning.

### Statistics
| Metric | Value |
|--------|-------|
| Total Questions | 11,550 |
| Image Count | 11,550 |
| Subjects | 30 |
| Disciplines | 6 |

### Disciplines Covered
1. Art & Design
2. Business
3. Science
4. Health & Medicine
5. Humanities & Social Sciences
6. Tech & Engineering

### Question Sources
- College exams
- Quizzes
- Textbooks

### Scoring
- Multiple choice format
- Accuracy percentage
- Subject-wise breakdown available

### Current Leaderboard (Top 5)

| Rank | Model | MMMU Score |
|------|-------|------------|
| 1 | InternVL3-78B | 72.2 |
| 2 | Gemini 2.5 Pro | ~71 |
| 3 | GPT-4o | ~69 |
| 4 | Qwen2.5-VL-72B | ~70 |
| 5 | Claude 3.5 Sonnet | ~68 |

### Resources
- [MMMU Benchmark](https://mmmu-benchmark.github.io/)
- [Leaderboard](https://mmmu-benchmark.github.io/#leaderboard)
- [Paper](https://arxiv.org/abs/2311.16502)

---

## MMBench

**Multi-Modal Benchmark**

### Overview
MMBench provides comprehensive evaluation across 20 different ability dimensions through multiple-choice questions.

### Statistics
| Metric | Value |
|--------|-------|
| Total Questions | 3,217 |
| Ability Dimensions | 20 |
| Languages | English, Chinese |
| Versions | v1.0, v1.1 |

### Ability Dimensions

**Perception Abilities:**
- Object Localization
- Image Quality Assessment
- Attribute Recognition
- Scene Understanding
- Spatial Relationship
- Action Recognition

**Reasoning Abilities:**
- Future Prediction
- Function Reasoning
- Identity Reasoning
- Social Relation
- Physical Relation
- Structuralized Image-Text Understanding

### Evaluation Format
- Multiple choice (A/B/C/D)
- Circular evaluation for robustness
- Split into dev (with answers) and test sets

### Current Performance

| Model | MMBench-EN v1.1 |
|-------|-----------------|
| InternVL3-78B | 84.4 |
| GPT-4o | 83.4 |
| Gemini 1.5 Pro | 81.3 |
| Qwen2.5-VL-72B | 83.1 |

### Resources
- [MMBench](https://opencompass.org.cn/mmbench)
- [GitHub](https://github.com/open-compass/MMBench)

---

## VQAv2

**Visual Question Answering v2**

### Overview
VQAv2 is the foundational benchmark for visual question answering, built on MS-COCO images.

### Statistics
| Metric | Value |
|--------|-------|
| Images | 200,000+ |
| Questions | 1,100,000+ |
| Answer Types | Yes/No, Number, Other |

### Improvements over VQAv1
- More balanced answer distributions
- Reduced language biases
- Complementary image pairs

### Question Types
- **Existence**: "Is there a dog?"
- **Counting**: "How many people?"
- **Color**: "What color is the car?"
- **Spatial**: "Where is the ball?"
- **Action**: "What is the person doing?"

### Performance

| Model | VQAv2 Accuracy |
|-------|----------------|
| Llama 3.2 90B | 73.6 |
| GPT-4V | 77.2 |
| Qwen2.5-VL-72B | ~80 |

### Resources
- [VQA Website](https://visualqa.org/)
- [Dataset](https://visualqa.org/download.html)

---

## TextVQA

### Overview
TextVQA benchmarks visual reasoning that requires reading and interpreting text within images.

### Focus Areas
- Scene text (signs, labels)
- Storefronts
- Advertisements
- Book covers
- Product packaging

### Statistics
| Metric | Value |
|--------|-------|
| Images | 28,408 |
| Questions | 45,336 |
| Unique Answers | 26,263 |

### Key Challenge
Models must:
1. Detect text in the image
2. Read/OCR the text
3. Reason about the text
4. Formulate an answer

### Performance

| Model | TextVQA |
|-------|---------|
| Qwen2.5-VL-72B | 84.3 |
| GPT-4V | 78.0 |
| Llama 3.2 90B | 73.5 |

---

## DocVQA

**Document Visual Question Answering**

### Overview
DocVQA evaluates understanding of document images including forms, tables, and complex layouts.

### Document Types
- Forms
- Tables
- Invoices
- Scientific papers
- Handwritten documents
- Multi-column layouts

### Statistics
| Metric | Value |
|--------|-------|
| Documents | 12,767 |
| Questions | 50,000+ |
| Industries | Various |

### Key Challenges
- Complex layouts
- Small text
- Handwriting recognition
- Table structure understanding
- Multi-page documents

### Performance

| Model | DocVQA |
|-------|--------|
| Qwen2.5-VL-72B | 96.5 |
| GPT-4o | 92.8 |
| InternVL3-78B | 93.1 |

---

## ChartQA

### Overview
Evaluates understanding and reasoning about charts and graphs.

### Chart Types
- Bar charts
- Line graphs
- Pie charts
- Scatter plots
- Complex multi-series charts

### Question Types
- Data extraction
- Trend analysis
- Comparison
- Calculation

### Performance

| Model | ChartQA |
|-------|---------|
| Qwen2.5-VL-72B | 88.3 |
| GPT-4o | 85.7 |
| Molmo-72B | 87.5 |

---

## RealWorldQA

### Overview
Tests models on practical, real-world visual understanding scenarios.

### Categories
- Everyday objects
- Street scenes
- Indoor environments
- Actions and activities
- Social interactions

### Performance

| Model | RealWorldQA |
|-------|-------------|
| InternVL3-78B | 70.0 |
| GPT-4o | 68.9 |
| Gemini 1.5 Pro | 67.5 |

---

## POPE

**Polling-based Object Probing Evaluation**

### Overview
POPE specifically evaluates hallucination in VLMs - whether models claim to see objects that don't exist.

### Evaluation Method
- Ask about object presence
- Mix existing and non-existing objects
- Measure false positive rate

### Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Hallucination Rate

### Performance

| Model | POPE Accuracy |
|-------|---------------|
| LLaVA-OV-7B | 87.2 |
| LLaVA-1.5-13B | 85.9 |
| InternVL2 | 88.0 |

---

## MINDCUBE

### Overview
A challenging benchmark evaluating spatial mental model formation from limited viewpoints.

### Core Capabilities Tested
1. **Position Representation**: Understanding where objects are
2. **Orientation Understanding**: Grasping object orientations
3. **Dynamics Simulation**: Predicting movement and change

### Statistics
| Metric | Value |
|--------|-------|
| Questions | 21,154 |
| Images | 3,268 |
| Tasks | 3 core capabilities |

### Key Finding
Current state-of-the-art VLMs perform only marginally better than random guessing on these spatial reasoning tasks, highlighting a major area for improvement.

---

## Evaluation Tools

### VLMEvalKit

Open-source toolkit supporting 220+ models and 80+ benchmarks.

**Supported Benchmarks:**
- HLE-Bench
- MMVP
- MM-AlignBench
- Creation-MMBench
- MM-IFEval
- OmniDocBench
- OCR-Reasoning
- And many more...

**GitHub:** [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)

### Open VLM Leaderboard

Hugging Face space tracking VLM performance across benchmarks.

**Link:** [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)

---

## Benchmark Selection Guide

| Use Case | Recommended Benchmark |
|----------|----------------------|
| General capability | MMMU, MMBench |
| OCR/Text reading | TextVQA, DocVQA |
| Scientific reasoning | MMMU, AI2D |
| Hallucination testing | POPE |
| Chart/Graph understanding | ChartQA, InfoVQA |
| Spatial reasoning | MINDCUBE |
| Real-world applications | RealWorldQA |
