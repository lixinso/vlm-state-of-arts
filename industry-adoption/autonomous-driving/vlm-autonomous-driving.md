# VLM Adoption in Self-Driving Car Industry

---

## Abstract

Vision Language Models (VLMs) are rapidly transforming the autonomous driving industry by enabling vehicles to understand complex driving scenarios through the integration of visual perception and natural language reasoning. This document examines the adoption of VLMs across major autonomous driving companies, analyzes the emerging Vision-Language-Action (VLA) paradigm, and evaluates the technological landscape as of 2025-2026. We find that companies like Waymo, Li Auto, and XPeng are leading VLM integration, with Chinese EV manufacturers showing particularly aggressive adoption strategies.

---

## 1. Introduction

### 1.1 Background

The autonomous driving industry has evolved from rule-based systems to deep learning approaches, and now to foundation model-based architectures. Vision Language Models represent the latest paradigm shift, enabling vehicles to leverage the common-sense reasoning and world knowledge embedded in large language models while maintaining robust visual perception capabilities.

### 1.2 Why VLMs for Autonomous Driving?

Traditional autonomous driving systems face challenges with:
- **Long-tail scenarios**: Rare events not well-represented in training data
- **Semantic understanding**: Understanding context beyond object detection
- **Generalization**: Adapting to novel situations without explicit programming

VLMs address these challenges by:
- Injecting human-like common sense and logical reasoning
- Understanding complex semantic scenarios (e.g., "vehicle on fire ahead")
- Providing natural language explanations for driving decisions
- Enabling better generalization through pre-trained world knowledge

---

## 2. The VLA Paradigm Shift

### 2.1 From Modular to End-to-End to VLA

The autonomous driving architecture has evolved through three major paradigms:

```
┌─────────────────────────────────────────────────────────────────────┐
│  PARADIGM 1: Modular Pipeline (Traditional)                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐         │
│  │Perception│ → │Prediction│ → │Planning  │ → │Control   │         │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘         │
│  - Separate hand-crafted modules                                    │
│  - Explicit interfaces between components                           │
│  - Limited by module boundaries                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  PARADIGM 2: End-to-End (E2E)                                       │
│  ┌──────────────────────────────────────────────────────┐          │
│  │     Sensors → Neural Network → Trajectory/Control    │          │
│  └──────────────────────────────────────────────────────┘          │
│  - Single unified network                                           │
│  - Data-driven optimization                                         │
│  - Lacks common-sense reasoning                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  PARADIGM 3: Vision-Language-Action (VLA)                           │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  Vision ─┐                                           │          │
│  │          ├→ Multimodal LLM → Reasoning → Actions    │          │
│  │  Language┘                                           │          │
│  └──────────────────────────────────────────────────────┘          │
│  - Combines perception, reasoning, and action                       │
│  - Leverages LLM world knowledge                                    │
│  - Human-like decision making                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 VLA Architecture Benefits

| Capability | Traditional E2E | VLA Model |
|------------|-----------------|-----------|
| Object Detection | ✓ | ✓ |
| Trajectory Planning | ✓ | ✓ |
| Common-sense Reasoning | ✗ | ✓ |
| Long-tail Handling | Limited | Strong |
| Explainability | ✗ | ✓ |
| Generalization | Limited | Strong |

### 2.3 Implementation Approaches

**End-to-End VLA:**
Unified architecture where VLM directly outputs driving actions.
- Example: Waymo's EMMA model
- Pros: Seamless integration, joint optimization
- Cons: High computational requirements

**Dual-System VLA:**
VLM provides high-level reasoning while specialized module handles control.
- Example: Li Auto's dual-system architecture
- Pros: Lower latency, practical deployment
- Cons: Potential information loss at interface

---

## 3. Industry Adoption Analysis

### 3.1 Waymo (Alphabet)

**VLM Strategy:** Production deployment of Gemini-based "Driving VLM"

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Waymo Foundation Model                        │
├─────────────────────────────────────────────────────────────────┤
│  Sensors (29 cameras, 6 radar, 5 lidar)                         │
│           ↓                                                      │
│  ┌─────────────────────────────────────────────┐                │
│  │  Encoder: Feature extraction + Fusion       │                │
│  │  (Spatial and temporal compression)         │                │
│  └─────────────────────────────────────────────┘                │
│           ↓                                                      │
│  ┌─────────────────────────────────────────────┐                │
│  │  Driving VLM (Gemini-based)                 │                │
│  │  - Fine-tuned on Waymo driving data         │                │
│  │  - Complex semantic reasoning               │                │
│  │  - Leverages Gemini world knowledge         │                │
│  └─────────────────────────────────────────────┘                │
│           ↓                                                      │
│  ┌─────────────────────────────────────────────┐                │
│  │  Decoder: Trajectory prediction             │                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

**Key Capabilities:**
- Semantic understanding of rare scenarios
- Example: Vehicle on fire → VLM signals to reroute (not just "space is clear")
- EMMA: Joint object detection and motion planning VLM

**Deployment Status:**
- 71 million rider-only miles through March 2025
- Robotaxi service operational in multiple US cities
- Production VLM integration confirmed

### 3.2 Tesla

**VLM Strategy:** Vision-only approach with massive data advantage

**Key Characteristics:**
- 8 cameras only (no lidar/radar)
- ~50 billion miles/year from "shadow mode" training data
- End-to-end neural network focus
- Less public disclosure of VLM-specific usage

**Recent Developments:**
- Robotaxi service launched in Austin (June 2025)
- FSD (Full Self-Driving) continuous improvements
- Emphasis on data quantity over sensor diversity

**Comparison with Waymo:**

| Aspect | Waymo | Tesla |
|--------|-------|-------|
| Sensors | 29 cameras + lidar + radar | 8 cameras only |
| VLM Integration | Explicit (Gemini) | Undisclosed |
| Training Data | 71M miles (rider) | ~50B miles/year (shadow) |
| Deployment | Robotaxi fleet | Consumer vehicles |
| Approach | Sensor fusion + VLM | Vision-only E2E |

### 3.3 Li Auto (China)

**VLM Strategy:** Most aggressive VLA adoption among automakers

**Architecture Evolution:**
```
2024: E2E + VLM Dual System
      ┌─────────────┐     ┌─────────────┐
      │ E2E Model   │     │ VLM Model   │
      │ (Fast)      │     │ (Reasoning) │
      └──────┬──────┘     └──────┬──────┘
             └──────────┬────────┘
                        ↓
                  Driving Actions

2025: Mind VLA (Unified)
      ┌─────────────────────────────────┐
      │     Vision-Language-Action      │
      │     (Single Unified Model)      │
      └─────────────────────────────────┘
                        ↓
                  Driving Actions
```

**Technical Specifications:**
- VLM model: 2.2B parameters (constrained for latency)
- Dual-chip system: One for E2E, one for compressed VLM
- Mind VLA: Mass production Q3 2025
- Deployed to all AD MAX vehicle users

**Key Innovations:**
- First automaker to mass-produce VLA system
- Cloud-based generative world model for validation
- Targeting supervised L3 autonomous driving

### 3.4 XPeng (China)

**VLM Strategy:** Custom silicon + massive cloud model

**Hardware:**
- **Turing Chip**: Self-developed AI chip
  - 700 TOPS AI computing power
  - Supports up to 30B parameter models on-vehicle
  - Mass production June 2025
  - First deployed in 2025 XPeng G7

**Software:**
- **VLA 2.0**: On-vehicle system with VLM capabilities
- **XPeng World Base Model**: 72B parameter cloud model
  - LLM backbone
  - Multi-modal driving data training
  - Visual understanding + chained reasoning + action generation

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                      XPeng AI Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Cloud: XPeng World Base Model (72B parameters)         │    │
│  │  - Training and validation                              │    │
│  │  - Complex scenario understanding                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            ↕ Sync                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Vehicle: Turing Chip + VLA 2.0                         │    │
│  │  - 700 TOPS computing power                             │    │
│  │  - Real-time inference                                  │    │
│  │  - Up to 30B parameters supported                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Target:** Level 4 autonomous driving capabilities

### 3.5 NIO (China)

**VLM Strategy:** AGI-focused with custom chip development

**Key Developments:**
- **Shenji NX9031**: Self-developed intelligent driving SoC
- **World Model**: Cloud-based + in-vehicle deployment
- **AGI Committee**: Established to drive AI adoption across business

**Focus Areas:**
- AI integration across R&D, manufacturing, supply chain
- Cloud-vehicle world model synchronization
- Heavy AI investment planned for 2026

### 3.6 Other Notable Players

| Company | VLM Approach | Status |
|---------|--------------|--------|
| **Huawei** | WEWA architecture with cloud world engine | Powering multiple OEMs |
| **Nvidia** | Thor chip (next-gen) | Mass production 2025 |
| **Mobileye** | EyeQ Ultra + RSS | Integrated approach |
| **Baidu Apollo** | Cloud-based VLM | Robotaxi deployment |

---

## 4. Technical Challenges

### 4.1 Computational Constraints

**On-Vehicle Limitations:**
- Current chips (NVIDIA Orin): Models capped at ~4B parameters
- Latency requirements: Real-time inference mandatory
- Power consumption: Limited by vehicle electrical system

**Solution Approaches:**
| Approach | Description | Example |
|----------|-------------|---------|
| Model Compression | Reduce VLM to fit on-vehicle | Li Auto 2.2B VLM |
| Dual-System | Separate fast/slow processing | Li Auto dual-chip |
| Cloud Offload | Heavy reasoning in cloud | XPeng World Model |
| Custom Silicon | Purpose-built AI chips | XPeng Turing, NIO Shenji |

**Timeline Prediction:**
> "The chip that can support the delivery of the VLA model in vehicles may only appear in 2026."

### 4.2 Latency Requirements

| System Type | Acceptable Latency | Challenge |
|-------------|-------------------|-----------|
| Perception | <50ms | VLM inference too slow |
| Planning | <100ms | Reasoning bottleneck |
| Control | <10ms | Direct action only |

**Mitigation Strategies:**
- Asynchronous VLM reasoning (non-blocking)
- Cached reasoning for common scenarios
- Hierarchical decision making

### 4.3 Safety and Validation

**Challenges:**
- VLM outputs can be unpredictable
- Hallucination risk in safety-critical decisions
- Regulatory uncertainty for AI-based driving

**Industry Approaches:**
- Waymo: "Demonstrably Safe AI" framework
- Li Auto: Cloud reconfiguration + generative world model validation
- XPeng: Extensive simulation with 72B cloud model

---

## 5. Benchmarks and Evaluation

### 5.1 Academic Benchmarks

**VLADBench:**
- 5 key domains, 11 secondary aspects, 29 tertiary tasks
- ~2,000 static scenes, ~3,000 dynamic scenarios
- 12,000 close-form questions

**Performance Results (2025):**

| Model | Benchmark | Score |
|-------|-----------|-------|
| Qwen2.5-VL-72B | RoboSense Challenge (IROS 2025) | 70.87% (clean), 72.85% (corrupted) |
| SOTA | ICCV 2025 Autonomous Grand Challenge | EPDMS 49.12 |

### 5.2 Real-World Metrics

| Company | Metric | Value |
|---------|--------|-------|
| Waymo | Rider-only miles | 71 million (March 2025) |
| Tesla | Shadow mode miles/year | ~50 billion |
| Li Auto | VLA deployment | All AD MAX vehicles |

---

## 6. Market Landscape

### 6.1 Computing Power Race

The three leading Chinese EV makers entered "four-digit computing power club" in 2025:

| Company | Chip | Computing Power | Status |
|---------|------|-----------------|--------|
| NIO | Shenji NX9031 | 1000+ TOPS | ES6/EC6 deployment |
| Li Auto | NVIDIA Thor | 1000+ TOPS | L Series Smart Refresh |
| XPeng | Turing | 700 TOPS | G7 debut |

### 6.2 Investment Trends

**Research Report Findings (2025):**
- End-to-end autonomous driving research intensifying
- VLA models identified as "key springboard" to full autonomy
- Chinese OEMs leading in AI-defined vehicle strategies

### 6.3 Competitive Positioning

```
                    VLM Integration Maturity
                    Low ←─────────────────→ High

    Aggressive  ┌─────────────────────────────────┐
    Deployment  │                    Li Auto      │
         ↑      │              XPeng    ●         │
                │                  ●              │
                │         NIO                     │
                │           ●      Waymo          │
                │                    ●            │
                │    Tesla                        │
                │      ●                          │
         ↓      │                                 │
    Conservative└─────────────────────────────────┘
```

---

## 7. Future Outlook

### 7.1 Short-term (2025-2026)

- VLA-capable chips entering mass production
- Li Auto, XPeng leading VLA deployment
- Waymo expanding Gemini VLM integration
- Tesla potentially revealing more AI architecture details

### 7.2 Medium-term (2026-2028)

- Level 4 autonomous driving in geofenced areas
- VLA becoming standard architecture
- Cloud-vehicle VLM synchronization maturing
- Regulatory frameworks adapting to AI-based driving

### 7.3 Long-term Predictions

- Convergence toward unified VLA architectures
- Custom AI chips becoming differentiator
- World models enabling better simulation and validation
- Human-like driving behavior through VLM reasoning

---

## 8. Conclusions

### 8.1 Key Findings

1. **VLA is the emerging paradigm**: The industry is converging on Vision-Language-Action models that unify perception, reasoning, and control.

2. **Chinese automakers lead adoption**: Li Auto and XPeng are ahead in mass-producing VLA systems, while Western companies (Waymo, Tesla) take different approaches.

3. **Waymo leads VLM deployment**: Production use of Gemini VLM for semantic reasoning in robotaxis.

4. **Compute is the bottleneck**: On-vehicle VLM deployment limited by chip capabilities; 2026 expected for VLA-capable silicon.

5. **Dual-system architecture dominates near-term**: Practical deployments use separate fast (E2E) and slow (VLM reasoning) systems.

### 8.2 Ranking: VLM Adoption Leaders

| Rank | Company | Rationale |
|------|---------|-----------|
| 1 | **Waymo** | Production Gemini VLM, demonstrable safety, robotaxi scale |
| 2 | **Li Auto** | First VLA mass production, aggressive rollout, technical transparency |
| 3 | **XPeng** | Custom chip + 72B model, VLA 2.0, Level 4 targeting |
| 4 | **NIO** | Self-developed chip, AGI focus, world model deployment |
| 5 | **Tesla** | Massive data advantage, but less VLM transparency |

### 8.3 Recommendations

**For Automakers:**
- Invest in VLA architecture development
- Consider custom AI silicon for competitive advantage
- Balance cloud and on-vehicle VLM capabilities

**For Researchers:**
- Focus on efficient VLM architectures for edge deployment
- Develop better benchmarks for driving-specific VLM evaluation
- Address safety validation for VLM-based decisions

**For Regulators:**
- Develop frameworks for AI-based driving system certification
- Consider explainability requirements for VLM decisions
- Balance innovation with safety requirements

---

## References

1. Waymo. (2025). "Demonstrably Safe AI For Autonomous Driving." Waymo Blog.

2. Li, et al. (2025). "Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future." arXiv:2512.16760.

3. Jiang, et al. (2025). "A Survey on Vision-Language-Action Models for Autonomous Driving." ICCV Workshop.

4. Research and Markets. (2025). "End-to-End Autonomous Driving Research Report, 2025."

5. Business Wire. (2025). "Chinese OEMs' AI-Defined Vehicle Strategy Report 2025."

6. CnEVPost. (2026). "How do NIO, XPeng, and Li Auto differ in AI approaches?"

7. 36Kr. (2025). "Assisted Driving Models Growing Larger: XPeng and Li Auto First Enter 7-Billion Parameter Order of Magnitude."

8. Li, et al. (2025). "Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving." ICCV 2025.

9. Zhou, et al. (2024). "VLM-AD: End-to-End Autonomous Driving through Vision-Language Model Supervision." arXiv:2412.14446.

10. Awesome-VLM-AD-ITS. GitHub Repository. https://github.com/ge25nab/Awesome-VLM-AD-ITS

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **VLM** | Vision Language Model - AI model combining visual and language understanding |
| **VLA** | Vision-Language-Action - Architecture unifying perception, reasoning, and control |
| **E2E** | End-to-End - Single neural network from sensors to actions |
| **TOPS** | Tera Operations Per Second - Measure of AI computing power |
| **FSD** | Full Self-Driving - Tesla's autonomous driving system |
| **EMMA** | Waymo's end-to-end multimodal model for autonomous driving |

---

## Appendix B: Company VLM Specifications

| Company | VLM Model | Parameters | Chip | Deployment |
|---------|-----------|------------|------|------------|
| Waymo | Gemini-based | Undisclosed | Custom | Production |
| Li Auto | Mind VLA | 2.2B (vehicle) | Dual Orin/Thor | All AD MAX |
| XPeng | VLA 2.0 + World Model | 30B (vehicle), 72B (cloud) | Turing (700 TOPS) | G7, Ultra |
| NIO | World Model | Undisclosed | Shenji NX9031 | ES6/EC6 |
| Tesla | FSD Neural Net | Undisclosed | HW4 | All vehicles |

---

*Last updated: January 2026*
