# **The Convergence of Sight and Thought: A Technical Analysis of Vision-Language Architectures in the 2026 Landscape**

## **1\. Executive Summary**

The early months of 2026 have marked a definitive inflection point in the trajectory of Artificial Intelligence. The longstanding architectural divide between computer vision—historically dominated by Convolutional Neural Networks (CNNs) and later Vision Transformers (ViTs)—and natural language processing has been effectively bridged. The era of "modular" Vision-Language Models (VLMs), characterized by the ad-hoc grafting of frozen visual encoders onto Large Language Models (LLMs) via trainable projection layers, has largely concluded. In its place, a new generation of **native multimodal architectures** has emerged, driven by the imperative to model cross-modal dependencies with the same fidelity, fluidity, and depth as text-only sequences.

This comprehensive research report provides an exhaustive technical analysis of the VLM ecosystem as of the first quarter of 2026\. It synthesizes developments from major proprietary releases, including Meta’s **Llama 4** family and Google’s **Gemini 2.5/3.0** series, alongside groundbreaking open-weight contributions such as Alibaba’s **Qwen3-VL**, DeepSeek’s **Janus-Pro** and **VL2**, and SenseTime’s **NEO**.

The analysis identifies five central pillars of architectural innovation defining the current state of the art:

1. **Native Unification and Early Fusion:** The industry is abandoning separate vision encoders in favor of unified transformer backbones that process raw visual tokens (pixels or patches) directly alongside text. This is exemplified by the **NEO** architecture and **Llama 4’s** early fusion mechanisms, which allow for bidirectional cross-modal attention from the initial layers, eliminating the "information bottleneck" of projection adapters.1  
2. **Spatiotemporal Encoding and Dynamic Resolution:** The rigid fixed-resolution constraints of early ViT-based systems have been replaced by dynamic tiling and "naive" resolution processing. **Qwen2.5-VL** and **Qwen3-VL** utilize **Multimodal Rotary Positional Embeddings (MRoPE)** to map time, width, and height into a unified 3D coordinate system, enabling the seamless handling of images of arbitrary aspect ratios and videos of extended duration.4  
3. **Mixture-of-Experts (MoE) for Multimodal Scalability:** To manage the prohibitively high computational costs of dense multimodal training, architectures have universally adopted sparse MoE designs. **DeepSeek-VL2** and **Llama 4 Maverick** demonstrate that activating a small subset of parameters (e.g., 17B out of 400B) per token allows for massive capacity scaling without linear increases in inference latency. The distinction between "total parameters" (memory cost) and "active parameters" (compute cost) has become the primary metric for model evaluation.7  
4. **System 2 "Thinking" Capabilities in Vision:** Following the trajectory of text-based reasoning models, VLMs in 2026 have integrated "slow thinking" mechanisms. Models like **Qwen3-VL-Thinking** and **Gemini 2.5 Deep-Think** employ reinforced chain-of-thought (CoT) processes to tackle complex visual reasoning tasks found in benchmarks like **Humanity’s Last Exam (HLE)** and **MMMU-Pro**, moving beyond pattern recognition to multi-step deductive reasoning.10  
5. **Native Streaming and Audio:** The boundary between modalities is further dissolving with models like **Gemini 2.0 Flash**, which process raw audio and video streams natively. This eliminates the latency and information loss associated with cascading systems (Speech-to-Text ![][image1] LLM ![][image1] Text-to-Speech), enabling real-time, emotionally resonant human-AI interaction.13

This report dissects these innovations, offering a granular view of the engineering choices, trade-offs, and performance implications that define the 2026 VLM frontier.

## ---

**2\. The Historical Context: The Evolution from Pipelines to Unification (2021–2025)**

To understand the magnitude of the shifts occurring in 2026, one must first contextualize the architectural limitations that characterized the preceding generation of models. The journey of Vision-Language Models has been one of increasing integration, moving from loose coupling to tight fusion.

### **2.1. The Era of Contrastive Alignment (2021–2023)**

The foundation of modern VLMs was laid by **CLIP (Contrastive Language-Image Pre-training)**. These models consisted of two distinct towers: a vision encoder (usually a ViT) and a text encoder. They were aligned via a contrastive loss function that pulled the representations of matched image-text pairs closer together in a shared embedding space.15

* **Limitation:** While powerful for zero-shot classification and retrieval, these models lacked generative capabilities. They could not "speak" about what they saw; they could only output similarity scores.

### **2.2. The Era of Modular Adapters (2023–2024)**

The explosion of Large Language Models (LLMs) led to the "connector" paradigm, popularized by architectures like **LLaVA (Large Language-and-Vision Assistant)** and **Flamingo**.

* **Mechanism:** These models froze a pre-trained vision encoder (e.g., CLIP-ViT-L) and a pre-trained LLM (e.g., Llama 2). They introduced a lightweight trainable adapter—typically a linear projection layer or a Q-Former (Query Transformer)—to translate visual features into the LLM’s token embedding space.2  
* **The Bottleneck:** This approach created a fundamental "information bottleneck." The vision encoder, having been pre-trained on contrastive tasks, often discarded fine-grained visual details (like exact object counts or small text) that were irrelevant to global image similarity but crucial for detailed captioning. Furthermore, the "late fusion" meant the LLM had no influence over the visual processing; it simply received a summarized report of the image.15

### **2.3. The 2025-2026 Shift: Native Multimodality**

By late 2025, the industry recognized that the "Frankenstein" approach of stitching together frozen modules had hit a ceiling. The "Scaling Laws" that applied to text were not transferring perfectly to multimodal tasks because the visual representations were not being learned in service of the language generation objective.17

The response was a move toward **Native Architectures**—models designed from the ground up to process pixels and text as equivalent, interleaved data streams. This shift is characterized by:

1. **End-to-End Training:** The entire network, including the visual processing layers, is trainable.  
2. **Unified Backbones:** A single transformer processes multimodal sequences, allowing for cross-modal attention at every layer.  
3. **Variable Resolution:** The abandonment of fixed ![][image2] or ![][image3] square inputs in favor of handling images in their native aspect ratios.6

## ---

**3\. The Native Unification Paradigm: SenseTime NEO and the Death of the Encoder**

The **NEO** architecture, released in late 2025 by SenseTime and Nanyang Technological University, represents the most radical adherence to the "first principles" of native multimodality. It challenges the entrenched orthodoxy that a VLM requires a separate, pre-trained vision encoder.1

### **3.1. Philosophy of Native Embeddings**

Most VLMs operate on the assumption that visual data requires specialized preprocessing (convolutions or hierarchical attention) before it can be digested by a transformer. NEO rejects this. It posits that a sufficiently large transformer, given raw pixel patches, can learn to construct high-level visual features that are intrinsically aligned with language, without the need for an external "vision teacher" like CLIP.19

### **3.2. Technical Architecture: Native Patch Embedding**

In place of a SigLIP or CLIP tower, NEO introduces a **Native Patch Embedding** layer.

* **Mechanism:** The input image is divided into patches (e.g., ![][image4] pixels). These patches are flattened and projected linearly into the model's embedding dimension ![][image5].  
* **Integration:** These visual embeddings are concatenated directly with text token embeddings. The sequence entering the transformer is $$.  
* **Significance:** By removing the bottleneck of a frozen encoder, NEO ensures **lossless information flow**. High-frequency details—often lost in the pooling layers of traditional encoders—are preserved in the patch tokens, allowing the attention layers to focus on microscopic details when necessary (e.g., reading small text on a distant sign).18

### **3.3. Native Multi-Head Attention**

A critical challenge in unified models is the conflict between the causal (unidirectional) nature of text generation and the non-causal (bidirectional) nature of visual perception. In an image, a pixel at the bottom right is contextually relevant to a pixel at the top left; there is no inherent "time" dimension.

* **NEO's Solution:** NEO implements **Native Multi-Head Attention**. Within the same transformer block, the masking pattern changes dynamically based on the token type.  
  * **Text Tokens:** Attend only to previous tokens (causal mask).  
  * **Visual Tokens:** Attend to all other visual tokens belonging to the same image (bidirectional mask), regardless of their sequence position.  
  * **Cross-Modal:** Text tokens can attend to all visual tokens that appeared earlier in the sequence.18 This "coexistence" of attention patterns allows NEO to maintain the generative coherence of an LLM while leveraging the global context capabilities of a ViT, all within a single set of weights.17

### **3.4. Performance and Data Efficiency**

One of the most startling findings from the NEO technical report is its data efficiency. The model achieves parity with flagship modular models while using only **1/10th of the training data** (approximately 390 million image-text pairs).18

* **Reasoning:** In modular approaches, the model burns significant capacity "learning to translate" the alien representations of the frozen vision encoder into its own space. In NEO, the visual representations are learned *specifically* to be useful for the language task, eliminating this translational overhead. This suggests that the "alignment tax" in traditional VLMs was much higher than previously realized.

## ---

**4\. The Meta Strategy: Llama 4, Early Fusion, and Extreme Context**

While SenseTime pushed for architectural purity, Meta’s **Llama 4** family (released April 2025\) focused on scale, context, and practical deployment. Llama 4 represents the culmination of the "Early Fusion" school of thought, distinct from both the pipeline approach and the pure "patch-is-a-token" approach of NEO.3

### **4.1. Early Fusion Mechanics**

Llama 4’s architecture is defined by its **Early Fusion** mechanism. Unlike LLaVA, which fuses vision and language at the input layer (Late Fusion is often used to describe deep fusion, but LLaVA is technically input-level shallow fusion), Llama 4 integrates visual information through a specialized "adaptor" that is deeply interwoven into the initial layers of the transformer stack.

* **The Process:** The model is pre-trained on massive amounts of interleaved text, image, and video data from step zero. There is no separate "visual instruction tuning" phase grafted onto a text-only model. The weights are "natively multimodal".3  
* **Benefit:** This prevents "catastrophic forgetting" often seen when fine-tuning text models for vision. The model does not lose its text-only IQ when gaining vision capabilities because the two modalities grew up together in the optimization landscape.

### **4.2. iRoPE: The Key to 10 Million Token Context**

Perhaps the most headline-grabbing feature of **Llama 4 Scout** is its **10 million token context window**. This capability allows the model to ingest entire corporate archives, hour-long video feeds, or massive codebases in a single forward pass.9

* **The Challenge:** Standard Rotary Positional Embeddings (RoPE) fail when extrapolating beyond their training length. As the sequence gets longer, the rotation frequencies become indistinguishable, and attention resolution collapses.  
* **The Innovation:** Llama 4 utilizes **iRoPE (Interpolated Rotary Positional Embeddings)**. Instead of extrapolating the rotation curves (guessing what positions ![][image6] looks like), iRoPE *interpolates* the new positions within the existing training range. If the model was trained on length ![][image7], and encounters input ![][image8], it effectively "squeezes" the positions to fit within ![][image7].  
* **Result:** This allows the attention mechanism to maintain high fidelity over distances that would turn into noise for standard models. It enables "Needle-in-a-Haystack" retrieval performance at scales previously thought impossible for transformer architectures.3

### **4.3. Scout vs. Maverick: The MoE Trade-off**

Llama 4 fully embraces the **Mixture-of-Experts (MoE)** paradigm to balance this massive capability with inference cost. The family is split into two distinct architectural philosophies 9:

| Feature | Llama 4 Scout | Llama 4 Maverick |
| :---- | :---- | :---- |
| **Role** | Efficiency & Long Context | Flagship Reasoning & Complexity |
| **Total Parameters** | 109 Billion | \~400 Billion |
| **Active Parameters** | 17 Billion | 17 Billion |
| **Experts** | **16 Experts** | **128 Experts** |
| **Context Window** | **10 Million Tokens** | 1 Million Tokens |
| **Hardware Target** | Single H100 Node | Multi-Node Cluster |

* **Scout's Design:** With only 16 experts, Scout is designed for *throughput*. The experts are broader generalists. The massive context window suggests its primary use case is retrieval and synthesis over large data, where the complexity of reasoning per token is lower, but the volume of data is high.  
* **Maverick's Design:** With 128 experts, Maverick is designed for *nuance*. The experts are hyper-specialized (e.g., an expert solely for coding in Rust, or an expert for analyzing medieval art). While the active parameter count is the same (17B), the *knowledge base* accessible to Maverick is four times larger. This model targets complex reasoning tasks where deep domain knowledge is required.21

### **4.4. The "Behemoth" Teacher**

Meta also teased **Llama 4 Behemoth**, a massive 2-trillion parameter model (288B active). Unlike Scout and Maverick, Behemoth is not primarily intended for direct deployment. Instead, it serves as a **Teacher Model**.

* **Distillation:** The reasoning traces and outputs of Behemoth are used to generate synthetic training data for Scout and Maverick. This "knowledge distillation" allows the smaller models to punch above their weight class, effectively mimicking the reasoning patterns of the giant model without paying the inference cost.22

## ---

**5\. Mastering Spatiotemporal Dynamics: The Qwen Lineage**

While Meta focused on context length, Alibaba’s **Qwen** team focused on the fidelity of spatial and temporal representation. The evolution from **Qwen2.5-VL** (January 2025\) to **Qwen3-VL** (May 2025\) illustrates a rapid iteration cycle aimed at solving the "aspect ratio" and "video time" problems.24

### **5.1. Naive Dynamic Resolution**

Traditional VLMs (like CLIP) resize all images to a fixed square (e.g., ![][image3]). This is disastrous for document understanding (tall, thin receipts) or panoramic images, as it introduces distortion and squashes text.

* **Qwen's Solution:** **Naive Dynamic Resolution**. Qwen2.5-VL processes images at their native resolution. It divides the image into ![][image9] patches based on its actual dimensions. A ![][image10] image becomes a sequence of patches reflecting that shape.  
* **Implementation:** These patches are packed into a single sequence. To prevent the model from getting confused by the variable sequence lengths, Qwen utilizes a specialized 2D attention mask that informs the model of the grid structure of the current image.4

### **5.2. MRoPE: 3D Positional Embeddings**

The true breakthrough in Qwen's architecture is **MRoPE (Multimodal Rotary Positional Embeddings)**.

* **The Problem:** A standard 1D positional embedding assigns a single number (index) to a token. In a flattened image, pixel ![][image11] might be index 10, and pixel ![][image12] might be index 1000\. The model struggles to understand that these pixels are actually neighbors in 2D space.  
* **The Solution:** MRoPE decomposes the positional embedding into three orthogonal components:  
  ![][image13]  
  * **t (Time):** Used for video frames.  
  * **h (Height):** Vertical position in the image grid.  
  * **w (Width):** Horizontal position in the image grid.  
* **Impact:** This allows the attention mechanism to compute the exact Euclidean distance between any two tokens in 3D space (Space \+ Time). A token in frame 5 can explicitly attend to a token in frame 1 based on their spatial overlap, enabling robust **object tracking** and **trajectory prediction**.5

### **5.3. Qwen3 Updates: DeepStack and Interleaved-MRoPE**

**Qwen3-VL** refined this further with **DeepStack**.

* **DeepStack Fusion:** Instead of taking only the final output of the vision encoder, DeepStack aggregates features from multiple layers of the ViT.  
  * *Shallow Layers:* Contain geometric information (edges, textures).  
  * *Deep Layers:* Contain semantic information (object identity).  
  * *Fusion:* By feeding this multi-level representation to the LLM, Qwen3-VL achieves superior performance on tasks requiring fine-grained visual discrimination, such as reading distorted text (OCR) or identifying small UI elements in a GUI agent task.5

### **5.4. The "Thinking" Models in Vision**

Qwen3-VL-Thinking represents the application of **System 2** reasoning to vision. Trained via Reinforcement Learning (RL) on visual reasoning traces, this model learns to "pause and think" before answering.

* **Process:** When asked "How many red cars are turning left?", the model generates an internal Chain-of-Thought: *"Let me scan the intersection... I see a group of cars in the top left... checking their signal lights... determining their trajectory... one red car is turning, another is going straight..."*  
* **Result:** This explicit reasoning step allows Qwen3-VL-Thinking to outperform larger models on benchmarks like **MathVista**, where "gut feeling" (System 1\) perception is insufficient.12

## ---

**6\. Efficiency and Decoupling: DeepSeek-VL2 and Janus-Pro**

DeepSeek has carved a niche by focusing on architectural efficiency and questioning the "unification at all costs" dogma.

### **6.1. DeepSeek-VL2: Dynamic Tiling and MLA**

**DeepSeek-VL2** targets high-efficiency deployment.

* **Dynamic Tiling:** Unlike Qwen’s native resolution, DeepSeek uses a tiling strategy. High-res images are cropped into fixed-size tiles (e.g., ![][image14]). A "global view" low-res image is also processed. The model learns to correlate the detailed tiles with the global context. This is computationally predictable and hardware-friendly.7  
* **Multi-Head Latent Attention (MLA):** This is DeepSeek’s "killer app" for memory efficiency.  
  * *The Bottleneck:* In standard attention, the Key-Value (KV) cache grows linearly with sequence length. For long multimodal sequences, the KV cache can exceed the GPU memory needed for the model weights themselves.  
  * *The MLA Solution:* MLA projects the Key and Value vectors into a low-rank latent space *before* storing them. During the attention step, they are reconstructed.  
  * *The Math:* Instead of storing ![][image15] vectors, MLA stores ![][image16] where ![][image17].  
  * *Impact:* This reduces KV cache memory usage by **93.3%**. It enables DeepSeek-VL2 to run long-context video analysis on consumer hardware (e.g., RTX 4090s) that would choke on a standard Llama or Qwen model.28

### **6.2. Janus-Pro: The Argument for Decoupled Pathways**

**Janus-Pro** (January 2025\) challenges the trend of using a single set of weights for both understanding and generation.30

* **The Conflict:** DeepSeek argues that the features needed to *understand* an image (semantic, abstract, invariant) are fundamentally different from those needed to *generate* an image (pixel-perfect, detailed, constructive). Forcing one transformer to do both leads to a "tug-of-war" that degrades performance on both.  
* **The Decoupled Architecture:** Janus-Pro uses a unified transformer backbone but distinct input/output heads:  
  * **Understanding Head:** Uses a SigLIP encoder for input.  
  * **Generation Head:** Uses a VQ-VAE tokenizer for output.  
* **Result:** The transformer learns a central multimodal reasoning capability but interacts with the visual world through specialized interfaces. This allows Janus-Pro to achieve SOTA on generation benchmarks (beating DALL-E 3\) without the "hallucinations" common in fully unified models.30

## ---

**7\. The Proprietary Frontier: Gemini 2.0/3.0 and Native Streaming**

Google’s **Gemini** series (2.0 Flash in late 2024, 3.0 in 2025\) has focused on the temporal and interactive aspects of multimodality, specifically aiming for the "Agentic" future.11

### **7.1. Native Audio and Video Tokenization**

Gemini 2.0 represents a breakthrough in **native streaming**.

* **No Transcoding:** Traditional "voice modes" in AI were cascades: Audio ![][image1] Whisper (Text) ![][image1] LLM ![][image1] TTS (Audio). This introduced latency (2-3 seconds) and lost paralinguistic cues (tone, hesitation, background noise).  
* **Native Processing:** Gemini 2.0 tokenizes raw audio directly using a specialized audio encoder (likely based on SoundStream or similar proprietary tech). The transformer processes these audio tokens natively.  
* **Implication:** The model can "hear" a user's frustration in their pitch, interrupt them naturally (turn-taking), and generate speech with matching emotional inflection. It turns the VLM into a true **Omnimodel** capable of passing the Turing Test in voice-only interactions.13

### **7.2. Real-Time Agentic Architectures**

The low latency of native processing allows Gemini 3.0 to function as a real-time agent.

* **Vision-Action Loop:** By processing video frames as a continuous stream of tokens, Gemini can observe a user's screen or camera feed and take actions (clicking buttons, writing code) with human-like reaction times.  
* **Deep Research:** Gemini 3.0 integrates this perception with "Deep Research" capabilities—the ability to spawn asynchronous "threads" to browse the web, read documents, and synthesize reports, while maintaining the main conversation loop with the user.33

### **7.3. System 2 Reasoning ("Deep-Think")**

Like Qwen, Gemini 2.5/3.0 incorporates **Deep-Think** capabilities.

* **Self-Correction:** In medical imaging or complex chart analysis, the model generates internal monologues to verify its findings. *"I see a shadow on the lung... checking contrast... it looks like an artifact from the scan process, not a nodule."*  
* **HLE Performance:** This rigorous internal verification is what allows Gemini 3.0 Pro to achieve the highest recorded score (**45.8%**) on **Humanity's Last Exam**, a benchmark designed to be unsolvable by "gut instinct" pattern matching.11

## ---

**8\. The New Benchmarking Reality: HLE, MMMU-Pro, and the "Reasoning Gap"**

The capabilities of 2026 models have rendered traditional benchmarks (COCO, VQAv2, MME) largely obsolete. The saturation of these metrics (with models routinely scoring \>90%) has led to the development of "Post-Turing" benchmarks.35

### **8.1. Deconstructing Humanity's Last Exam (HLE)**

**HLE** is currently the gold standard for AGI-level assessment.

* **Composition:** 3,000 questions from law, medicine, engineering, and advanced mathematics, tailored to require multi-step reasoning.  
* **The Gap:** While simple VLMs score near 0% (random chance), the best models (Gemini 3 Pro, Kimi K2) score around 45%. This vast gap indicates that while *perception* (identifying objects) is solved, *reasoning* (understanding implications) is the new frontier.  
* **Failure Modes:** Models often fail HLE questions not because they don't see the image, but because they cannot chain the visual evidence with abstract knowledge correctly. For example, identifying a specific part of a rocket engine (perception) is easy; explaining why that specific part's failure led to a historical launch abort (reasoning \+ knowledge) is hard.35

### **8.2. MMMU-Pro: Breaking the Text Shortcut**

**MMMU-Pro** was introduced to fix the flaws in the original MMMU benchmark.

* **The Flaw:** Researchers found that many MMMU questions could be answered by text-only models (LLMs) simply by guessing based on the question text, without looking at the image.  
* **The Fix:** MMMU-Pro embeds the question text *into the image* itself (Vision-Only Input). The model *must* perform OCR and visual structural analysis to even know what is being asked.  
* **Impact:** Performance dropped by \~20% across the board when moving from MMMU to MMMU-Pro, exposing the "blind guessing" behavior of many 2024-era models. Qwen3-VL and Gemini 3.0, with their native resolution and superior OCR, show the smallest drops, validating their architectural choices.37

### **8.3. MathVista and the Necessity of "Thinking"**

**MathVista** evaluates mathematical reasoning in visual contexts (function plots, geometry).

* **Trend:** The leaderboard is dominated by models with "Thinking" (System 2\) capabilities (OpenAI o1, Qwen3-Thinking). Standard "System 1" models (like GPT-4o or Claude 3.5 Sonnet) struggle here because they try to predict the next token immediately. The "Thinking" models take time to trace the graph, calculate coordinates, and derive the answer, proving that **inference-time compute** is the key to solving visual math.34

## ---

**9\. Hardware and Deployment Implications**

The architectural innovations of 2026 are deeply intertwined with hardware economics.

### **9.1. Inference Economics: Active vs. Total Parameters**

The VLM market has shifted from boasting about model size ("We have a 1 Trillion parameter model\!") to boasting about efficiency ("We have a 10B *active* parameter model\!").

* **The Metric:** **Active Parameters** determine the FLOPs (Floating Point Operations) per token, which translates directly to latency and electricity cost. **Total Parameters** determine the "knowledge capacity" or "IQ" of the model.  
* **The MoE Advantage:** Architectures like **Llama 4 Maverick** (400B Total / 17B Active) offer the intelligence of a massive model with the speed of a small one. This decoupling is essential for deploying VLMs in real-world applications where latency is critical (e.g., autonomous driving, robotics).8

### **9.2. Edge vs. Cloud Architectures**

A bifurcation is occurring in deployment strategies:

* **Cloud (The "God" Models):** Models like Gemini 3.0 and Llama 4 Behemoth, with massive expert counts and context windows, reside in data centers. They handle complex, low-frequency tasks (e.g., "Analyze this 2-hour movie and write a screenplay based on it").  
* **Edge (The "Fast" Models):** Models like **DeepSeek-VL2-Small** (2.8B) and **Llama 4 Scout** (quantized) are designed for local deployment. Innovations like **MLA** (DeepSeek) and **iRoPE** (Llama) allow these models to run on high-end consumer GPUs (RTX 5090\) or even mobile chips (Apple M5), enabling privacy-preserving, offline multimodal intelligence.29

## ---

**10\. Conclusion and Future Outlook**

The landscape of Vision-Language Architectures in 2026 is defined by the **convergence of modalities** and the **divergence of inference strategies**.

**Summary of Key Trends:**

* **Unification:** The separate vision encoder is dead. Models like NEO and Llama 4 prove that transformers can learn to see from scratch.  
* **Sparsity:** Mixture-of-Experts is the standard for scaling. Dense models are becoming relics of the past.  
* **Spatiotemporal Native:** Qwen and Gemini have taught models to understand time and space natively, transforming them from static image classifiers into dynamic reality engines.  
* **Thinking:** The integration of System 2 reasoning loops into vision is the current frontier for solving "hard" problems.

**The Road to 2027:**

As we look forward, the distinction between "text model," "image model," and "audio model" will likely vanish entirely. We are moving toward **Omnimodels**—single, sparse, gigantic neural networks that consume and generate any digital signal. The challenge will no longer be "how do we connect vision to language," but "how do we teach these omnipotent models to reason reliably about the physical world?" The architectures of 2026—NEO, Llama 4, Qwen3, and Gemini 3—have laid the foundation for this AGI-adjacent future.

### **Comparative Specification Table: The 2026 Leaders**

| Feature | Llama 4 Maverick | Qwen3-VL-235B | Gemini 3.0 Pro | DeepSeek-VL2 | SenseTime NEO |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Core Architecture** | MoE (Sparse) | MoE (Sparse) | MoE (Sparse) | MoE \+ MLA | Native Dense |
| **Vision Strategy** | Early Fusion Adaptor | Native Resolution \+ DeepStack | Native Multimodal | Dynamic Tiling | Native Patch Embed |
| **Positional Enc.** | **iRoPE** (Interpolated) | **Interleaved-MRoPE** (3D) | RoPE Variants | RoPE | Native-RoPE |
| **Context Window** | 1M (Maverick) / **10M (Scout)** | 256K (Extends to 1M) | **2M+** | 128K | 32K |
| **Special Capability** | Extreme Context Retrieval | 3D Grounding & GUI | Agentic & Native Audio | Memory Efficiency | Data Efficiency |
| **Reasoning Mode** | Standard CoT | **Thinking (System 2\)** | **Deep-Think (System 2\)** | Standard | Standard |
| **Release** | April 2025 | May 2025 | Late 2025 | Late 2024 / 2025 | Late 2025 |

This table encapsulates the diverse engineering philosophies driving the field: Meta chasing context, Alibaba chasing precision, Google chasing interaction, DeepSeek chasing efficiency, and SenseTime chasing architectural purity. Together, they form the cutting edge of human knowledge in multimodal artificial intelligence.

#### **Works cited**

1. From Pixels to Words – Towards Native Vision-Language Primitives at Scale \- arXiv, accessed January 23, 2026, [https://arxiv.org/html/2510.14979v1](https://arxiv.org/html/2510.14979v1)  
2. The Paradigm Shift to Native Multimodality: Architectural Unification in Foundation Models, accessed January 23, 2026, [https://uplatz.com/blog/the-paradigm-shift-to-native-multimodality-architectural-unification-in-foundation-models/](https://uplatz.com/blog/the-paradigm-shift-to-native-multimodality-architectural-unification-in-foundation-models/)  
3. Llama 4's Architecture Deconstructed: MoE, iRoPE, and Early Fusion Explained \- Medium, accessed January 23, 2026, [https://medium.com/@mandeep0405/llama-4s-architecture-deconstructed-moe-irope-and-early-fusion-explained-e58eb9403067](https://medium.com/@mandeep0405/llama-4s-architecture-deconstructed-moe-irope-and-early-fusion-explained-e58eb9403067)  
4. \[2502.13923\] Qwen2.5-VL Technical Report \- arXiv, accessed January 23, 2026, [https://arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923)  
5. Qwen/Qwen3-VL-8B-Instruct \- Hugging Face, accessed January 23, 2026, [https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)  
6. Qwen2.5-VL: AI at the Intersection of Vision & Language | by U.V. | Medium, accessed January 23, 2026, [https://uv020.medium.com/qwen2-5-vl-ai-at-the-intersection-of-vision-language-569fe85bf1bf](https://uv020.medium.com/qwen2-5-vl-ai-at-the-intersection-of-vision-language-569fe85bf1bf)  
7. DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding \- arXiv, accessed January 23, 2026, [https://arxiv.org/html/2412.10302v1](https://arxiv.org/html/2412.10302v1)  
8. Llama 4 Technical Analysis: Decoding the Architecture Behind Meta's Multimodal MoE Revolution | by Karan\_bhutani | Medium, accessed January 23, 2026, [https://medium.com/@karanbhutani477/llama-4-technical-analysis-decoding-the-architecture-behind-metas-multimodal-moe-revolution-535b2775d07d](https://medium.com/@karanbhutani477/llama-4-technical-analysis-decoding-the-architecture-behind-metas-multimodal-moe-revolution-535b2775d07d)  
9. Welcome Llama 4 Maverick & Scout on Hugging Face, accessed January 23, 2026, [https://huggingface.co/blog/llama4-release](https://huggingface.co/blog/llama4-release)  
10. MMMU leaderboard, accessed January 23, 2026, [https://mmmu-benchmark.github.io/](https://mmmu-benchmark.github.io/)  
11. LLM Leaderboard 2025 \- Vellum AI, accessed January 23, 2026, [https://www.vellum.ai/llm-leaderboard](https://www.vellum.ai/llm-leaderboard)  
12. Qwen/Qwen3-VL-8B-Thinking \- Hugging Face, accessed January 23, 2026, [https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)  
13. How to use Gemini Live API Native Audio in Vertex AI | Google Cloud Blog, accessed January 23, 2026, [https://cloud.google.com/blog/topics/developers-practitioners/how-to-use-gemini-live-api-native-audio-in-vertex-ai](https://cloud.google.com/blog/topics/developers-practitioners/how-to-use-gemini-live-api-native-audio-in-vertex-ai)  
14. Gemini API I/O updates \- Google Developers Blog, accessed January 23, 2026, [https://developers.googleblog.com/gemini-api-io-updates/](https://developers.googleblog.com/gemini-api-io-updates/)  
15. Vision Encoders in Vision-Language Models: A Survey \- Jina AI, accessed January 23, 2026, [https://jina.ai/vision-encoder-survey.pdf](https://jina.ai/vision-encoder-survey.pdf)  
16. Exploring the Frontier of Vision-Language Models: A Survey of Current Methodologies and Future Directions \- arXiv, accessed January 23, 2026, [https://arxiv.org/html/2404.07214v3](https://arxiv.org/html/2404.07214v3)  
17. The World's First Native Multimodal Architecture NEO Arrives Right After Ilya's Prediction: Vision and Language Fully Integrated \- 36氪, accessed January 23, 2026, [https://eu.36kr.com/en/p/3582215483980929](https://eu.36kr.com/en/p/3582215483980929)  
18. Evolving From "Data Fusion" to "Native Architecture", SenseTime Releases NEO Architecture Redefining the Efficiency Boundaries of Multimodal Models-News and Stories-SenseTime, accessed January 23, 2026, [https://www.sensetime.com/en/news-detail/51170267?categoryId=1072](https://www.sensetime.com/en/news-detail/51170267?categoryId=1072)  
19. NEO, the World's First Native Multimodal Architecture, Launches—Achieving Deep Vision-Language Fusion and Breaking Industry Bottlenecks \- Pandaily, accessed January 23, 2026, [https://pandaily.com/neo-the-world-s-first-native-multimodal-architecture-launches-achieving-deep-vision-language-fusion-and-breaking-industry-bottlenecks](https://pandaily.com/neo-the-world-s-first-native-multimodal-architecture-launches-achieving-deep-vision-language-fusion-and-breaking-industry-bottlenecks)  
20. What Is LLaMA 4? Everything You Need to Know \- Resemble AI, accessed January 23, 2026, [https://www.resemble.ai/what-is-llama-4-everything-you-need-to-know/](https://www.resemble.ai/what-is-llama-4-everything-you-need-to-know/)  
21. Specializations of Llama 4 Scout & Maverick Models: A Comparative Analysis \- Medium, accessed January 23, 2026, [https://medium.com/@rajraftaar3/specializations-of-llama-4-scout-maverick-models-a-comparative-analysis-344b20e7f002](https://medium.com/@rajraftaar3/specializations-of-llama-4-scout-maverick-models-a-comparative-analysis-344b20e7f002)  
22. Meta Unleashes Llama 4 Scout & Maverick: The Future of Open, Multimodal AI Has Arrived, accessed January 23, 2026, [https://pressconnect.ai/news/meta-unleashes-llama-4-scout-maverick-the-future-of-open-multimodal-ai-has-arrived/](https://pressconnect.ai/news/meta-unleashes-llama-4-scout-maverick-the-future-of-open-multimodal-ai-has-arrived/)  
23. Meta releases Llama 4 AI models to rival DeepSeek and OpenAI \- The American Bazaar, accessed January 23, 2026, [https://americanbazaaronline.com/2025/04/08/meta-releases-llama-4-ai-models-to-rival-deepseek-and-openai/](https://americanbazaaronline.com/2025/04/08/meta-releases-llama-4-ai-models-to-rival-deepseek-and-openai/)  
24. Qwen2.5 VL\! Qwen2.5 VL\! Qwen2.5 VL\! | Qwen, accessed January 23, 2026, [https://qwenlm.github.io/blog/qwen2.5-vl/](https://qwenlm.github.io/blog/qwen2.5-vl/)  
25. Qwen3-VL is the multimodal large language model series developed by Qwen team, Alibaba Cloud. \- GitHub, accessed January 23, 2026, [https://github.com/QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)  
26. Introduction to Qwen3-VL \- DebuggerCafe, accessed January 23, 2026, [https://debuggercafe.com/introduction-to-qwen3-vl/](https://debuggercafe.com/introduction-to-qwen3-vl/)  
27. Qwen3-VL: Advanced Multimodal LLM Family \- Emergent Mind, accessed January 23, 2026, [https://www.emergentmind.com/topics/qwen3-vl-model](https://www.emergentmind.com/topics/qwen3-vl-model)  
28. DeepSeek-VL2 : A Giant Leap in Open Source Multimodal Intelligence | by KoshurAI, accessed January 23, 2026, [https://koshurai.medium.com/deepseek-vl2-a-giant-leap-in-open-source-multimodal-intelligence-7520f815bc1e](https://koshurai.medium.com/deepseek-vl2-a-giant-leap-in-open-source-multimodal-intelligence-7520f815bc1e)  
29. DeepSeek-VL2: MoE Vision-Language Model \- Emergent Mind, accessed January 23, 2026, [https://www.emergentmind.com/topics/deepseek-vl2](https://www.emergentmind.com/topics/deepseek-vl2)  
30. Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling, accessed January 23, 2026, [https://arxiv.org/html/2501.17811v1](https://arxiv.org/html/2501.17811v1)  
31. Janus Pro AI, accessed January 23, 2026, [https://janusai.pro/](https://janusai.pro/)  
32. deepseek-ai/Janus-Pro-7B \- Hugging Face, accessed January 23, 2026, [https://huggingface.co/deepseek-ai/Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)  
33. Introducing Gemini 2.0: our new AI model for the agentic era \- Google Blog, accessed January 23, 2026, [https://blog.google/innovation-and-ai/models-and-research/google-deepmind/google-gemini-ai-update-december-2024/](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/google-gemini-ai-update-december-2024/)  
34. MathVista testmini Leaderboard \- Kaggle, accessed January 23, 2026, [https://www.kaggle.com/benchmarks/open-benchmarks/mathvista-testmini](https://www.kaggle.com/benchmarks/open-benchmarks/mathvista-testmini)  
35. Humanity's Last Exam \- Wikipedia, accessed January 23, 2026, [https://en.wikipedia.org/wiki/Humanity%27s\_Last\_Exam](https://en.wikipedia.org/wiki/Humanity%27s_Last_Exam)  
36. Humanity's Last Exam, accessed January 23, 2026, [https://agi.safe.ai/](https://agi.safe.ai/)  
37. MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark \- arXiv, accessed January 23, 2026, [https://arxiv.org/html/2409.02813v3](https://arxiv.org/html/2409.02813v3)  
38. MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark \- ACL Anthology, accessed January 23, 2026, [https://aclanthology.org/2025.acl-long.736.pdf](https://aclanthology.org/2025.acl-long.736.pdf)  
39. Vision Model Leaderboard \- SiliconFlow, accessed January 23, 2026, [https://www.siliconflow.com/articles/benchmark/vision-models](https://www.siliconflow.com/articles/benchmark/vision-models)  
40. MathVista Leaderboard | Kaggle, accessed January 23, 2026, [https://www.kaggle.com/benchmarks/open-benchmarks/mathvista](https://www.kaggle.com/benchmarks/open-benchmarks/mathvista)  
41. NVIDIA Accelerates Inference on Meta Llama 4 Scout and Maverick, accessed January 23, 2026, [https://developer.nvidia.com/blog/nvidia-accelerates-inference-on-meta-llama-4-scout-and-maverick/](https://developer.nvidia.com/blog/nvidia-accelerates-inference-on-meta-llama-4-scout-and-maverick/)  
42. Top 10 Vision Language Models in 2026 | Benchmark, Use Cases \- Dextra Labs, accessed January 23, 2026, [https://dextralabs.com/blog/top-10-vision-language-models/](https://dextralabs.com/blog/top-10-vision-language-models/)  
43. Qwen3-VL Technical Report | alphaXiv, accessed January 23, 2026, [https://www.alphaxiv.org/overview/2511.21631/sso-callback](https://www.alphaxiv.org/overview/2511.21631/sso-callback)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAXCAYAAADpwXTaAAAAbUlEQVR4XmNgGAWjgKqgEF2AErAQiFXRBckF1kC8DV2QEpANxGnogiAgBMRSZOClQLwWyoaDTiBeTgY+CcT/gLiegUKgAsR7GSDhRxHgAOIrQCyDLkEOSAHiYnRBcsF+IGZBFyQXSKILjIJBAAAj9xTbjwG/KAAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFIAAAAXCAYAAACYuRhEAAAD9klEQVR4Xu2XW4hWVRTH/2qZEV5KzZRMUIl6SFHQh0xT00J9URHtypR4YQztTcFESlOY8kFLQrBgyCxUvCCK4HW8pKUvUhmCmKNWhg+p5QVEq//fdfbMOnvOfN/5BhQfzg/+6F57z3fWXmfttdcBCgoKCgry0JGqpj6h5lDd09MNDKU+TDQ2msviIWob1SueuAf0peZSNVQV1S49fYfW1BTYmnnUgPR0Jlr/UWwUT1NHqdnUDOpn6io13i8iy6g11BvUcuo2tRvZDgbk4H/Us/HEXWYStZd6GxbMS9RZqo9bI7/3UIupt6it1L/UUrcmpgfst9bFE6IO9sYCXalb1LXk/2I4dZzqkIzFCliQPnY2z2DYC7nXgZSPF5B+5gSYH0qYwAfUaqqNs2mPWtfcadsIS6AmgVRq34S9CUU7oLepH5yWjBck41UNK4Bhie03ZwvoSO+HZXHeQD5OPRIbHa2op2JjBi/AnnnS2R6GJYbsOvJCp0njV8MisjCxfe1sgTepRdR1ZARSfE5tpx6IbPrBqcl4EPUD7M0GtHGtOedsgSXU67BjlTeQyuBdSGd9QC+8FraZcujvlQg6sp5fYL70Tsba2xFYaQtMhq35ytlEN1hitEWJQGZxDHa8w9HOYhzsoZ9GdhVsHQFRSSDFKFip6eRsCqI2NtPZKkXB1X5+jCcidNnK34mRXYFTMoncgRwN+7Ev4omI76i/kQ6SsnofGstEpYEUev4BWDAVRF1w1akVlaMTIj9KZbSS5jL1E6wUBHRx1bhxrkC2h9WWTbA0bg7d7v9QIyP7+9Q7btySQIqXYcFcS70bzVVKP+oGzJdSfAPb+5PO9hjsSPvOpGwglU07qC+RvsliRlB/oDHVAwqWWghPSwMpX1TntLFHo7lKUD/8KxovzeZQAqicxaWslhoS2coGUgH0KfwcLGge2ZT6zzjbe8m/s2C9Wj3MeekKLJDnqe+TdeVQENfDjvOLsIzwNTMvqot6pq93+n9PNxZVsBtcp1HoWOvEiVOwPZ2B7aceth+1dRq/lqxrQF8q6uw982H1ISAHlCXekQdht3lzfIbKMtIHMaBgqu7q6ysv8msL9VJk30l1ceNXqA2wdi3wPLXSjT2dYfvJzMhQ7+SsAlVHHYZlkzJQKCNOwL569DWgtaphainCDZ2F+k49uH88kYHKybfU9HgC1rPKtzzBVL+pPlD9rf5GUlarGb/o1g2EXZaHYHuqow5Sv8O+8rJ4ArafzfGE0lhtgSZjqYMPRTbcelnK+qRSD3kadgvK2b9gPVspxiD9hRWjWpXn4gldR5aUIAF1HfF8kC67GPn/J2w/kvrnJke7oKCgoKCg4H7mfzAt87SKRbVpAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFIAAAAXCAYAAACYuRhEAAAEr0lEQVR4Xu2XZ6heRRCGxxJ7b9iNPTaiIGLD2HtDVOxijxosKFiiCSh2sUCsIImGWBLQH6LYMJZgIvaKPbHHLhbsmve5s3u+OXtz7vchen+dB17u3dn5zvbZWbOWlpaWll6YR9pbukw6VFqtXl2xkjRaOkvaXpq/VltnU+lC6UBpkaJuMKDNk6Sx0j7SkvXqCvp5qXSiNLyoiyxhPjfnSxsWdX2sLD0mXSBtJ90q/SIdEp3EOdJd0kHSftIX0jRpsegkFpSmShOknaTLpfujwyCws/SwNFLawbyfH0sbB5/5zMdznbSLdJr0VyqX7CvNlI4x931f2rzmYf7Dr6UtU3kB6Q/pO2nhZFtG+lualMpwkfSPdF6wwR3SbaHMpNLBpYPt/+Zl6Q3r7MIdzfv6UOVhtr/0u3RKsD1l7rdVsLFLv5XWTuUR5j7XVh6Jq1LFkam8kHkDP1nnSDKRv5mvRCZP5Jhg43hg2yDYCBkc8W6sIC1aGgOEn9VLYwPPmS/e0FTOE8miZvZKtrjoeSKZLCB0fSTdU3mYLS5dKW0WbH3MK60bylubf4xtHyE+Eicyj5j7rR9s90nfp//ZDcuFum5sIT1q9TYy9HGCdERhb4IFWTWUiWv0lZgZWcc6cZ6/P0ufmrcH7EZ+d4b5Qq5hHhK6woS+JD1rndUs4YPESxo9s6gjDr0lXWJ+xPnWM+a7rReIbU9ISwUbg+Jb5ST0yh7SV+YL0XTpscvYOJ+bx/XMceYTyel7UBovfWn1cNCPm8zjCo6bFHUZdiuBm4vmFquvDpNFo39KJyQbk36v9Jp1VrkbBHOOGJPJbyZKJ9c8eoPJ4RL9QHre6osTYYFekH6QjirqbjQf0zvmlyjkXXpwdmqCXfGjDRzX+CiB+0VpWLJtZN4AsTROMDcndmJSr+xqPplcbqcWdf8G4jiLz+3bBCnfLGmydTKRKeZ9J6OJzDaPw13hOPKBueZLCQaLTw7Ey6cyqxchrmG/obAPBPHqcfMw8V/c9hxpTgrxb6Dcl3ySvuYTQJ8pH1Z5OO8lO2OuOF0aFQ3idnNHEnRg111t9YtlTevsQI4Rx5Dj8WrwgcPN/W4u7E0wUHYFgxkhPWnNx3JucFnRbxY68qF5P3ZL5d3NYzl9z5An4jM9lc9O5QMqD+fdZF8xG4YmAyIxz0xNtrwy41OZuJhhkNjY5sRCuFv6pPJwckrUNaZYfRIztEN/ml4mJUebt/d2sA0xT+mw59RsZirzWsmMTbZ8ythAlI+vPJzPzO+TPO6+WMdK3ZkN5qnDN0mrJBtxjrSGFCVzrnkjVwQbgyZ/i37EuRkWGm0gvzTyRRXhxcVR72UyuQx+Ne9zhscGfSUbybDor1g9oyDu0/8Yz7mwHgjl9cy/RTZQgwpi0TjzNzTbmtXkDRq5xvwC4K3N5PHyud76TxDf4Ka8WHraPN7mC2kg6Bi7qYltrPeLh/f9m+YxjxxylnmOGuMtcZOdzu7jCLOIZCxlHzi+r5vnyIQ3dmP5mqtgMkh52OZrFXURYhWrRfzhBdQEjRNv9rT+Ez1YcNq2NZ/UZYu6CLc1YYe3c1OKxhhIy461eqLf0tLS0tIyOMwBFuj8MS9/q60AAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD4AAAAXCAYAAABTYvy6AAAB7klEQVR4Xu2WSygFURjHP3mVQgoJhZSwkTxKKRsWLCws5Jk8VlaSDVmxkZSNx9LWqyR5LJQSNrJj45FShIWFsmCB/9eZ0cyXc8fkXJH51W9x/+fMN9+ce+65QxQQEPDfyIYdMhQkwGY4BAvF2E+STd/stQCOwSP4Ctfdwy7q4SXshDXwApa6ZoQXo72Wwy5YAp9JX6wIPsBc63MVfIOTHzPCT9h61RWLgldwwZHFw3FY7Mh05MhAkAEjZeiB0V51xXgFecX6YATMIn+NzsMmGVqUwR0YJwc8MNqrrlg3qWIjcAPOwXvY65wUghi4CttEzlt3FyaL/CsY7VVXbIZUsVMYa2X2yjbakzzg69Zgu/WZH3oPpnzM8IfRXrkYr5JkidSFwyK/hYciC4X98KNwH6a6h31htFcutilDME2qWIvIz63cz7dWDZ9gvxzwidFedcUGSF3UIPIzK08TuY5KeADT4TJ5v4CEwmivXGxLhiCf1EU9Ir+BJ6ROTy/sh7ZXPJrUtrR/834x1is38kLqlOX/Qsk2uQ+TPFI3qHVkOiro89Ob77kIW0XuhZFe60htg2v4aHlH6lRMdMzjLXIMV+AEqRUcdIyHYhYmydCCG58i9V7txU/0+im8Tfi9l18bM8XYb+Mv9RoQEBDwfd4BeVCbNoDRf2IAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAYCAYAAABqWKS5AAACiklEQVR4Xu2WSehOURjGH/OcMaIMyRAKmbNAUiyMsVIsKGyMyRAi08ZYFBEpCzspZCOFZCNkj39sJBkWxkzP03Ou73X8N7766n71/erXd+977j3fOe9577kXaNCgJrSjPfJg2ZlO39Nf9H7WVhco4xr8obyhHpgFD35u3lAPHKA/aLe8oayMo1PS8T36MLSVlgH0Lr1Kd6bfb/RovKiM9KUv6OkQOwjX+4IQKyXn6VfaM8T2wfXePcRqwRn6ig7NGwKz6QN6JG/oDQ9SJRO5Qx9lsVowhL6hLfKGjOt0eR7UrFQeynRBB3gljodYrVhJL+fBjFb0Ax2YN8yABz8nxGam2EI6ma6BM7SXjqDz4G10PTzRjfQYHaObA53oJnqYjg9x3bODrqNX4H4K+tC18EYxMsUm0Gd/rgi0pR/psnTeCy4XDX4QPUlHwwNfQl/TafAAXtJztDNdSm+gwiR6kw6nLWkT3PdguET7wf3/RGXSSpr+uwvtDz+LYnM4/odVcOen6Fm4E+0+t+iJdM1EuLSUYaEBvYMHI1bj7+XXAONOpQduEZzpPSmme9/CfSkZT+FV0orof0al666hmXqP6OZh4bw1nJ2IBjQ/HY+FJ1ygrK9Ix1PpZ7hWRVf6He7/C7xyQtdrMkKloTbVtaqhQH3oA1HvoappTz+h8qmwgV5Ix6pttakslDVlrCm1CWVR7xBlWDuLJiO0UnpetsOlojIsdp028GaiJD2H639xavtv9MlwO5xfgp8B0ZE+oVvh+hdafr2p98MPnwYuNLnddBu8m11EZcV0/y54QnpJau9Xsh7TLXB1VIWWT9kviMdCS513rhUpSieiLKssRd6P+tB9keb6btCgGn4DKz1yK4S20IkAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAYCAYAAABqWKS5AAABW0lEQVR4Xu2WzStEURjG3yIWLNhY2LOyESUbU8qaUjY2NvwF42PJwh9hZSVbCkkRK3+E77IQKd8r4Tm9Z2o87pjzHmOoOb/61e193nt7mqZzr0giUdsMwhN4Bo/89THcLl76A3LeINbhOxzmoIoMwSXRH9J1mf4cl+YKPsB6DiJY5UEgo3AMjouhfKfo8iYHkezxwEi/GMpPii7nOYhknwdGTOVXRJd7OYjkgAdGTOUv4R2s4yCSqpXvEF3c4MDTAht46GmG7RkeZsycrXpbWQrlZzhgpuT7xTXYxkPPhOjJwl5nzJzz7qYACuVnOWDcQ91iHwegC27xMIBK/W3Klnfn+6N8Pd8b4Y7omWulUuXnOCimW3SJPwV64C58gk2UhfDT8jnRXoscOAbgKXwWXXqB5/AC3sI3+AqX/b6V2PILot9XN/Be9I3vOpU6TH6F2PL/ghEeJBKJGuQD93RVA7aRNQsAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAYCAYAAAAh8HdUAAAAwElEQVR4Xu3SzQpBURSG4VXEwMTIHTAyEWVm4AZcgDFXILdjJFMKnZQr8V8GJkr+RsJ7Wltp6eyhgXz11Gl9e7UH+4j8fqpYYIWZ+54jeD8UlQEeqNnClx2OiNsiKjnRW0a28KUhutSyhS9d0aWSLXzZ4oCYLaKSFb1laAuXNBJ22BRdatvCpY+MHfZEl8q2IHmM7TBM+D4n+XyfJCaom7kURG+xv0wRU5yReg0rWOIiunTFGhvscccNHXf+n+/nCQT+J/YZaZDxAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAYCAYAAAARfGZ1AAABXElEQVR4Xu2UPSiFURjHn5CPQSaJ2cdkEMqqmCwGu5JEYhKyyCibwcTIokSRjTtRrAYlLL5WX4V8/Z6ec+u8T2/cjeH91a/3nud/zrnnnnvOK5Lxl1ThCC7gONYm4wSbeInneIYX4dkTd8rTiEc4hkN4gk/YG3dytOAXHmKZyxLksD9qV+M7PofPaQyLTT7pg5gifMNPrIvqe2KDB6NazJpY3u4DzxLuYImr6eCBqBZzhfdY7INCOBbbmrRtaRD74m0fFEK32OBlHwR0qzSf8MFvVOIpbmCpy/Ksik3e5oNAjS8ouue7uCI/7+UNPkh6n1Zc9EVFJ52P2s3YGbWVJrFV6yLSWMcOX5zDKVebwT5X01usk0+7utIltqUJ9FY+4r7Y+c7hgdhR09XHbIlNHq9O/6dRfMHZqC4VYkdOB3g/sDz00597K3bZNLsTe79c42voq8/60D8j47/wDbTJTpsq/zc9AAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAABF0lEQVR4Xu2TvUoDQRRGr1qZSkTSpBQMIjZ5AAvjA0jE0jYJ1jZ2go1VQLCxsgkk8SUELRQUkQTSBEJIo502kkLUnHFm2J27m8J+Dxx25n7D/C0jkvFftnGIr/iGzTD+4xFHOBA7thGkijZ+4A+uqmwBT/AWC2GUpItH+CvpK57hvi5qiniNS/iJ75gLRojcYV7VEtTw0LUvxe6qGsWyiM+x/kxauO7am2InMkf1lPEi1p/Ji+rfiJ1sy/VPcS+K0/H3E6cidiJfN/ezEsXp1CW6H4/53WP8wjV8CuN0Orihi3Asdlf3eK6yBHPYd1+NOcpE7GS7KktwgD2c14HjCr9xWQeeHbHvyqxoNE/DvDlNCR90MSMDpvMbNCf6RtASAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFsAAAAXCAYAAABkrDOOAAAEdUlEQVR4Xu2YechWRRSHT7tEm4Ga7YtJ0UZ7RoVFFCWFCpkREUaLoVZGtFiB7Rnt0EZQ2YbS8ke0WJRJ2WJQiFtpBimVWv1VRAtZ/R7PzHvnzvve73u9H1nCfeDH986ZuXNnzp05c+Yza2hoaGjYmNlTOj83JuwtTZRul0ZkdZFNpTOk26TJ0uBydYujpeulG6RDs7oNyfHSTUGnZ3WRHaTzpGnSBdJW5eoWvc5pf+lO6VPpL+m1cnWLE6UV0uXSSdJsaWaphdmW0ovmfZwgXS19Kw1LG4mrpMXSaOkcaZn5R9zQ3C09I50r3S+tld6W+iVtdpWWSPeYz+NR6XNpx6QNdDWno8y/1uHS79bZ2ZtJ35h3GOFlP1t5J4yX1khbJzZW+JfS5qHMx/1bOqLVwuwU83cPTWz/NsOl+dJ2ie0B87HdldhmSK8mZfhAejIp15pTlbPPNO/ssMw+13wlRD6TXknKcLL5s8eF8r3ST0X1OtgR7Cq2X0/QbpfcmLFHbqiAdzEuVmqE3YiNhQUDzH1yZauFc6v0ixXhpNacqpzNFmMQe2V2HMszrFriGm2ml1r4B8J+YyizmghHOQz2ndyY0V96XzogrwhcLD2WGys4UponjUpsA83HujKUR4byuFYLB+djJ95DrTlVOfs5887zw+6FYB8k7Rt+55PFMdgfCWVi+NKiusUP5rGwN3aTPpYOzuyXSE+bH9B14dBnrA+G8kWhTAxOmRDsZ4dyrTlVOftNK5yawgGJnZg1LPyOTo3EeBYPU97RaQDEetQNu0ufSIeEMmfFs+ZnS18gFrMaGTNcZ2WnRi4Ndv5CrTnx0Ou5Ucwy73ynzB6dzSFwTPidxkCIzn4+lH+TviiqWzCo73JjD0SHk4bSd18dTQjiwCfTilxrPvaxiQ2is3kGas0JZ7+RG81XDZ3nhxNpHnYykyHh9+OlFmYHBnvcmhw+ZCc5P0oLcmMv3GLuIOJvXyCtxSl5Pxeaj53UMIWUDvvoUK41pypnTzPvPG6vyFvmp/Im0jbSn9aeex9r/uw1oTzP2rcWz1e9uwouTCwCYviHVoSU9eUgaaG0X2LjLgGnWTlcRKYEO5cYqDUnKgkZOcPNOz81s7N1Xk7Kc6SPkjKwBXk2HmhTzVOi9OJAeKLNZYmtJ66wcoze2Tze5odmb/ChuJzxN7KFufOA+wKL6Y6ieh2Eyu+teP9UW8858ZI/pPesuIBE6JSvTwoYIU7/ap5HRzhIGByTj5AhzEnK+5hvfdKqCJeq1dZ+JnRikvSUtWcdg83zflZqN5CqcuNbZJ6evWs+d26LLyXtHja/P8T3kT8vN3dwpOs58f8A4g3pCycxYktw3dw+aceB9LX0hHmeyQvzlAiIo5zMbEVWH1s8vaUB23OVdLP5vwq+surcOYVs6D5rd3SEiXHB6AZutqy8TuLQjeBcLm6IcMIHfSipj9SdUyWseFK8s6w9505h0mPM27JjOkGM50MzyG2zuv8jxHQWV35upWxsc2poaGhoaPhv+QfP7CEsivdeXwAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAXCAYAAACrggdNAAADCElEQVR4Xu2XWaiNURiGP5R5npIhSWS4InPhQhGlJGXKBSEylaFMFzspFMoFuSAnIqXkQkmmkygSIfMYmYcUmTO8r2//p7Vf/9pnnx3a5Kmn3X7X2uf837+/tf61zf7zd9ME1tGwhGijQWW0hqdhIx0oIabCjRrGqAVPwck6APrCZXA57CFjVaUeXKBhQGM4Ca6BU8yvK6QaPAMXSZ7KWvPJ/FDIQngZjobj4Q04O2dG5bClF8PD8AN8mztcQVt4Ba6D/eFmeBU2DSeBfvAz7CJ5Dg3hazhA8q7wG+wVZEPhR9g5yCqjJZwDB8PjFi9qN9wv2Um4TTKy3Xx+lHnwkoZgvXmxITXhV/NWLIaDll5UC/ObNV/ylebztQ0HwS+wmeQVHIL7NATn4T0NzQs9omGBxIoaZd4VuqZZJPOBknNTYz5G8gpuma8p5SG8riF4bt7rxRArapr5RXLdhszK5mMl59rn39kg+Q9qwE9whg6Yt0PaxT/NWgyxopZY+sXPzOZ8VS7CPRqS9uYfGqID5jvVNQ3NC3qkYYGwqHcamu+OvI5xkidFTZec7IXHNCQ9zT/EV+UBvKkheGF+l4qBRb3X0PyhyuuYKDkfH8z5SFG2whMakg4WX3A8XWibsZfZlgckL5RYUcMtvc2WZnMeAJSjcKeGpIH5h/j1Kxnz7bt2kLUynz83yLjNdwre54NFsa2VuuZrbZXkfAA/M1/7yn3zLT8VttkWDUFH+MZ8u03g0eWJeXEJZebFa+ukwTXAjam+DoBN8Bysnn3Pm8WdOZNMCOCN5v/k9aRSBsslS2BbPIYr4Gp4G3bPmeGnhZdwh+QJfEBybd41f8bRV+ab0LBgHovgUYqyDbleYofXbuYd00cHErie+OwJ2yyEd3WEeYFs1zR4FNqlYZHwTMfnFY9pMXjovaBhCL9uHiR5Ei+WCfbzaeB3wSMTO4Y7Zl5Gmq+fKv8IM//9VW7+s+JPwJt/1uKdlQMPqbF1kQ+e4qO9/YvhTb8D2+lAPjKwuYYlBM+IvTX8p/kO6gilQqkGIIkAAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAXCAYAAACS5bYWAAACbUlEQVR4Xu2WS0hVURSGVwVlVPZUogiJqAghSKLAQQ4MpSCoqEE1KggalA2SCHIgDXpARE2jgSA6aNQokV7SpKJJRC97UlRaGYLREyr/n3VObP977uHee+CC0Acfwr8u13XPXnufbfaf8jIbTtWwDEyGVRqmsQDegTO1UAb4gG7BFVpIYgq8DXdrIWIaPKRhkayFR2EbXCU1sgm+hNVaUE7Du3BCkHEkjsCr8Af8GtSKpRU+hFvhDvgU7h/zCacHdmkYUglHYL3k/IUHYAO8aaU3y6X9C1cHWRP8CZcFGVlu/mBqJf/HQfhAQ6HXSm/2jPnDCOGG+mM+Esp1eE7DmCvwkoZClmbvwdcamv+AaxqC85by8J6bz2waWZp9B/s1BJ/gYw3BYfOxmaOFSfAX3KcFIUuznM2kpj5EKtyEbDZnbmuiQqMWBDb7TcMC4YZ5oqF5o+81BCvNe+LGHkNdVODfNNjsdw0L5C18piEYgvc1BIvMe1qvhcVRYZsWhCzN8q2oy83znOPBc1VZZ97TUi3MiAo8/NNgs1zOJHgM5XxxQLv5MVURZPPN/29LkMXsgb/NvzcHLtMFDYUb5htxuhZAh3kzuySPWQK/wM1BxoYGzZtWjsM3GsZ0wD7JyFzzWXtlfibSYfPN0hx8jm+5z7AzyJQNcAAegyfhC0vY7REX4WUNYzivPPPCZSoWvpq7NRS4KhvNG+f4JTHR/AFt0UIMP/DI/EZUKjst/42tGPaav+14/ueF1zPO1UItFADvv33m18gszIIf4XYtJMFLRdrc5YO3qTUalsBZeELDNNrhPA3LAFfnlPlIjm9GARoQfDiWhZiwAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAjCAYAAAApBFa1AAADEElEQVR4Xu3dW8ilUxgH8OWcRjk0UjKaC0QOzY1ByphwOSkhjBSJ5kYzkSbJHXEhJZNDU5pBUySJ4opxIeVGalJSIhdOSU43FJ6ntfbsZXn7vm9mClu/X/3baz37bV8/Petd31cKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/O8dHTl5LAIAMHf/WFihE8bCQTo28t5YBABYVEdE9kXObfuTIn9EHtn/xIH5KbJpqO0e9lPOimwfi4fohbEAALCo3o+c3u1fK7VpO1B5FHntUDsu8vBQm/J05MixeIjeGgsAAIsqG7Y13f6T8teG7ajIXZErulq6PXJ95LK2vzSybv51OSNyc+TOyJldfcrHkVWRjeMXEw6LXDkWm9Xd+pluDQCw0LJheyhyb+TlyFORU9t32UTld+nwyIelTtK+brX0RPt8oKvNvDsWJtxQaoOYx7Npuene3ZGzI7e1/ZftMxvLN9o6XVPmR729SyLvDNkbebvUqVz/GwAA/wnZsOUE7ery9ynatsiF3T6bqVsiP0TuKPUoNSdp6dnZQ002e78OtSlPRn7p9ss1bHmpYGtkbamNW//8o936gsiGbg8AsLCyYTtvLDZ5YeD8bp/N0Y5Sp2w3RnaWOp1Ku9rnTE7mfhtqU/I372vrY8ryR5mnlHmTlhcLZhO21N80zaPaqaPYEyNXLZGxaQUA+Nd9EFk/Fpu8CPB6W2ej80pb91OtvGWa8p2207r6V6U2dDmJu6fVfo9s3v9E9XOpx5kpLyhkM3hx2+fz+X5bLy8y5IQv5a3Uz9o6J3W92VEuAMBC+zTyY+SbUi8bTMnjxz2lTrJyApY+jzxe6jTsolZbG7m8rVP+mY7vIw92te8iz3X79Hy3zt96qdt/VGpDN8oj3LxZmrdSv4i8WubvtM28OOwBAAjflqX/AO5NkVvH4hLyRuibY3GFsqkEAGCQ78M9NhY7e0u9jLBSeelhy1hcgWwarxuLAABU54yFf9jx5eD/UwMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACM/gTVa2x+eTrNVwAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFIAAAAXCAYAAACYuRhEAAAESUlEQVR4Xu2XechuUxTGH7NL5kLGixARCplFhm6hbsiQIUXmJKXMJUNCkZnoImOJ5A9TLjJdc4oMcT+kzPOUIZ6ftff37rPfc3zv+wn3j/Orp96z937P3nvttdZeR+rp6enpGYWFrL2sC62DrDWb3ZOsbh1pnWFtUfW1sYT1gLV23fEfsJR1tHWOtbe1XLN7kk2tU63jrbWqvjYOsM6rG2E161HrTGsn63rrJ+vAcpDZ0npBMeG21h3WfY0Rw1xk/WFtVHf8y+xmPWQdY+1iPWV9aG1SDjLHWo9Y+1qzrOetkxsjmmCrr6y76g64zPrc2iY9L279qvjDjDzIzNewZ2HYQ6u2zNbW9/p/DPmq9boGXrirYh0PTo6QNrDeUERjZkXr69TXxj3W7+ow5MWKSbJBlrR+URiB8IA10pjt03OGQ7iqagNC+gnrEo1uyJWtpevGAjY8SugBB8yGZ6bnbMi5eYA5ROFAyxZtwCHsX7UB48+1flSHIRe21i+et1NMSuhmGMOk72jguRj5TUU6qDnfOliRe0Y1JB5MmNUbA+afo9jMKHAgHH7mdMU6yJmZzVIbXkruhw2tTzW8hlUUjkG0dhqyBIO+Ys3T4DQzJG0m5qQvVyzgxMaIgEuIEIBxDAnktset5Ys2jHiLmkYYB3LfZ4qDyBGWeUyxvm+s06wXrR0bIwIMt1X6PaUhr1HkFU6Em6wNxjAx+kQRMiWLKsKHpAzjGhJ2t55UGBMj3qq4FMZlGcUl+p7CQOXhZEhBOE7eE5fNOo0R0n6KSzMzpSEzeMV31llVOzn0ZWtP6xnFxOTSnYsxlEVHFM/TMSTsoTDmbYoq4Z9ytuLg9ynaOKSrrTsVxuJWZ63vK0IZuHwIae6NzMiGhGyojdMzbs0tzoUALIIy4WfrrdSGse5PvzPTNSSeTdiRg1eo+qYDIf2b9ZHi3XCK9bQGtzZ58SbFeq9LbXM0fMF2GvIk64Sq7WbFCynQ4QbrxkH3JNxijOPkjlOc5oQinBC5h35O+7n4y5Sw0bsV4Yy34xFtYdkFBmHdeHUJa2MtRBS8rfbSjUh4Lf3mcuV/8xX7mVC8g4qGZz5e/mJm6kA5r8Hc1JZzE2VOmyG5+Sje8ynXXKHxPLI0YgZjsp6uL5OawxVz5kiBxRRpqFwLZU6bIXEsqoc2VlK8Y8gjSbZY/PaijdLhi6RcFrCZL6318qDEpYoLqItrFRNj8KlYRFFyHVV3KEosQn0UYzIXKYevmgwlG+uYV7SRN3meUbTxm68gPgPbWFXxnnvrDqCKJxddqcgbzypOc/NykDnM+kFR2lAqPKxI1HVJAdSQ7yq+Er5VHALv/TtmKbypC3LVqBcPlwdfLRcoasgJhZeV+ZY8jxOwNvbO9/NLivFtsP6PFftBH6gI7QwJl5KHjnWrvhI8glsdQ9XfrQsaRNsOCqMSkl0QdbOTyiK+p6enp6dnweFPGTf09XjMeTkAAAAASUVORK5CYII=>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEwAAAAYCAYAAABQiBvKAAADXUlEQVR4Xu2XWahNcRTGPzOhTJkyPlCGzGOGKxSieKBEHiiSWYRMKfGAEMpUSCQPQoZQhgxRijwh81QikZlM39f6784+617OPtx07+n86uvuvdb/nr33+q+19tpAnjx5ShmVqFremKcwBdQb6id12flykf7UPeoBdScc36VOxBdlQpmlgK32jhzmMOyZh3tHEgbC/nmYd+Qwz6m3VHnvSMIK6jtVwztKAM29wVEd2ffelrAEOeYdf6IT1SMcX6KuxXwlienUBqqMd5A61DmqjbNnYiIsYHO9oyiaUBeoI9Ti8PcrtTa+qISxiNqC9KApWOepbjFbUvbCAtbFOzwNqMewi0esxD80v//IEmorLGgKlja9e9qK5DyFTQblvMOzg/pC1Y7ZlsP6V82YbQJ1m5oWsxUX/air1HZnT4KCtpO6iFQ7yZYWsAQ56h0B9fGKOqgLC4x2Jo7S+rqziVtUT28sJvbAelO21KceUYdQdE9LwiRYwOZ5R0C/rVhhEGyhMiqiCizj1sdsQjf2jqrg7MXFE6qdN2ZA93QFVoZq1srQvwnaPlgciirnttTx6KQfbOHgyACbemUbAfuBycE+mjpLDaVWUb2DPaIeLEP0omgds+uC86lR1BCkP1AvWL8cT71yvkzEgxUxB6melg2av5QMfv7Sp+EpamxkUF1+oMaFczVOlaIC1ozahNSubw4+9boC6kawCwVZPs0/jWF9UQygTsIurB74ItiFek+U2epBB2O+TKg8NPZ09Q5Y0HSvSYPWEfa8/jOoM3Waek9VjTtUvw9hF1FK6+H11jxDbUwtw02qTzjW9K/dFSphfXutoWZQ62AzUFnqGSyLhbLoQDjWTWqj9L9iNzUrHCdhAew3fscUFK4AT1/qPuw+FLCPsDioHyrbf1DfYJtZCN24Jt0IpWbD2Ll2VJGO+pcySBkiNLd8ppoivEkCKrdPsOwSu2ABFZr1tCER2qAOsfNSz0hYLQul+kuqFbUMVoZq2FEJKKh6mbRH6k2rDdGcozRXdsyEZaJQyb+Gfe6MCbZSj8otKhkFRoFYitSIoaaujJsNa+Kaa7RuGyw4yqj9sJLX51cjWKpPpRbCxhi9lSsjR0hrerBpuJqzKYv8OhEffn1AIp+CG5Vunjx58iThF5SwnvWFaDyEAAAAAElFTkSuQmCC>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFQAAAAYCAYAAABk8drWAAADtElEQVR4Xu2YR4hVSRSGfwPmMWdEHEyzUTBHtBXTMGIAF6bRjRFREUUFwyBiFhUVZphBbRVH0I2i4kIxiyCoOwNiQEzMyowKhv/vUzWvXvFs75NedLgf/Lxb59z0Tp06VXWBlJSUlCyqUw1jY0r+DKReUF+oK5GvInKEekDdo+5S993vb+FJ30OZqYBuih0VlC7IJJhGbt4Mgd1gZOyooMyCxWNx7EjKGuoTVT92lHKaUrVjY0AlqnVsTMC/sID2iB3F0ZXq7Y4vU9cDX1mhJ3WKqhs7SGWqkJoc2ZPwmHpJVYkduVCPXaSOUcvd70dqS3hSGULl6hyyR5eCuY+aGdiS0h6WncdjRy5aUI+ovwLbWtgNRge2ssZQ6gIsqArmfmp21hnJmQaLx6LYkYvd1AeqUWBbDaufDVxbdaeQ+g9Wo5KgP1FSJBpmORgGC+oBak7kywddr4B2jx2OZv5AwVHgNNxD9BI3IpuKsdZfSahKPUTxk0NSplMbYmNC9B5nqNvIJMeP8JR6hdwd243a7hvDYZFXRnpqwjJ2W2ATSvc9ke1b9IL9iZLgJDU+NiZAwTwEG+baqJzHj61YOsJipPfIxWFkJnIUwE4e4Q1ksLONgQVG6y+hgjzFn+RQh6yg+sOuE6q/mmW1k1jvbEIdNQm2UfBrW12/CjZSNFlshg1ToSBspV7D6l+BsychDKZH9ztL1QtsSdA9FI+lsQM2+WUlTjXqLfW7azeGDXXdoA21k+oMS3UtGcL120pY8FRf98J6StSgTsOWJn5H0QQW4E7OdsL9LoM9Q89s6fxangg9swD2wrpn0pqs6w7CSkXMAFgJyCeoR2Hx+D8LyU+wmvye+iOwFzEDVu/+pP6BZZpmfT14hztHxVj7WM8v1DtY1gkFdK47Vnao3jR3baFFsTJcWahOGAfrPHWQ6rVfTfSlbrpjsQTZq48k/EpNjY0B/ZBsglKCPKM+wwL6HLaffwIriZp79NvOXxCiwHQI2gqKMsazkNoVtJVZCrjnISyThcrEnYyriDfUWKpOZNdz1TG+tm2k1mXcRXVrQtAuN+hri3pd9VLMh9U3oUDqy9TP1ETY5PU3bEirLAitInzAha+hg6hrgf0W1YdaACslynR1rDojry1faUeB1NDzn6tawWb8ebBirWGrVYFq3ShY8VdGt3XnaxiqVqrA6xuBn3h0rSYlzyWYX50jrrq278hyRa5CHi78/QQktI+WLUSTSvyxuhayz9NxuAdXWxNASkpKSmnjK3DIqulMPn6XAAAAAElFTkSuQmCC>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHQAAAAYCAYAAAArrNkGAAAE6ElEQVR4Xu2ZZ4hkRRSFjzm75oRZzDlhxjFgjmCOqJgwo6CYF3PAVRQV4xoQxAgmDKiYAyr+MSDKsuofRcwRRM83t2q7ppjZ6e5ppFveB4fpd9/rfvVu3br31hupoaGhoaGPmMtapDY2zJS+9Nm21g/WP9ab1bmG0el7nxFlDO6a+kTDmPS1z3ZUDG6P+sT/gJWtm6xZK/s21nmVrRP62meXWX9bC9UnBpjFrBusJ6z1Cvta1qPWLdaShb1T+s5nG1mbp8+vW+8X5waZea1zrVesnQr7Mtbtislcs7B3Ql/6bHnrVUXknp/+/mVdV140gMxmHWO9ZR2pVopdwLrUeknRzHRD3/psaWu6dWthu1xRC/YubIPGbtZr1jnWPMk2u3WSYoIPtGZJ9k7pa5/dZf1pLVrYLlHUgoXTMQ8+1frGWiLZxqNuOCYCK61dqI1PK2olNTOzpyIlnm7NWdi7oR2fwdHWp9bJha1XDFnvKkrGDJgcBkHqKKHWfFDZNrW+qGxjwUqYZs1X2bvhWOuq2jgTHlbUxNLZbCc+si7SxCezE5/BJ9YWtbFH3G+dUhp2VqQJoitDeiL6ri9scJZ1d2Ubi80UD9ILnrEOqo3jQLp9wzrbmjvZCDJWCun2gGTrhk58tpT1szVHZe8VX2pktz68bBncLoVt+2TbRzExJyT7k9YR+aIED3eBtbXie0Ated76zLoy2YCHPlSx6c77NL4/WRH1x1vXqtWF0qxMUTjkPsVYO4E0zep+WzHuXAImKcZIQ8S+s1OG1L7PCETus7t1tcJPJWyRWGE0UmydMusognF/a1eNrPVbKcZ/lPVddW44/fxqHZ6OqTmkDQa3omLzTQTgnB8VnV3mQsUP84P3WA8lOyviBeswxXtNWFwxwesm21PpL5t47sE92UJw/qv0He45pFjp/Ga3NZm0z1hJiWz8M8sqauEj1uqFfTza9Rmwr+Uc6Z8A/TDZgSDgHB33coqxwA7Wswr/UIPpWzIsnpwZyJaPFedmcJyi3nFzCiw3mm69aN2YrtnE+jx9hjWs39TqHpnQnMtJbT8p0k3mAcUKZxUSBPspHEGA4OjcGW6pqHUZorTsJCcCq+FmxfaCwMngfGyca7fha8dn8LFaWYCsRLoH/IY/yUinKjLR2oqg/VqtbMQqJOBgQ0UgZZ/fq2jwRoWLViuOmRRWTOZM687imJXF4DPT1IpKUg6dXckv1r7W/JWd+xIY+c0KaemK1unh+nlwcdwLeE6yyR0a2T2zMl5WOLgdxvMZwcFz5/rJCmSFAQvkD2sFjWzSSKe/q5XZpqo1Hva6pc8JoA2K4454XLExzwM6TRFVwETyX4aVrEMUzdNtikGR6oCOsCzeuYZuZ71X2IloOsIzFKmclY6TCAa67F4yWprlnuUkTQSy0HPpM7/7reJt1MWKNEtDk+sfk04/sb5anTIBQ/nZWLGXrn3+vVo+7xgmktRHcQfqDzmc6OFmpE06PGrdXopGgBW9Srqewk5tOVHxvjM3Pnx3cvoMvAjgPAOFd9JxDqRBgnSaUyITx0QR4HkLQznhuQheytCq6ToWA5PHinxQkdJ5vZh9zouR/Boz+7wrJtUGjXzxkNMELJhsJdSH+h+/vGMtr+Mz3y2PieZBpN6Dk95HKzn1dVC+nKgnbCyfNzQ0NDT8d/wLdI/5X6ekjM0AAAAASUVORK5CYII=>