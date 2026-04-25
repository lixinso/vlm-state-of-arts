# VLM and VLA Adoption in Robotics and Embodied AI

---

## Abstract

Vision Language Models (VLMs) are becoming the cognitive layer for modern robotics, extending from visual perception into planning, control, and physical interaction. In robotics this shift is usually expressed through Vision-Language-Action (VLA) models: systems that ingest visual observations and language instructions, then produce robot actions rather than text alone. This document summarizes the 2026 robotics VLM/VLA landscape, including architecture patterns, safety challenges, reinforcement learning pipelines, middleware integration, edge deployment constraints, and commercial adoption in humanoids, industrial automation, navigation, and human-robot interaction.

---

## 1. Introduction

Robotics has historically relied on modular software stacks: perception, localization, mapping, planning, and control were built as separate components connected through explicit interfaces. This worked well in structured environments such as manufacturing lines, but it struggled in open-world settings where robots face novel objects, ambiguous instructions, and changing physical conditions.

Foundation models are changing this design. VLMs provide semantic understanding, common-sense reasoning, and language grounding. VLAs extend that capability into physical control by mapping multimodal inputs into action sequences.

### 1.1 From VLM to VLA

| Model Type | Input | Output | Robotics Role |
|------------|-------|--------|---------------|
| VLM | Images/video + text | Text, labels, explanations | Scene understanding, task planning, inspection |
| VLA | Images/video + text + robot state | Actions, trajectories, joint targets | Manipulation, navigation, loco-manipulation |
| VLA+ | Vision + language + proprioception/tactile/force | High-frequency action corrections | Contact-rich manipulation and physical adaptation |

### 1.2 Why Robotics Needs VLA Models

VLA models address several long-standing robotics bottlenecks:

- **Long-tail generalization**: Robots can reason about unseen objects and rare situations.
- **Natural instruction following**: Operators can specify tasks in language instead of code.
- **Cross-task transfer**: Learned policies can adapt across robots, environments, and object categories.
- **Explainability**: VLM reasoning can expose why a robot selected a plan or refused an instruction.
- **Data efficiency**: Foundation-model priors reduce the amount of task-specific demonstration data required.

---

## 2. Robotics VLM/VLA Landscape

The 2026 robotics stack is split between large frontier models used for high-level planning and compact open-weight models optimized for edge inference. Large proprietary models are useful for semantic planning, simulation, instruction decomposition, and operator interaction. Smaller models are more practical for on-device perception and closed-loop control.

### 2.1 Model Tiers

| Tier | Typical Use | Strength | Limitation |
|------|-------------|----------|------------|
| Frontier proprietary VLMs | High-level task reasoning, planning, dialogue | Strong semantic reasoning and long context | Cloud dependency, latency, cost, limited control over weights |
| Open-weight VLMs | Robot perception, grounding, fine-tuning | Customizable and deployable on local hardware | Requires integration and optimization work |
| VLA policies | Direct action generation | End-to-end perception-to-action learning | Data scarcity, safety validation, latency constraints |
| Specialist action heads | Grasping, navigation, bimanual control | Faster and more controllable | Narrower task coverage |

### 2.2 Representative Robotics Models and Frameworks

| Model / Framework | Focus | Key Idea |
|-------------------|-------|----------|
| RT-2 | Robot instruction following | Transfers web-scale VLM knowledge into robot actions |
| OpenVLA | Open robotics VLA | Fine-tunable VLA model for manipulation research |
| Physical Intelligence pi0 / pi0-FAST | General robot policies | Cross-embodiment robotics foundation models |
| Figure Helix | Humanoid loco-manipulation | Hierarchical high-level reasoning plus fast whole-body control |
| TwinVLA | Bimanual manipulation | Couples two single-arm VLA priors with lightweight coordination |
| FALCON | Spatial grounding | Injects 3D spatial tokens into action heads |
| ProgressVLA | Long-horizon manipulation | Adds task progress estimation to diffusion policies |
| Robometer | Reward modeling | Uses trajectory comparisons to train general robotic reward models |
| RoboNeuron | Middleware integration | Bridges foundation models and ROS 2 through MCP-style tool interfaces |

---

## 3. Architecture Patterns

Robotics imposes stricter architectural requirements than image captioning or visual question answering. A robot policy must coordinate slow semantic reasoning with fast physical control while respecting geometry, contact, and safety constraints.

### 3.1 Modular Robotics vs. End-to-End VLA

```
Traditional Robotics
Sensors -> Perception -> Mapping -> Planning -> Control -> Actuators

End-to-End VLA
Vision + Language + Robot State -> VLA Policy -> Actions

Hybrid VLA Deployment
Vision + Language -> VLM Planner -> Action Head / Controller -> Actuators
Robot State + Tactile Feedback --------------------------^
```

Most near-term systems are hybrid. They use VLMs for semantic understanding and task planning, while lower-level controllers handle real-time stability, force control, and actuator safety.

### 3.2 System 0/1/2 Hierarchy

Humanoid and mobile manipulation systems increasingly use a layered hierarchy to reconcile different timing requirements.

| Layer | Frequency | Responsibility | Example Capability |
|-------|-----------|----------------|--------------------|
| System 2 | 1-10 Hz | Slow semantic reasoning and task planning | "Unload the dishwasher and sort plates by size" |
| System 1 | 50-200 Hz | Visuomotor policy and whole-body coordination | Reach while adjusting stance |
| System 0 | 500-1000 Hz | Reflex-level stability and torque control | Balance, contact response, actuator smoothing |

This split is important because a VLM cannot safely run the entire robot loop at token-generation speed. The VLM can think slowly, while learned controllers maintain continuous physical stability.

### 3.3 Spatial Grounding

General VLMs are usually trained on 2D images, but robots operate in 3D space. This creates a spatial reasoning gap: a model may identify an object correctly but still fail to infer reachability, depth, collision constraints, or object pose.

Recent approaches address this by separating semantic and geometric reasoning:

- **Semantic VLM backbone**: Understands objects, instructions, and goals.
- **Embodied spatial model**: Processes depth, pose, point clouds, and camera geometry.
- **Spatial action head**: Fuses semantic intent with 3D spatial tokens to produce feasible actions.

```
RGB / Video --------> VLM Backbone --------> Semantic Action Token
Depth / Pose -------> Spatial Encoder -----> 3D Spatial Tokens
                                           |
                                           v
                               Spatial-Enhanced Action Head
                                           |
                                           v
                                  Continuous Robot Action
```

### 3.4 Bimanual and Whole-Body Coordination

Single-arm manipulation has benefited from larger datasets than bimanual manipulation. For two-arm systems, data scarcity and collision constraints make monolithic training difficult. Modular approaches such as TwinVLA reuse strong single-arm policies and add coordination layers for synchronization, collision avoidance, and shared task intent.

Whole-body humanoids add another layer of difficulty: locomotion and manipulation cannot be treated separately. A humanoid reaching for an object changes its center of mass, so walking, balance, posture, and grasping must be coordinated continuously.

---

## 4. Physical Grounding and Safety

Robotics exposes VLM failure modes that are less severe in digital tasks. A hallucinated caption is a quality issue; a hallucinated action can damage property or injure people.

### 4.1 Action Hallucination Modes

| Failure Mode | Description | Example |
|--------------|-------------|---------|
| Topological failure | Ignores physical connectivity and obstacles | Moving an arm through a solid object |
| Precision failure | Cannot produce fine-grained actuator accuracy | Missing a peg insertion by millimeters |
| Horizon failure | Loses task progress over time | Repeating a subtask or releasing too early |
| Affordance hallucination | Treats an impossible action as feasible | Trying to grasp a transparent reflection |
| Instruction conflict | Follows visual priors over language | Picking the visible object despite a contradictory instruction |

### 4.2 Linguistic Blindness

Some VLA models overweight visual affordances and underweight language. When the instruction conflicts with the visual scene, the policy may perform the most visually plausible action rather than obeying the command or refusing safely.

Mitigation strategies include:

- Instruction-conditioned attention recalibration.
- Explicit contradiction detection.
- Refusal or clarification behavior for impossible instructions.
- Safety monitors outside the VLA policy.

### 4.3 Progress Awareness

Long-horizon tasks require an internal sense of completion. A robot must know whether a drawer is fully closed, whether a cup has been placed securely, or whether a subtask has already succeeded. Progress-aware VLA policies add progress estimation to prevent looping, premature termination, and state drift.

### 4.4 Tactile and Force Feedback

Vision alone is insufficient for contact-rich manipulation. During grasping, insertion, wiping, or assembly, the key information may be tactile rather than visual. VLA+ systems route force and tactile feedback into high-frequency action experts, enabling fast reactions to slipping, resistance, and contact changes while leaving slower semantic planning to the VLM.

---

## 5. Training, Reinforcement Learning, and Data Generation

The biggest constraint for robotics foundation models is not only model size; it is high-quality physical trajectory data. Unlike web text or images, robot data must be embodied, kinematically valid, and tied to real-world physical outcomes.

### 5.1 Training Data Sources

| Source | Strength | Weakness |
|--------|----------|----------|
| Human teleoperation | High-quality demonstrations | Expensive and slow to scale |
| Robot fleet logs | Real deployment distribution | Limited to deployed tasks and hardware |
| Simulation | Cheap, controllable, scalable | Sim-to-real gap |
| Video pretraining | Massive visual diversity | Lacks action labels and robot state |
| Synthetic trajectories | Scalable task variation | Requires validation against physics and hardware |

### 5.2 Simulation-First Pipelines

Simulation-first training is increasingly important for scaling embodied AI. Diverse resets, procedural environments, randomized physics, and synthetic demonstrations can expose policies to more states than physical data collection alone.

Key techniques include:

- Programmatic environment resets that start policies from many boundary conditions.
- Domain randomization for lighting, textures, object shape, friction, and camera pose.
- Procedural scene generation for kitchens, warehouses, factories, and homes.
- Simulated failures and recovery states, not only expert successes.

### 5.3 Reward Modeling

Reward modeling is critical because dense hand-authored rewards do not scale across thousands of tasks. Modern approaches train reward models on trajectory comparisons, including failed and partially successful attempts.

| Reward Signal | What It Captures | Robotics Value |
|---------------|------------------|----------------|
| Absolute progress | How close a frame is to task completion | Tracks partial completion |
| Pairwise preference | Which trajectory is better for the same task | Learns from failures and near misses |
| Human correction | Operator feedback during deployment | Supports continuous improvement |
| Safety penalty | Unsafe proximity, force, or collision states | Reduces dangerous exploration |

---

## 6. Middleware and Systems Integration

Most robots run on Robot Operating System 2 (ROS 2), industrial control buses, or vendor-specific middleware. Foundation models, however, tend to emit unstructured text, JSON, tool calls, or high-dimensional tensors. Bridging these worlds requires typed interfaces, safety boundaries, and latency-aware routing.

### 6.1 Foundation Models and ROS 2

ROS 2 uses strongly typed topics, services, and actions. VLMs and LLMs are probabilistic generators. Directly connecting them with ad hoc wrappers creates brittle integration and unclear safety boundaries.

Better integration patterns expose robot capabilities as typed tools:

- Discover available sensors, controllers, services, and URDF metadata.
- Translate them into structured tool schemas.
- Let the cognitive model invoke tools within allowed policies.
- Validate generated commands before publishing to robot topics.

### 6.2 Simple and Complex Execution Paths

| Path | Use Case | Flow |
|------|----------|------|
| Simple path | Immediate commands and safety overrides | Tool call -> ROS command -> controller |
| Complex path | Spatial reasoning and manipulation | Sensors -> VLA inference -> action head -> controller |

This split prevents every action from passing through a slow VLM. Reactive commands, emergency stops, and low-level stabilization must remain close to the controller.

### 6.3 Safety Boundary

```
Operator Instruction
        |
        v
VLM / Task Planner
        |
        v
Typed Tool or VLA Action Request
        |
        v
Policy Validator / Safety Monitor
        |
        v
ROS 2 / Robot Controller
        |
        v
Physical Robot
```

The safety monitor should be independent of the generative model. It can enforce workspace limits, velocity bounds, force thresholds, protected zones, and task-specific constraints.

---

## 7. Edge Deployment and Hardware Constraints

Robots cannot depend exclusively on cloud inference. Network latency, jitter, privacy, and reliability all push robotics toward edge deployment.

### 7.1 Latency Requirements

| Function | Typical Frequency | Deployment Implication |
|----------|-------------------|------------------------|
| High-level reasoning | 1-10 Hz | VLM can run slowly or asynchronously |
| Visual servoing / action policy | 30-200 Hz | Requires optimized local inference |
| Balance / torque control | 500-1000 Hz | Must run outside the VLM |
| Safety interlock | Real time | Deterministic controller-side logic |

### 7.2 Model Compression

Running VLMs on embedded hardware requires aggressive optimization:

- 4-bit or 8-bit quantization.
- Distillation from larger teacher models.
- Smaller dynamic-resolution vision encoders.
- KV-cache and memory optimizations for video inputs.
- Hardware-specific acceleration on NVIDIA Jetson, edge GPUs, NPUs, or custom robotics silicon.

### 7.3 Edge-Cloud Split

Some deployments use a split architecture:

- **On robot**: Fast perception, control, safety, local fallback.
- **Near edge / MEC**: Heavier VLM reasoning with lower latency than public cloud.
- **Cloud**: Fleet learning, simulation, model training, world-model validation.

This allows robots to keep moving safely even when connectivity degrades.

---

## 8. Specialized Robotics Domains

### 8.1 Industrial and Warehouse Robotics

VLA models are useful for flexible picking, sorting, packaging, kitting, inspection, and assembly. Their main advantage is adaptability: a robot can handle new SKUs or instructions without full reprogramming.

| Task | VLA Advantage | Remaining Challenge |
|------|---------------|--------------------|
| Pick-and-place | Generalizes across object categories | Grasp reliability and cycle time |
| Assembly | Combines visual checks with action planning | Precision, force control, fixtures |
| Quality inspection | Explains visual defects in language | False positives and domain drift |
| Warehouse sorting | Handles varied packaging and labels | Throughput and edge compute |

### 8.2 Household Robotics

Home environments are unstructured, cluttered, and user-specific. VLMs help robots interpret natural language instructions and identify objects, but household robotics remains limited by safety, reliability, cost, and manipulation success rates.

### 8.3 Navigation

VLM-based navigation enables semantic goals such as "find the manager's office" or "go to the nearest open seat." Vision-language frontier maps and semantic scene descriptions can guide exploration in unfamiliar environments without pre-authored metric maps.

### 8.4 Autonomous Driving

Autonomous driving is a specialized robotics domain with stricter safety and validation requirements. Driving VLA systems combine high-level context understanding with world models and trajectory prediction. VLMs help interpret rare scenarios, construction zones, unusual actor behavior, and natural-language explanations of driving decisions.

See [VLM Adoption in Self-Driving Car Industry](../autonomous-driving/vlm-autonomous-driving.md) for a dedicated autonomous-driving analysis.

### 8.5 Human-Robot Interaction

Human-robot interaction uses VLMs for scene understanding, intent inference, explanation, and mixed-initiative recovery. In collaborative settings, robots must communicate limitations clearly instead of silently failing.

Applications include:

- AR explanations of robot plans and limitations.
- Assistive feeding and manipulation.
- Navigation support for visually impaired users.
- Socially aware proxemics and gaze-aware interaction.

---

## 9. Commercial Adoption

VLA-enabled robotics is moving from research demonstrations into commercial pilots and early deployment, especially in logistics, manufacturing, and humanoid robotics.

### 9.1 Leading Deployment Areas

| Area | Adoption Status | Why VLM/VLA Helps |
|------|-----------------|-------------------|
| Warehouses | Active pilots and production niches | Varied objects, labels, and layouts |
| Manufacturing | Inspection and assembly pilots | Visual reasoning plus procedural instructions |
| Humanoids | Early industrial pilots | General-purpose manipulation in human spaces |
| Assistive robotics | Research and controlled deployment | Natural interaction and scene understanding |
| Mobile service robots | Growing adoption | Semantic navigation and human interaction |

### 9.2 Humanoid Robotics

Humanoids are a major commercial test case for VLA models because they require perception, language, whole-body control, balance, manipulation, and human-safe behavior in a single platform.

Representative systems include:

- **Figure 02 / Helix-style architectures**: Hierarchical reasoning and full-body action policies for industrial tasks.
- **Tesla Optimus**: Vision-based manipulation and factory-oriented learning from video and fleet data.
- **NVIDIA Groot ecosystem**: Foundation-model infrastructure and simulation tooling for humanoid robotics.
- **Unitree and other low-cost platforms**: Faster experimentation with humanoid hardware and edge inference.

### 9.3 Deployment Bottlenecks

| Bottleneck | Impact |
|------------|--------|
| Battery life | Limits continuous industrial shifts |
| Actuator cost and durability | Raises platform cost and maintenance burden |
| Reliability gap | Lab success may degrade in real environments |
| Safety certification | Slows deployment in shared human spaces |
| Data flywheel | Requires fleet scale and high-quality feedback loops |
| Supply chain | Critical components such as actuators and gearboxes can constrain scaling |

---

## 10. Key Challenges

| Challenge | Description | Research Direction |
|-----------|-------------|--------------------|
| Spatial reasoning | 2D VLMs lack native 3D grounding | Spatial tokens, depth fusion, world models |
| Contact-rich manipulation | Vision fails under occlusion and force interaction | Tactile VLA+, force-aware policies |
| Long-horizon reliability | Policies drift or loop across many subtasks | Progress estimation and hierarchical planning |
| Real-time control | VLM inference is too slow for actuator loops | Hierarchical controllers and async execution |
| Data scarcity | Robot trajectories are expensive | Simulation, video pretraining, reward models |
| Safety validation | Generative actions are hard to certify | Independent safety monitors and formal constraints |
| Edge deployment | Models exceed embedded compute budgets | Quantization, distillation, hardware acceleration |

---

## 11. Future Outlook

### 11.1 Short Term: 2026-2027

- More hybrid VLM planner + specialist controller deployments.
- Wider use of open VLA fine-tuning for manipulation tasks.
- Better benchmarks for physical reasoning, progress tracking, and action safety.
- More edge-optimized robotics VLMs under 10B parameters.

### 11.2 Medium Term: 2027-2030

- Commercial humanoid pilots expand in factories and warehouses.
- Simulation-first training pipelines become standard.
- Tactile and force feedback become common in high-end manipulation systems.
- Middleware layers standardize typed interfaces between foundation models and robot controllers.

### 11.3 Long Term

Robotics VLMs will likely converge toward embodied foundation models that combine visual perception, language, proprioception, tactile feedback, world modeling, and action generation. The remaining bottlenecks are as much physical and operational as algorithmic: energy density, actuator reliability, cost, safety certification, and deployment-scale data collection.

---

## 12. Conclusions

VLMs have moved robotics beyond visual recognition toward semantic planning and grounded action. The most practical 2026 systems are not pure end-to-end black boxes; they are layered architectures that combine VLM reasoning, spatial grounding, learned action policies, typed middleware, and deterministic safety controllers.

The central trend is clear: robots are becoming language-instructed, visually grounded, and increasingly general-purpose. The core challenge is turning impressive foundation-model reasoning into reliable physical behavior under latency, safety, and hardware constraints.

---

## References

1. Google DeepMind. "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." 2023.
2. Physical Intelligence. "pi0 and pi0-FAST: Robotics Foundation Models." 2025.
3. OpenVLA. "OpenVLA: An Open-Source Vision-Language-Action Model." 2024.
4. Zhang et al. "From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors." ICLR 2026.
5. Zhang et al. "Restoring Linguistic Grounding in VLA Models via Train-Free Attention Recalibration." arXiv:2603.06001, 2026.
6. Yan et al. "ProgressVLA: Progress-Guided Diffusion Policy for Vision-Language Robotic Manipulation." arXiv, 2026.
7. Yin et al. "Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning." ICLR 2026.
8. Liang et al. "Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons." arXiv:2603.02115, 2026.
9. Guan et al. "RoboNeuron: A Middle-Layer Infrastructure for Agent-Driven Orchestration in Embodied AI." arXiv:2512.10394, 2026.
10. Englmeier et al. "WorldVLM: Combining World Model Forecasting and Vision-Language Reasoning." arXiv:2603.14497, 2026.

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| VLM | Vision Language Model; model that jointly reasons over visual and text inputs |
| VLA | Vision-Language-Action; model that maps visual/language inputs to robot actions |
| VLA+ | VLA extended with additional physical modalities such as tactile or force feedback |
| ROS 2 | Robot Operating System 2, common middleware for robotics software |
| MCP | Model Context Protocol, a tool/interface protocol useful for typed model integration |
| Sim-to-real | Transfer of policies trained in simulation to real robot hardware |
| Loco-manipulation | Coordinated locomotion and manipulation, especially for humanoids |
| Proprioception | Robot internal sensing such as joint position, velocity, and torque |
| Action head | Model component that converts high-level intent or embeddings into robot actions |

---

*Last updated: April 2026*