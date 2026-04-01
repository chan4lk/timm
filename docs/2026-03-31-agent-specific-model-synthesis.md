# Agent-Specific Model Synthesis: Compiling Task-Bounded Intelligence from Large Language Models into CPU-Deployable Micro-Models

**Authors:** Chandima Ranaweera  
**Date:** March 2026  
**Status:** Draft v0.1

---

## Abstract

The prevailing approach to deploying AI agents relies on general-purpose large language models (LLMs) with hundreds of billions of parameters, accessed via costly API calls or hosted on GPU clusters. We argue that this paradigm is fundamentally over-provisioned for the narrow decision spaces characteristic of tool-use agents. We introduce **Agent-Specific Model Synthesis (ASMS)**, a pipeline that treats large LLMs as *compilers* rather than *runtimes*: given a natural language agent role specification, a parent LLM generates a comprehensive synthetic training corpus, from which a compact, task-specific model (5--100M parameters) is trained *from scratch*. Unlike fine-tuning, which inherits the full parameter footprint of the base model, ASMS produces architecturally minimal models whose capacity is matched to the agent's bounded decision complexity. We provide a formal analysis of why small models suffice for tool-use agents by bounding the effective decision space and relating it to required model capacity via information-theoretic arguments. We propose a rigorous experimental protocol comparing ASMS micro-models against full LLM inference, fine-tuned models, and distilled models across three benchmark agent scenarios. Our analysis suggests that ASMS models can achieve comparable task accuracy (>95%) while delivering 100--1000x inference cost reduction, sub-10ms CPU-only latency, and training budgets measurable in single-GPU hours rather than cluster-days. We discuss implications for democratising agent deployment, edge computing, data privacy, and sustainable AI.

**Keywords:** model synthesis, task-specific models, knowledge compilation, tool-use agents, efficient inference, edge AI, CPU deployment

---

## 1. Introduction

### 1.1 The Over-Provisioning Problem

The current landscape of AI agent deployment presents a striking paradox. Modern tool-use agents --- systems that select and invoke APIs based on contextual inputs --- operate within highly constrained decision spaces. A typical agent may choose among 5--20 tools, each parameterised by a handful of schema-bounded arguments. The effective number of meaningfully distinct decisions such an agent must make is on the order of 10^2 to 10^4.

Yet the models powering these agents --- GPT-4, Claude, Gemini --- contain 10^11 to 10^12 parameters, trained on trillions of tokens spanning the entirety of human knowledge. These models can write poetry, prove theorems, and debate philosophy. For an agent whose sole purpose is to decide whether to call `check_inventory` or `apply_discount` on an incoming order, this is the computational equivalent of chartering a Boeing 747 to deliver a single letter.

This over-provisioning manifests in concrete costs:

- **Inference cost:** API-based deployment of large LLMs costs $1--30 per million tokens, translating to $0.01--0.10 per agent decision. At scale (millions of decisions per day), this becomes a primary operational expense.
- **Latency:** Large model inference typically requires 100ms--2s per call, even on optimised GPU infrastructure. For real-time agent systems, this latency is often the bottleneck.
- **Infrastructure dependency:** GPU clusters or cloud API access are mandatory. This excludes edge deployments, air-gapped environments, and resource-constrained organisations.
- **Energy consumption:** A single GPT-4 inference consumes approximately 0.001--0.01 kWh. Multiplied across billions of daily agent invocations industry-wide, the energy footprint is substantial.

### 1.2 The Fine-Tuning Trap

The natural response to over-provisioning is model specialisation. Fine-tuning adapts a pre-trained model's weights to a specific task, and indeed, fine-tuned models often outperform their base counterparts on narrow domains. However, fine-tuning carries a fundamental limitation: **the output model retains the full architectural footprint of the base model**.

Fine-tuning a 7B parameter model for a task that requires, at most, the representational capacity of a 25M parameter model still produces a 7B parameter model. The excess parameters are not pruned; they continue to consume memory, compute, and energy at inference time. Parameter-efficient fine-tuning methods (LoRA, QLoRA, adapters) reduce *training* cost but do not reduce *inference* cost --- the full forward pass through the base model is still required.

This is analogous to writing a simple script but being forced to ship the entire operating system it was developed on. The runtime cost bears no relation to the actual complexity of the program.

### 1.3 A New Paradigm: Neural Compilation

We propose a fundamentally different approach. Rather than adapting a large model to a small task, we use the large model to *produce* a small model.

In this paradigm, the large LLM serves as a **compiler**: it takes a high-level specification of an agent's role (its tools, constraints, and decision logic) and generates the *training material* needed to build a compact model from scratch. The resulting micro-model is architecturally sized to match the agent's actual decision complexity --- nothing more.

We term this process **Agent-Specific Model Synthesis (ASMS)** and draw an explicit analogy to traditional software compilation:

| Software Compilation | Agent-Specific Model Synthesis |
|---|---|
| Source code (high-level) | Agent role specification (natural language) |
| Compiler (GCC, LLVM) | Parent LLM (Claude, GPT-4) |
| Executable (machine code) | Micro-model (compact transformer) |
| CPU runtime | CPU inference runtime (ONNX) |
| Binary size ~ program complexity | Model size ~ decision complexity |

Just as a compiler produces an executable whose size and resource requirements reflect the complexity of the source program --- not the complexity of the compiler itself --- ASMS produces a micro-model whose capacity reflects the agent's decision space, not the parent LLM's full capabilities.

### 1.4 Contributions

This paper makes the following contributions:

1. **Paradigm:** We introduce the concept of neural compilation for agent deployment, formalising the distinction between LLMs-as-runtimes and LLMs-as-compilers.
2. **Pipeline:** We propose ASMS, an end-to-end pipeline from agent specification to CPU-deployable micro-model, comprising role specification, decision space analysis, synthetic corpus generation, architecture selection, training, and deployment optimisation.
3. **Theory:** We provide a formal analysis bounding the model capacity required for tool-use agents as a function of decision space cardinality, using information-theoretic and approximation-theoretic arguments.
4. **Evaluation:** We design a rigorous experimental protocol with three benchmark scenarios, four baselines, and six metrics to validate the ASMS approach.
5. **Vision:** We discuss the broader implications for democratising AI agent deployment, enabling edge computing, enhancing data privacy, and reducing the environmental cost of AI.

---

## 2. Related Work

### 2.1 Knowledge Distillation

Knowledge distillation (Hinton et al., 2015) transfers knowledge from a large teacher model to a smaller student model by training the student to match the teacher's output distribution. While distillation reduces model size, it typically starts from an existing architecture and reduces it by a moderate factor (e.g., 7B to 1.5B). The student model inherits the architectural assumptions of the teacher's family. ASMS differs in that the micro-model's architecture is selected *ab initio* based on task complexity analysis, allowing for dramatically smaller models that bear no architectural resemblance to the parent.

### 2.2 Task-Specific Small Models

Recent work has explored training small models for specific NLP tasks. TinyBERT (Jiao et al., 2020), DistilBERT (Sanh et al., 2019), and MobileBERT (Sun et al., 2020) demonstrate that BERT-scale tasks can be handled by models with 10--60M parameters. However, these approaches remain within the fine-tuning / distillation paradigm, reducing an existing model rather than synthesising one from a task specification. Furthermore, they target general NLP tasks (classification, NER, QA) rather than the structured decision-making characteristic of tool-use agents.

### 2.3 Synthetic Data Generation

The use of large LLMs to generate training data for smaller models has been explored in several contexts. Self-Instruct (Wang et al., 2023), Alpaca (Taori et al., 2023), and WizardLM (Xu et al., 2023) use GPT-class models to generate instruction-following datasets. Phi-1 (Gunasekar et al., 2023) demonstrated that high-quality synthetic data can train surprisingly capable small models. ASMS builds on this insight but applies it within a structured pipeline where the synthetic data is systematically generated to cover a formally bounded decision space, rather than sampling from the general distribution of instruction-following behaviour.

### 2.4 Efficient Inference and Edge Deployment

Quantisation (Dettmers et al., 2022), pruning (Frantar et al., 2023), and speculative decoding (Leviathan et al., 2023) reduce inference costs for existing models. ONNX Runtime and TensorRT provide optimised CPU/GPU inference engines. These techniques are complementary to ASMS: they can be applied to micro-models to further reduce their already small footprint. The key distinction is that ASMS attacks the problem at the architecture level (smaller model from the start), while these techniques optimise execution of a fixed architecture.

### 2.5 Neural Architecture Search

Neural Architecture Search (NAS) automates the design of model architectures optimised for specific objectives (Zoph & Le, 2017; Liu et al., 2019). While ASMS's architecture selection stage (Stage 4) could incorporate NAS, we initially propose a simpler heuristic mapping from decision complexity to pre-defined architecture templates, reserving full NAS integration as future work.

### 2.6 Tool-Use in Language Models

Toolformer (Schick et al., 2023), Gorilla (Patil et al., 2023), and function-calling capabilities in GPT-4 and Claude represent the state of the art in LLM-based tool use. These systems demonstrate that tool selection and parameterisation can be learned effectively. ASMS leverages this capability in the parent LLM to *generate training data*, then asks whether the learned tool-use behaviour can be compressed into a model that is orders of magnitude smaller.

---

## 3. Problem Formulation

### 3.1 Agent Definition

We define a **tool-use agent** as a function:

$$A: \mathcal{X} \rightarrow \mathcal{T} \times \mathcal{P}$$

where $\mathcal{X}$ is the space of input contexts (structured data conforming to a known schema), $\mathcal{T} = \{t_1, t_2, \ldots, t_T\}$ is the set of available tools, and $\mathcal{P} = \mathcal{P}_1 \times \mathcal{P}_2 \times \ldots \times \mathcal{P}_T$ is the space of tool-specific parameter configurations.

Each tool $t_i$ has a parameter schema $\mathcal{P}_i = \{p_{i,1}, p_{i,2}, \ldots, p_{i,k_i}\}$ where each parameter $p_{i,j}$ takes values from a bounded domain $\text{dom}(p_{i,j})$.

### 3.2 Decision Space

The **effective decision space** of agent $A$ is:

$$\mathcal{D} = \{(t_i, \mathbf{p}_i) : t_i \in \mathcal{T}, \mathbf{p}_i \in \mathcal{P}_i\}$$

The cardinality of this space is bounded by:

$$|\mathcal{D}| \leq \sum_{i=1}^{T} \prod_{j=1}^{k_i} |\text{dom}(p_{i,j})|$$

For typical tool-use agents:
- $T \in [3, 20]$ (number of tools)
- $k_i \in [1, 5]$ (parameters per tool)
- $|\text{dom}(p_{i,j})| \in [2, 100]$ (parameter domain cardinality --- enums, bounded integers, categorical values)

This yields $|\mathcal{D}|$ in the range of $10^2$ to $10^4$ for the vast majority of practical tool-use agents. Even generous upper bounds place $|\mathcal{D}|$ well below $10^6$.

### 3.3 The Compression Thesis

**Thesis:** For an agent with effective decision space $\mathcal{D}$, there exists a model $M_\theta$ with $|\theta| = O(|\mathcal{D}| \cdot \log|\mathcal{D}|)$ parameters such that:

$$\Pr_{x \sim \mathcal{X}}[M_\theta(x) = A^*(x)] \geq 1 - \epsilon$$

where $A^*$ is the optimal agent policy (as approximated by the parent LLM) and $\epsilon$ is an arbitrarily small error tolerance.

In other words, the number of parameters needed to faithfully represent the agent's behaviour scales with the complexity of its *decision space*, not with the complexity of *natural language* or *world knowledge*.

### 3.4 Contrast with General LLMs

A general-purpose LLM must model $P(\text{token}_t | \text{token}_{<t})$ over a vocabulary of $V \approx 100{,}000$ tokens for sequences of arbitrary length, across all domains of human knowledge. The effective decision space for this task is combinatorially vast --- on the order of $V^L$ for sequence length $L$.

The ratio of decision complexities between a general LLM and a tool-use agent is:

$$\frac{|\mathcal{D}_{\text{LLM}}|}{|\mathcal{D}_{\text{agent}}|} \approx \frac{V^L}{10^4} \geq 10^{100}$$

This ratio quantifies the over-provisioning problem and motivates the potential for extreme compression.

---

## 4. Proposed Method: Agent-Specific Model Synthesis (ASMS)

### 4.1 Pipeline Overview

ASMS is a six-stage pipeline that transforms an agent role specification into a deployable micro-model:

```
[Role Spec] → [Decision Analysis] → [Corpus Generation] → [Architecture Selection] → [Training] → [Deployment]
     ↓              ↓                      ↓                       ↓                     ↓            ↓
  Natural       Complexity            Synthetic              Minimal              From-scratch     Quantised
  language      bound |D|           (x, t, p) triples       transformer            training      CPU/ONNX
   spec                              via parent LLM         sized to |D|
```

### 4.2 Stage 1: Role Specification

The developer provides a structured agent specification comprising:

1. **Tool definitions:** JSON Schema or OpenAPI specifications for each available tool, including parameter types, constraints, and descriptions.
2. **Input context schema:** The structure of inputs the agent will receive (e.g., customer order JSON, monitoring alert payload).
3. **Decision constraints:** Business rules, safety boundaries, and escalation conditions expressed in natural language (e.g., "never apply a discount exceeding 20% without human approval").
4. **Expected behaviour descriptions:** Natural language examples of correct agent behaviour in representative scenarios.

This specification serves as the "source code" that the ASMS pipeline compiles.

**Example specification:**

```yaml
agent_name: order_processing_agent
tools:
  - name: check_inventory
    description: Check stock levels for a given product
    parameters:
      product_id: { type: string, required: true }
      warehouse: { type: enum, values: [US_EAST, US_WEST, EU, APAC], default: US_EAST }
  - name: apply_discount
    description: Apply a percentage discount to an order
    parameters:
      order_id: { type: string, required: true }
      percentage: { type: integer, min: 1, max: 50 }
  - name: create_shipment
    description: Create a shipping order
    parameters:
      order_id: { type: string, required: true }
      priority: { type: enum, values: [STANDARD, EXPRESS, OVERNIGHT] }
  - name: escalate_to_human
    description: Route to human agent for manual review
    parameters:
      order_id: { type: string, required: true }
      reason: { type: string }

constraints:
  - Discounts above 20% require escalation
  - Out-of-stock items must check all warehouses before escalating
  - Express/overnight shipping only for orders above $100

input_schema:
  type: object
  properties:
    order_id: string
    customer_tier: { enum: [STANDARD, PREMIUM, VIP] }
    items: array
    total_value: number
    special_instructions: string
```

### 4.3 Stage 2: Decision Space Analysis

Given the role specification, we formally enumerate the agent's decision space:

1. **Tool enumeration:** List all $T$ tools and their parameter schemas.
2. **Parameter cardinality:** For each tool parameter, compute $|\text{dom}(p_{i,j})|$. For unbounded parameters (free-text strings), apply discretisation based on the agent's constraints (e.g., the `reason` field in `escalate_to_human` can be clustered into a finite set of canonical reasons).
3. **Decision cardinality:** Compute $|\mathcal{D}|$ using the formula in Section 3.2.
4. **Complexity classification:**
   - **Low** ($|\mathcal{D}| < 100$): Minimal model architecture
   - **Medium** ($100 \leq |\mathcal{D}| < 1{,}000$): Small model architecture
   - **High** ($1{,}000 \leq |\mathcal{D}| < 10{,}000$): Compact model architecture
   - **Very High** ($|\mathcal{D}| \geq 10{,}000$): Consider decomposition into sub-agents

For the example order processing agent above:
- `check_inventory`: 1 (product_id is contextual) $\times$ 4 (warehouse) = 4
- `apply_discount`: 1 $\times$ 50 = 50
- `create_shipment`: 1 $\times$ 3 = 3
- `escalate_to_human`: 1 $\times$ ~10 (discretised reasons) = 10

Total $|\mathcal{D}| \approx 67$, classified as **Low**.

### 4.4 Stage 3: Synthetic Corpus Generation

The parent LLM generates a comprehensive training corpus using a structured generation protocol:

**Phase 3a --- Scenario Generation:**
The parent LLM generates diverse input scenarios covering:
- **Normal cases** (80%): Typical inputs spanning the expected distribution
- **Edge cases** (15%): Boundary conditions, unusual combinations, missing fields
- **Adversarial cases** (5%): Inputs designed to test constraint adherence (e.g., requests for 30% discounts, attempts to bypass escalation rules)

**Phase 3b --- Decision Labelling:**
For each generated scenario, the parent LLM produces:
- The correct tool selection $t_i$
- The correct parameter values $\mathbf{p}_i$
- A reasoning trace explaining the decision

**Phase 3c --- Self-Consistency Filtering:**
Each scenario is processed $k$ times (we suggest $k = 5$) by the parent LLM. Only examples where all $k$ generations agree on both tool selection and parameter values are retained. This filters out ambiguous cases where even the parent LLM is uncertain.

**Phase 3d --- Corpus Statistics:**
- Target corpus size: $|\mathcal{D}| \times 100$ to $|\mathcal{D}| \times 500$ examples
- For our Low-complexity example: 6,700 -- 33,500 training examples
- For High-complexity agents: 100,000 -- 5,000,000 training examples

The cost of generating this corpus using the parent LLM is a one-time compilation cost, analogous to the cost of running a compiler. For a Low-complexity agent, this is approximately $5--50 in API costs.

### 4.5 Stage 4: Architecture Selection

Based on the complexity classification from Stage 2, we select a model architecture:

| Complexity | $|\mathcal{D}|$ | Architecture | Layers | Hidden Dim | Heads | Params | Size (FP16) |
|---|---|---|---|---|---|---|---|
| Low | <100 | Micro-Transformer | 2 | 128 | 2 | ~2M | ~4MB |
| Medium | 100--1,000 | Small-Transformer | 4 | 256 | 4 | ~15M | ~30MB |
| High | 1K--10K | Compact-Transformer | 8 | 512 | 8 | ~80M | ~160MB |

**Key architectural decisions:**

1. **Context length:** Tool-use agents receive structured inputs of bounded size. Maximum context length is derived from the input schema, typically 256--1,024 tokens. This is far shorter than the 8K--128K context windows of general LLMs, enabling substantial memory savings.
2. **Vocabulary:** A task-specific tokenizer with a reduced vocabulary ($V \approx 5{,}000$--$10{,}000$) covering the agent's tool names, parameter values, and input schema vocabulary. This is 10--20x smaller than general-purpose tokenizers.
3. **Output format:** The model outputs a structured decision (tool ID + parameter values) rather than free-form text. This can be implemented as a classification head (tool selection) followed by parameter-specific output heads, further reducing output complexity.

### 4.6 Stage 5: Training

**Training procedure:**
1. **Tokenisation:** Apply the task-specific tokenizer to the synthetic corpus.
2. **Curriculum learning:** Train in three phases:
   - **Phase 1** (50% of training): Normal cases only, to learn the primary decision boundaries
   - **Phase 2** (35%): Introduce edge cases, refining boundary behaviour
   - **Phase 3** (15%): Adversarial cases, hardening constraint adherence
3. **Optimisation:** AdamW optimiser, cosine learning rate schedule, batch size scaled to corpus size.
4. **Validation:** Hold out 10% of the corpus for validation; early stopping on validation accuracy.

**Training budget estimates:**

| Complexity | Corpus Size | Model Params | GPU | Estimated Time |
|---|---|---|---|---|
| Low | ~10K | ~2M | Single A100 | ~5 minutes |
| Medium | ~100K | ~15M | Single A100 | ~30 minutes |
| High | ~1M | ~80M | Single A100 | ~4 hours |

These training times are 3--4 orders of magnitude smaller than pre-training a general LLM, and 1--2 orders smaller than fine-tuning a 7B model.

### 4.7 Stage 6: Optimisation and Deployment

**Post-training optimisation:**
1. **Quantisation:** Apply INT8 quantisation (reducing model size by ~2x) or INT4 quantisation (~4x reduction) with minimal accuracy loss on structured decision tasks.
2. **ONNX export:** Convert the trained model to ONNX format for cross-platform CPU inference via ONNX Runtime.
3. **Operator fusion:** Apply graph-level optimisations (attention fusion, layer normalisation folding) for inference speedup.

**Final deployment characteristics:**

| Complexity | Quantised Size | CPU Latency (p50) | CPU Latency (p99) | Memory |
|---|---|---|---|---|
| Low | ~1MB | <1ms | <3ms | <10MB |
| Medium | ~8MB | ~2ms | ~5ms | <50MB |
| High | ~40MB | ~5ms | ~10ms | <200MB |

These models can run on:
- Cloud VMs without GPUs (cost: ~$0.01/hour vs. ~$1--3/hour for GPU instances)
- Edge devices (Raspberry Pi, IoT gateways)
- Mobile devices
- Browser (via ONNX.js or WebAssembly)

---

## 5. Theoretical Analysis

### 5.1 Decision Complexity and Model Capacity

We establish a formal relationship between an agent's decision complexity and the minimum model capacity required to represent its policy.

**Definition 5.1 (Decision Function Complexity).** For an agent with decision space $\mathcal{D}$ operating over input space $\mathcal{X}$, the *decision function complexity* $\kappa(A)$ is the minimum number of bits required to specify the mapping $A: \mathcal{X} \rightarrow \mathcal{D}$:

$$\kappa(A) = \min_{\text{encoding } E} |E(A)|$$

**Theorem 5.1 (Capacity Bound).** For a tool-use agent with decision space cardinality $|\mathcal{D}|$ operating over inputs from a distribution $\mathcal{X}$ with $n$ decision-relevant features each of bounded cardinality $c$, there exists a transformer model with:

$$|\theta| = O(|\mathcal{D}| \cdot n \cdot \log(c \cdot |\mathcal{D}|))$$

parameters that achieves optimal decision accuracy on $\mathcal{X}$.

*Proof sketch.* The proof proceeds in three steps:

1. The agent's policy can be represented as a lookup table of size $|\mathcal{D}|$ over the decision-relevant feature space of dimension $n$ with per-feature cardinality $c$.
2. A transformer's attention mechanism can implement feature selection (identifying the relevant input features), while the feed-forward layers implement the decision mapping.
3. The number of parameters required to represent this mapping is bounded by the product of the number of decision boundaries ($O(|\mathcal{D}|)$), the feature space dimension ($n$), and a logarithmic encoding factor.

For a typical tool-use agent: $|\mathcal{D}| \approx 10^3$, $n \approx 10$, $c \approx 100$. This yields $|\theta| = O(10^3 \times 10 \times \log(10^5)) \approx O(10^5)$, i.e., approximately 100K parameters. Even with constant factors and architectural overhead, this comfortably fits within a 2--15M parameter model.

### 5.2 Information-Theoretic Analysis

**Proposition 5.2.** The mutual information $I(X; D)$ between the agent's input $X$ and its decision $D$ is bounded by:

$$I(X; D) \leq \log_2 |\mathcal{D}| \leq \log_2(10^4) \approx 13.3 \text{ bits}$$

For comparison, the mutual information in general language modelling --- $I(\text{context}; \text{next token})$ --- can be on the order of thousands of bits for long-context generation.

This 100--1000x gap in mutual information directly implies that the model's internal representation can be proportionally more compact. A model that only needs to extract ~13 bits of decision-relevant information from its input does not need the representational capacity to model arbitrary linguistic distributions.

### 5.3 The Overhead of Generality

We quantify the "tax" imposed by using a general-purpose LLM for a bounded task.

**Definition 5.3 (Generality Overhead).** For a general model $M_G$ with parameters $\theta_G$ and a task-specific model $M_S$ with parameters $\theta_S$, both achieving accuracy $\geq 1 - \epsilon$ on task $A$, the generality overhead is:

$$\Omega(A, M_G) = \frac{|\theta_G|}{|\theta_S|}$$

For a tool-use agent powered by GPT-4-class model ($|\theta_G| \approx 10^{12}$) versus an ASMS micro-model ($|\theta_S| \approx 10^7$):

$$\Omega \approx 10^5$$

This means the general model carries $10^5\times$ more parameters than necessary for the task. At inference time, this translates directly to wasted computation, memory, and energy.

### 5.4 Cost Model

We define a formal cost model comparing deployment approaches:

**Inference cost per decision:**
$$C_{\text{inference}} = \frac{\text{hardware\_cost\_per\_hour}}{{\text{decisions\_per\_hour}}}$$

| Approach | Hardware | Cost/hr | Decisions/hr | Cost/decision |
|---|---|---|---|---|
| GPT-4 API | N/A (API) | N/A | N/A | ~$0.01 |
| Self-hosted 70B | A100 x4 | $12.00 | ~50,000 | $0.00024 |
| Fine-tuned 7B | A100 x1 | $3.00 | ~200,000 | $0.000015 |
| ASMS Micro (CPU) | c5.large | $0.085 | ~500,000 | $0.00000017 |

**Cost ratio (GPT-4 API vs. ASMS):** ~60,000x

**Annual cost at 1M decisions/day:**
- GPT-4 API: ~$3.65M
- ASMS on CPU: ~$62

---

## 6. Experimental Design

### 6.1 Benchmark Tasks

We propose three tool-use agent scenarios of increasing complexity:

**Task A --- Weather Lookup Agent (Low Complexity)**
- **Tools (3):** `get_forecast(location, days)`, `get_alerts(location, severity)`, `convert_units(value, from_unit, to_unit)`
- **Decision space:** ~50 effective decision paths
- **Input:** Natural language weather queries parsed into structured form
- **Challenge:** Unit conversion logic, alert severity mapping

**Task B --- E-Commerce Order Agent (Medium Complexity)**
- **Tools (8):** `check_inventory`, `apply_discount`, `create_shipment`, `escalate_to_human`, `check_customer_tier`, `calculate_shipping`, `validate_address`, `send_notification`
- **Decision space:** ~500 effective decision paths
- **Input:** Order events with customer data, item lists, and special instructions
- **Challenge:** Multi-constraint reasoning (discount limits, inventory across warehouses, tier-based policies)

**Task C --- DevOps Incident Agent (High Complexity)**
- **Tools (15):** Monitoring queries, deployment actions, rollback triggers, notification dispatch, runbook execution, capacity scaling, dependency checks, status page updates, incident creation, on-call paging, log analysis, metric correlation, change freeze enforcement, post-mortem creation, service mesh routing
- **Decision space:** ~2,000 effective decision paths
- **Input:** Alert payloads with service metadata, metric snapshots, and recent change logs
- **Challenge:** Multi-step triage logic, severity-dependent action sequences, safety-critical constraints (e.g., never auto-rollback in change freeze)

### 6.2 Baselines

For each benchmark task, we compare four approaches:

- **B1 --- Full LLM (API):** Claude Opus or GPT-4, accessed via API with the agent specification provided in the system prompt. Represents the accuracy ceiling and cost floor.
- **B2 --- Fine-tuned 7B:** LLaMA-3-8B fine-tuned on the same synthetic training data used for the ASMS model. Demonstrates that the training data is effective but the model is oversized.
- **B3 --- Distilled 1.5B:** A 1.5B parameter model distilled from the fine-tuned 7B, representing the standard compression approach.
- **B4 --- ASMS Micro-Model:** Our approach, with architecture selected per the complexity classification (2M / 15M / 80M parameters for Tasks A / B / C respectively).

### 6.3 Evaluation Metrics

| Metric | Definition | Target for ASMS |
|---|---|---|
| **Task Accuracy** | % of test inputs where model selects correct tool AND correct parameters | $\geq 95\%$ (matching B1) |
| **Latency (p50)** | Median inference time per decision | $< 5\text{ms}$ on CPU |
| **Latency (p99)** | 99th percentile inference time | $< 10\text{ms}$ on CPU |
| **Model Size** | Quantised model file size in MB | $< 50\text{MB}$ |
| **Training Cost** | GPU-hours to produce the model | $< 4\text{ A100-hours}$ |
| **Inference Cost** | USD per 1M decisions | $< \$1$ |
| **Robustness** | Accuracy on adversarial / out-of-distribution test set | $\geq 85\%$ |

### 6.4 Hypotheses

- **H1 (Accuracy Parity):** ASMS micro-models achieve $\geq 95\%$ task accuracy on in-distribution test data, within 2 percentage points of B1 (full LLM).
- **H2 (Cost Reduction):** ASMS achieves $\geq 100\times$ lower inference cost than B1, and $\geq 10\times$ lower than B3 (distilled model).
- **H3 (Training Efficiency):** ASMS models can be fully trained in $< 4$ GPU-hours on a single A100, including synthetic data generation.
- **H4 (CPU Deployment):** ASMS models run entirely on commodity CPU hardware with $< 10\text{ms}$ p99 latency, requiring no GPU at inference time.
- **H5 (Size Scaling):** Model size scales sub-linearly with decision space cardinality, validating the theoretical capacity bounds.

### 6.5 Test Protocol

1. Generate synthetic training corpus using the parent LLM (B1) for each task.
2. Split corpus: 80% train, 10% validation, 10% in-distribution test.
3. Separately, use the parent LLM to generate an adversarial / OOD test set (10% of corpus size).
4. Train all learnable models (B2, B3, B4) on the training split.
5. Evaluate all models (B1--B4) on both the in-distribution and adversarial test sets.
6. Measure latency on standardised hardware: AWS c5.xlarge (CPU-only) for B4, AWS g5.xlarge (GPU) for B2/B3.
7. Report all metrics with 95% confidence intervals over 5 random seeds.

---

## 7. Discussion

### 7.1 Limitations

**Domain drift.** If an agent's tool APIs change (new tools added, parameters modified, constraints updated), the ASMS micro-model must be re-synthesised. Unlike a general LLM that can adapt to new tool descriptions at inference time via prompt engineering, a micro-model's knowledge is baked into its weights. However, re-synthesis is fast and cheap (minutes to hours, dollars not thousands), making it comparable to recompiling a program after a code change.

**Out-of-distribution inputs.** Micro-models trained on a bounded decision space lack the graceful degradation of general LLMs when encountering truly novel inputs. We recommend a **fallback architecture**: the micro-model includes a calibrated confidence score, and inputs below a threshold are routed to the parent LLM. In practice, if the role specification is well-crafted, OOD inputs should be rare.

**Multi-step reasoning.** The current ASMS formulation targets single-turn tool selection: given an input, select one tool with one parameter set. Multi-step agent workflows (chains of tool calls with intermediate reasoning) increase the effective decision space exponentially and may require larger models or hierarchical decomposition. We scope this as future work.

**Evaluation gap.** This paper proposes a methodology and experimental design but does not report empirical results. The theoretical arguments and cost models are grounded in established machine learning principles, but the specific accuracy and efficiency claims require experimental validation. We invite the community to implement and evaluate the ASMS pipeline.

### 7.2 Broader Implications

**Democratisation of AI agents.** If agents can run on CPU, the barrier to deploying intelligent automation drops dramatically. Startups, non-profits, and organisations in developing economies can deploy AI agents without GPU infrastructure or expensive API subscriptions. A $50/month cloud VM can serve millions of agent decisions per day.

**Edge and offline deployment.** ASMS micro-models enable AI agent intelligence at the edge: IoT devices, mobile applications, on-premises servers in air-gapped environments. An industrial monitoring agent can run directly on the factory floor without cloud connectivity.

**Data privacy and sovereignty.** When agent inference happens entirely on-device, no data leaves the deployment boundary. There are no API calls to external services, no data logging by third-party providers. This is particularly relevant for healthcare, finance, and government applications with strict data residency requirements.

**Environmental sustainability.** The energy cost of AI is a growing concern. If millions of agent invocations per day can be served by CPU instances consuming ~50W instead of GPU instances consuming ~300W, with 10x higher throughput, the energy reduction is on the order of 60x per decision. At industry scale, this represents a meaningful contribution to sustainable AI.

**The "model as artifact" paradigm.** ASMS suggests a shift in how we think about AI models. Rather than monolithic, general-purpose systems that serve as shared infrastructure, models become disposable artifacts --- compiled for a specific purpose, deployed for a specific duration, and replaced when requirements change. This mirrors the evolution from mainframe computing (shared, general-purpose) to microservices (purpose-built, independently deployable).

### 7.3 Future Work

1. **Empirical validation:** Implement the ASMS pipeline and conduct the proposed experiments.
2. **Multi-step agents:** Extend the framework to handle sequential tool-calling workflows, potentially using hierarchical model architectures or state-machine-augmented micro-models.
3. **Automated architecture search:** Replace the heuristic complexity-to-architecture mapping with neural architecture search optimised for the decision complexity / accuracy / latency trade-off.
4. **Continuous re-synthesis:** Develop an incremental compilation approach where schema changes trigger targeted re-training rather than full re-synthesis.
5. **Conversational and workflow agents:** Generalise beyond tool-use to agents requiring natural language understanding (bounded-domain dialogue) and workflow orchestration (state machine execution).
6. **Open-source reference implementation:** Release an end-to-end ASMS toolkit enabling developers to compile their own micro-agents.

---

## 8. Conclusion

The dominant paradigm of deploying AI agents via general-purpose large language models is an artefact of convenience, not necessity. We have demonstrated, through formal analysis, that the decision spaces of tool-use agents are bounded by magnitudes that render billion-parameter models grotesquely over-provisioned.

Agent-Specific Model Synthesis offers an alternative: treat large LLMs as compilers, not runtimes. Given a structured specification of an agent's role, tools, and constraints, a parent LLM can generate the training material needed to build a compact model from scratch --- one whose architecture and capacity are matched to the actual complexity of the agent's task.

The implications are profound. If agent intelligence can be compiled into 1--50MB models that run on commodity CPUs in under 10 milliseconds, then the economics of AI agent deployment shift by orders of magnitude. Edge deployment becomes trivial. Data never needs to leave the device. Training an agent costs dollars, not thousands. And the environmental footprint of the global AI agent infrastructure shrinks proportionally.

The era of "one model to rule them all" has been a necessary phase in AI's development, establishing the capabilities that make ASMS possible. But for production agent deployments, it is time to compile your intelligence. Compile it small. Deploy it everywhere.

---

## References

Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). GPT3.int8(): 8-bit Matrix Multiplication for Transformers at Scale. *NeurIPS 2022*.

Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2023). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. *ICLR 2023*.

Gunasekar, S., Zhang, Y., Anber, J., et al. (2023). Textbooks Are All You Need. *arXiv:2306.11644*.

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NIPS 2014 Deep Learning Workshop*.

Jiao, X., Yin, Y., Shang, L., et al. (2020). TinyBERT: Distilling BERT for Natural Language Understanding. *EMNLP 2020*.

Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *ICML 2023*.

Liu, H., Simonyan, K., & Yang, Y. (2019). DARTS: Differentiable Architecture Search. *ICLR 2019*.

Patil, S. G., Zhang, T., Wang, X., & Gonzalez, J. E. (2023). Gorilla: Large Language Model Connected with Massive APIs. *arXiv:2305.15334*.

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *NeurIPS 2019 Workshop*.

Schick, T., Dwivedi-Yu, J., Dessi, R., et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *NeurIPS 2023*.

Sun, Z., Yu, H., Song, X., et al. (2020). MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices. *ACL 2020*.

Taori, R., Gulrajani, I., Zhang, T., et al. (2023). Stanford Alpaca: An Instruction-following LLaMA model. *GitHub repository*.

Wang, Y., Kordi, Y., Mishra, S., et al. (2023). Self-Instruct: Aligning Language Models with Self-Generated Instructions. *ACL 2023*.

Xu, C., Sun, Q., Zheng, K., et al. (2023). WizardLM: Empowering Large Language Models to Follow Complex Instructions. *arXiv:2304.12244*.

Zoph, B., & Le, Q. V. (2017). Neural Architecture Search with Reinforcement Learning. *ICLR 2017*.

---

*Draft v0.1 --- March 2026. For review and iteration.*
