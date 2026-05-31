# Instruct-to-Act: Decoupling Planning and Control for Instructable Agents

Reproduction of "Decoupling Planning and Control for Instructable Agents" (COLM 2026 under review),
using the [official DreamerV3 repository](https://github.com/danijar/dreamerv3) as the controller backbone.

---

## Setup

```bash
cd instruct_to_act/
bash setup.sh        # clones dreamerv3/, installs all Python deps
source ~/.bashrc     # activate PYTHONPATH

export OPENAI_API_KEY=sk-...   # required for gpt4o planner/annotator
```

Optional heavy environments:

```bash
pip install minerl>=0.4.4        # Minecraft Diamond (MC-Dia)
pip install dm-control>=1.0.0    # DMLab
```

---

## Training a controller

```bash
python train.py \
  --config configs/base.yaml configs/minecraft.yaml \
  --logdir /data/terran/instruct_to_act/seed0 \
  --seed 0
```

All config keys are in [configs/base.yaml](configs/base.yaml).
Per-environment overrides live in `configs/<env>.yaml`.
Checkpoints are saved as `logdir/<env>/seed<s>/checkpoint.pkl`.
Annotation logs are written to `logdir/<env>/seed<s>/annotations/`.

Supported environments: `crafter`, `minecraft`, `atari`, `dmlab`.
(Multi-agent env stubs exist in `envs/wrappers.py` but are not used for these experiments.)

---

## Evaluation

```bash
python evaluate.py \
  --config configs/minecraft.yaml configs/base.yaml \
  --checkpoint logdir/minecraft/seed0/checkpoint.pkl \
  --planner gpt4o \
  --mode online \      # online (Algorithm 1) | offline (Algorithm 2)
  --episodes 100 \
  --outdir results/
```

Additional environment example (Minecraft Diamond):

```bash
python evaluate.py \
  --config configs/minecraft.yaml configs/base.yaml \
  --checkpoint logdir/minecraft/seed0/checkpoint.pkl \
  --planner gpt4o --mode online --episodes 100
```

---

## Reviewer-response experiments

Ordered fastest → slowest. All scripts are in `experiments/`.

---

### 1. Loss curves during training
**Reviewer qnYd Q3** — *"How well does each loss go down? Was training dominated by any subset?"*

**Prereqs:** A completed (or in-progress) training run producing `metrics.jsonl`.
**GPU:** none. **VLM:** none. **Runtime:** seconds.

```bash
python experiments/exp_loss_curves.py \
  --logdirs logdir/minecraft/seed0 logdir/minecraft/seed1 logdir/minecraft/seed2 \
  --env minecraft \
  --smooth 100 \
  --outdir results/loss_curves

# Live monitor during training:
python experiments/exp_loss_curves.py \
  --logdirs logdir/minecraft/seed0 \
  --env minecraft --watch
```

Outputs: `results/loss_curves/loss_curves_minecraft.{pdf,png}` — per-component
loss curves (L_model, L_value, L_actor, L_BC, L_stop, KL), dominant-loss pie
chart, task score curve, and a text dominance report.

---

### 2. Instruction taxonomy
**Reviewer 2B11 Q6** — *"Are instructions mostly subgoals, action descriptions, or retrospective summaries?"*

**Prereqs:** Annotation logs from training (`annotations/annotations_<env>.jsonl`).
If logs are missing, the script generates synthetic demo data.
**GPU:** CPU-only (MiniLM + sklearn k-means). **VLM:** optional GPT-4o for cluster labelling (degrades gracefully to heuristic). **Runtime:** 1–5 min.

```bash
python experiments/exp_instruction_taxonomy.py \
  --annotation_logs \
    logdir/minecraft/seed0/annotations/annotations_minecraft.jsonl \
    logdir/minecraft/seed1/annotations/annotations_minecraft.jsonl \
  --env minecraft \
  --n_clusters 8 \
  --outdir results/instruction_taxonomy
```

Outputs: UMAP/PCA scatter coloured by taxonomy class, frequency/reward/follow-accuracy
breakdown, LaTeX table, qualitative exemplars per cluster, correlation between
instruction type and episode reward.

---

### 3. Annotator sensitivity (eval only, pre-trained checkpoints)
**Reviewer 2B11 Q4** — *"How sensitive are results to the VLM used for post-hoc annotation?"*

**Prereqs:** One pre-trained checkpoint per annotator condition. If conditions
`template`, `cluster`, `random` are already trained they require no VLM at
eval time.
**GPU:** eval only (controller forward passes). **VLM:** gpt4o planner at eval
(not annotation). **Runtime:** ~1 h per env per condition with 100 episodes.

```bash
# Evaluate pre-trained checkpoints (skip --train):
python experiments/exp_annotator_sensitivity.py \
  --env minecraft \
  --conditions gpt4o template cluster random \
  --eval_planner gpt4o \
  --episodes 100 \
  --from_checkpoints logdir/annotator_sweep/ \
  --outdir results/annotator_sensitivity

# To train from scratch first (expensive):
python experiments/exp_annotator_sensitivity.py \
  --env minecraft \
  --conditions gpt4o template cluster random \
  --train \
  --seeds 3 \
  --outdir results/annotator_sensitivity
```

Outputs: bar chart, LaTeX Table 4 extension, JSON results.

---

### 4. Instruction robustness
**Reviewer 2B11 Q7** — *"Does the controller exploit instruction artifacts? How does it behave under paraphrases, contradictory, underspecified, or adversarial instructions?"*

**Prereqs:** One checkpoint. **GPU:** controller inference. **VLM:** GPT-4o for
`paraphrase`/`contradictory` perturbations and for the judge; `underspecified`
and `adversarial` use local word lists (no API).
**Runtime:** ~1–2 h for 50 episodes × 5 conditions.

```bash
python experiments/exp_instruction_robustness.py \
  --env minecraft \
  --checkpoint logdir/minecraft/seed0/checkpoint.pkl \
  --planner gpt4o \
  --conditions original paraphrase contradictory underspecified adversarial \
  --episodes 50 \
  --logdir results/instruction_robustness

# Cheapest run (no GPT-4o calls for perturbation):
python experiments/exp_instruction_robustness.py \
  --env crafter \
  --checkpoint logdir/crafter/seed0/checkpoint.pkl \
  --planner gpt4o \
  --conditions original underspecified adversarial \
  --episodes 30 \
  --logdir results/instruction_robustness
```

Outputs: reward + instruction-following accuracy table per condition, JSON results.

---

### 5. Instruction horizon and length sensitivity
**Reviewer EkK5 Q5 / 2B11 Q13** — *"How does performance change as the instruction horizon varies?"*

**Prereqs:** One checkpoint per env. **GPU:** controller inference. **VLM:** gpt4o planner.
**Runtime:** ~1 h per env for default episode count.

```bash
# Both sweeps (token length + cadence K):
python experiments/exp_instruction_horizon.py \
  --envs crafter dmlab \
  --checkpoint_root logdir/ \
  --planner gpt4o \
  --episodes 20 \
  --sweep both \
  --logdir results/instruction_horizon

# Length sweep only:
python experiments/exp_instruction_horizon.py \
  --envs crafter \
  --checkpoint_root logdir/ \
  --sweep length \
  --episodes 15
```

Sweep A — token budget: `tiny` (4–6), `short` (6–12), `medium` (12–18), `long` (18–30), `free`.
Sweep B — fixed cadence: K=8, 16, 32, 64, `adaptive` (p_stop head).

Outputs: grouped bar charts per sweep, length-vs-reward line plot, LaTeX tables.

---

### 6. Async latency and instruction staleness
**Reviewer EkK5 Q4 / Q2(c)** — *"Latency distributions and instruction-staleness statistics. Token breakdown between advance and emit."*

**Prereqs:** One checkpoint per env. **GPU:** controller inference. **VLM:** gpt4o.
Runs short episodes (max_steps / 10) for speed.
**Runtime:** ~30 min per env.

```bash
python experiments/exp_async_latency.py \
  --envs crafter minecraft \
  --checkpoint_root logdir/ \
  --planner gpt4o \
  --episodes 20 \
  --logdir results/async_latency

# Offline (mock VLM, no API calls):
python experiments/exp_async_latency.py \
  --envs crafter \
  --checkpoint_root logdir/ \
  --no_actual_vlm \
  --episodes 20
```

Outputs: staleness histograms per env, emit vs. advance latency boxplots (log scale),
summary table with p95 staleness and throughput.

---

### 7. Online vs. offline planning across all tasks
**Reviewer 2B11 Q10** — *"A more direct online versus offline ablation across all tasks would help."*

**Prereqs:** One checkpoint per env. **GPU:** controller inference. **VLM:** gpt4o.
**Runtime:** ~1–2 h per env (2 modes × 20 episodes).

```bash
python experiments/exp_online_vs_offline.py \
  --envs crafter minecraft atari dmlab \
  --checkpoint_root logdir/ \
  --planner gpt4o \
  --episodes 20 \
  --logdir results/online_vs_offline
```

Outputs: grouped bar charts (reward, throughput), violin plots per env, summary
table with speedup column and staleness (online only), LaTeX table.

---

### 8. Planner-controller ablations
**Reviewer 2B11** — *"Ablations separating communication, planner reasoning, parameter sharing, centralized vs. decentralized coordination."*

**Prereqs:** Minecraft Diamond checkpoint (`logdir/minecraft/seed0/checkpoint.pkl`).
Runs single-agent (n_agents=1); comm conditions degenerate gracefully and serve
as a sanity-check baseline.
**GPU:** controller inference. **VLM:** gpt4o. **Runtime:** ~2 h.

```bash
python experiments/exp_multiagent_ablation.py \
  --envs minecraft \
  --checkpoint_root logdir/ \
  --planners gpt4o \
  --conditions full no_comm centralized separate_ctrl controller_only \
  --n_agents 1 \
  --episodes 50 \
  --logdir results/multiagent_ablation
```

Conditions:
- `full` — decentralized async planning + communication (paper default)
- `no_comm` — planners act independently (chatroom disabled)
- `centralized` — hub-and-spoke communication
- `separate_ctrl` — separate controller parameters per agent
- `controller_only` — null planners (empty instructions)

Outputs: grouped bar chart, JSON results.

---

### 9. Multi-seed CI for main results
**Reviewer 2B11 Q3** — *"Report standard deviations or confidence intervals. How many seeds were used?"*

**Prereqs:** N pre-trained checkpoints per env (or run with `--train` to train them).
**GPU:** eval only if checkpoints exist. **VLM:** gpt4o.
**Runtime:** ~30 min per seed × env (eval); days if training from scratch.

```bash
# Eval only (assuming checkpoints at logdir/<env>/seed{0..4}/):
python experiments/exp_multiseed_ci.py \
  --envs crafter minecraft atari dmlab \
  --planner gpt4o \
  --seeds 5 \
  --episodes 100 \
  --from_checkpoints logdir/ \
  --logdir results/multiseed_ci

# Train + eval (expensive):
python experiments/exp_multiseed_ci.py \
  --envs crafter \
  --seeds 5 \
  --train \
  --gpus 0 1 2 3 \
  --logdir results/multiseed_ci
```

Outputs: bootstrap 95% CI table matching Table 1 format, LaTeX output.

---

### 10. Cross-environment generalisation
**Reviewer 2B11 Q8** — *"Can the same controller generalise across environments, or is one controller trained per environment?"*

**Prereqs:** Per-env checkpoints + a joint checkpoint (or train one with `configs/joint.yaml`).
**GPU:** controller inference. **VLM:** gpt4o.
**Runtime:** ~1–2 h per (train_env, eval_env) pair.

```bash
# Eval only (pre-trained checkpoints):
python experiments/exp_cross_env_generalization.py \
  --train_envs crafter minecraft atari dmlab \
  --eval_envs  crafter minecraft atari dmlab \
  --checkpoint_root logdir/ \
  --joint_ckpt logdir/joint/seed0/checkpoint.pkl \
  --planner gpt4o \
  --episodes 50 \
  --skip_train \
  --logdir results/cross_env

# Train joint controller first:
python train.py \
  --config configs/joint.yaml configs/base.yaml \
  --logdir logdir/joint/seed0 \
  --seed 0
```

Outputs: transfer heatmap (rows=trained on, cols=evaluated on), per-vs-joint
bar chart, full LaTeX table of all pairs.

---

## Known gaps and paper vs. codebase discrepancies

The following issues were found by cross-checking the code against the paper.
They do not affect the analysis scripts that read pre-existing results, but
matter for training and inference correctness.

### Critical

**1. BC and stop losses are logged but not optimized** (`agent/lang_agent.py:126–140`)

`LangCondAgent.train()` computes `_bc_loss` and `_stop_loss` and logs them,
but DreamerV3's ninjax training step is already JIT-compiled by `super().train()`.
The additional losses are computed *after* the optimizer has stepped and do not
contribute to parameter updates. The paper's combined objective
`L = L_model + λ_V L_value + λ_A L_actor + L_BC + L_stop` is therefore only
partially implemented — `L_BC` and `L_stop` are reported but have zero gradient.

**Fix needed:** integrate `_bc_loss` and `_stop_loss` into dreamerv3's ninjax
`_train_step` before the optimizer call, or use separate optimizer steps inside
a `nj.pure` context.

**2. `LangCondAgent._lang_size` is never set** (`agent/lang_agent.py:151`)

`policy()` references `self._lang_size` but `__init__` only stores `lang_size`
as a local variable. This causes `AttributeError` at inference time.

**Fix:** add `self._lang_size = lang_size` to `__init__` after the `super().__init__()` call.

**3. `LangReplayBuffer.dataset()` uses approximate random indexing** (`training/replay_lang.py:165–191`)

Language embeddings are injected at random buffer positions per batch, not at
the positions that correspond to the actual sampled episodes. The BC and stop
losses would train on mismatched (observation, instruction) pairs.

**Fix:** expose episode indices from `embodied.replay.Replay` (e.g. via a
custom replay class that records write pointers per episode) and use them to
look up the correct `lang_embed`/`annotated`/`complete` values.

**4. `wm.rssm.get_feat_size()` does not exist in DreamerV3** (`agent/lang_agent.py:117`)

DreamerV3's RSSM exposes `get_feat(state)` but not `get_feat_size()`. Calling
`StopHead.__init__` will fail with `AttributeError`.

**Fix:** compute the feature size from config: `feat_size = config.rssm_deter + config.rssm_stoch * config.rssm_classes`.

### Moderate

**5. `no_comm` and `full` conditions are identical in `exp_multiagent_ablation.py`** (experiments line 74–80)

Both create `MultiAgentInference` with `mode='decentralized'` and no chatroom
change. The `no_comm` condition should pass `comm=False` or use a `NoChatroom`
stub that discards all messages.

**6. `separate_ctrl` condition is identical to `shared_ctrl`** (experiments line 93–99)

Both load the same single checkpoint. `separate_ctrl` should load one checkpoint
per agent. A simple fix is to load `n_agents` independent `LangCondAgent`
instances from separate checkpoint files.

**7. Open-source VLM planners use OpenAI message format** (`planners/vlm_planner.py:307–322`)

`QwenPlanner`, `GemmaPlanner`, and `LlavaPlanner` pass OpenAI-style
`{"type": "image_url", ...}` content dicts to `apply_chat_template`, which
expects a different format for HuggingFace vision models. This causes a
`ValueError` when running with local models.

**8. `annotator.py` always instantiates an OpenAI client** for Qwen/Gemma/LLaVA

`_openai_client()` is called regardless of `annotator_type`. For open-source
annotators the annotator needs a local HuggingFace `pipeline` instead.

**9. `exp_annotator_sensitivity.py` passes dreamerv3-style `key=value` to `train.py`** (line 88)

```python
cmd = ['python', 'train.py', ..., f'annotator_type={condition}', ...]
```

`train.py` uses `argparse`, which does not accept positional `key=value` overrides.
This subprocess call will raise an `unrecognized arguments` error.

**Fix:** either add `--annotator_type` as an explicit argparse argument to
`train.py`, or pass it as an additional YAML snippet: `--config ... <(echo "annotator_type: {condition}")`.

**10. `exp_async_latency.py` advance token counts are synthetic** (line 135)

```python
self.advance_tokens.append(np.random.randint(20, 80))  # from paper Fig 2d
```

These are random integers, not actual token counts from the VLM. The plotted
emit-vs-advance token breakdown is therefore not meaningful when using a real VLM.

### Minor

**11. `base.yaml` model size label is misleading**

The comment at the top of `base.yaml` reads "800M default, Table 2" but
`rssm_deter: 4096` corresponds to the 50M model (the inline comments confirm
this). The 800M scale is only active in `configs/minecraft.yaml`.

**12. Annotation log paths do not match between annotator and taxonomy script**

`PostHocAnnotator` writes to `<logdir>/annotations/annotations_<env>.jsonl`
(i.e. `logdir/minecraft/seed0/annotations/annotations_minecraft.jsonl`), but
`exp_instruction_taxonomy.py`'s default is
`logdir/minecraft/seed0/annotations.jsonl` (one level up).
Pass the correct path explicitly with `--annotation_logs`.

**13. `instr_scale` config key is unused**

`base.yaml` defines `instr_scale: 1.0` but no code reads it.

**14. Multi-agent environments are stubs (not used for rebuttal)**

`_make_overcooked`, `_make_pico_park`, and `_make_mindcraft` in `envs/wrappers.py`
raise `NotImplementedError`. All rebuttal experiment scripts now default to
Minecraft Diamond (`--envs minecraft --n_agents 1`), so this does not block
any of the scripts in this repository.
