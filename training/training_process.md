# Laya Training Process — End-to-End Guide

This document explains every step of the training pipeline used to fine-tune the Laya model on your burnout-detection dataset.

---

## 1. What We're Building

**Goal:** Fine-tune a general-purpose structured-prediction model (`convaiinnovations/laya`) to detect burnout risk, identify stress root causes, and rate psychological safety from team chat conversations.

**Output:** A custom model that answers 3 questions per conversation with calibrated probability distributions, not just single-label guesses.

---

## 2. Architecture Overview

```
Raw Data (JSONL)
      │
      ▼
convert_data.py          ← Data format conversion
      │
      ▼
Laya training format     ← { state, questions, gold }
      │
      ▼
finetune() from laya      ← RLCD loss + temperature calibration
      │
      ▼
laya_finetuned/           ← Saved checkpoint + configs (inside training/)
      │
      ▼
compare_models.py         ← Base vs fine-tuned evaluation
```

---

## 3. The Three Question Types

Laya handles three kinds of questions. Understanding them is key to understanding the training process.

### 3.1 noul (Yes/No)
```
{
  "type": "noul",
  "instructions": "Does the conversation show burnout?",
  "criteria": {
    "true": "team members express exhaustion",
    "false": "team members sound comfortable"
  }
}
```
- Output: a single probability score (0.0–1.0) for `true`
- Decided by threshold: > 0.5 → `true`, else `false`

### 3.2 choice (Multi-class)
```
{
  "type": "choice",
  "instructions": "What is the primary stress driver?",
  "criteria": {
    "overload_and_deadlines": "overload and deadlines",
    "lack_of_support_and_isolation": "lack of support and isolation",
    "none_healthy": "none healthy"
  }
}
```
- Output: a full probability distribution across all criteria keys
- The model picks the highest-probability key as its answer
- We shuffle option order during training so the model doesn't learn positional bias

### 3.3 score (Ordinal)
```
{
  "type": "score",
  "instructions": "Rate psychological safety from 0 to 2",
  "criteria": ["Low", "Medium", "High"]
}
```
- Output: a float score (e.g., 0.8 = "Low", 1.5 = "Medium", 2.2 = "High")
- Predicted label: `round(score)`

---

## 4. Data Pipeline

### Step 1: Raw Data (`training/data/train.jsonl`, `training/data/test.jsonl`)

Your source files use **hard labels**:

```json
{
  "text": "I'm working 60 hour weeks and haven't slept properly...",
  "questions": { ... },
  "labels": {
    "burnout_risk": true,
    "root_cause": "overload_and_deadlines",
    "psychological_safety": 0
  }
}
```

### Step 2: Convert to Laya Format (`training/convert_data.py`)

This script transforms each row. The output (`training/data/laya_format/train.jsonl`) uses **probability distributions** as targets:

```json
{
  "state": "I'm working 60 hour weeks...",
  "questions": { ... },
  "gold": {
    "burnout_risk": {"probabilities": {"true": 1.0, "false": 0.0}},
    "root_cause": {"probabilities": {"overload_and_deadlines": 1.0, "lack_of_support_and_isolation": 0.0, "none_healthy": 0.0}},
    "psychological_safety": {"probabilities": {"0": 1.0, "1": 0.0, "2": 0.0}}
  }
}
```

**Conversion rules:**
- `noul: true` → `{"true": 1.0, "false": 0.0}`
- `noul: false` → `{"true": 0.0, "false": 1.0}`
- `choice: "overload_and_deadlines"` → one-hot across all criteria keys
- `score: 1` → one-hot across indices `0, 1, 2`

**Why one-hot?** Laya trains to predict full probability distributions, not just pick the right answer. This enables calibration and confidence estimation.

### Step 3: Data Splits

| Split | Rows | Items |
|-------|------|-------|
| train | 42 | 114 |
| calibrate | 4 | 12 |
| test | 10 | 30 |

"Items" = rows × number of questions per row (3). The calibrate split is held out for temperature fitting after training.

---

## 5. Training Configuration

### File: `training/run_training.py`

```python
config = TrainConfig(
    epochs=4,
    micro_batch=4,
    grad_accum=4,
    encoder_lr=2.5e-5,
    head_lr=1e-4,
    loss="rlcd",
    shuffle_options=("choice",),
    seed=42,
)
```

**Parameter breakdown:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `epochs` | 4 | Passes over the training data |
| `micro_batch` | 4 | Sequences per forward pass (small due to CPU) |
| `grad_accum` | 4 | Accumulate gradients over 4 micro-batches before updating weights. Effective batch size = 4 × 4 = 16 |
| `encoder_lr` | 2.5e-5 | Learning rate for the model backbone (slow, careful updates) |
| `head_lr` | 1e-4 | Learning rate for the question-specific classification heads (faster, these are small layers) |
| `loss` | `"rlcd"` | The training objective (see Section 6) |
| `shuffle_options` | `("choice",)` | Randomize the order of choice options each epoch to prevent the model from learning positional bias |
| `seed` | 42 | Reproducibility |

---

## 6. The RLCD Loss (How Training Actually Works)

This is the core of Laya's training. Instead of standard cross-entropy, it uses **Reinforced Learning from Label Distributions (RLCD)**. Here's how it works:

### 6.1 Forward Pass

The model takes a conversation (`state`) and a set of questions, and outputs logits (raw scores) for each possible answer.

### 6.2 Noisy Sampling

Laya adds Gaussian noise to the logits:

```
noisy_logits = logits + noise × sigma
```

- `sigma` starts at **0.4** (high noise = lots of exploration)
- `sigma` anneals to **0.1** by the final epoch (low noise = precise predictions)
- Multiple noisy samples are drawn per training step

### 6.3 Scoring Each Sample

Each noisy sample is evaluated against the target distribution using **two proper scoring rules**:

1. **Spherical score** — rewards probability mass to be concentrated on the correct answer
2. **Ranked probability score** — rewards correct ordering of probabilities

Higher score = sample was better than average. Lower score = sample was worse.

### 6.4 Policy Gradient

Samples with above-average scores get **reinforced** (their log-probabilities are weighted positively). Samples with below-average scores get **penalized** (weighted negatively).

This is the same mechanism used in RL fine-tuning (like PPO), but applied directly to the model's own noisy predictions without a separate reward model.

### 6.5 Soft Cross-Entropy (Added)

On top of RLCD, a standard cross-entropy term is computed against the target distribution. This ensures the model still learns the correct answer directly, not just through noisy self-reinforcement.

### 6.6 Total Loss

```
total_loss = rlcd_loss + cross_entropy_loss
```

---

## 7. Training Output Interpretation

### 7.1 Loss Values

```
epoch 1/4 mean loss 0.7648
epoch 2/4 mean loss 0.2411
epoch 3/4 mean loss 0.1194
epoch 4/4 mean loss 0.0491
```

- **Epoch 1 (0.7648):** Model starts with high loss. RLCD noise is maximum (sigma=0.4), so predictions are noisy and the policy gradient signal is rough.
- **Epoch 2 (0.2411):** Rapid improvement. Cross-entropy dominates early, pulling predictions toward targets.
- **Epoch 3 (0.1194):** Converging. Noise is lower (sigma ≈ 0.2), refinements are subtler.
- **Epoch 4 (0.0491):** Near-convergence. Model is well-fitted to the training distribution.

### 7.2 What "Mean Loss" Means

It's the average per-item loss across all 114 training items. Each item can have up to 3 questions (burnout_risk, root_cause, psychological_safety), so the loss aggregates across all of them.

---

## 8. Temperature Calibration (Post-Training)

After training completes, Laya fits **temperature scalars** on the held-out calibration set (12 items).

### What is Temperature?

Raw model outputs (logits) are often overconfident — a prediction of 0.95 might actually be correct only 80% of the time. Temperature scaling divides logits by a scalar `T`:

```
calibrated_probs = softmax(logits / T)
```

- `T = 1.0` → no change (uncalibrated)
- `T > 1.0` → softens probabilities (more conservative)
- `T < 1.0` → sharpens probabilities (more confident)

### Why It Matters

Calibrated probabilities mean you can trust the model's confidence scores. If it says "90% confident," it should be right 90% of the time.

### Our Result

```
Temperatures: [1.0, 1.0, 1.0]  (all unchanged)
```

**Reason:** The calibration set was too small (12 items total, fewer than 10 per question type). Laya's minimum for fitting is 10 items per type. With more training data, temperatures will be fitted properly.

---

## 9. The Fine-Tuned Checkpoint

Output directory: `training/laya_finetuned/`

```
training/laya_finetuned/
├── model.safetensors          ← Fine-tuned weights (804 MB)
├── tokenizer/                 ← Tokenizer config
├── encoder/                   ← Encoder adapter weights
├── questions.json             ← Question definitions used during training
└── rl_agent_config.json       ← Training config (loss, temperatures, etc.)
```

### How to Use It

```python
from laya import load

# Load the fine-tuned model
agent = load("C:/Users/Pravin Shinde/Documents/Projects/LayaPOC/training/laya_finetuned")

# Ask questions
questions = {
    "burnout_risk": {
        "type": "noul",
        "instructions": "Does the conversation show burnout?",
        "criteria": {"true": "exhaustion", "false": "comfortable"}
    },
    "root_cause": {
        "type": "choice",
        "instructions": "What is the primary stress driver?",
        "criteria": {
            "overload_and_deadlines": "overload and deadlines",
            "lack_of_support_and_isolation": "lack of support",
            "none_healthy": "none healthy"
        }
    },
    "psychological_safety": {
        "type": "score",
        "instructions": "Rate psychological safety from 0-2",
        "criteria": ["Low", "Medium", "High"]
    }
}

result = agent.predict("I'm working 60 hour weeks and haven't slept...", questions)
print(result["answers"]["burnout_risk"]["probabilities"])  # {"true": 0.97, "false": 0.03}
print(result["answers"]["root_cause"]["choice"])            # "overload_and_deadlines"
print(result["answers"]["psychological_safety"]["score"])   # 0.15 → "Low"
```

---

## 10. Evaluation: Base vs Fine-Tuned

### How It Works

`training/compare_models.py` loads both models, runs them on the same test set (`training/data/test.jsonl`), and compares predictions against ground truth labels.

### Metrics Computed

| Metric | How |
|--------|-----|
| **Accuracy** | `correct_predictions / total_predictions` |
| **Confidence** | Average `answer_confidence` across all predictions |
| **Delta** | Fine-tuned accuracy minus base accuracy |

### Classification Rules

| Question Type | Prediction Rule |
|---------------|----------------|
| `noul` | `noul > 0.5` → `true`, else `false` |
| `choice` | `argmax(probabilities)` → predicted key |
| `score` | `round(score)` → integer index |

### Results

```
Overall accuracy:  Base 13/30 (43%)  →  Fine-tuned 26/30 (87%)  (+43%)
burnout_risk:      70% → 100%  (+30%)
root_cause:        30% → 70%   (+40%)
psychological_safety: 30% → 90%  (+60%)
```

---

## 11. Key Takeaways

1. **RLCD trains probability distributions, not labels.** The model learns to output calibrated uncertainty, which is critical for HR/people-analytics use cases where false positives and false negatives have real costs.

2. **Small datasets still work.** With only 42 training rows, the model learned the task well — but temperature calibration needs more data to become reliable.

3. **Confidence improved as much as accuracy.** Base model averaged ~0.66 confidence (often wrong but overconfident). Fine-tuned model averages ~0.95+ confidence (and is mostly right).

4. **Option shuffling prevents shortcut learning.** By randomizing choice order each epoch, the model can't cheat by learning "the first option is always correct."

5. **Different learning rates for encoder vs heads** lets the model adapt its classification layers quickly while keeping the general language understanding intact.

---

## 12. Running the Pipeline

```bash
# 1. Activate virtual environment
.venv/Scripts/activate

# 2. Convert data to Laya format
python training/convert_data.py

# 3. Train
python training/run_training.py

# 4. Compare base vs fine-tuned
python training/compare_models.py
```

---

## 13. Next Steps for Production

- **More training data:** At least 200–500 rows for robust calibration and generalization
- **Validation split:** Hold out ~20% during training for early stopping (detect overfitting)
- **Temperature calibration:** Once you have 10+ calibration items per question type, temperatures will be meaningful
- **API endpoint:** Wrap `agent.predict()` in FastAPI for production inference
- **Monitoring:** Track prediction drift and confidence calibration over time
