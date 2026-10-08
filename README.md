# LayaPOC — Burnout Detection with Fine-Tuned Laya

Proof-of-concept for detecting employee burnout risk, stress root causes, and psychological safety from team chat conversations using the [Laya](https://github.com/convaiinnovations/laya) structured-prediction model.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Laya Installation](#laya-installation)
3. [Training the Laya Base Model](#training-the-laya-base-model)
4. [Comparing Base vs Fine-Tuned Model](#comparing-base-vs-fine-tuned-model)
5. [Guide to Re-Train an Existing Model](#guide-to-re-train-an-existing-model)

---

## Project Structure

```
LayaPOC/
├── .venv/                          # Python virtual environment (Laya installed here)
├── cache/                          # HuggingFace model cache
├── laya_repo/                      # Laya source code (git submodule/clone)
│   └── laya/                       # Laya Python package
├── out/
│   ├── predictions_laya.json       # Laya predictions on windows
│   └── predictions_keyword.json    # Keyword baseline predictions
├── src/
│   ├── serve.py                    # Flask REST API for Laya predictions
│   ├── run_laya.py                 # Score windows with Laya model
│   ├── run_keyword.py              # Score windows with keyword baseline
│   ├── questions.py                # Question definitions
│   ├── compare.py                  # Compare Laya vs keyword predictions
│   ├── build_windows.py            # Build sliding windows from raw conversations
│   ├── demo.py                     # Demo script
│   └── export_from_mongo.py        # Export data from MongoDB
├── training/
│   ├── dataset.jsonl               # Combined dataset (source of truth for training)
│   ├── data/
│   │   ├── train.jsonl             # Raw training data
│   │   ├── test.jsonl              # Held-out test data
│   │   ├── calibrate.jsonl         # Calibration split
│   │   ├── windows.json            # Sliding-window view of conversations
│   │   └── laya_format/            # Converted Laya training format (auto-generated)
│   │       ├── train.jsonl
│   │       ├── test.jsonl
│   │       └── calibrate.jsonl
│   ├── laya_finetuned/             # Fine-tuned model checkpoint (output of training)
│   │   ├── model.safetensors
│   │   ├── tokenizer/
│   │   ├── encoder/
│   │   ├── questions.json
│   │   └── rl_agent_config.json
│   ├── prep_data.py                # Split dataset.jsonl into train/calibrate/test
│   ├── convert_data.py             # Convert JSONL splits to Laya training format
│   ├── run_training.py             # Main training runner
│   ├── compare_models.py           # Compare base vs fine-tuned model
│   ├── training_process.md         # Detailed training documentation
│   └── (other helper scripts)
├── requirements.txt
├── label_tool.py
└── README.md
├── label_tool.py                   # Data labeling utility
└── README.md                       # This file
```

---

## Laya Installation

### Prerequisites

- Python 3.10+
- pip or uv
- 4 GB+ free disk space (model weights are ~800 MB)
- GPU recommended for training (CPU works but is slow)

### Setup

```bash
# 1. Clone the repository (if not already done)
git clone <your-repo-url>
cd LayaPOC

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Laya from the local repo
cd laya_repo/laya
pip install -e .
cd ../..

# 5. Verify installation
python -c "from laya import load; print('Laya installed successfully')"
```

### What Gets Installed

- `laya` — the core Python package for structured prediction
- `torch` — PyTorch backend
- `transformers` — HuggingFace transformers for the encoder
- `flask` — for the REST API server
- `accelerate` — for efficient model loading

---

## Training the Laya Base Model

Training fine-tunes the base Laya model (`convaiinnovations/laya`) on your burnout-detection dataset so it learns to answer your specific questions with calibrated probability distributions.

### Step 1: Prepare Your Data

Format your training data as JSONL (`training/data/train.jsonl`). Each line is a JSON object:

```json
{
  "text": "I'm working 60 hour weeks and haven't slept properly...",
  "questions": {
    "burnout_risk": {
      "type": "noul",
      "instructions": "Does the conversation show signs of burnout?",
      "criteria": {
        "true": "team members express exhaustion",
        "false": "team members sound comfortable"
      }
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
      "instructions": "Rate psychological safety from 0 to 2",
      "criteria": ["Low", "Medium", "High"]
    }
  },
  "labels": {
    "burnout_risk": true,
    "root_cause": "overload_and_deadlines",
    "psychological_safety": 0
  }
}
```

**Field definitions:**
- `text` — the conversation or observation to classify
- `questions` — the schema defining what to predict (types: `noul`, `choice`, `score`)
- `labels` — ground truth values for training

### Step 2: Convert Data to Laya Format

Laya trains on probability distributions, not hard labels. The converter transforms your labels:

```bash
.venv\Scripts\activate
python training/convert_data.py
```

This reads `training/data/train.jsonl`, `training/data/calibrate.jsonl`, and `training/data/test.jsonl` and writes converted versions to `training/data/laya_format/`.

Conversion rules:
- `burnout_risk: true` → `{"probabilities": {"true": 1.0, "false": 0.0}}`
- `root_cause: "overload_and_deadlines"` → one-hot across all criteria keys
- `psychological_safety: 1` → `{"probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}}`

### Step 3: Run Training

```bash
python training/run_training.py
```

This will:
1. Load the base model from `convaiinnovations/laya`
2. Read and tokenize your converted training data
3. Split off 10% for calibration (temperature fitting)
4. Train for 4 epochs using RLCD loss
5. Fit temperature scalars on the calibration set
6. Save the fine-tuned checkpoint to `training/laya_finetuned/`

**Expected output:**

```
epoch 1/4 mean loss 0.7648
epoch 2/4 mean loss 0.2411
epoch 3/4 mean loss 0.1194
epoch 4/4 mean loss 0.0491

=== Training Complete ===
  Train items: 114
  Calibration items: 12
  Epoch losses: [0.7648, 0.2411, 0.1194, 0.0491]
  Temperatures (choice, score, noul): [1.0, 1.0, 1.0]
```

Loss should decrease across epochs. Temperatures will be 1.0 if calibration set is too small (fewer than 10 items per question type).

### Step 4: Use the Fine-Tuned Model

```python
from laya import load

agent = load("C:/Users/Pravin Shinde/Documents/Projects/LayaPOC/training/laya_finetuned")

questions = {
    "burnout_risk": {
        "type": "noul",
        "instructions": "Does the conversation show signs of burnout?",
        "criteria": {
            "true": "team members express exhaustion",
            "false": "team members sound comfortable"
        }
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
        "instructions": "Rate psychological safety from 0 to 2",
        "criteria": ["Low", "Medium", "High"]
    }
}

result = agent.predict("I'm working 60 hour weeks and haven't slept properly...", questions)
print(result["answers"]["burnout_risk"]["noul"])        # 0.97 (probability of true)
print(result["answers"]["root_cause"]["choice"])         # "overload_and_deadlines"
print(result["answers"]["psychological_safety"]["score"]) # 0.15 → "Low"
```

---

## Comparing Base vs Fine-Tuned Model

The comparison script runs both models on the same held-out test set and reports accuracy and confidence side by side.

### Run Comparison

```bash
python training/compare_models.py
```

### Output

```
======================================================================
  COMPARISON SUMMARY
======================================================================
  Question                 Base Acc     FT Acc    Base Conf       FT Conf     Delta
  ---------------------- ---------- ---------- ------------ ---------- ----------
  burnout_risk                  70%       100%       0.6654      0.9774       +30%
  root_cause                    30%        70%       0.5747      0.9945       +40%
  psychological_safety          30%        90%       0.7061      0.9093       +60%

  Overall                       43%        87%                            +43%
  Base: 13/30  |  Fine-tuned: 26/30
```

**What the columns mean:**
- **Base Acc** — accuracy of the untrained base model
- **FT Acc** — accuracy of your fine-tuned model
- **Base Conf / FT Conf** — average confidence score across all predictions
- **Delta** — improvement from fine-tuning (positive = better)

**Classification rules:**
- `noul` (yes/no): `noul > 0.5` → `true`, else `false`
- `choice` (multi-class): `argmax(probabilities)` → predicted option
- `score` (ordinal): `round(score)` → integer index

---

## Guide to Re-Train an Existing Model

When you have more training data (e.g., 200 rows), re-training from your existing fine-tuned checkpoint is called **warm-start training**. It's faster and preserves what the model already learned.

### Step 1: Add New Training Data

Add your new rows to `training/dataset.jsonl`. Keep the same format (`text`, `questions`, `labels`). You can append to the existing file or replace it entirely.

```json
{"text": "New conversation example...", "questions": {...}, "labels": {...}}
{"text": "Another example...", "questions": {...}, "labels": {...}}
```

### Step 2: Re-Convert the Data

```bash
python training/convert_data.py
```

This re-generates `training/data/laya_format/train.jsonl` with the updated dataset.

### Step 3: Update Training Configuration

Edit `training/run_training.py` and make these changes:

```python
config = TrainConfig(
    epochs=8,           # Increase: model already knows the task, needs fewer passes
    micro_batch=8,      # Can increase now (was 4 for small CPU runs)
    grad_accum=4,
    encoder_lr=2.5e-5,  # Keep small to avoid catastrophic forgetting
    head_lr=1e-4,       # Keep the same
    loss="rlcd",
    shuffle_options=("choice",),
    seed=42,
)

# Point to your EXISTING fine-tuned checkpoint as the starting point
base_model = os.path.join(SCRIPT_DIR, "laya_finetuned")  # warm-start from existing checkpoint
```

**Key hyperparameter guidance for re-training:**

| Parameter | First Training | Re-training | Why |
|-----------|---------------|-------------|-----|
| `epochs` | 4 | 8 | More data = more passes needed; model starts smarter |
| `micro_batch` | 4 | 8 | Larger dataset = can handle bigger batches |
| `encoder_lr` | 2.5e-5 | 2.5e-5 | Keep small to preserve general language knowledge |
| `head_lr` | 1e-4 | 1e-4 | Same — these adapt quickly |
| `grad_accum` | 4 | 4 | Effective batch = micro_batch × grad_accum |

### Step 4: Re-Train

```bash
python training/run_training.py
```

The output goes to the same `training/laya_finetuned/` directory. Your existing checkpoint is overwritten with the new one.

**What's different this time:**
- Training starts from your fine-tuned weights, not the base model
- With 200 rows, you'll get ~20 calibration items (enough for temperature fitting)
- Temperatures should now be fitted properly (not stuck at 1.0)
- Training takes longer but converges faster since the model already knows the task

### Step 5: Evaluate

```bash
python training/compare_models.py
```

This uses the same held-out `training/data/test.jsonl`, so you get a clean before/after comparison on identical data.

### Tips for Scaling

- **200–500 rows:** Good for a functional model with basic calibration
- **500–2000 rows:** Reliable calibration, meaningful confidence scores
- **2000+ rows:** Consider adding a validation split for early stopping to prevent overfitting
- **Keep the test set fixed:** Don't modify `training/data/test.jsonl` between runs — it's your consistency anchor
- **Monitor loss curves:** If loss starts going up after epoch 3–4, you're overfitting — reduce epochs or add dropout
