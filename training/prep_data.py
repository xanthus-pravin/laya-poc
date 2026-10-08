import json
import random
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Load your generated dataset
with open(os.path.join(SCRIPT_DIR, 'dataset.jsonl'), 'r') as f:
    data = [json.loads(line) for line in f]

# Shuffle to ensure random distribution
random.shuffle(data)

# Calculate splits (70% Train, 15% Calibrate, 15% Test)
total = len(data)
train_idx = int(total * 0.70)
calib_idx = int(total * 0.85)

splits = {
    'train.jsonl': data[:train_idx],
    'calibrate.jsonl': data[train_idx:calib_idx],
    'test.jsonl': data[calib_idx:]
}

# Write to data/ directory
for filename, dataset in splits.items():
    out_path = os.path.join(DATA_DIR, filename)
    with open(out_path, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
    print(f"  {filename}: {len(dataset)} records -> {out_path}")