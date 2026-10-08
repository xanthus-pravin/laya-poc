import json

for fname in ['training/data/train.jsonl', 'training/data/calibrate.jsonl', 'training/data/test.jsonl']:
    try:
        lines = open(fname, 'r', encoding='utf-8').readlines()
    except FileNotFoundError:
        continue
    for i, line in enumerate(lines, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"{fname}:{i}: {e.msg} at char {e.pos}")
            print(f"  Line ({len(line)} chars): {repr(line[:200])}")
            # Show context around the error position
            start = max(0, e.pos - 40)
            end = min(len(line), e.pos + 40)
            print(f"  Context: ...{repr(line[start:end])}...")
