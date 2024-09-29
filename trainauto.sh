#!/bin/bash
which python
MODEL="bert-base-cased"
JSON_FILE="dev-v2.0.json"
EPOCHS=10
STRATEGY="auto"

python train.py --model "$MODEL" --json "$JSON_FILE" --strategy "$STRATEGY" --epochs "$EPOCHS" --devices 1 --nodes 1
