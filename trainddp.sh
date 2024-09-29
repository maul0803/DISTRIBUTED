#!/bin/bash
which python
MODEL="bert-base-cased"
JSON_FILE="dev-v2.0.json"
EPOCHS=10
STRATEGY="ddp"
python train.py --model "$MODEL" --json "$JSON_FILE" --strategy "$STRATEGY" --epochs "$EPOCHS" --devices 2 --nodes 2 --use_tensor_cores
