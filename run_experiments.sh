#!/bin/bash
# Load WANDB_API_KEY from .env if not already set in the environment
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

if [ -z "$WANDB_API_KEY" ]; then
  echo "ERROR: WANDB_API_KEY is not set. Add it to a .env file or export it manually."
  exit 1
fi

echo "Activating Virtual Environment..."
source .venv/bin/activate

echo "Starting Experiment Pipeline..."
python experiments.py --all

echo "Done!"
