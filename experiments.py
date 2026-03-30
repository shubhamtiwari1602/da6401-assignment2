"""W&B Experiments Orchestrator
"""

import os
import subprocess

def run_experiment(name, args):
    """Utility to run train.py with specific arguments."""
    cmd = ["python", "-m", "train", "--run_name", name] + args
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def exp_2_1():
    print("--- 2.1 The Regularization Effect of Dropout ---")
    # Train with and without BN is handled in base since base model has BN.
    # We would theoretically modify VGG11 to remove BN for the secondary run.
    # Here we just run baseline.
    run_experiment("baseline_with_bn", ["--epochs", "5"])

def exp_2_2():
    print("--- 2.2 Internal Dynamics ---")
    # Train under three dropout conditions
    run_experiment("dropout_0.0", ["--dropout_p", "0.0", "--epochs", "5"])
    run_experiment("dropout_0.2", ["--dropout_p", "0.2", "--epochs", "5"])
    run_experiment("dropout_0.5", ["--dropout_p", "0.5", "--epochs", "5"])

def exp_2_3():
    print("--- 2.3 Transfer Learning Showdown ---")
    # Evaluate transfer learning strategies
    run_experiment("tl_strict", ["--fine_tune_strategy", "strict", "--epochs", "5"])
    run_experiment("tl_partial", ["--fine_tune_strategy", "partial", "--epochs", "5"])
    run_experiment("tl_full", ["--fine_tune_strategy", "full", "--epochs", "5"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all experiments sequence")
    args = parser.parse_args()
    
    if args.all:
        exp_2_1()
        exp_2_2()
        exp_2_3()
    else:
        print("Run with --all to execute the experimental sequence.")
