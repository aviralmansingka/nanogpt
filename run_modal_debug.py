#!/usr/bin/env python3
# Script to run train_gpt2.py in debug mode on Modal to save credits

from train_gpt2 import run_model_multi

# Run with debug mode explicitly enabled
if __name__ == "__main__":
    print("Running train_gpt2.py in DEBUG mode on Modal")
    print("This will only run a few steps to save credits")
    run_model_multi.remote(debug=True)
