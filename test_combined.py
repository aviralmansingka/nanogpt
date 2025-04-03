import os
import argparse
import torch

# Test combining debug flag with LOCAL_TEST environment variable

DEBUG_STEPS = 2
MAX_STEPS = 10


def run_with_flags(debug=False):
    # Get steps based on debug flag
    steps = DEBUG_STEPS if debug else MAX_STEPS
    print(f"Running with {'DEBUG' if debug else 'NORMAL'} mode: {steps} steps")

    # Check for LOCAL_TEST
    local_test = os.environ.get("LOCAL_TEST", "0") == "1"
    print(f"Using {'SYNTHETIC' if local_test else 'REAL'} data")

    # Run simulation
    for i in range(steps):
        print(f"Step {i}")

    print("Completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    run_with_flags(debug=args.debug)
