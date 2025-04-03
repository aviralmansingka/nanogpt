import torch

# A very simple script to test the debug flag feature
print("Starting test_debug.py")

DEBUG_STEPS = 1


def train(debug=False):
    steps = DEBUG_STEPS if debug else 10
    print(f"Running with {'DEBUG' if debug else 'NORMAL'} mode: {steps} steps")

    for i in range(steps):
        print(f"Step {i}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    train(debug=args.debug)
