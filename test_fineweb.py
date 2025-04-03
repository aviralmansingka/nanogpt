import argparse
import os
import time
import torch

# Simple test script for the debug flag with synthetic data

DEBUG_STEPS = 1
MAX_STEPS = 5


class SyntheticDataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        # Create random token buffer
        self.token_buffer = torch.randint(0, 50257, (B * T * 20 + 1,))
        self.current_position = 0
        print(f"Created synthetic data loader with {len(self.token_buffer)} tokens")

    def next_batch(self):
        B, T = self.B, self.T

        # Reset position if needed
        if self.current_position + (B * T + 1) > len(self.token_buffer):
            self.current_position = 0

        # Get current batch
        buf = self.token_buffer[
            self.current_position : self.current_position + B * T + 1
        ]

        # Create inputs and targets
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        # Update position
        self.current_position += B * T

        return x, y


def train_model(debug=False):
    """Simple training loop to test debug flag"""
    # Determine steps based on debug flag
    steps = DEBUG_STEPS if debug else MAX_STEPS

    print(f"Running in {'DEBUG' if debug else 'NORMAL'} mode with {steps} steps")

    # Create dataloader
    loader = SyntheticDataLoader(B=4, T=128)

    # Simple training loop
    for step in range(steps):
        t0 = time.time()

        # Get batch
        x, y = loader.next_batch()

        # Simulate training (do nothing)
        time.sleep(0.1)  # Pretend we're doing work

        t1 = time.time()
        dt = (t1 - t0) * 1000

        print(f"Step {step} | dt: {dt:.2f}ms")

    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test debug flag with synthetic data")
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with fewer steps"
    )
    args = parser.parse_args()

    # Run training with debug flag
    train_model(debug=args.debug)
