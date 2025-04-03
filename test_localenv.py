import os
import torch

# Test LOCAL_TEST environment variable


def run_test():
    # Check if we're in a testing environment
    local_debug = os.environ.get("LOCAL_TEST", "0") == "1"

    if local_debug:
        print("Using synthetic data for local testing")
        token_buffer = torch.randint(0, 100, (100,))
    else:
        print("Would load real dataset here")
        token_buffer = torch.zeros(10)

    print(f"token_buffer size: {len(token_buffer)}")


if __name__ == "__main__":
    run_test()
