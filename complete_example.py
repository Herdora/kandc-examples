import os
import time
import random
import torch
import torch.nn as nn

import kandc

import ignore_code_example


@kandc.capture_model_class(model_name="SimpleNet")
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    print("🚀 Simple kandc Example")
    
    # Initialize with configuration
    run = kandc.init(
        project="simple-demo",
        name="basic-example",
        tags=["simple-demo", "basic-example"],
        code_exclude_patterns=[
            "logs", "data/", "outputs/*"
        ]
        
    )

    # Create and run model
    model = SimpleNet()
    data = torch.randn(32, 784)
    print("📊 Running model forward pass...")
    output = model(data)
    loss = output.mean()

    @kandc.timed(name="random_wait")
    def random_wait():
        print("⏳ Starting random wait...")
        time.sleep(random.random() * 2)
        print("✅ Random wait complete")

    for i in range(5):
        random_wait()

    def another_random_wait(a, b, c=3):
        print(f"a: {a}, b: {b}, c: {c}")
        time.sleep(random.random() * 2)
        print("✅ Another random wait complete")
    
    for i in range(5):
        kandc.timed_call("another_random_wait", another_random_wait, 1, b=2, c=3)

    try:
        backend_run_id = (
            getattr(run, "_run_data", {}).get("id") if getattr(run, "_run_data", None) else None
        )
    except Exception:
        backend_run_id = None
    if backend_run_id:
        print(f"✅ Run ID: {backend_run_id}")
        print("✅ API client initialized")
    else:
        # We'll always have a local run id, even when offline
        print(f"✅ Local Run ID: {run.id}")
        print("⚠️ Backend run not created; operating offline")

    kandc.finish()

    print("✅ Done! Check your dashboard for results.")


if __name__ == "__main__":
    main()