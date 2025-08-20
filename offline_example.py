#!/usr/bin/env python3
"""
Offline Mode Example
===================

This example demonstrates kandc running in offline mode - no internet required!
All data is saved locally without any cloud synchronization.

Requirements:
    pip install torch kandc

Usage:
    python offline_example.py
"""

import os
import time
import random
import torch
import torch.nn as nn
import kandc


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
    print("🔌 kandc Offline Mode Example")
    print("=" * 50)
    print("This example works without internet connection!")
    print("All data is saved locally in the ./kandc directory.")
    print("")
    
    # Initialize in offline mode
    run = kandc.init(
        project="offline-demo",
        name="local-experiment",
        mode="offline",  # 🔑 This is the key difference!
    )

    print(f"✅ Initialized offline run: {run.name}")
    print(f"📁 Data will be saved to: {run.dir}")
    print("")

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

    # Run multiple timed operations
    for i in range(5):
        print(f"🔄 Running timed operation {i + 1}/5")
        random_wait()
        
        # Log metrics (saved locally)
        kandc.log({
            "step": i,
            "loss": loss.item() * random.random(),
            "accuracy": random.random(),
            "batch_size": 32
        })

    # In offline mode, we only have local run ID
    print(f"✅ Local Run ID: {run.id}")
    print("📁 All data saved locally - no cloud sync")

    kandc.finish()

    print("")
    print("🎉 Offline demo completed!")
    print("📁 Check the local directory for saved data:")
    print(f"   - Traces: {run.dir}/traces/")
    print(f"   - Artifacts: {run.dir}/artifacts/")
    print(f"   - Metrics: {run.dir}/metrics.jsonl")
    print(f"   - Config: {run.dir}/config.json")
    print("")
    print("💡 This all worked without internet! Perfect for:")
    print("   - Development and debugging")
    print("   - Air-gapped environments")
    print("   - CI/CD pipelines")
    print("   - When you just want local profiling")


if __name__ == "__main__":
    main()
