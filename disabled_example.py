#!/usr/bin/env python3
"""
Disabled Mode Example
====================

This example demonstrates kandc running in disabled mode - zero overhead!
All kandc calls become no-ops, perfect for production deployment.

Requirements:
    pip install torch kandc

Usage:
    python disabled_example.py
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
    print("🚫 kandc Disabled Mode Example")
    print("=" * 50)
    print("This example shows zero-overhead mode for production!")
    print("All kandc calls will be no-ops with minimal performance impact.")
    print("")
    
    # Initialize in disabled mode
    run = kandc.init(
        project="disabled-demo",
        name="production-experiment",
        mode="disabled",  # 🔑 This disables all kandc functionality!
        tags=["disabled", "production", "no-overhead"],
    )

    print("✅ kandc initialized in disabled mode")
    print("🚫 No data will be captured, saved, or synced")
    print("")

    # Create and run model - profiling decorators become no-ops
    model = SimpleNet()
    data = torch.randn(32, 784)
    print("📊 Running model forward pass...")
    output = model(data)
    loss = output.mean()

    @kandc.timed(name="random_wait")  # This decorator does nothing in disabled mode
    def random_wait():
        print("⏳ Starting random wait...")
        time.sleep(random.random() * 0.5)  # Shorter wait for demo
        print("✅ Random wait complete")

    # Run multiple operations - timing is disabled
    for i in range(5):
        print(f"🔄 Running operation {i + 1}/5")
        random_wait()
        
        # Log metrics - these calls do nothing in disabled mode
        kandc.log({
            "step": i,
            "loss": loss.item() * random.random(),
            "accuracy": random.random(),
            "batch_size": 32
        })

    # In disabled mode, run object exists but most functionality is disabled
    if run:
        print(f"✅ Run object exists: {run.name}")
        print("🚫 But no data capture, profiling, or sync occurred")
    else:
        print("🚫 No run object created (fully disabled)")

    kandc.finish()  # This also does nothing in disabled mode

    print("")
    print("🎉 Disabled mode demo completed!")
    print("")
    print("💡 In disabled mode:")
    print("   ✅ Zero performance overhead")
    print("   ✅ No file I/O operations")
    print("   ✅ No network calls")
    print("   ✅ No memory usage for tracking")
    print("   ✅ Perfect for production deployment")
    print("")
    print("🔧 To enable tracking, change mode to 'online' or 'offline'")


if __name__ == "__main__":
    main()
