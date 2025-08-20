#!/usr/bin/env python3
"""
No Code Sync Example
===================

This example demonstrates kandc with code synchronization disabled.
Experiment tracking and profiling work normally, but no source code is captured.

Requirements:
    pip install torch kandc

Usage:
    python no_code_sync_example.py
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
    print("📸🚫 kandc No Code Sync Example")
    print("=" * 50)
    print("This example shows full tracking WITHOUT code capture!")
    print("Perfect when you have large codebases or sensitive code.")
    print("")
    
    # Initialize with code capture disabled
    run = kandc.init(
        project="no-code-sync-demo",
        name="tracking-without-code",
        capture_code=False,  # 🔑 This disables code snapshot capture!
        tags=["no-code-sync", "privacy", "large-codebase"],
    )

    print("✅ kandc initialized with code sync disabled")
    print("📊 Experiment tracking and profiling still work normally")
    print("🚫 No source code will be captured or uploaded")
    print("")

    # Create and run model - profiling still works!
    model = SimpleNet()
    data = torch.randn(32, 784)
    print("📊 Running model forward pass (with profiling)...")
    output = model(data)
    loss = output.mean()

    @kandc.timed(name="random_wait")  # Timing still works
    def random_wait():
        print("⏳ Starting random wait...")
        time.sleep(random.random() * 2)
        print("✅ Random wait complete")

    # Run multiple timed operations
    for i in range(5):
        print(f"🔄 Running timed operation {i + 1}/5")
        random_wait()
        
        # Log metrics - this still works normally
        kandc.log({
            "step": i,
            "loss": loss.item() * random.random(),
            "accuracy": random.random(),
            "batch_size": 32,
            "model_params": sum(p.numel() for p in model.parameters())
        })

    # Check run status
    try:
        backend_run_id = (
            getattr(run, "_run_data", {}).get("id") if getattr(run, "_run_data", None) else None
        )
    except Exception:
        backend_run_id = None
    
    if backend_run_id:
        print(f"✅ Backend Run ID: {backend_run_id}")
        print("📊 Metrics and traces uploaded (but no code)")
    else:
        print(f"✅ Local Run ID: {run.id}")
        print("📁 Data saved locally (but no code)")

    kandc.finish()

    print("")
    print("🎉 No code sync demo completed!")
    print("")
    print("✅ What was captured:")
    print("   📊 Experiment metrics and logs")
    print("   🔍 Model execution traces")
    print("   ⏱️  Function timing data")
    print("   ⚙️  Configuration and metadata")
    print("")
    print("🚫 What was NOT captured:")
    print("   📄 Source code files")
    print("   📁 Project structure")
    print("   🔧 Configuration files")
    print("")
    print("💡 Perfect for:")
    print("   - Large codebases (>100MB)")
    print("   - Proprietary/sensitive code")
    print("   - When you only need metrics/profiling")
    print("   - Compliance requirements")


if __name__ == "__main__":
    main()
