#!/usr/bin/env python3
"""
Ignore Code Example
==================

This example demonstrates kandc with custom code exclusion patterns.
Experiment tracking and profiling work normally, but specific files/directories
are excluded from code capture (e.g., model files, data, logs).

Requirements:
    pip install torch kandc

Usage:
    python ignore_code_example.py
"""

import os
import time
import random
import torch
import torch.nn as nn
import kandc


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
    print("📸🔍 kandc Selective Code Capture Example")
    print("=" * 50)
    print("This example shows how to exclude specific files from code capture!")
    print("Perfect for ignoring model files, data, logs, etc.")
    print("")
    
    # Initialize with custom code exclusion patterns
    run = kandc.init(
        project="ignore-code-demo",
        name="selective-code-capture",
        capture_code=True,  # Code capture is enabled
        code_exclude_patterns=[  # 🔑 But these patterns are excluded!
            # Model files
            "*.pth", "*.safetensors", "*.bin", "*.ckpt",
            
            # Data directories and files (Git-style: use / to match directory contents)
            "data/", "datasets/", "cache/",
            "*.csv", "*.parquet", "*.h5", "*.hdf5",
            
            # Experiment outputs (Git-style: use / to match directory contents)
            "outputs/", "results/", "logs/", "runs/",
            "experiments/", "checkpoints/",
            
            # Temporary and log files
            "*.log", "*.tmp", "*.temp", "temp_*",
            "*.out", "*.err",
            
            # Large media files
            "*.mp4", "*.avi", "*.mov", "*.png", "*.jpg", "*.jpeg",
            
            # IDE and system files (Git-style: use / to match directory contents)
            ".vscode/", ".idea/", "*.swp", ".DS_Store",
            
            # Custom exclusions for this project
            "secret_config.json", "private_keys/",
        ],
        tags=["ignore-code", "selective-capture", "custom-exclusions"],
    )

    print("✅ kandc initialized with selective code capture")
    print("📊 Experiment tracking and profiling work normally")
    print("🔍 Only specific files/patterns are excluded from code capture")
    print("")
    print("🚫 Excluded patterns:")
    print("   - Model files (*.pth, *.safetensors, etc.)")
    print("   - Data directories (data/, datasets/, cache/)")
    print("   - Experiment outputs (outputs/, results/, logs/)")
    print("   - Temporary files (*.tmp, *.log, temp_*)")
    print("   - Large media files (*.mp4, *.png, etc.)")
    print("   - IDE files (.vscode/, .idea/, *.swp)")
    print("")
    
    # Check if we're actually authenticated and can capture code
    try:
        has_api_client = hasattr(run, '_api_client') and run._api_client is not None
        has_run_data = hasattr(run, '_run_data') and run._run_data is not None
        print(f"🔍 Code capture status:")
        print(f"   - API client available: {'✅' if has_api_client else '❌'}")
        print(f"   - Backend run created: {'✅' if has_run_data else '❌'}")
        print(f"   - Code capture enabled: ✅")
        print(f"   - Mode: {run.config.mode}")
        
        if has_api_client and has_run_data:
            print("   ➡️  Code capture should happen when this script runs")
        else:
            print("   ⚠️  Code capture may not happen (authentication/backend issues)")
        print("")
    except Exception as e:
        print(f"   ⚠️  Could not check code capture status: {e}")
        print("")

    # Create and run model - profiling still works!
    model = SimpleNet()
    data = torch.randn(32, 784)
    print("📊 Running model forward pass (with profiling)...")
    output = model(data)
    loss = output.mean()

    def random_wait():
        print("⏳ Starting random wait...")
        time.sleep(random.random() * 2)
        print("✅ Random wait complete")

    # Check for actual files that should be excluded
    print("📁 Checking for files that should be excluded:")
    excluded_files = [
        "model_checkpoint.pth",
        "model.safetensors",
        "data/training_data.csv", 
        "logs/training.log",
        "outputs/results.json",
        "experiment.log",
        "temp_results.tmp",
        "temp_processing_file",
        "secret_config.json",
        "private_keys/api_key.txt"
    ]
    
    import os
    for filename in excluded_files:
        exists = "✅ exists" if os.path.exists(filename) else "❌ missing"
        print(f"   🚫 {filename} ({exists}, should be excluded)")

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
            "model_params": sum(p.numel() for p in model.parameters()),
            "excluded_files_count": len(excluded_files)
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
        print("📊 Metrics, traces, and selective code uploaded")
    else:
        print(f"✅ Local Run ID: {run.id}")
        print("📁 Data and selective code saved locally")

    kandc.finish()

    print("")
    print("🎉 Selective code capture demo completed!")
    print("")
    print("✅ What was captured:")
    print("   📊 Experiment metrics and logs")
    print("   🔍 Model execution traces")
    print("   ⏱️  Function timing data")
    print("   ⚙️  Configuration and metadata")
    print("   📄 Source code files (.py, .js, .md, etc.)")
    print("   🔧 Config files (requirements.txt, etc.)")
    print("")
    print("🚫 What was excluded from code capture:")
    print("   🤖 Model checkpoint files (*.pth, *.safetensors)")
    print("   📁 Data directories and files")
    print("   📋 Log and temporary files")
    print("   🖼️  Large media files")
    print("   🔒 Sensitive configuration files")
    print("")
    print("💡 Perfect for:")
    print("   - Projects with large model files")
    print("   - Excluding sensitive configuration")
    print("   - Ignoring data directories")
    print("   - Selective code versioning")
    print("   - Compliance with file size limits")


if __name__ == "__main__":
    main()
