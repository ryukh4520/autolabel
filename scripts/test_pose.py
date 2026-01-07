#!/usr/bin/env python3
"""
Test script to verify YOLO-Pose installation and basic functionality
"""

import torch
import sys

def test_yolo_pose():
    print("=" * 60)
    print("YOLO-Pose Installation Test")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    
    # Try importing Ultralytics
    print("\n2. Importing Ultralytics YOLO...")
    try:
        from ultralytics import YOLO
        print("   ✓ Ultralytics imported successfully")
    except ImportError as e:
        print(f"   ✗ Failed to import Ultralytics: {e}")
        return False
    
    # Check model checkpoint
    print("\n3. Checking model checkpoint...")
    import os
    checkpoint_path = "/workspace/models/yolov8m-pose.pt"
    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / 1024**2
        print(f"   ✓ Model checkpoint found: {checkpoint_path}")
        print(f"   Size: {size_mb:.2f} MB")
    else:
        print(f"   ✗ Model checkpoint not found: {checkpoint_path}")
        return False
    
    # Try loading the model
    print("\n4. Loading YOLO-Pose model...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load YOLO-Pose model
        model = YOLO(checkpoint_path)
        model.to(device)
        
        print(f"   ✓ YOLO-Pose model loaded successfully on {device}")
        
        # Print model info
        print(f"   Task: {model.task}")
        print(f"   Model type: {type(model.model).__name__}")
        
        # Check VRAM usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"   VRAM Allocated: {allocated:.2f} GB")
            print(f"   VRAM Reserved: {reserved:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Failed to load YOLO-Pose model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_yolo_pose()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ YOLO-Pose Test PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("✗ YOLO-Pose Test FAILED")
        print("=" * 60)
        sys.exit(1)
