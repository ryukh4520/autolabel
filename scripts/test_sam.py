#!/usr/bin/env python3
"""
Test script to verify SAM2 installation and basic functionality
"""

import torch
import sys

def test_sam2():
    print("=" * 60)
    print("SAM2 Installation Test")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Try importing SAM2
    print("\n2. Importing SAM2...")
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        print("   ✓ SAM2 imported successfully")
    except ImportError as e:
        print(f"   ✗ Failed to import SAM2: {e}")
        return False
    
    # Check model checkpoint
    print("\n3. Checking model checkpoint...")
    import os
    checkpoint_path = "/workspace/models/sam2_hiera_base_plus.pt"
    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / 1024**2
        print(f"   ✓ Model checkpoint found: {checkpoint_path}")
        print(f"   Size: {size_mb:.2f} MB")
    else:
        print(f"   ✗ Model checkpoint not found: {checkpoint_path}")
        return False
    
    # Try loading the model
    print("\n4. Loading SAM2 model...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Build SAM2 model
        sam2 = build_sam2(
            config_file="sam2_hiera_b+.yaml",
            ckpt_path=checkpoint_path,
            device=device
        )
        
        # Create mask generator
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
        )
        
        print(f"   ✓ SAM2 model loaded successfully on {device}")
        
        # Check VRAM usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"   VRAM Allocated: {allocated:.2f} GB")
            print(f"   VRAM Reserved: {reserved:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Failed to load SAM2 model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sam2()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ SAM2 Test PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("✗ SAM2 Test FAILED")
        print("=" * 60)
        sys.exit(1)
