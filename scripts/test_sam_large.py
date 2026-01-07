#!/usr/bin/env python3
"""
SAM2 Large 모델 구동 가능 여부 테스트
"""

import torch
import sys
import gc

def test_sam2_large():
    print("=" * 80)
    print("SAM2 Large Model Load Test")
    print("=" * 80)
    
    # GPU 정보
    print(f"\n[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 초기 VRAM 상태
    torch.cuda.empty_cache()
    gc.collect()
    initial_allocated = torch.cuda.memory_allocated(0) / 1024**3
    initial_reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"\n[INFO] Initial VRAM:")
    print(f"       - Allocated: {initial_allocated:.2f} GB")
    print(f"       - Reserved: {initial_reserved:.2f} GB")
    
    # SAM2 import
    print("\n[INFO] Importing SAM2...")
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        print("       ✓ SAM2 imported")
    except ImportError as e:
        print(f"       ✗ Failed: {e}")
        return False
    
    # Large 모델 로드 시도
    print("\n[INFO] Loading SAM2 Large model...")
    checkpoint_path = "/workspace/models/sam2_hiera_large.pt"
    
    try:
        # 모델 빌드
        sam2_large = build_sam2(
            config_file="sam2_hiera_l.yaml",
            ckpt_path=checkpoint_path,
            device="cuda"
        )
        
        print("       ✓ Model loaded successfully")
        
        # 로드 후 VRAM
        after_load_allocated = torch.cuda.memory_allocated(0) / 1024**3
        after_load_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\n[INFO] After model load:")
        print(f"       - Allocated: {after_load_allocated:.2f} GB")
        print(f"       - Reserved: {after_load_reserved:.2f} GB")
        print(f"       - Delta Allocated: +{after_load_allocated - initial_allocated:.2f} GB")
        
        # Mask Generator 생성
        print("\n[INFO] Creating mask generator...")
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_large,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
        )
        
        print("       ✓ Mask generator created")
        
        # Generator 생성 후 VRAM
        after_gen_allocated = torch.cuda.memory_allocated(0) / 1024**3
        after_gen_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\n[INFO] After generator creation:")
        print(f"       - Allocated: {after_gen_allocated:.2f} GB")
        print(f"       - Reserved: {after_gen_reserved:.2f} GB")
        
        # 더미 이미지로 추론 테스트
        print("\n[INFO] Testing inference with dummy image...")
        import numpy as np
        dummy_image = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
        
        try:
            masks = mask_generator.generate(dummy_image)
            print(f"       ✓ Inference successful ({len(masks)} masks)")
            
            # 추론 후 VRAM
            after_infer_allocated = torch.cuda.memory_allocated(0) / 1024**3
            after_infer_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"\n[INFO] After inference:")
            print(f"       - Allocated: {after_infer_allocated:.2f} GB")
            print(f"       - Reserved: {after_infer_reserved:.2f} GB")
            print(f"       - Peak Allocated: +{after_infer_allocated - initial_allocated:.2f} GB")
            
            # 여유 공간 계산
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            remaining = total_vram - after_infer_reserved
            print(f"\n[INFO] Remaining VRAM: {remaining:.2f} GB")
            
            if remaining < 0.5:
                print("       ⚠️  WARNING: Very low remaining VRAM!")
                print("       May cause OOM errors with larger images")
            elif remaining < 1.0:
                print("       ⚠️  Caution: Limited remaining VRAM")
            else:
                print("       ✓ Sufficient remaining VRAM")
            
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"       ✗ OOM Error during inference!")
                print(f"       Error: {e}")
                return False
            else:
                raise
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"       ✗ OOM Error during model load!")
            print(f"       Error: {e}")
            
            # 현재 VRAM 상태
            current_allocated = torch.cuda.memory_allocated(0) / 1024**3
            current_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"\n[INFO] VRAM at failure:")
            print(f"       - Allocated: {current_allocated:.2f} GB")
            print(f"       - Reserved: {current_reserved:.2f} GB")
            return False
        else:
            raise
    
    except Exception as e:
        print(f"       ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    success = test_sam2_large()
    
    print("\n" + "=" * 80)
    if success:
        print("✓ SAM2 Large Model Test PASSED")
        print("  → Large model can be used on RTX 3070 8GB")
        print("=" * 80)
        sys.exit(0)
    else:
        print("✗ SAM2 Large Model Test FAILED")
        print("  → Recommend using Base Plus model instead")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
