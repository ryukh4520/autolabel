#!/usr/bin/env python3
"""
SAM2 Auto Mask Generation Test on Video
동영상에서 300프레임마다 자동 세그멘테이션을 수행하고 결과를 시각화
"""

import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def show_anns(anns, ax):
    """
    SAM2 마스크를 시각화하는 함수
    
    Args:
        anns: SAM2가 생성한 마스크 리스트
        ax: matplotlib axis
    """
    if len(anns) == 0:
        return
    
    # 면적 기준으로 정렬
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    # 전체 마스크 이미지 생성
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], 
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    
    for ann in sorted_anns:
        m = ann['segmentation']
        # 랜덤 색상 생성
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
    
    ax.imshow(img)


def process_video(video_path, output_dir, frame_interval=300):
    """
    동영상을 읽어서 지정된 간격마다 SAM2 세그멘테이션 수행
    
    Args:
        video_path: 입력 동영상 경로
        output_dir: 결과 저장 디렉토리
        frame_interval: 프레임 간격 (기본값: 300)
    """
    print("=" * 80)
    print("SAM2 Video Auto Segmentation Test")
    print("=" * 80)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # GPU 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Device: {device}")
    
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # SAM2 모델 로드
    print("\n[INFO] Loading SAM2 model...")
    checkpoint_path = "/workspace/models/sam2_hiera_base_plus.pt"
    
    try:
        sam2 = build_sam2(
            config_file="sam2_hiera_b+.yaml",
            ckpt_path=checkpoint_path,
            device=device
        )
        
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
        )
        
        print("[INFO] ✓ SAM2 model loaded successfully")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"[INFO] VRAM Allocated: {allocated:.2f} GB")
        
    except Exception as e:
        print(f"[ERROR] Failed to load SAM2 model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 동영상 열기
    print(f"\n[INFO] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return
    
    # 동영상 정보
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] Video Info:")
    print(f"       - Total Frames: {total_frames}")
    print(f"       - FPS: {fps:.2f}")
    print(f"       - Resolution: {width}x{height}")
    print(f"       - Frame Interval: {frame_interval}")
    
    # 처리할 프레임 번호 계산
    frames_to_process = list(range(0, total_frames, frame_interval))
    print(f"[INFO] Frames to process: {len(frames_to_process)}")
    
    # 프레임 처리
    print("\n[INFO] Processing frames...")
    
    for frame_idx in tqdm(frames_to_process, desc="Processing"):
        # 프레임 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"[WARNING] Failed to read frame {frame_idx}")
            continue
        
        # BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # SAM2 세그멘테이션 수행
        try:
            masks = mask_generator.generate(image_rgb)
            
            # VRAM 사용량 체크
            if torch.cuda.is_available() and frame_idx == frames_to_process[0]:
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"\n[INFO] VRAM during inference:")
                print(f"       - Allocated: {allocated:.2f} GB")
                print(f"       - Reserved: {reserved:.2f} GB")
            
            # 결과 시각화
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            # 원본 이미지
            axes[0].imshow(image_rgb)
            axes[0].set_title(f'Original Frame {frame_idx}', fontsize=16)
            axes[0].axis('off')
            
            # 세그멘테이션 결과
            axes[1].imshow(image_rgb)
            show_anns(masks, axes[1])
            axes[1].set_title(f'SAM2 Segmentation ({len(masks)} masks)', fontsize=16)
            axes[1].axis('off')
            
            # 저장
            output_path = os.path.join(output_dir, f'frame_{frame_idx:06d}_seg.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[INFO] Frame {frame_idx}: {len(masks)} masks detected -> {output_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 정리
    cap.release()
    
    print("\n" + "=" * 80)
    print(f"[INFO] ✓ Processing complete!")
    print(f"[INFO] Results saved to: {output_dir}")
    print("=" * 80)


def main():
    """메인 함수"""
    # 경로 설정
    source_dir = "/workspace/test_codes/source"
    result_dir = "/workspace/test_codes/result"
    
    # source 디렉토리에서 동영상 파일 찾기
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    video_files = []
    
    if os.path.exists(source_dir):
        for file in os.listdir(source_dir):
            if any(file.endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(source_dir, file))
    
    if not video_files:
        print(f"[ERROR] No video files found in {source_dir}")
        print(f"[INFO] Please place video files in {source_dir}")
        print(f"[INFO] Supported formats: {', '.join(video_extensions)}")
        sys.exit(1)
    
    # 첫 번째 동영상 처리
    video_path = video_files[0]
    print(f"[INFO] Found video: {video_path}")
    
    # 처리 실행
    process_video(
        video_path=video_path,
        output_dir=result_dir,
        frame_interval=300
    )


if __name__ == "__main__":
    main()
