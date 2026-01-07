
import os
import sys
import argparse
import glob
import cv2
import time
import shutil
from tqdm import tqdm

# 프로젝트 루트 경로 추가 (필요시)
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.auto_labeler import AutoLabeler
from src.label_converter import LabelConverter
from src.visualizer import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(description="SAM-HQ Auto-Labeling Pipeline")
    
    # 필수 인자
    parser.add_argument('--video_path', type=str, required=True, help="Path to input video file")
    parser.add_argument('--output_dir', type=str, required=True, help="Root directory for output dataset")
    
    # 옵션 인자
    parser.add_argument('--interval', type=int, default=30, help="Frame sampling interval (default: 30)")
    parser.add_argument('--visualize', action='store_true', help="Enable comprehensive visualization (slower)")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use (cuda/cpu)")
    parser.add_argument('--conf_threshold', type=float, default=0.5, help="Confidence threshold for person detection")
    
    # 덮어쓰기 옵션
    parser.add_argument('--overwrite', action='store_true', help="Overwrite output directory if exists")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 입력 확인
    if not os.path.exists(args.video_path):
        print(f"[ERROR] Input video not found: {args.video_path}")
        return
        
    print(f"[INFO] Input Video: {args.video_path}")
    print(f"[INFO] Output Dir : {args.output_dir}")
    print(f"[INFO] Interval   : {args.interval}")
    print(f"[INFO] Device     : {args.device}")
    
    # 2. 출력 디렉토리 구조 생성
    # dataset/
    #   ├── images/
    #   ├── labels/
    #   └── vis/ (optional)
    
    if os.path.exists(args.output_dir):
        if args.overwrite:
            print(f"[INFO] Cleaning up output directory...")
            shutil.rmtree(args.output_dir)
        else:
            print(f"[WARNING] Output directory already exists. New files will be added/overwritten.")
            
    images_dir = os.path.join(args.output_dir, 'images')
    labels_dir = os.path.join(args.output_dir, 'labels')
    vis_dir = os.path.join(args.output_dir, 'vis')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(vis_dir, exist_ok=True)
        
    # 3. 모듈 초기화
    print("\n[INFO] Initializing pipelines...")
    try:
        # Config는 기본값 사용 (src/auto_labeler.py 내부에서 로드)
        labeler = AutoLabeler() 
        converter = LabelConverter()
        visualizer = Visualizer(labeler.mask_mapper.label_names) if args.visualize else None
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        return

    # 4. 비디오 처리 루프
    cap = cv2.VideoCapture(args.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[INFO] Video loaded: {total_frames} frames, {fps:.2f} FPS")
    print("\n[INFO] Starting Auto-Labeling Process...")
    
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    pbar = tqdm(total=total_frames)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 지정된 간격마다 처리
        if frame_count % args.interval == 0:
            frame_name = f"frame_{frame_count:06d}"
            
            # (1) 이미지 저장
            img_save_path = os.path.join(images_dir, f"{frame_name}.jpg")
            cv2.imwrite(img_save_path, frame)
            
            # (2) Auto-Labeling 실행
            # process_image는 이미지 파일 경로를 받음
            result = labeler.process_image(img_save_path, verbose=False)
            
            labeled_masks = result.get('labeled_masks', [])
            person_info = result.get('person_info', {})
            
            # (3) YOLO Label 저장
            label_save_path = os.path.join(labels_dir, f"{frame_name}.txt")
            height, width = frame.shape[:2]
            converter.save_yolo_label(labeled_masks, label_save_path, width, height)
            
            # (4) 시각화 (옵션)
            if args.visualize:
                # 시각화용 Raw Mask 생성 (전체 이미지)
                # AutoLabeler 내부에서 생성된 마스크를 가져오면 좋겠지만, 
                # 종합 시각화 패널 3번(Raw SAM)을 위해 별도 생성 (비용 발생)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_masks = labeler.sam_segmenter.generate_masks(img_rgb)
                
                vis_save_path = os.path.join(vis_dir, f"{frame_name}_vis.png")
                visualizer.visualize_comprehensive_result(
                    img_rgb,
                    person_info,
                    all_masks,
                    labeled_masks,
                    label_save_path,
                    vis_save_path
                )
            
            processed_count += 1
            
        frame_count += 1
        pbar.update(1)
        
    pbar.close()
    cap.release()
    
    elapsed_time = time.time() - start_time
    print(f"\n[SUCCESS] Processing Complete!")
    print(f"Total Processed: {processed_count} frames")
    print(f"Total Time: {elapsed_time:.2f}s")
    print(f"Avg Time per Frame: {elapsed_time/processed_count if processed_count > 0 else 0:.2f}s")
    print(f"Output Directory: {args.output_dir}")

if __name__ == "__main__":
    main()
