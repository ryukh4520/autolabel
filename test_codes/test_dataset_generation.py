
import sys
import os
import cv2
import numpy as np
import argparse
import glob

# 프로젝트 루트 경로 추가 (모듈 import를 위함)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.auto_labeler import AutoLabeler
from src.label_converter import LabelConverter

def parse_args():
    parser = argparse.ArgumentParser(description="Test Dataset Generation Pipeline")
    parser.add_argument('--video_path', type=str, help="Path to input video")
    parser.add_argument('--output_dir', type=str, default='test_yolo_output', help="Directory to save test results")
    return parser.parse_args()

def process_video_generation_test():
    args = parse_args()
    
    # 1. 비디오 경로 확인
    video_path = args.video_path
    if not video_path:
        mp4_files = glob.glob(os.path.join(project_root, '*.mp4'))
        if mp4_files:
            video_path = mp4_files[0]
        else:
            default_path = '/workspace/storage/videos/match2.mp4'
            if os.path.exists(default_path):
                video_path = default_path
            else:
                mp4_files = glob.glob('*.mp4')
                if mp4_files:
                    video_path = mp4_files[0]

    if not video_path or not os.path.exists(video_path):
        print("[ERROR] Could not find any input video. Please specify --video_path")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[TEST] Output Directory: {args.output_dir}")
    print(f"[TEST] Using video: {video_path}")
    
    # 2. 모델 초기화
    print("\n[TEST] 1. Initializing models...")
    
    try:
        # AutoLabeler는 내부적으로 config.yaml을 로드하여 모델 경로를 결정함
        labeler = AutoLabeler() 
        converter = LabelConverter()
    except Exception as e:
        print(f"[ERROR] Model initialization failed: {e}")
        return
    
    # 3. 비디오 처리 루프
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[TEST] Video info: {total_frames} frames, {fps:.2f} FPS")
    print("\n[TEST] 2. Starting video processing (Interval: 300 frames)...")
    
    frame_interval = 300
    frame_count = 0
    processed_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            print(f"\n[{processed_count+1}] Processing Frame {frame_count}/{total_frames}...")
            
            # (1) 샘플 이미지 저장 (AutoLabeler 입력용)
            img_height, img_width = frame.shape[:2]
            sample_img_path = os.path.join(args.output_dir, f'frame_{frame_count:06d}.jpg')
            cv2.imwrite(sample_img_path, frame)
            
            # (2) 파이프라인 실행
            result = labeler.process_image(sample_img_path, verbose=False) # verbose 끔
            labeled_masks = result.get('labeled_masks', [])
            print(f"    - Generated {len(labeled_masks)} labeled masks")
            
            # (3) YOLO 포맷 변환 및 저장
            sample_txt_path = os.path.join(args.output_dir, f'frame_{frame_count:06d}.txt')
            num_polygons = converter.save_yolo_label(labeled_masks, sample_txt_path, img_width, img_height)
            print(f"    - Saved {num_polygons} polygons to .txt")
            
            # (4) 검증 (시각화)
            print("    - Verifying...")
            verify_img_path = os.path.join(args.output_dir, f'frame_{frame_count:06d}_verified.png')
            verify_yolo_label_and_save(sample_img_path, sample_txt_path, verify_img_path, labeler.mask_mapper.label_names)
            
            processed_count += 1
            
        frame_count += 1
        
    cap.release()
    print(f"\n[SUCCESS] Test pipeline complete! Processed {processed_count} frames.")
    print(f"Check the output directory: {args.output_dir}")

def verify_yolo_label_and_save(img_path, txt_path, save_path, label_names):
    # 기존 verify_yolo_label 함수를 재활용하되 저장 경로를 인자로 받도록 수정
    img = cv2.imread(img_path)
    if img is None: return

    height, width = img.shape[:2]
    if not os.path.exists(txt_path): return

    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    # 색상 팔레트 (클래스별 고정 색상 사용 권장하지만 여기선 랜덤)
    np.random.seed(42)
    colors = np.random.randint(0, 255, (20, 3), dtype=np.uint8).tolist()
    
    overlay = img.copy()
    
    for line in lines:
        parts = list(map(float, line.strip().split()))
        class_id = int(parts[0])
        coords = parts[1:]
        
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * width)
            y = int(coords[i+1] * height)
            points.append([x, y])
            
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        color = [int(c) for c in colors[class_id % len(colors)]]
        
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        
        label_text = label_names.get(class_id, str(class_id))
        
        text_pos = (points[0][0], points[0][1]-10)
        # 텍스트가 화면 밖으로 나가는 것 방지
        text_pos = (max(0, text_pos[0]), max(20, text_pos[1]))
        
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (text_pos[0], text_pos[1] - th - 5), (text_pos[0] + tw, text_pos[1] + 5), color, -1)
        cv2.putText(img, label_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 2)
    
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    cv2.imwrite(save_path, img)

if __name__ == "__main__":
    process_video_generation_test()
