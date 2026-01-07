
import sys
import os
import cv2
import numpy as np
import argparse
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.auto_labeler import AutoLabeler
from src.label_converter import LabelConverter

# 시각화용 색상 정의 (RGB)
LABEL_COLORS = {
    'head_cover': [255, 0, 0],       # Red
    'goggles': [0, 255, 0],          # Green
    'mask': [0, 0, 255],             # Blue
    'upper_body': [255, 255, 0],     # Yellow
    'pants': [255, 0, 255],          # Magenta
    'gloves': [0, 255, 255],         # Cyan
    'arm_covers': [255, 165, 0],     # Orange
    'shoe_covers': [128, 0, 128],    # Purple
    'unknown': [128, 128, 128]       # Gray
}

def parse_args():
    parser = argparse.ArgumentParser(description="Test Dataset Generation & Verification")
    parser.add_argument('--video_path', type=str, help="Path to input video")
    # 마운트된 볼륨 내의 절대 경로 사용 -> 호스트에서 즉시 확인 가능
    parser.add_argument('--output_dir', type=str, default='/workspace/test_codes/result_datageneration', help="Directory to save results")
    return parser.parse_args()

def process_video_generation_test():
    args = parse_args()
    
    # 1. 비디오 경로 확인
    video_path = args.video_path
    if not video_path:
        # 자동 탐색
        search_paths = [
            os.path.join(project_root, '*.mp4'),
            '/workspace/storage/videos/match2.mp4',
            '/workspace/test_codes/source/test_video.mp4',
            '*.mp4'
        ]
        for path in search_paths:
            found = glob.glob(path)
            if found:
                video_path = found[0]
                break

    if not video_path or not os.path.exists(video_path):
        print("[ERROR] No input video found. Please specify --video_path")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Output: {args.output_dir}")
    
    # 2. 모델 초기화
    print("\n[INFO] Initializing models...")
    try:
        labeler = AutoLabeler()
        converter = LabelConverter()
    except Exception as e:
        print(f"[ERROR] Init failed: {e}")
        return
    
    # 3. 비디오 루프
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_interval = 300
    frame_count = 0
    processed_count = 0
    
    print("\n[INFO] Starting pipeline visualization test...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            print(f"\n[Frame {frame_count}] Processing...")
            
            # (1) 임시 이미지 저장
            temp_img_path = os.path.join(args.output_dir, f'temp_frame_{frame_count}.jpg')
            cv2.imwrite(temp_img_path, frame)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            
            # (2) SAM-HQ Raw Mask 생성 (시각화용)
            # 전체 마스크 시각화를 위해 따로 생성 (AutoLabeler 내부 로직과 독립적 확인)
            print("    - Generating SAM-HQ raw masks...")
            sam_input = labeler.load_image(temp_img_path)
            all_masks = labeler.sam_segmenter.generate_masks(sam_input)
            
            # (3) Auto-Labeling Pipeline 실행
            print("    - Running auto-labeling pipeline...")
            result = labeler.process_image(temp_img_path, verbose=False)
            labeled_masks = result.get('labeled_masks', [])
            person_info = result.get('person_info', {})
            
            # (4) YOLO 포맷 변환 및 저장
            label_txt_path = os.path.join(args.output_dir, f'frame_{frame_count:06d}.txt')
            converter.save_yolo_label(labeled_masks, label_txt_path, width, height)
            print(f"    - Saved label: {label_txt_path}")
            
            # (5) 종합 시각화 및 저장
            vis_save_path = os.path.join(args.output_dir, f'frame_{frame_count:06d}_comprehensive.png')
            print("    - Creating comprehensive visualization...")
            visualize_comprehensive_result(
                img_rgb, 
                person_info, 
                all_masks, 
                labeled_masks, 
                label_txt_path, 
                vis_save_path,
                labeler.mask_mapper.label_names
            )
            
            processed_count += 1
            
            # 임시 파일 삭제
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
                
        frame_count += 1
        
    cap.release()
    print("\n[SUCCESS] Pipeline Test Complete!")

def visualize_comprehensive_result(image, person_info, all_masks, labeled_masks, label_txt_path, output_path, label_mapping):
    """
    5단계 종합 시각화
    Grid 2x3:
    1. Original      2. Pose Detection     3. SAM-HQ (Raw)
    4. Labeled Masks 5. Final Dataset Verify 6. Statistics
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # 1. Original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('1. Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Pose Detection
    axes[0, 1].imshow(image)
    if person_info:
        bbox = person_info.get('bbox', [0, 0, 0, 0])
        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                         fill=False, edgecolor='red', linewidth=2)
        axes[0, 1].add_patch(rect)
        
        # Keypoints
        kpts = person_info.get('keypoints', [])
        # Skeleton 연결 정보 (간략화)
        skeleton = [[5,7],[7,9],[6,8],[8,10],[11,13],[13,15],[12,14],[14,16],[5,6],[11,12],[5,11],[6,12]]
        
        # Draw skeleton
        kpts_np = np.array(kpts)
        if len(kpts_np) > 0:
            for p1, p2 in skeleton:
                if p1 < len(kpts_np) and p2 < len(kpts_np) and kpts_np[p1][2] > 0.3 and kpts_np[p2][2] > 0.3:
                    x = [kpts_np[p1][0], kpts_np[p2][0]]
                    y = [kpts_np[p1][1], kpts_np[p2][1]]
                    axes[0, 1].plot(x, y, 'g-', linewidth=2)
                    
            # Draw points
            for kp in kpts:
                if kp[2] > 0.3:
                    axes[0, 1].plot(kp[0], kp[1], 'ro', markersize=4)
                
    axes[0, 1].set_title('2. YOLO-Pose', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. SAM-HQ Raw
    axes[0, 2].imshow(image)
    if len(all_masks) > 0:
        sorted_anns = sorted(all_masks, key=(lambda x: x['area']), reverse=True)
        
        # 마스크 오버레이 생성
        img_h, img_w = image.shape[:2]
        mask_overlay = np.ones((img_h, img_w, 4))
        mask_overlay[:,:,3] = 0
        
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.45]])
            mask_overlay[m] = color_mask
            
        axes[0, 2].imshow(mask_overlay)
        
    axes[0, 2].set_title(f'3. SAM-HQ Raw ({len(all_masks)} masks)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 4. Labeled Results
    # 마스크 색상 적용
    mask_img = image.copy()
    overlay_img = np.zeros_like(image)
    
    for mask in labeled_masks:
        label = mask.get('label', 'unknown')
        color = LABEL_COLORS.get(label, [128, 128, 128]) # RGB List
        
        seg = mask['segmentation']
        overlay_img[seg] = color
        
    # 합성
    mask_vis = cv2.addWeighted(mask_img, 0.6, overlay_img, 0.4, 0)
    axes[1, 0].imshow(mask_vis)
    axes[1, 0].set_title(f'4. Auto-Labeling Result ({len(labeled_masks)} labeled)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 5. Final Dataset Verification (Load from .txt)
    axes[1, 1].imshow(image)
    
    if os.path.exists(label_txt_path):
        with open(label_txt_path, 'r') as f:
            lines = f.readlines()
            
        h, w = image.shape[:2]
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if not parts: continue
            cls_id = int(parts[0])
            coords = parts[1:]
            
            # Polygon 좌표 복원
            poly_points = []
            for i in range(0, len(coords), 2):
                px = coords[i] * w
                py = coords[i+1] * h
                poly_points.append([px, py])
            
            if len(poly_points) > 2:
                poly_np = np.array(poly_points)
                
                # Draw Polygon
                poly_patch = plt.Polygon(poly_np, fill=False, edgecolor='cyan', linewidth=2)
                axes[1, 1].add_patch(poly_patch)
                
                # Label Text
                label_name = label_mapping.get(cls_id, str(cls_id))
                axes[1, 1].text(poly_points[0][0], poly_points[0][1], label_name, 
                               color='white', fontsize=10, fontweight='bold',
                               bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
            
    axes[1, 1].set_title('5. Final Dataset (Loaded from .txt)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 6. Statistics / Legend
    axes[1, 2].axis('off')
    
    # 통계 텍스트 작성
    stats_text = "Label Statistics:\n"
    stats_text += "-" * 30 + "\n"
    
    label_counts = {}
    for mask in labeled_masks:
        l = mask.get('label', 'unknown')
        label_counts[l] = label_counts.get(l, 0) + 1
        
    for label, count in sorted(label_counts.items()):
        stats_text += f"\u2022 {label}: {count}\n"
        
    stats_text += "\n\nFiles Generated:\n"
    stats_text += "-" * 30 + "\n"
    stats_text += f"Label: {os.path.basename(label_txt_path)}\n"
    stats_text += f"Polygons: {len(label_counts)}"
    
    axes[1, 2].text(0.1, 0.9, stats_text, fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    # 저장
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"    - Vis saved: {output_path}")

if __name__ == "__main__":
    process_video_generation_test()
