
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

class Visualizer:
    def __init__(self, label_mapping=None):
        self.label_mapping = label_mapping or {}
        
    def visualize_comprehensive_result(self, image, person_info, all_masks, labeled_masks, label_txt_path, output_path):
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
                    label_name = self.label_mapping.get(cls_id, str(cls_id))
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
