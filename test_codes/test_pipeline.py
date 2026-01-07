#!/usr/bin/env python3
"""
통합 파이프라인 테스트 스크립트

전체 자동 라벨링 파이프라인을 테스트:
1. AutoLabeler 초기화
2. 단일 이미지 처리
3. 결과 확인
4. 간단한 시각화
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 프로젝트 경로 추가
sys.path.insert(0, '/workspace')

from src.auto_labeler import AutoLabeler


# 라벨별 색상 정의 (8개 클래스)
LABEL_COLORS = {
    'head_cover': [255, 100, 100],    # 연한 빨강
    'goggles': [100, 200, 255],       # 하늘색
    'mask': [255, 200, 100],          # 주황색
    'upper_body': [100, 255, 100],    # 연한 초록
    'pants': [255, 255, 100],         # 노랑
    'gloves': [200, 100, 255],        # 보라
    'arm_covers': [100, 255, 255],    # 청록
    'shoe_covers': [255, 100, 200],   # 분홍
    'unknown': [128, 128, 128]        # 회색
}


def show_masks_samhq(ax, masks, image_rgb):
    """SAM-HQ 마스크를 시각화 (모든 마스크)"""
    if len(masks) == 0:
        return
    
    # 면적 기준으로 정렬
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # 전체 마스크 이미지 생성
    mask_image = np.ones((image_rgb.shape[0], image_rgb.shape[1], 4))
    mask_image[:, :, 3] = 0
    
    for mask in sorted_masks:
        m = mask['segmentation']
        # 랜덤 색상
        color = np.concatenate([np.random.random(3), [0.6]])
        mask_image[m] = color
    
    ax.imshow(image_rgb)
    ax.imshow(mask_image)


def visualize_result(result, output_path, all_masks=None):
    """
    결과 시각화 (개선된 버전)
    
    Args:
        result: 파이프라인 결과
        output_path: 저장 경로
        all_masks: SAM-HQ 전체 마스크 (필터링 전)
    """
    image_path = result['image_path']
    labeled_masks = result['labeled_masks']
    person_info = result['person_info']
    
    # 이미지 로드
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2x2 그리드
    fig, axes = plt.subplots(2, 2, figsize=(24, 24))
    
    # ========== (0, 0) 원본 이미지 ==========
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Step 1: Original Image', fontsize=18, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 이미지 정보 텍스트
    h, w = image_rgb.shape[:2]
    axes[0, 0].text(10, 30, f'Size: {w}x{h}', 
                   fontsize=14, color='white', 
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # ========== (0, 1) YOLO-Pose 결과 ==========
    axes[0, 1].imshow(image_rgb)
    
    if person_info:
        # Person bbox
        bbox = person_info['bbox']
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                            fill=False, edgecolor='red', linewidth=3, linestyle='--')
        axes[0, 1].add_patch(rect)
        
        # 키포인트
        keypoints = person_info['keypoints']
        for i, kpt in enumerate(keypoints):
            if kpt[2] >= 0.3:
                axes[0, 1].plot(kpt[0], kpt[1], 'ro', markersize=8)
                axes[0, 1].text(kpt[0]+5, kpt[1]-5, str(i), 
                              fontsize=8, color='yellow',
                              bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        
        # 신체 영역 표시
        body_regions = person_info.get('body_regions', {})
        region_colors = {
            'head_cover_region': 'red',
            'goggles_region': 'cyan',
            'mask_region': 'orange',
            'upper_body_region': 'green',
            'pants_region': 'yellow',
            'left_glove_region': 'blue',
            'right_glove_region': 'blue',
            'left_arm_cover_region': 'purple',
            'right_arm_cover_region': 'purple',
            'left_shoe_cover_region': 'magenta',
            'right_shoe_cover_region': 'magenta'
        }
        
        for region_name, region_data in body_regions.items():
            if region_data is None:
                continue
            
            color = region_colors.get(region_name, 'white')
            
            # 원형 영역 (장갑)
            if 'glove' in region_name:
                x, y, radius = region_data
                circle = plt.Circle((x, y), radius, 
                                  fill=False, edgecolor=color, 
                                  linewidth=2, linestyle=':', alpha=0.8)
                axes[0, 1].add_patch(circle)
            # 사각형 영역
            else:
                x1, y1, x2, y2 = region_data
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   fill=False, edgecolor=color,
                                   linewidth=2, linestyle=':', alpha=0.8)
                axes[0, 1].add_patch(rect)
        
        # 정보 텍스트
        info_text = f"Person: {person_info['confidence']:.2f}\n"
        info_text += f"Keypoints: {np.sum(keypoints[:, 2] >= 0.3)}/17"
        axes[0, 1].text(10, 30, info_text,
                       fontsize=14, color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    axes[0, 1].set_title('Step 2: YOLO-Pose Detection', fontsize=18, fontweight='bold')
    axes[0, 1].axis('off')
    
    # ========== (1, 0) SAM-HQ 자동 마스크 ==========
    if all_masks is not None:
        show_masks_samhq(axes[1, 0], all_masks, image_rgb)
        mask_count = len(all_masks)
    else:
        axes[1, 0].imshow(image_rgb)
        mask_count = 0
    
    axes[1, 0].set_title(f'Step 3: SAM-HQ Auto Masks ({mask_count} masks)', 
                        fontsize=18, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 정보 텍스트
    axes[1, 0].text(10, 30, f'Generated: {mask_count} masks',
                   fontsize=14, color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # ========== (1, 1) 최종 라벨된 마스크 ==========
    mask_overlay = image_rgb.copy().astype(float)
    
    # 라벨별로 그룹화
    masks_by_label = {}
    for mask in labeled_masks:
        label = mask.get('label', 'unknown')
        if label not in masks_by_label:
            masks_by_label[label] = []
        masks_by_label[label].append(mask)
    
    # 각 라벨별로 다른 색상으로 표시
    for label, masks in masks_by_label.items():
        color = np.array(LABEL_COLORS.get(label, [128, 128, 128]))
        
        for mask in masks:
            segmentation = mask['segmentation']
            mask_overlay[segmentation] = mask_overlay[segmentation] * 0.4 + color * 0.6
    
    axes[1, 1].imshow(mask_overlay.astype(np.uint8))
    
    # 범례 추가
    legend_y = 30
    for label in sorted(masks_by_label.keys()):
        count = len(masks_by_label[label])
        color = np.array(LABEL_COLORS.get(label, [128, 128, 128])) / 255.0
        
        # 색상 박스
        rect = plt.Rectangle((10, legend_y), 30, 20,
                            facecolor=color, edgecolor='white', linewidth=2)
        axes[1, 1].add_patch(rect)
        
        # 라벨 텍스트
        axes[1, 1].text(50, legend_y+15, f'{label}: {count} masks',
                       fontsize=14, color='white', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        legend_y += 35
    
    axes[1, 1].set_title(f'Step 4: Labeled Masks ({len(labeled_masks)} masks)', 
                        fontsize=18, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 전체 제목
    stats = result['statistics']
    time_stats = result['processing_time']
    
    title_text = f"Auto-Labeling Pipeline Result | "
    title_text += f"Total Time: {time_stats.get('total', 0):.2f}s | "
    title_text += f"Labels: {len(stats['by_label'])} types"
    
    fig.suptitle(title_text, fontsize=20, fontweight='bold', y=0.98)
    
    # 저장
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Visualization saved: {output_path}")


def main():
    """메인 함수"""
    print("=" * 80)
    print("Integrated Pipeline Test")
    print("=" * 80)
    
    # 테스트 이미지
    test_image = "/workspace/test_codes/source/test_image.jpg"
    
    if not os.path.exists(test_image):
        print(f"[ERROR] Test image not found: {test_image}")
        sys.exit(1)
    
    # 출력 디렉토리
    output_dir = "/workspace/test_codes/result_pipeline"
    os.makedirs(output_dir, exist_ok=True)
    
    # AutoLabeler 초기화
    print("\n[INFO] Initializing AutoLabeler...")
    labeler = AutoLabeler()
    
    # 이미지 로드 (SAM-HQ 전체 마스크를 얻기 위해)
    print("\n[INFO] Loading image...")
    image = labeler.load_image(test_image)
    
    # SAM-HQ 전체 마스크 생성 (필터링 전)
    print("\n[INFO] Generating SAM-HQ masks...")
    all_masks = labeler.sam_segmenter.generate_masks(image)
    
    # 이미지 처리
    print("\n[INFO] Processing image...")
    result = labeler.process_image(test_image, verbose=True)
    
    # 결과 시각화 (전체 마스크 포함)
    print("\n[INFO] Generating visualization...")
    output_path = os.path.join(output_dir, 'pipeline_result.png')
    visualize_result(result, output_path, all_masks=all_masks)
    
    # VRAM 사용량
    vram = labeler.get_vram_usage()
    if vram:
        print(f"\n[INFO] VRAM Usage:")
        print(f"       - Allocated: {vram['allocated']:.2f} GB")
        print(f"       - Reserved: {vram['reserved']:.2f} GB")
    
    print("\n" + "=" * 80)
    print("✓ Pipeline Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
