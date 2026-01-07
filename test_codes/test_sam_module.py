#!/usr/bin/env python3
"""
SAM2 Segmentation 모듈 테스트 스크립트

테스트 내용:
1. 이미지에서 마스크 생성
2. 마스크 필터링
3. 중심점 계산
4. 결과 시각화
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 프로젝트 경로 추가
sys.path.insert(0, '/workspace')

from src.sam_segmentation import SAMSegmenter


def show_masks(ax, masks, image_shape):
    """마스크 시각화"""
    if len(masks) == 0:
        return
    
    # 면적 기준으로 정렬 (큰 것부터)
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # 전체 마스크 이미지 생성
    mask_image = np.ones((image_shape[0], image_shape[1], 4))
    mask_image[:, :, 3] = 0  # 투명
    
    for mask in sorted_masks:
        m = mask['segmentation']
        # 랜덤 색상
        color = np.concatenate([np.random.random(3), [0.5]])
        mask_image[m] = color
    
    ax.imshow(mask_image)


def show_centroids(ax, masks):
    """중심점 시각화"""
    for mask in masks:
        if mask.get('centroid') is not None:
            cx, cy = mask['centroid']
            ax.plot(cx, cy, 'r+', markersize=10, markeredgewidth=2)


def test_sam_segmentation(image_path, output_dir):
    """SAM2 Segmentation 테스트"""
    print("=" * 80)
    print("SAM2 Segmentation Module Test")
    print("=" * 80)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # SAM Segmenter 초기화
    print("\n[INFO] Initializing SAMSegmenter...")
    model_path = "/workspace/models/sam2_hiera_large.pt"
    config_path = "/workspace/config/config.yaml"
    
    segmenter = SAMSegmenter(
        model_path=model_path,
        device='cuda',
        config_path=config_path
    )
    
    # 이미지 로드
    print(f"\n[INFO] Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    print(f"[INFO] Image size: {width}x{height}")
    
    # 1. 마스크 생성 (필터링 없음)
    print("\n" + "=" * 80)
    print("[STEP 1] Generating all masks (no filtering)...")
    print("=" * 80)
    
    all_masks = segmenter.generate_masks(image_rgb)
    stats_all = segmenter.get_mask_stats(all_masks)
    
    print(f"\n[INFO] All masks statistics:")
    print(f"       - Count: {stats_all['count']}")
    print(f"       - Avg area: {stats_all['avg_area']:.0f} pixels")
    print(f"       - Min area: {stats_all['min_area']:.0f} pixels")
    print(f"       - Max area: {stats_all['max_area']:.0f} pixels")
    print(f"       - Avg IoU: {stats_all['avg_iou']:.3f}")
    print(f"       - Avg Stability: {stats_all['avg_stability']:.3f}")
    
    # 2. 마스크 필터링
    print("\n" + "=" * 80)
    print("[STEP 2] Filtering masks...")
    print("=" * 80)
    
    filtered_masks = segmenter.filter_masks(all_masks, image_rgb.shape)
    stats_filtered = segmenter.get_mask_stats(filtered_masks)
    
    print(f"\n[INFO] Filtered masks statistics:")
    print(f"       - Count: {stats_filtered['count']}")
    print(f"       - Avg area: {stats_filtered['avg_area']:.0f} pixels")
    print(f"       - Removed: {stats_all['count'] - stats_filtered['count']} masks")
    
    # 3. 중심점 계산
    print("\n" + "=" * 80)
    print("[STEP 3] Calculating centroids...")
    print("=" * 80)
    
    filtered_masks = segmenter.calculate_centroids(filtered_masks)
    
    centroids_count = sum(1 for m in filtered_masks if m.get('centroid') is not None)
    print(f"[INFO] Calculated {centroids_count} centroids")
    
    # 4. 시각화
    print("\n" + "=" * 80)
    print("[STEP 4] Visualizing results...")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # (0, 0) 원본 이미지
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image', fontsize=16)
    axes[0, 0].axis('off')
    
    # (0, 1) 모든 마스크
    axes[0, 1].imshow(image_rgb)
    show_masks(axes[0, 1], all_masks, image_rgb.shape)
    axes[0, 1].set_title(f'All Masks ({len(all_masks)} masks)', fontsize=16)
    axes[0, 1].axis('off')
    
    # (1, 0) 필터링된 마스크
    axes[1, 0].imshow(image_rgb)
    show_masks(axes[1, 0], filtered_masks, image_rgb.shape)
    axes[1, 0].set_title(f'Filtered Masks ({len(filtered_masks)} masks)', fontsize=16)
    axes[1, 0].axis('off')
    
    # (1, 1) 필터링된 마스크 + 중심점
    axes[1, 1].imshow(image_rgb)
    show_masks(axes[1, 1], filtered_masks, image_rgb.shape)
    show_centroids(axes[1, 1], filtered_masks)
    axes[1, 1].set_title(f'Filtered Masks + Centroids ({centroids_count} points)', fontsize=16)
    axes[1, 1].axis('off')
    
    # 저장
    output_path = os.path.join(output_dir, 'sam_segmentation_test.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved result: {output_path}")
    
    # 5. 통합 처리 테스트
    print("\n" + "=" * 80)
    print("[STEP 5] Testing integrated process()...")
    print("=" * 80)
    
    processed_masks = segmenter.process(image_rgb)
    print(f"[INFO] Processed {len(processed_masks)} masks")
    
    # 중심점 확인
    with_centroids = sum(1 for m in processed_masks if m.get('centroid') is not None)
    print(f"[INFO] Masks with centroids: {with_centroids}/{len(processed_masks)}")
    
    print("\n" + "=" * 80)
    print("✓ SAM2 Segmentation Test Complete!")
    print("=" * 80)


def main():
    """메인 함수"""
    # 테스트 이미지 경로
    test_image = "/workspace/test_codes/source/test_image.jpg"
    
    if not os.path.exists(test_image):
        print(f"[ERROR] Test image not found: {test_image}")
        sys.exit(1)
    
    # 출력 디렉토리
    output_dir = "/workspace/test_codes/result_sam"
    
    # 테스트 실행
    test_sam_segmentation(test_image, output_dir)


if __name__ == "__main__":
    main()
