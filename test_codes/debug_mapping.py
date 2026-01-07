#!/usr/bin/env python3
"""
마스크 상세 분석 스크립트
각 마스크의 위치, 크기, 키포인트와의 거리를 분석
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, '/workspace')

from src.auto_labeler import AutoLabeler


def analyze_masks(image_path):
    """마스크 상세 분석"""
    print("=" * 80)
    print("Detailed Mask Analysis")
    print("=" * 80)
    
    # AutoLabeler 초기화
    labeler = AutoLabeler()
    
    # 이미지 로드
    image = labeler.load_image(image_path)
    h, w = image.shape[:2]
    
    # YOLO-Pose
    persons = labeler.pose_estimator.detect(image)
    person = persons[0]
    keypoints = person['keypoints']
    
    # 키포인트 출력
    print(f"\n[INFO] Keypoints:")
    kp_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    
    for i, (kp, name) in enumerate(zip(keypoints, kp_names)):
        if kp[2] >= 0.3:
            print(f"  {i}: {name}: ({kp[0]:.0f}, {kp[1]:.0f})")
    
    # SAM-HQ
    masks = labeler.sam_segmenter.process(image)
    
    # Person Filter
    filtered_masks = labeler.person_filter.filter_by_person(masks, person['bbox'])
    
    print(f"\n[INFO] Total filtered masks: {len(filtered_masks)}")
    
    # 각 마스크 분석
    print(f"\n{'='*80}")
    print("Mask Analysis (sorted by Y position)")
    print(f"{'='*80}\n")
    
    # Y 위치로 정렬
    mask_info = []
    for i, mask in enumerate(filtered_masks):
        centroid = mask.get('centroid')
        area = mask.get('area')
        bbox = mask.get('bbox')  # [x, y, w, h]
        
        # 키포인트와의 거리 계산
        distances = {}
        
        # 손목 (장갑)
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        if left_wrist[2] >= 0.3:
            dist = np.sqrt((centroid[0] - left_wrist[0])**2 + (centroid[1] - left_wrist[1])**2)
            distances['left_wrist'] = dist
        if right_wrist[2] >= 0.3:
            dist = np.sqrt((centroid[0] - right_wrist[0])**2 + (centroid[1] - right_wrist[1])**2)
            distances['right_wrist'] = dist
        
        # 팔꿈치 (토시)
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        if left_elbow[2] >= 0.3:
            dist = np.sqrt((centroid[0] - left_elbow[0])**2 + (centroid[1] - left_elbow[1])**2)
            distances['left_elbow'] = dist
        if right_elbow[2] >= 0.3:
            dist = np.sqrt((centroid[0] - right_elbow[0])**2 + (centroid[1] - right_elbow[1])**2)
            distances['right_elbow'] = dist
        
        # 발목 (신발)
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        if left_ankle[2] >= 0.3:
            dist = np.sqrt((centroid[0] - left_ankle[0])**2 + (centroid[1] - left_ankle[1])**2)
            distances['left_ankle'] = dist
        if right_ankle[2] >= 0.3:
            dist = np.sqrt((centroid[0] - right_ankle[0])**2 + (centroid[1] - right_ankle[1])**2)
            distances['right_ankle'] = dist
        
        mask_info.append({
            'idx': i,
            'mask': mask,
            'centroid': centroid,
            'area': area,
            'bbox': bbox,
            'distances': distances
        })
    
    # Y 위치로 정렬
    mask_info.sort(key=lambda x: x['centroid'][1])
    
    # 출력
    for info in mask_info:
        cx, cy = info['centroid']
        area = info['area']
        
        print(f"Mask #{info['idx']+1}:")
        print(f"  Centroid: ({cx:.0f}, {cy:.0f})")
        print(f"  Area: {area:.0f} px ({area/(h*w)*100:.2f}%)")
        
        # 가장 가까운 키포인트
        if info['distances']:
            closest = min(info['distances'].items(), key=lambda x: x[1])
            print(f"  Closest keypoint: {closest[0]} ({closest[1]:.0f}px)")
            
            # 다른 거리도 출력
            for kp_name, dist in sorted(info['distances'].items(), key=lambda x: x[1]):
                if dist < 300:  # 300px 이내만
                    print(f"    - {kp_name}: {dist:.0f}px")
        
        # 현재 할당된 라벨
        label = info['mask'].get('label', 'unknown')
        print(f"  Current label: {label}")
        print()


def main():
    test_image = "/workspace/test_codes/source/test_image.jpg"
    
    if not os.path.exists(test_image):
        print(f"[ERROR] Test image not found: {test_image}")
        sys.exit(1)
    
    analyze_masks(test_image)


if __name__ == "__main__":
    main()
