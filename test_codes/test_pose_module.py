#!/usr/bin/env python3
"""
YOLO-Pose 모듈 테스트 스크립트

테스트 내용:
1. 이미지에서 사람 검출
2. 키포인트 추출
3. 신체 영역 계산
4. 결과 시각화
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 프로젝트 경로 추가
sys.path.insert(0, '/workspace')

from src.pose_estimation import PoseEstimator, KEYPOINT_NAMES


# 키포인트 연결 정의 (스켈레톤 그리기용)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),  # 코-눈
    (1, 3), (2, 4),  # 눈-귀
    (5, 6),  # 어깨
    (5, 7), (7, 9),  # 왼팔
    (6, 8), (8, 10),  # 오른팔
    (5, 11), (6, 12),  # 어깨-엉덩이
    (11, 12),  # 엉덩이
    (11, 13), (13, 15),  # 왼다리
    (12, 14), (14, 16),  # 오른다리
]


def draw_keypoints_and_skeleton(ax, keypoints, conf_threshold=0.3):
    """키포인트와 스켈레톤 그리기"""
    # 스켈레톤 연결선
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        start_kpt = keypoints[start_idx]
        end_kpt = keypoints[end_idx]
        
        if start_kpt[2] >= conf_threshold and end_kpt[2] >= conf_threshold:
            ax.plot([start_kpt[0], end_kpt[0]], 
                   [start_kpt[1], end_kpt[1]], 
                   'g-', linewidth=2, alpha=0.7)
    
    # 키포인트
    for idx, kpt in enumerate(keypoints):
        if kpt[2] >= conf_threshold:
            ax.plot(kpt[0], kpt[1], 'ro', markersize=8)
            ax.text(kpt[0], kpt[1], str(idx), 
                   color='white', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))


def draw_body_regions(ax, body_regions):
    """신체 영역 박스 그리기"""
    colors = {
        'head_region': 'red',
        'left_hand_region': 'blue',
        'right_hand_region': 'cyan',
        'torso_region': 'green',
        'pants_region': 'yellow',
        'boots_region': 'magenta'
    }
    
    labels = {
        'head_region': 'Helmet',
        'left_hand_region': 'L-Glove',
        'right_hand_region': 'R-Glove',
        'torso_region': 'Upper Body',
        'pants_region': 'Pants',
        'boots_region': 'Boots'
    }
    
    for region_name, region_data in body_regions.items():
        if region_data is None:
            continue
        
        color = colors.get(region_name, 'white')
        label = labels.get(region_name, region_name)
        
        # 원형 영역 (장갑)
        if 'hand' in region_name:
            x, y, radius = region_data
            circle = patches.Circle((x, y), radius, 
                                   fill=False, edgecolor=color, 
                                   linewidth=2, linestyle='--', alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y - radius - 10, label, 
                   color=color, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 사각형 영역
        else:
            x1, y1, x2, y2 = region_data
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle((x1, y1), width, height,
                                    fill=False, edgecolor=color,
                                    linewidth=2, linestyle='--', alpha=0.8)
            ax.add_patch(rect)
            ax.text(x1, y1 - 10, label,
                   color=color, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))


def test_pose_estimation(image_path, output_dir):
    """Pose estimation 테스트"""
    print("=" * 80)
    print("YOLO-Pose Module Test")
    print("=" * 80)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # Pose Estimator 초기화
    print("\n[INFO] Initializing PoseEstimator...")
    model_path = "/workspace/models/yolov8m-pose.pt"
    config_path = "/workspace/config/config.yaml"
    
    estimator = PoseEstimator(
        model_path=model_path,
        device='cuda',
        conf_threshold=0.5,
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
    
    # Pose detection
    print("\n[INFO] Detecting persons and keypoints...")
    persons = estimator.detect(image_rgb)
    
    print(f"[INFO] Detected {len(persons)} person(s)")
    
    if len(persons) == 0:
        print("[WARNING] No persons detected!")
        return
    
    # 각 검출된 사람에 대해 시각화
    for person_idx, person in enumerate(persons):
        print(f"\n[INFO] Person {person_idx + 1}:")
        print(f"       - Confidence: {person['confidence']:.3f}")
        print(f"       - Bbox: {person['bbox']}")
        
        # 키포인트 정보
        keypoints = person['keypoints']
        valid_kpts = np.sum(keypoints[:, 2] >= 0.3)
        print(f"       - Valid keypoints: {valid_kpts}/17")
        
        # 신체 영역 정보
        body_regions = person['body_regions']
        print(f"       - Body regions:")
        for region_name, region_data in body_regions.items():
            if region_data is not None:
                print(f"         ✓ {region_name}")
            else:
                print(f"         ✗ {region_name} (not detected)")
        
        # 시각화
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # 1. 원본 이미지
        axes[0].imshow(image_rgb)
        axes[0].set_title(f'Original Image', fontsize=16)
        axes[0].axis('off')
        
        # 2. 키포인트 + 스켈레톤
        axes[1].imshow(image_rgb)
        draw_keypoints_and_skeleton(axes[1], keypoints)
        axes[1].set_title(f'Keypoints & Skeleton ({valid_kpts}/17)', fontsize=16)
        axes[1].axis('off')
        
        # 3. 신체 영역
        axes[2].imshow(image_rgb)
        draw_keypoints_and_skeleton(axes[2], keypoints)
        draw_body_regions(axes[2], body_regions)
        axes[2].set_title('Body Regions for Equipment Detection', fontsize=16)
        axes[2].axis('off')
        
        # 저장
        output_path = os.path.join(output_dir, f'pose_test_person_{person_idx}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Saved result: {output_path}")
    
    print("\n" + "=" * 80)
    print("✓ Pose Estimation Test Complete!")
    print("=" * 80)


def main():
    """메인 함수"""
    # 테스트 이미지 경로
    test_image = "/workspace/test_codes/source/test_image.jpg"
    
    # test_codes/source에서 이미지 찾기
    source_dir = "/workspace/test_codes/source"
    if not os.path.exists(test_image):
        # 동영상에서 프레임 추출
        video_path = "/workspace/test_codes/source/KakaoTalk_20260107_155259635.mp4"
        if os.path.exists(video_path):
            print("[INFO] Extracting frame from video...")
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(test_image, frame)
                print(f"[INFO] Frame saved: {test_image}")
            cap.release()
    
    if not os.path.exists(test_image):
        print(f"[ERROR] Test image not found: {test_image}")
        print(f"[INFO] Please place a test image in {source_dir}")
        sys.exit(1)
    
    # 출력 디렉토리
    output_dir = "/workspace/test_codes/result_pose"
    
    # 테스트 실행
    test_pose_estimation(test_image, output_dir)


if __name__ == "__main__":
    main()
