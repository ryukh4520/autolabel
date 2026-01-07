"""
Pose Estimation Module using YOLO-Pose

이 모듈은 YOLO-Pose를 사용하여:
1. 이미지에서 사람을 검출
2. 17개 신체 키포인트를 추출
3. 키포인트로부터 신체 영역(헬멧, 장갑, 상의, 하의, 신발)을 계산
"""

import numpy as np
from ultralytics import YOLO
import yaml


# COCO 키포인트 인덱스 정의
KEYPOINT_NAMES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}


class PoseEstimator:
    """
    YOLO-Pose 기반 자세 추정 클래스
    """
    
    def __init__(self, model_path, device='cuda', conf_threshold=0.5, config_path=None):
        """
        Args:
            model_path (str): YOLO-Pose 모델 경로
            device (str): 'cuda' or 'cpu'
            conf_threshold (float): Person detection 신뢰도 임계값
            config_path (str): 설정 파일 경로 (옵션)
        """
        self.device = device
        self.conf_threshold = conf_threshold
        
        # 설정 로드
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.mapping_config = config.get('mapping', {})
        else:
            # 기본 설정
            self.mapping_config = {
                'head_offset_ratio': 0.15,
                'hand_radius': 60,
                'torso_expand_ratio': 0.1,
                'ankle_offset': 30
            }
        
        # YOLO-Pose 모델 로드
        print(f"[PoseEstimator] Loading YOLO-Pose model from {model_path}...")
        self.model = YOLO(model_path)
        self.model.to(device)
        print(f"[PoseEstimator] Model loaded on {device}")
    
    def detect(self, image):
        """
        이미지에서 사람 검출 및 키포인트 추출
        
        Args:
            image (np.ndarray): RGB 이미지
            
        Returns:
            list of dict: 검출된 각 사람의 정보
                {
                    'keypoints': np.array([[x, y, conf], ...]),  # 17x3
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'body_regions': dict
                }
        """
        # YOLO-Pose 추론
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        persons = []
        
        for result in results:
            # 검출된 사람이 없으면 스킵
            if result.keypoints is None or len(result.keypoints) == 0:
                continue
            
            # 각 검출된 사람에 대해
            for i in range(len(result.boxes)):
                # Bounding box
                box = result.boxes[i]
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                confidence = float(box.conf[0])
                
                # Keypoints (17개)
                kpts = result.keypoints[i].data[0].cpu().numpy()  # [17, 3] (x, y, conf)
                
                # 신체 영역 계산
                body_regions = self.get_body_regions(kpts, image.shape)
                
                person_info = {
                    'keypoints': kpts,
                    'bbox': bbox.tolist(),
                    'confidence': confidence,
                    'body_regions': body_regions
                }
                
                persons.append(person_info)
        
        return persons
    
    def get_body_regions(self, keypoints, image_shape):
        """
        키포인트로부터 신체 영역 계산
        
        Args:
            keypoints (np.ndarray): [17, 3] 키포인트 배열
            image_shape (tuple): (height, width, channels)
            
        Returns:
            dict: 신체 영역 정보
                {
                    'head_region': [x1, y1, x2, y2],
                    'left_hand_region': [x, y, radius],
                    'right_hand_region': [x, y, radius],
                    'torso_region': [x1, y1, x2, y2],
                    'pants_region': [x1, y1, x2, y2],
                    'boots_region': [x1, y1, x2, y2]
                }
        """
        height, width = image_shape[:2]
        regions = {}
        
        # 1. 헬멧 영역 (머리 위쪽)
        regions['head_region'] = self._calculate_head_region(keypoints, height, width)
        
        # 2. 장갑 영역 (손목 주변)
        regions['left_hand_region'] = self._calculate_hand_region(keypoints, 9)  # left wrist
        regions['right_hand_region'] = self._calculate_hand_region(keypoints, 10)  # right wrist
        
        # 3. 상의 영역 (어깨-엉덩이)
        regions['torso_region'] = self._calculate_torso_region(keypoints, width)
        
        # 4. 하의 영역 (엉덩이-발목)
        regions['pants_region'] = self._calculate_pants_region(keypoints, width)
        
        # 5. 신발 영역 (발목 아래)
        regions['boots_region'] = self._calculate_boots_region(keypoints, height, width)
        
        return regions
    
    def _calculate_head_region(self, keypoints, height, width):
        """헬멧 영역 계산 (코 위쪽)"""
        nose = keypoints[0]  # [x, y, conf]
        
        if nose[2] < 0.3:  # 신뢰도가 너무 낮으면
            return None
        
        # 머리 크기 추정 (이미지 높이의 비율)
        head_height = height * self.mapping_config['head_offset_ratio']
        head_width = head_height * 0.8  # 가로는 세로의 80%
        
        x1 = max(0, nose[0] - head_width / 2)
        y1 = max(0, nose[1] - head_height)
        x2 = min(width, nose[0] + head_width / 2)
        y2 = nose[1]
        
        return [x1, y1, x2, y2]
    
    def _calculate_hand_region(self, keypoints, wrist_idx):
        """장갑 영역 계산 (손목 주변 원형)"""
        wrist = keypoints[wrist_idx]
        
        if wrist[2] < 0.3:  # 신뢰도가 너무 낮으면
            return None
        
        radius = self.mapping_config['hand_radius']
        
        return [wrist[0], wrist[1], radius]
    
    def _calculate_torso_region(self, keypoints, width):
        """상의 영역 계산 (어깨-엉덩이)"""
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        # 신뢰도 체크
        valid_points = [p for p in [left_shoulder, right_shoulder, left_hip, right_hip] 
                       if p[2] >= 0.3]
        
        if len(valid_points) < 2:
            return None
        
        # 확장 비율
        expand_ratio = self.mapping_config['torso_expand_ratio']
        
        # X 좌표 범위
        x_coords = [p[0] for p in valid_points]
        x1 = min(x_coords)
        x2 = max(x_coords)
        expand_x = (x2 - x1) * expand_ratio
        x1 = max(0, x1 - expand_x)
        x2 = min(width, x2 + expand_x)
        
        # Y 좌표 범위 (어깨 ~ 엉덩이)
        shoulder_y = [left_shoulder[1], right_shoulder[1]]
        hip_y = [left_hip[1], right_hip[1]]
        
        y1 = min([y for y in shoulder_y if y > 0], default=0)
        y2 = max([y for y in hip_y if y > 0], default=0)
        
        if y2 <= y1:
            return None
        
        return [x1, y1, x2, y2]
    
    def _calculate_pants_region(self, keypoints, width):
        """하의 영역 계산 (엉덩이-발목)"""
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        # 신뢰도 체크
        valid_points = [p for p in [left_hip, right_hip, left_ankle, right_ankle] 
                       if p[2] >= 0.3]
        
        if len(valid_points) < 2:
            return None
        
        # 확장 비율
        expand_ratio = self.mapping_config['torso_expand_ratio']
        
        # X 좌표 범위
        x_coords = [p[0] for p in valid_points]
        x1 = min(x_coords)
        x2 = max(x_coords)
        expand_x = (x2 - x1) * expand_ratio
        x1 = max(0, x1 - expand_x)
        x2 = min(width, x2 + expand_x)
        
        # Y 좌표 범위 (엉덩이 ~ 발목)
        hip_y = [left_hip[1], right_hip[1]]
        ankle_y = [left_ankle[1], right_ankle[1]]
        
        y1 = min([y for y in hip_y if y > 0], default=0)
        y2 = max([y for y in ankle_y if y > 0], default=0)
        
        if y2 <= y1:
            return None
        
        return [x1, y1, x2, y2]
    
    def _calculate_boots_region(self, keypoints, height, width):
        """신발 영역 계산 (발목 아래)"""
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        # 신뢰도 체크
        valid_ankles = [p for p in [left_ankle, right_ankle] if p[2] >= 0.3]
        
        if len(valid_ankles) == 0:
            return None
        
        ankle_offset = self.mapping_config['ankle_offset']
        
        # X 좌표 범위
        x_coords = [p[0] for p in valid_ankles]
        x1 = min(x_coords) - 30  # 좌우 여유
        x2 = max(x_coords) + 30
        x1 = max(0, x1)
        x2 = min(width, x2)
        
        # Y 좌표 범위 (발목부터 아래)
        y1 = min([p[1] for p in valid_ankles])
        y2 = min(height, y1 + ankle_offset)
        
        return [x1, y1, x2, y2]
    
    def get_keypoint_name(self, idx):
        """키포인트 인덱스로 이름 반환"""
        return KEYPOINT_NAMES.get(idx, f'unknown_{idx}')
