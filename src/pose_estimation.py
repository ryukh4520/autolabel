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
        키포인트로부터 신체 영역 계산 (8개 클래스)
        
        Args:
            keypoints (np.ndarray): [17, 3] 키포인트 배열
            image_shape (tuple): (height, width, channels)
            
        Returns:
            dict: 신체 영역 정보
                {
                    'head_cover_region': [x1, y1, x2, y2],  # 두건 복면 (머리~어깨)
                    'goggles_region': [x1, y1, x2, y2],     # 보안경 (눈 영역)
                    'mask_region': [x1, y1, x2, y2],        # 마스크 (코/입 영역)
                    'upper_body_region': [x1, y1, x2, y2],  # 상의 (어깨~엉덩이)
                    'pants_region': [x1, y1, x2, y2],       # 하의 (엉덩이~발목)
                    'left_glove_region': [x, y, radius],    # 왼손 장갑
                    'right_glove_region': [x, y, radius],   # 오른손 장갑
                    'left_arm_cover_region': [x1, y1, x2, y2],  # 왼팔 토시
                    'right_arm_cover_region': [x1, y1, x2, y2], # 오른팔 토시
                    'left_shoe_cover_region': [x1, y1, x2, y2], # 왼발 신발커버
                    'right_shoe_cover_region': [x1, y1, x2, y2] # 오른발 신발커버
                }
        """
        height, width = image_shape[:2]
        regions = {}
        
        # 1. 두건 복면 (머리 전체 ~ 어깨)
        regions['head_cover_region'] = self._calculate_head_cover_region(keypoints, height, width)
        
        # 2. 보안경 (눈 영역)
        regions['goggles_region'] = self._calculate_goggles_region(keypoints, height, width)
        
        # 3. 마스크 (코/입 영역)
        regions['mask_region'] = self._calculate_mask_region(keypoints, height, width)
        
        # 4. 상의 (어깨 ~ 엉덩이)
        regions['upper_body_region'] = self._calculate_torso_region(keypoints, width)
        
        # 5. 하의 (엉덩이 ~ 발목)
        regions['pants_region'] = self._calculate_pants_region(keypoints, width)
        
        # 6. 장갑 (손목 주변)
        regions['left_glove_region'] = self._calculate_hand_region(keypoints, 9)   # left wrist
        regions['right_glove_region'] = self._calculate_hand_region(keypoints, 10)  # right wrist
        
        # 7. 토시 (팔꿈치 ~ 손목)
        regions['left_arm_cover_region'] = self._calculate_arm_cover_region(keypoints, 'left', width)
        regions['right_arm_cover_region'] = self._calculate_arm_cover_region(keypoints, 'right', width)
        
        # 8. 신발 커버 (발목 아래)
        regions['left_shoe_cover_region'] = self._calculate_shoe_cover_region(keypoints, 'left', height, width)
        regions['right_shoe_cover_region'] = self._calculate_shoe_cover_region(keypoints, 'right', height, width)
        
        return regions
    
    def _calculate_head_cover_region(self, keypoints, height, width):
        """두건 복면 영역 계산 (머리 전체 ~ 어깨)"""
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        
        # 신뢰도 체크
        if nose[2] < 0.3:
            return None
        
        # 머리 크기 추정 (더 크게 - 두건이 머리를 덮음)
        head_height = height * 0.25  # 이미지 높이의 25%
        head_width = head_height * 1.2  # 가로는 세로의 120%
        
        # X 범위 (어깨 포함)
        shoulder_x = []
        if left_shoulder[2] >= 0.3:
            shoulder_x.append(left_shoulder[0])
        if right_shoulder[2] >= 0.3:
            shoulder_x.append(right_shoulder[0])
        
        if shoulder_x:
            x1 = max(0, min(shoulder_x) - 20)
            x2 = min(width, max(shoulder_x) + 20)
        else:
            x1 = max(0, nose[0] - head_width / 2)
            x2 = min(width, nose[0] + head_width / 2)
        
        # Y 범위 (머리 위 ~ 어깨)
        y1 = max(0, nose[1] - head_height)
        
        if shoulder_x:
            shoulder_y = []
            if left_shoulder[2] >= 0.3:
                shoulder_y.append(left_shoulder[1])
            if right_shoulder[2] >= 0.3:
                shoulder_y.append(right_shoulder[1])
            y2 = max(shoulder_y) if shoulder_y else nose[1] + head_height * 0.5
        else:
            y2 = nose[1] + head_height * 0.5
        
        return [x1, y1, x2, y2]
    
    def _calculate_goggles_region(self, keypoints, height, width):
        """보안경 영역 계산 (눈 영역)"""
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        nose = keypoints[0]
        
        # 신뢰도 체크
        valid_points = [p for p in [left_eye, right_eye, nose] if p[2] >= 0.3]
        if len(valid_points) < 2:
            return None
        
        # 눈 영역 크기
        eye_height = height * 0.08  # 이미지 높이의 8%
        eye_width = height * 0.25   # 이미지 높이의 25%
        
        # 중심점 (코 또는 두 눈의 중간)
        if nose[2] >= 0.3:
            center_x = nose[0]
            center_y = nose[1] - eye_height * 0.5  # 코보다 약간 위
        else:
            center_x = np.mean([p[0] for p in valid_points])
            center_y = np.mean([p[1] for p in valid_points])
        
        x1 = max(0, center_x - eye_width / 2)
        y1 = max(0, center_y - eye_height / 2)
        x2 = min(width, center_x + eye_width / 2)
        y2 = min(height, center_y + eye_height / 2)
        
        return [x1, y1, x2, y2]
    
    def _calculate_mask_region(self, keypoints, height, width):
        """마스크 영역 계산 (코/입 영역)"""
        nose = keypoints[0]
        
        if nose[2] < 0.3:
            return None
        
        # 마스크 크기 (코 아래)
        mask_height = height * 0.12  # 이미지 높이의 12%
        mask_width = height * 0.20   # 이미지 높이의 20%
        
        x1 = max(0, nose[0] - mask_width / 2)
        y1 = nose[1]  # 코부터 시작
        x2 = min(width, nose[0] + mask_width / 2)
        y2 = min(height, nose[1] + mask_height)
        
        return [x1, y1, x2, y2]
    
    def _calculate_arm_cover_region(self, keypoints, side, width):
        """토시 영역 계산 (팔꿈치 ~ 손목)"""
        if side == 'left':
            elbow = keypoints[7]
            wrist = keypoints[9]
        else:  # right
            elbow = keypoints[8]
            wrist = keypoints[10]
        
        # 신뢰도 체크
        if elbow[2] < 0.3 or wrist[2] < 0.3:
            return None
        
        # 팔 두께 추정
        arm_thickness = 40  # 픽셀
        
        x1 = min(elbow[0], wrist[0]) - arm_thickness
        x2 = max(elbow[0], wrist[0]) + arm_thickness
        y1 = min(elbow[1], wrist[1])
        y2 = max(elbow[1], wrist[1])
        
        x1 = max(0, x1)
        x2 = min(width, x2)
        
        return [x1, y1, x2, y2]
    
    def _calculate_shoe_cover_region(self, keypoints, side, height, width):
        """신발 커버 영역 계산 (발목 아래)"""
        if side == 'left':
            ankle = keypoints[15]
        else:  # right
            ankle = keypoints[16]
        
        if ankle[2] < 0.3:
            return None
        
        # 신발 크기
        shoe_width = 60
        shoe_height = self.mapping_config.get('ankle_offset', 50)
        
        x1 = max(0, ankle[0] - shoe_width / 2)
        x2 = min(width, ankle[0] + shoe_width / 2)
        y1 = ankle[1]
        y2 = min(height, ankle[1] + shoe_height)
        
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
