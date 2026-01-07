"""
Mask Mapper Module (고도화 버전 - 거리 기반 매핑)

이 모듈은:
1. 키포인트와의 거리 기반으로 마스크에 장비 라벨 할당
2. 영역 겹침 + 거리 기반 하이브리드 전략
3. 8개 클래스 지원
"""

import numpy as np


class MaskMapper:
    """
    마스크-라벨 매핑 클래스 (고도화 버전)
    
    매핑 전략:
    1. 키포인트와의 거리 기반 매핑 (1차)
    2. 영역 겹침 기반 매핑 (2차)
    3. 가장 가까운 영역 할당 (3차)
    """
    
    def __init__(self, config=None):
        """
        Args:
            config (dict): 매핑 설정
        """
        if config is None:
            config = {}
        
        self.config = config
        
        # 라벨 정의 (8개 클래스)
        self.label_names = {
            0: 'head_cover',
            1: 'goggles',
            2: 'mask',
            3: 'upper_body',
            4: 'pants',
            5: 'gloves',
            6: 'arm_covers',
            7: 'shoe_covers'
        }
        
        self.label_ids = {v: k for k, v in self.label_names.items()}
        
        # 거리 임계값 (픽셀)
        self.distance_thresholds = {
            'gloves': 120,       # 손목에서 120px 이내
            'arm_covers': 150,   # 팔꿈치에서 150px 이내
            'shoe_covers': 100,  # 발목에서 100px 이내
            'goggles': 80,       # 눈에서 80px 이내
            'mask': 100,         # 코에서 100px 이내
        }
    
    def map_masks_to_labels(self, masks, keypoints, body_regions):
        """
        마스크에 라벨 할당
        
        Args:
            masks (list): 마스크 리스트
            keypoints (np.ndarray): [17, 3] 키포인트 배열
            body_regions (dict): 신체 영역 정보
            
        Returns:
            list of dict: 라벨이 할당된 마스크
        """
        labeled_masks = []
        
        for mask in masks:
            label_name, label_id, confidence = self.assign_label(mask, keypoints, body_regions)
            
            mask['label'] = label_name
            mask['label_id'] = label_id
            mask['label_confidence'] = confidence
            
            labeled_masks.append(mask)
        
        # 통계 출력
        label_counts = {}
        for mask in labeled_masks:
            label = mask['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"[MaskMapper] Labeled {len(labeled_masks)} masks:")
        for label, count in sorted(label_counts.items()):
            print(f"[MaskMapper]   {label}: {count} masks")
        
        return labeled_masks
    
    def assign_label(self, mask, keypoints, body_regions):
        """
        단일 마스크에 라벨 할당 (거리 기반 우선)
        
        우선순위:
        1. 거리 기반 매핑 (장갑, 토시, 신발)
        2. 영역 기반 매핑 (보안경, 마스크, 두건)
        3. 면적 기반 매핑 (상의, 하의)
        4. 가장 가까운 영역
        
        Args:
            mask (dict): 마스크 정보
            keypoints (np.ndarray): 키포인트
            body_regions (dict): 신체 영역
            
        Returns:
            tuple: (label_name, label_id, confidence)
        """
        centroid = mask.get('centroid')
        area = mask.get('area', 0)
        
        if centroid is None:
            return ('unknown', -1, 0.0)
        
        cx, cy = centroid
        
        # ========== 1단계: 거리 기반 매핑 (사지 부위) ==========
        
        # 1-1. 신발 커버 (발목 근처 + Y 위치가 하단)
        shoe_result = self.check_shoe_by_distance(centroid, keypoints, area)
        if shoe_result:
            return shoe_result
        
        # 1-2. 장갑 (손목 근처 + 면적이 작은 편)
        glove_result = self.check_gloves_by_distance(centroid, keypoints, area)
        if glove_result:
            return glove_result
        
        # 1-3. 토시 (팔꿈치~손목 사이 + 중간 면적)
        arm_result = self.check_arm_covers_by_distance(centroid, keypoints, area)
        if arm_result:
            return arm_result
        
        # ========== 2단계: 영역 기반 매핑 (얼굴/머리) ==========
        
        # 2-1. 보안경 (눈 영역)
        goggles_result = self.check_goggles_by_region(centroid, mask, keypoints, body_regions)
        if goggles_result:
            return goggles_result
        
        # 2-2. 마스크 (코/입 영역)
        mask_result = self.check_mask_by_region(centroid, mask, keypoints, body_regions)
        if mask_result:
            return mask_result
        
        # 2-3. 두건 복면 (머리~어깨)
        head_cover_result = self.check_head_cover_by_region(centroid, mask, body_regions)
        if head_cover_result:
            return head_cover_result
        
        # ========== 3단계: 면적 기반 매핑 (상의/하의) ==========
        
        torso_result = self.check_body_by_overlap(mask, keypoints, body_regions)
        if torso_result:
            return torso_result
        
        # ========== 4단계: 가장 가까운 영역 ==========
        
        closest = self.find_closest_region(centroid, keypoints)
        if closest:
            return closest
        
        return ('unknown', -1, 0.0)
    
    def check_shoe_by_distance(self, centroid, keypoints, area):
        """신발 커버 - 발목 기준 거리 매핑"""
        cx, cy = centroid
        threshold = self.distance_thresholds['shoe_covers']
        
        # 왼발
        left_ankle = keypoints[15]
        if left_ankle[2] >= 0.3:
            dist = np.sqrt((cx - left_ankle[0])**2 + (cy - left_ankle[1])**2)
            # 발목보다 아래에 있고 거리가 가까우면
            if dist <= threshold and cy >= left_ankle[1]:
                confidence = 1 - (dist / threshold)
                return ('shoe_covers', 7, 0.8 + 0.2 * confidence)
        
        # 오른발
        right_ankle = keypoints[16]
        if right_ankle[2] >= 0.3:
            dist = np.sqrt((cx - right_ankle[0])**2 + (cy - right_ankle[1])**2)
            if dist <= threshold and cy >= right_ankle[1]:
                confidence = 1 - (dist / threshold)
                return ('shoe_covers', 7, 0.8 + 0.2 * confidence)
        
        return None
    
    def check_gloves_by_distance(self, centroid, keypoints, area):
        """장갑 - 손목 기준 거리 매핑"""
        cx, cy = centroid
        threshold = self.distance_thresholds['gloves']
        
        # 장갑은 보통 작은 마스크 (면적 50000px 이하)
        if area > 50000:
            return None
        
        # 왼손
        left_wrist = keypoints[9]
        if left_wrist[2] >= 0.3:
            dist = np.sqrt((cx - left_wrist[0])**2 + (cy - left_wrist[1])**2)
            if dist <= threshold:
                confidence = 1 - (dist / threshold)
                return ('gloves', 5, 0.85 + 0.15 * confidence)
        
        # 오른손
        right_wrist = keypoints[10]
        if right_wrist[2] >= 0.3:
            dist = np.sqrt((cx - right_wrist[0])**2 + (cy - right_wrist[1])**2)
            if dist <= threshold:
                confidence = 1 - (dist / threshold)
                return ('gloves', 5, 0.85 + 0.15 * confidence)
        
        return None
    
    def check_arm_covers_by_distance(self, centroid, keypoints, area):
        """토시 - 팔꿈치~손목 사이 거리 매핑"""
        cx, cy = centroid
        threshold = self.distance_thresholds['arm_covers']
        
        # 토시도 비교적 작은 마스크 (면적 100000px 이하)
        if area > 100000:
            return None
        
        # 왼팔 (팔꿈치~손목 중간점)
        left_elbow = keypoints[7]
        left_wrist = keypoints[9]
        if left_elbow[2] >= 0.3 and left_wrist[2] >= 0.3:
            # 팔꿈치~손목 중간점
            mid_x = (left_elbow[0] + left_wrist[0]) / 2
            mid_y = (left_elbow[1] + left_wrist[1]) / 2
            dist = np.sqrt((cx - mid_x)**2 + (cy - mid_y)**2)
            
            # 팔꿈치 근처이면서 손목보다 떨어져 있으면 토시
            elbow_dist = np.sqrt((cx - left_elbow[0])**2 + (cy - left_elbow[1])**2)
            wrist_dist = np.sqrt((cx - left_wrist[0])**2 + (cy - left_wrist[1])**2)
            
            if dist <= threshold and elbow_dist < wrist_dist:
                confidence = 1 - (dist / threshold)
                return ('arm_covers', 6, 0.75 + 0.25 * confidence)
        
        # 오른팔
        right_elbow = keypoints[8]
        right_wrist = keypoints[10]
        if right_elbow[2] >= 0.3 and right_wrist[2] >= 0.3:
            mid_x = (right_elbow[0] + right_wrist[0]) / 2
            mid_y = (right_elbow[1] + right_wrist[1]) / 2
            dist = np.sqrt((cx - mid_x)**2 + (cy - mid_y)**2)
            
            elbow_dist = np.sqrt((cx - right_elbow[0])**2 + (cy - right_elbow[1])**2)
            wrist_dist = np.sqrt((cx - right_wrist[0])**2 + (cy - right_wrist[1])**2)
            
            if dist <= threshold and elbow_dist < wrist_dist:
                confidence = 1 - (dist / threshold)
                return ('arm_covers', 6, 0.75 + 0.25 * confidence)
        
        return None
    
    def check_goggles_by_region(self, centroid, mask, keypoints, body_regions):
        """보안경 - 눈 영역 + 거리"""
        cx, cy = centroid
        threshold = self.distance_thresholds['goggles']
        
        # 코 위치 기준
        nose = keypoints[0]
        if nose[2] >= 0.3:
            # 코보다 위에 있고 거리가 가까우면
            dist = np.sqrt((cx - nose[0])**2 + (cy - nose[1])**2)
            if dist <= threshold and cy < nose[1]:
                overlap = self.calculate_region_overlap(mask, body_regions.get('goggles_region'))
                if overlap > 0.3:
                    return ('goggles', 1, 0.9)
        
        # 영역 기반 체크
        if self.is_in_region(centroid, body_regions.get('goggles_region')):
            overlap = self.calculate_region_overlap(mask, body_regions.get('goggles_region'))
            if overlap > 0.4:
                return ('goggles', 1, overlap)
        
        return None
    
    def check_mask_by_region(self, centroid, mask, keypoints, body_regions):
        """마스크 - 코/입 영역"""
        cx, cy = centroid
        threshold = self.distance_thresholds['mask']
        
        # 코 위치 기준
        nose = keypoints[0]
        if nose[2] >= 0.3:
            # 코 아래에 있고 거리가 가까우면
            dist = np.sqrt((cx - nose[0])**2 + (cy - nose[1])**2)
            if dist <= threshold and cy >= nose[1]:
                overlap = self.calculate_region_overlap(mask, body_regions.get('mask_region'))
                if overlap > 0.3:
                    return ('mask', 2, 0.9)
        
        return None
    
    def check_head_cover_by_region(self, centroid, mask, body_regions):
        """두건 복면 - 머리~어깨 영역"""
        if self.is_in_region(centroid, body_regions.get('head_cover_region')):
            overlap = self.calculate_region_overlap(mask, body_regions.get('head_cover_region'))
            if overlap > 0.25:
                return ('head_cover', 0, overlap)
        
        return None
    
    def check_body_by_overlap(self, mask, keypoints, body_regions):
        """상의/하의 - 겹침 기반"""
        centroid = mask.get('centroid')
        area = mask.get('area', 0)
        
        # 매우 큰 마스크만 상의/하의 (전체 이미지의 5% 이상)
        if area < 30000:
            return None
        
        torso_overlap = self.calculate_region_overlap(mask, body_regions.get('upper_body_region'))
        pants_overlap = self.calculate_region_overlap(mask, body_regions.get('pants_region'))
        
        if torso_overlap > pants_overlap and torso_overlap > 0.35:
            return ('upper_body', 3, torso_overlap)
        elif pants_overlap > 0.35:
            return ('pants', 4, pants_overlap)
        
        return None
    
    def find_closest_region(self, centroid, keypoints):
        """가장 가까운 키포인트 기반 분류"""
        cx, cy = centroid
        
        # 키포인트별 거리 계산
        distances = []
        
        # 손목 (장갑)
        for idx, name in [(9, 'gloves'), (10, 'gloves')]:
            kp = keypoints[idx]
            if kp[2] >= 0.3:
                dist = np.sqrt((cx - kp[0])**2 + (cy - kp[1])**2)
                distances.append((name, dist, 5))
        
        # 팔꿈치 (토시)
        for idx, name in [(7, 'arm_covers'), (8, 'arm_covers')]:
            kp = keypoints[idx]
            if kp[2] >= 0.3:
                dist = np.sqrt((cx - kp[0])**2 + (cy - kp[1])**2)
                distances.append((name, dist, 6))
        
        # 발목 (신발)
        for idx, name in [(15, 'shoe_covers'), (16, 'shoe_covers')]:
            kp = keypoints[idx]
            if kp[2] >= 0.3:
                dist = np.sqrt((cx - kp[0])**2 + (cy - kp[1])**2)
                distances.append((name, dist, 7))
        
        if not distances:
            return None
        
        # 가장 가까운 것
        closest = min(distances, key=lambda x: x[1])
        
        # 250px 이내면 할당
        if closest[1] <= 250:
            confidence = max(0.5, 1 - closest[1] / 250)
            return (closest[0], closest[2], confidence)
        
        return None
    
    def is_in_region(self, centroid, region):
        """중심점이 영역 안에 있는지 확인"""
        if region is None:
            return False
        
        x, y = centroid
        x1, y1, x2, y2 = region
        
        return (x1 <= x <= x2) and (y1 <= y <= y2)
    
    def calculate_region_overlap(self, mask, region):
        """영역 겹침 계산"""
        if region is None:
            return 0.0
        
        x1, y1, x2, y2 = region
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        segmentation = mask['segmentation']
        
        if y2 > segmentation.shape[0]:
            y2 = segmentation.shape[0]
        if x2 > segmentation.shape[1]:
            x2 = segmentation.shape[1]
        if y1 < 0:
            y1 = 0
        if x1 < 0:
            x1 = 0
        
        if y2 <= y1 or x2 <= x1:
            return 0.0
        
        mask_in_region = segmentation[y1:y2, x1:x2]
        intersection = np.sum(mask_in_region)
        
        mask_area = mask['area']
        if mask_area == 0:
            return 0.0
        
        return intersection / mask_area
    
    def get_label_name(self, label_id):
        """라벨 ID로 이름 반환"""
        return self.label_names.get(label_id, 'unknown')
    
    def get_label_id(self, label_name):
        """라벨 이름으로 ID 반환"""
        return self.label_ids.get(label_name, -1)
