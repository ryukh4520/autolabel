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
        
        # 0단계: 힙 라인 기준으로 큰 마스크 분할 (상의/하의 분리)
        masks = self.split_large_masks(masks, keypoints)
        
        for mask in masks:
            label_name, label_id, confidence = self.assign_label(mask, keypoints, body_regions)
            
            mask['label'] = label_name
            mask['label_id'] = label_id
            mask['label_confidence'] = confidence
            
            labeled_masks.append(mask)
            
        # [Refinement] 라벨 재검증 (오분류 수정)
        # 1. upper_body로 분류된 마스크 중, 팔 벡터에 가깝고 크기가 작으면 arm_covers로 변경
        for mask in labeled_masks:
            if mask['label'] == 'upper_body':
                # 면적이 80000 이하면 의심 (상체 전체는 보통 20만 이상)
                if mask['area'] < 80000:
                    # check_arm_covers_by_distance 호출
                    # (이미 상체 내부에 있음이 보장되므로, 팔 벡터 근처이기만 하면 됨)
                    arm_res = self.check_arm_covers_by_distance(mask['centroid'], keypoints, mask['area'])
                    # check_arm_covers_by_distance 내부에서 area 체크(100000이하)도 하므로 안전
                    
                    if arm_res:
                        print(f"[MaskMapper] Refined label: upper_body -> arm_covers (Area: {mask['area']})")
                        mask['label'] = arm_res[0]
                        mask['label_id'] = arm_res[1]
                        mask['label_confidence'] = arm_res[2]

        # 통계 출력
        label_counts = {}
        for mask in labeled_masks:
            label = mask['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"[MaskMapper] Labeled {len(labeled_masks)} masks:")
        for label, count in sorted(label_counts.items()):
            print(f"[MaskMapper]   {label}: {count} masks")
        
        return labeled_masks
            
    def split_large_masks(self, masks, keypoints):
        """
        힙 라인(키포인트 11, 12) 기준으로 큰 마스크 분할
        """
        # 힙 키포인트 (11: left_hip, 12: right_hip)
        l_hip = keypoints[11]
        r_hip = keypoints[12]
        
        # 힙 키포인트가 유효하지 않으면 분할하지 않음
        if l_hip[2] < 0.3 or r_hip[2] < 0.3:
            return masks
            
        # 힙 라인 Y좌표 (허리선으로 간주)
        waist_y = (l_hip[1] + r_hip[1]) / 2
        
        new_masks = []
        
        for mask in masks:
            area = mask.get('area', 0)
            bbox = mask.get('bbox', None)
            
            should_split = False
            
            # 면적이 충분히 크고 (예: > 30000), bbox가 유효한 경우
            if area > 30000 and bbox:
                x, y, w, h = bbox
                
                # 마스크가 waist_y 위아래로 걸쳐 있는지 확인
                # 위쪽으로 최소 20%, 아래쪽으로 최소 20% 뻗어 있어야 함
                if y < waist_y and (y + h) > waist_y:
                    top_h = waist_y - y
                    bottom_h = (y + h) - waist_y
                    
                    if top_h > h * 0.2 and bottom_h > h * 0.2:
                        should_split = True
            
            if should_split:
                # 마스크 분할 수행
                seg = mask['segmentation']
                
                # 상체 마스크 (waist_y 위쪽)
                upper_seg = seg.copy()
                upper_seg[int(waist_y):, :] = False
                
                # 하체 마스크 (waist_y 아래쪽)
                lower_seg = seg.copy()
                lower_seg[:int(waist_y), :] = False
                
                # 유효한 분할인지 확인 (너무 작은 조각이 되면 무시)
                upper_area = np.sum(upper_seg)
                lower_area = np.sum(lower_seg)
                
                if upper_area > 1000 and lower_area > 1000:
                    print(f"[MaskMapper] Splitting mask (area: {area}) at y={waist_y:.1f} -> Upper: {upper_area}, Lower: {lower_area}")
                    
                    # 상체 마스크 추가
                    upper_mask = mask.copy()
                    upper_mask['segmentation'] = upper_seg
                    upper_mask['area'] = int(upper_area)
                    upper_mask['centroid'] = self._calculate_centroid(upper_seg)
                    upper_mask['bbox'] = self._get_bbox(upper_seg)
                    new_masks.append(upper_mask)
                    
                    # 하체 마스크 추가
                    lower_mask = mask.copy()
                    lower_mask['segmentation'] = lower_seg
                    lower_mask['area'] = int(lower_area)
                    lower_mask['centroid'] = self._calculate_centroid(lower_seg)
                    lower_mask['bbox'] = self._get_bbox(lower_seg)
                    new_masks.append(lower_mask)
                else:
                    # 분할 실패 시 원본 유지
                    new_masks.append(mask)
            else:
                new_masks.append(mask)
                
        return new_masks

    def _calculate_centroid(self, segmentation):
        """세그멘테이션 마스크의 중심점 계산"""
        y_coords, x_coords = np.where(segmentation)
        if len(y_coords) > 0:
            return [float(np.mean(x_coords)), float(np.mean(y_coords))]
        return None

    def _get_bbox(self, segmentation):
        """세그멘테이션 마스크의 Bounding Box 계산 [x, y, w, h]"""
        y_coords, x_coords = np.where(segmentation)
        if len(y_coords) > 0:
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
        return [0, 0, 0, 0]
        
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
        """
        신발 커버 - 무릎~발목(13-16) 관계 기반 매핑
        제안: 무릎과 발목의 중간 지점보다 아래에 있으면 신발로 간주
        """
        cx, cy = centroid
        # 거리 임계값을 조금 여유 있게 적용 (부츠형 커버 고려)
        threshold = self.distance_thresholds['shoe_covers'] * 1.5
        
        # 키포인트 인덱스
        # 13: L.Knee, 15: L.Ankle
        # 14: R.Knee, 16: R.Ankle
        
        # 왼발 검사
        l_knee = keypoints[13]
        l_ankle = keypoints[15]
        
        if l_knee[2] >= 0.3 and l_ankle[2] >= 0.3:
            # 종아리 중간 지점 (Calf center)
            calf_y = (l_knee[1] + l_ankle[1]) / 2
            
            # 마스크 중심이 종아리 중간보다 아래에 있어야 함
            if cy > calf_y:
                dist = np.sqrt((cx - l_ankle[0])**2 + (cy - l_ankle[1])**2)
                if dist <= threshold:
                    confidence = 1 - (dist / threshold)
                    return ('shoe_covers', 7, 0.8 + 0.2 * confidence)
        
        # 오른발 검사
        r_knee = keypoints[14]
        r_ankle = keypoints[16]
        
        if r_knee[2] >= 0.3 and r_ankle[2] >= 0.3:
            calf_y = (r_knee[1] + r_ankle[1]) / 2
            
            if cy > calf_y:
                dist = np.sqrt((cx - r_ankle[0])**2 + (cy - r_ankle[1])**2)
                if dist <= threshold:
                    confidence = 1 - (dist / threshold)
                    return ('shoe_covers', 7, 0.8 + 0.2 * confidence)
                
        return None
    
    def check_gloves_by_distance(self, centroid, keypoints, area):
        """
        장갑 - 벡터 기반 매핑 (손목 너머에 위치)
        """
        # 장갑은 보통 작은 마스크 (면적 60000px 이하로 완화)
        if area > 60000:
            return None
            
        cx, cy = centroid
        threshold = self.distance_thresholds['gloves']
        
        # 왼손 (팔꿈치 -> 손목 벡터)
        l_elbow = keypoints[7][:2]
        l_wrist = keypoints[9][:2]
        if keypoints[7][2] > 0.3 and keypoints[9][2] > 0.3:
            dist = np.sqrt((cx - l_wrist[0])**2 + (cy - l_wrist[1])**2)
            if dist <= threshold:
                # 벡터 투영 (t > 0.9 이면 손목 근처나 그 너머)
                t = self._project_point_on_line(centroid, l_elbow, l_wrist)
                if t > 0.85:
                    return ('gloves', 5, 0.9)

        # 오른손
        r_elbow = keypoints[8][:2]
        r_wrist = keypoints[10][:2]
        if keypoints[8][2] > 0.3 and keypoints[10][2] > 0.3:
            dist = np.sqrt((cx - r_wrist[0])**2 + (cy - r_wrist[1])**2)
            if dist <= threshold:
                t = self._project_point_on_line(centroid, r_elbow, r_wrist)
                if t > 0.85:
                    return ('gloves', 5, 0.9)
        
        return None
    
    def check_arm_covers_by_distance(self, centroid, keypoints, area):
        """
        토시 - 벡터 기반 매핑 (팔꿈치와 손목 사이)
        """
        # 토시도 비교적 작은 마스크
        if area > 100000:
            return None
            
        cx, cy = centroid
        threshold = self.distance_thresholds['arm_covers']
        
        # 왼팔
        l_elbow = keypoints[7][:2]
        l_wrist = keypoints[9][:2]
        if keypoints[7][2] > 0.3 and keypoints[9][2] > 0.3:
            # 거리 체크 (팔꿈치~손목 중간점 기준 안씀, 선분과의 거리로 변경 가능하나 일단 유지)
            # 여기서는 벡터 투영 비율이 핵심
            t = self._project_point_on_line(centroid, l_elbow, l_wrist)
            
            # 팔꿈치(0.0) ~ 손목(1.0) 사이의 20% ~ 90% 구간
            if 0.15 < t <= 0.85:
                # 선분과의 수직 거리 체크 (너무 멀리 떨어진 오검출 방지)
                # 간략히 손목이나 팔꿈치 중 하나랑은 거리가 threshold 이내여야 함
                dist_w = np.sqrt((cx - l_wrist[0])**2 + (cy - l_wrist[1])**2)
                dist_e = np.sqrt((cx - l_elbow[0])**2 + (cy - l_elbow[1])**2)
                
                if dist_w <= threshold or dist_e <= threshold:
                    # 손목에 가까울수록(t가 0.85에 가까울수록) 토시 확률 높음
                    return ('arm_covers', 6, 0.8)
        
        # 오른팔
        r_elbow = keypoints[8][:2]
        r_wrist = keypoints[10][:2]
        if keypoints[8][2] > 0.3 and keypoints[10][2] > 0.3:
            t = self._project_point_on_line(centroid, r_elbow, r_wrist)
            
            if 0.15 < t <= 0.85:
                dist_w = np.sqrt((cx - r_wrist[0])**2 + (cy - r_wrist[1])**2)
                dist_e = np.sqrt((cx - r_elbow[0])**2 + (cy - r_elbow[1])**2)
                
                if dist_w <= threshold or dist_e <= threshold:
                    return ('arm_covers', 6, 0.8)
        
        return None

    def _project_point_on_line(self, point, line_start, line_end):
        """
        점을 선분에 투영하여 위치 비율 반환
        return t:
          t < 0: start 이전
          0 <= t <= 1: 선분 위
          t > 1: end 이후
        """
        line_vec = np.array(line_end) - np.array(line_start)
        point_vec = np.array(point) - np.array(line_start)
        
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq == 0:
            return 0.0
            
        t = np.dot(point_vec, line_vec) / line_len_sq
        return t
    
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
