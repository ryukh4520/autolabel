"""
Person Filter Module

이 모듈은:
1. SAM-HQ 마스크 중 사람 영역 내부의 마스크만 선택
2. 배경 객체(건물, 차량 등) 제거
3. IoU 기반 겹침 계산
"""

import numpy as np


class PersonFilter:
    """
    Person 영역 기반 마스크 필터링 클래스
    """
    
    def __init__(self, overlap_threshold=0.3):
        """
        Args:
            overlap_threshold (float): 마스크와 person bbox 겹침 최소 비율
        """
        self.overlap_threshold = overlap_threshold
    
    def filter_by_person(self, masks, person_bbox):
        """
        Person bbox와 겹치는 마스크만 선택
        
        Args:
            masks (list): 마스크 리스트
            person_bbox (list): [x1, y1, x2, y2] Person bounding box
            
        Returns:
            list of dict: 필터링된 마스크
        """
        if person_bbox is None:
            print("[PersonFilter] No person bbox provided, returning all masks")
            return masks
        
        filtered = []
        
        for mask in masks:
            overlap = self.calculate_overlap(mask, person_bbox)
            
            if overlap >= self.overlap_threshold:
                mask['person_overlap'] = overlap
                filtered.append(mask)
        
        print(f"[PersonFilter] Filtered {len(masks)} -> {len(filtered)} masks")
        print(f"[PersonFilter] Removed {len(masks) - len(filtered)} background masks")
        
        return filtered
    
    def calculate_overlap(self, mask, bbox):
        """
        마스크와 bbox의 겹침 비율 계산
        
        Args:
            mask (dict): 마스크 정보 (segmentation 포함)
            bbox (list): [x1, y1, x2, y2]
            
        Returns:
            float: 겹침 비율 (0.0 ~ 1.0)
        """
        # Person bbox 영역
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 마스크 전체 면적
        mask_area = mask['area']
        
        if mask_area == 0:
            return 0.0
        
        # 마스크와 bbox 교집합
        segmentation = mask['segmentation']
        
        # bbox 영역 내의 마스크 픽셀 수
        mask_in_bbox = segmentation[y1:y2, x1:x2]
        intersection = np.sum(mask_in_bbox)
        
        # 겹침 비율 = 교집합 / 마스크 면적
        overlap_ratio = intersection / mask_area
        
        return overlap_ratio
    
    def filter_by_multiple_persons(self, masks, person_bboxes):
        """
        여러 사람이 있을 때 각 마스크를 가장 가까운 사람에게 할당
        
        Args:
            masks (list): 마스크 리스트
            person_bboxes (list): Person bounding box 리스트
            
        Returns:
            dict: {person_idx: [masks]} 형태로 할당된 마스크
        """
        assigned_masks = {i: [] for i in range(len(person_bboxes))}
        
        for mask in masks:
            best_person_idx = -1
            best_overlap = 0
            
            # 각 사람과의 겹침 계산
            for i, bbox in enumerate(person_bboxes):
                overlap = self.calculate_overlap(mask, bbox)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_person_idx = i
            
            # 임계값 이상이면 할당
            if best_overlap >= self.overlap_threshold:
                mask['person_overlap'] = best_overlap
                mask['person_idx'] = best_person_idx
                assigned_masks[best_person_idx].append(mask)
        
        # 통계 출력
        total_assigned = sum(len(masks) for masks in assigned_masks.values())
        print(f"[PersonFilter] Assigned {total_assigned}/{len(masks)} masks to {len(person_bboxes)} persons")
        for i, person_masks in assigned_masks.items():
            print(f"[PersonFilter]   Person {i}: {len(person_masks)} masks")
        
        return assigned_masks
    
    def get_person_region_masks(self, masks, person_bbox, body_regions):
        """
        Person 영역 내 마스크를 신체 영역별로 분류
        
        Args:
            masks (list): 필터링된 마스크 리스트
            person_bbox (list): Person bounding box
            body_regions (dict): 신체 영역 정보
            
        Returns:
            dict: {region_name: [masks]} 형태로 분류된 마스크
        """
        region_masks = {
            'head': [],
            'hands': [],
            'torso': [],
            'legs': [],
            'feet': [],
            'unknown': []
        }
        
        for mask in masks:
            centroid = mask.get('centroid')
            if centroid is None:
                region_masks['unknown'].append(mask)
                continue
            
            cx, cy = centroid
            
            # 각 신체 영역과의 겹침 확인
            assigned = False
            
            # 헬멧 영역
            if body_regions.get('head_region'):
                x1, y1, x2, y2 = body_regions['head_region']
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    region_masks['head'].append(mask)
                    assigned = True
                    continue
            
            # 장갑 영역 (원형)
            for hand_key in ['left_hand_region', 'right_hand_region']:
                if body_regions.get(hand_key):
                    hx, hy, radius = body_regions[hand_key]
                    dist = np.sqrt((cx - hx)**2 + (cy - hy)**2)
                    if dist <= radius:
                        region_masks['hands'].append(mask)
                        assigned = True
                        break
            
            if assigned:
                continue
            
            # 상의 영역
            if body_regions.get('torso_region'):
                x1, y1, x2, y2 = body_regions['torso_region']
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    region_masks['torso'].append(mask)
                    assigned = True
                    continue
            
            # 하의 영역
            if body_regions.get('pants_region'):
                x1, y1, x2, y2 = body_regions['pants_region']
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    region_masks['legs'].append(mask)
                    assigned = True
                    continue
            
            # 신발 영역
            if body_regions.get('boots_region'):
                x1, y1, x2, y2 = body_regions['boots_region']
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    region_masks['feet'].append(mask)
                    assigned = True
                    continue
            
            # 할당되지 않은 마스크
            if not assigned:
                region_masks['unknown'].append(mask)
        
        # 통계 출력
        print(f"[PersonFilter] Masks by body region:")
        for region, masks_list in region_masks.items():
            if len(masks_list) > 0:
                print(f"[PersonFilter]   {region}: {len(masks_list)} masks")
        
        return region_masks
