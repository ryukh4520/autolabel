"""
SAM-HQ Segmentation Module

이 모듈은 SAM-HQ ViT-L을 사용하여:
1. 이미지에서 모든 객체를 자동 세그멘테이션
2. 마스크 필터링 (크기, 품질 기반)
3. 마스크 중심점 계산
"""

import numpy as np
import torch
import yaml
from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator


class SAMHQSegmenter:
    """
    SAM-HQ 기반 자동 세그멘테이션 클래스
    """
    
    def __init__(self, model_path, device='cuda', config_path=None):
        """
        Args:
            model_path (str): SAM-HQ 모델 경로
            device (str): 'cuda' or 'cpu'
            config_path (str): 설정 파일 경로 (옵션)
        """
        self.device = device
        
        # 설정 로드
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                sam_config = config.get('sam', {})
                self.filter_config = config.get('mask_filter', {})
        else:
            # 기본 설정
            sam_config = {
                'points_per_side': 32,
                'pred_iou_thresh': 0.88,
                'stability_score_thresh': 0.95
            }
            self.filter_config = {
                'min_area': 500,
                'max_area_ratio': 0.5,
                'person_overlap_threshold': 0.3
            }
        
        # 모델 타입 결정 (파일명에서 추출)
        if 'vit_h' in model_path.lower():
            model_type = "vit_h"
        elif 'vit_l' in model_path.lower():
            model_type = "vit_l"
        elif 'vit_b' in model_path.lower():
            model_type = "vit_b"
        else:
            model_type = "vit_l"  # 기본값
        
        print(f"[SAMHQSegmenter] Loading SAM-HQ {model_type.upper()} model from {model_path}...")
        
        # SAM-HQ 모델 로드
        self.sam = sam_model_registry[model_type](checkpoint=model_path)
        self.sam.to(device=device)
        
        # Mask Generator 생성
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=sam_config.get('points_per_side', 32),
            pred_iou_thresh=sam_config.get('pred_iou_thresh', 0.88),
            stability_score_thresh=sam_config.get('stability_score_thresh', 0.95),
        )
        
        print(f"[SAMHQSegmenter] Model loaded on {device}")
        
        # VRAM 사용량 출력
        if device == 'cuda' and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"[SAMHQSegmenter] VRAM allocated: {allocated:.2f} GB")
    
    def generate_masks(self, image, person_mask=None):
        """
        이미지에서 모든 마스크 생성
        
        Args:
            image (np.ndarray): RGB 이미지
            person_mask (np.ndarray): 사람 영역 마스크 (옵션)
            
        Returns:
            list of dict: 생성된 마스크 정보
                {
                    'segmentation': np.array (bool mask),
                    'area': int,
                    'bbox': [x, y, w, h],
                    'predicted_iou': float,
                    'stability_score': float
                }
        """
        print(f"[SAMHQSegmenter] Generating masks for image {image.shape}...")
        
        # 사람 마스크가 제공되면 해당 영역만 처리
        if person_mask is not None:
            print(f"[SAMHQSegmenter] Using person mask (area: {np.sum(person_mask)} px)")
            # 마스크 적용된 이미지 생성
            masked_image = image.copy()
            masked_image[~person_mask] = 0
            masks = self.mask_generator.generate(masked_image)
            
            # 생성된 마스크를 person_mask와 교집합
            filtered_masks = []
            for mask in masks:
                seg = mask['segmentation']
                # person_mask와 교집합
                intersected = np.logical_and(seg, person_mask)
                intersection_area = np.sum(intersected)
                
                # 교집합이 원래 마스크의 50% 이상이면 유지
                if intersection_area >= mask['area'] * 0.5:
                    mask['segmentation'] = intersected
                    mask['area'] = int(intersection_area)
                    filtered_masks.append(mask)
            
            print(f"[SAMHQSegmenter] Generated {len(masks)} masks, filtered to {len(filtered_masks)} (person area only)")
            return filtered_masks
        else:
            masks = self.mask_generator.generate(image)
            print(f"[SAMHQSegmenter] Generated {len(masks)} masks")
            return masks

    def generate_masks_from_crop(self, image, crop_box):
        """
        BBox로 잘린(Crop) 이미지 영역에 대해 마스크를 생성하고,
        다시 원본 이미지 좌표계로 복원하여 반환함.
        
        Args:
            image (np.ndarray): 원본 전체 이미지
            crop_box (tuple): (x, y, w, h) 형태의 Crop 영역
            
        Returns:
            list: 원본 좌표계로 변환된 마스크 리스트
        """
        x, y, w, h = crop_box
        orig_h, orig_w = image.shape[:2]
        
        # 1. 안전한 Crop (범위 체크)
        x = max(0, int(x))
        y = max(0, int(y))
        w = min(int(w), orig_w - x)
        h = min(int(h), orig_h - y)
        
        if w <= 0 or h <= 0:
            return []
            
        crop_img = image[y:y+h, x:x+w]
        
        # 2. 마스크 생성 (Crop된 작은 이미지 기준)
        # 여기서 생성된 마스크 좌표는 (0,0) ~ (w,h) 기준임
        print(f"[SAMHQSegmenter] Generating masks from crop: {w}x{h}")
        masks = self.mask_generator.generate(crop_img)
        
        # 3. 좌표 및 마스크 복원 (원본 좌표계로)
        restored_masks = []
        for mask in masks:
            # (1) Segmentation 복원
            # 원본 크기의 빈 캔버스 생성
            full_mask = np.zeros((orig_h, orig_w), dtype=bool)
            # 해당 위치에 붙여넣기
            full_mask[y:y+h, x:x+w] = mask['segmentation']
            mask['segmentation'] = full_mask
            
            # (2) BBox 복원
            bx, by, bw, bh = mask['bbox']
            mask['bbox'] = [bx + x, by + y, bw, bh]
            
            # (3) Point Coords 복원 (crop_nms_thresh 등에서 사용)
            if 'point_coords' in mask:
                if isinstance(mask['point_coords'], list):
                    coords = np.array(mask['point_coords'])
                    coords[:, 0] += x
                    coords[:, 1] += y
                    mask['point_coords'] = coords.tolist()
                elif isinstance(mask['point_coords'], np.ndarray):
                    mask['point_coords'][:, 0] += x
                    mask['point_coords'][:, 1] += y
                    
            # (4) Crop Box 정보 추가 (디버깅용)
            mask['crop_box'] = [x, y, w, h]
            
            restored_masks.append(mask)
            
        # 4. 필터링 (너무 작거나 큰 노이즈는 제거)
        filtered = self.filter_masks(restored_masks, (orig_h, orig_w))
        
        return filtered
    
    def filter_masks(self, masks, image_shape, min_area=None, max_area_ratio=None):
        """
        마스크 필터링 (크기 기반)
        
        Args:
            masks (list): 마스크 리스트
            image_shape (tuple): (height, width, channels)
            min_area (int): 최소 면적 (pixels)
            max_area_ratio (float): 최대 면적 비율 (0.0 ~ 1.0)
            
        Returns:
            list of dict: 필터링된 마스크
        """
        if min_area is None:
            min_area = self.filter_config.get('min_area', 500)
        if max_area_ratio is None:
            max_area_ratio = self.filter_config.get('max_area_ratio', 0.5)
        
        height, width = image_shape[:2]
        image_area = height * width
        max_area = image_area * max_area_ratio
        
        filtered = []
        
        for mask in masks:
            area = mask['area']
            
            # 너무 작은 마스크 제거 (노이즈)
            if area < min_area:
                continue
            
            # 너무 큰 마스크 제거 (배경)
            if area > max_area:
                continue
            
            filtered.append(mask)
        
        print(f"[SAMHQSegmenter] Filtered {len(masks)} -> {len(filtered)} masks")
        print(f"[SAMHQSegmenter] Removed: {len(masks) - len(filtered)} masks")
        print(f"[SAMHQSegmenter]   - Too small (< {min_area}px)")
        print(f"[SAMHQSegmenter]   - Too large (> {max_area_ratio*100:.0f}% of image)")
        
        return filtered
    
    def calculate_centroids(self, masks):
        """
        각 마스크의 중심점 계산
        
        Args:
            masks (list): 마스크 리스트
            
        Returns:
            list of dict: centroid 정보가 추가된 마스크
        """
        for mask in masks:
            segmentation = mask['segmentation']
            
            # 마스크의 True 픽셀 좌표 찾기
            y_coords, x_coords = np.where(segmentation)
            
            if len(y_coords) > 0:
                # 중심점 계산 (평균)
                centroid_x = np.mean(x_coords)
                centroid_y = np.mean(y_coords)
                mask['centroid'] = [float(centroid_x), float(centroid_y)]
            else:
                mask['centroid'] = None
        
        return masks
    
    def process(self, image, person_mask=None, filter_masks=True, calculate_centroids=True):
        """
        전체 처리 파이프라인
        
        Args:
            image (np.ndarray): RGB 이미지
            person_mask (np.ndarray): 사람 영역 마스크 (옵션)
            filter_masks (bool): 마스크 필터링 여부
            calculate_centroids (bool): 중심점 계산 여부
            
        Returns:
            list of dict: 처리된 마스크 리스트
        """
        # 1. 마스크 생성
        masks = self.generate_masks(image, person_mask=person_mask)
        
        # 2. 필터링
        if filter_masks:
            masks = self.filter_masks(masks, image.shape)
        
        # 3. 중심점 계산
        if calculate_centroids:
            masks = self.calculate_centroids(masks)
        
        return masks
    
    def get_mask_stats(self, masks):
        """
        마스크 통계 정보 반환
        
        Returns:
            dict: 통계 정보
        """
        if len(masks) == 0:
            return {
                'count': 0,
                'total_area': 0,
                'avg_area': 0,
                'min_area': 0,
                'max_area': 0
            }
        
        areas = [m['area'] for m in masks]
        
        return {
            'count': len(masks),
            'total_area': sum(areas),
            'avg_area': np.mean(areas),
            'min_area': min(areas),
            'max_area': max(areas),
            'avg_iou': np.mean([m['predicted_iou'] for m in masks]),
            'avg_stability': np.mean([m['stability_score'] for m in masks])
        }
