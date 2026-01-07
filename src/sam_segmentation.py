"""
SAM2 Segmentation Module

이 모듈은 SAM2 Large를 사용하여:
1. 이미지에서 모든 객체를 자동 세그멘테이션
2. 마스크 필터링 (크기, 품질 기반)
3. 마스크 중심점 계산
"""

import numpy as np
import torch
import yaml
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class SAMSegmenter:
    """
    SAM2 기반 자동 세그멘테이션 클래스
    """
    
    def __init__(self, model_path, device='cuda', config_path=None):
        """
        Args:
            model_path (str): SAM2 모델 경로
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
        
        # SAM2 모델 로드
        print(f"[SAMSegmenter] Loading SAM2 model from {model_path}...")
        
        # 모델 타입 결정 (파일명에서 추출)
        if 'large' in model_path.lower():
            config_file = "sam2_hiera_l.yaml"
        elif 'base_plus' in model_path.lower() or 'base+' in model_path.lower():
            config_file = "sam2_hiera_b+.yaml"
        elif 'small' in model_path.lower():
            config_file = "sam2_hiera_s.yaml"
        elif 'tiny' in model_path.lower():
            config_file = "sam2_hiera_t.yaml"
        else:
            config_file = "sam2_hiera_b+.yaml"  # 기본값
        
        self.sam2 = build_sam2(
            config_file=config_file,
            ckpt_path=model_path,
            device=device
        )
        
        # Mask Generator 생성
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2,
            points_per_side=sam_config.get('points_per_side', 32),
            pred_iou_thresh=sam_config.get('pred_iou_thresh', 0.88),
            stability_score_thresh=sam_config.get('stability_score_thresh', 0.95),
        )
        
        print(f"[SAMSegmenter] Model loaded on {device}")
        
        # VRAM 사용량 출력
        if device == 'cuda' and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"[SAMSegmenter] VRAM allocated: {allocated:.2f} GB")
    
    def generate_masks(self, image):
        """
        이미지에서 모든 마스크 생성
        
        Args:
            image (np.ndarray): RGB 이미지
            
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
        print(f"[SAMSegmenter] Generating masks for image {image.shape}...")
        masks = self.mask_generator.generate(image)
        print(f"[SAMSegmenter] Generated {len(masks)} masks")
        
        return masks
    
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
        
        print(f"[SAMSegmenter] Filtered {len(masks)} -> {len(filtered)} masks")
        print(f"[SAMSegmenter] Removed: {len(masks) - len(filtered)} masks")
        print(f"[SAMSegmenter]   - Too small (< {min_area}px)")
        print(f"[SAMSegmenter]   - Too large (> {max_area_ratio*100:.0f}% of image)")
        
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
    
    def process(self, image, filter_masks=True, calculate_centroids=True):
        """
        전체 처리 파이프라인
        
        Args:
            image (np.ndarray): RGB 이미지
            filter_masks (bool): 마스크 필터링 여부
            calculate_centroids (bool): 중심점 계산 여부
            
        Returns:
            list of dict: 처리된 마스크 리스트
        """
        # 1. 마스크 생성
        masks = self.generate_masks(image)
        
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
