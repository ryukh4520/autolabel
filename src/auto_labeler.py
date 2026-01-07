"""
Auto Labeler - 통합 파이프라인

이 모듈은 전체 자동 라벨링 파이프라인을 관리:
1. YOLO-Pose로 키포인트 및 신체 영역 추출
2. SAM-HQ로 마스크 생성
3. Person Filter로 배경 제거
4. Mask Mapper로 라벨 할당
"""

import os
import time
import yaml
import cv2
import numpy as np
import torch

from .person_segmenter import PersonSegmenter
from .pose_estimation import PoseEstimator
from .samhq_segmentation import SAMHQSegmenter
from .person_filter import PersonFilter
from .mask_mapper import MaskMapper


class AutoLabeler:
    """
    전체 자동 라벨링 파이프라인을 관리하는 메인 클래스
    """
    
    def __init__(self, config_path='/workspace/config/config.yaml'):
        """
        Args:
            config_path (str): 설정 파일 경로
        """
        print("=" * 80)
        print("AutoLabeler Initialization")
        print("=" * 80)
        
        # 설정 로드
        self.config = self.load_config(config_path)
        
        # 각 모듈 초기화
        print("\n[AutoLabeler] Initializing modules...")
        
        # 1. Person Segmenter (새로 추가!)
        print("\n[1/5] Initializing Person Segmenter...")
        self.person_segmenter = PersonSegmenter(
            model_path='/workspace/models/yolov8s-seg.pt',
            device=self.config['pose']['device'],
            conf_threshold=self.config['pose']['conf_threshold']
        )
        
        # 2. YOLO-Pose
        print("\n[2/5] Initializing YOLO-Pose...")
        self.pose_estimator = PoseEstimator(
            model_path=self.config['pose']['checkpoint'],
            device=self.config['pose']['device'],
            conf_threshold=self.config['pose']['conf_threshold'],
            config_path=config_path
        )
        
        # 3. SAM-HQ
        print("\n[3/5] Initializing SAM-HQ...")
        self.sam_segmenter = SAMHQSegmenter(
            model_path=self.config['sam']['checkpoint'],
            device=self.config['sam']['device'],
            config_path=config_path
        )
        
        # 4. Person Filter
        print("\n[4/5] Initializing Person Filter...")
        self.person_filter = PersonFilter(
            overlap_threshold=self.config['mask_filter']['person_overlap_threshold']
        )
        
        # 5. Mask Mapper
        print("\n[5/5] Initializing Mask Mapper...")
        self.mask_mapper = MaskMapper(config=self.config.get('mapping', {}))
        
        print("\n[AutoLabeler] ✓ All modules initialized successfully")
        print("=" * 80)
    
    def load_config(self, config_path):
        """설정 파일 로드"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_image(self, image_path):
        """이미지 로드 (BGR -> RGB)"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb
    
    def process_image(self, image_path, verbose=True):
        """
        단일 이미지 처리
        
        Args:
            image_path (str): 입력 이미지 경로
            verbose (bool): 상세 로그 출력 여부
            
        Returns:
            dict: 라벨된 결과
                {
                    'image_path': str,
                    'image_shape': tuple,
                    'person_info': dict,
                    'labeled_masks': list,
                    'statistics': dict,
                    'processing_time': dict
                }
        """
        if verbose:
            print("\n" + "=" * 80)
            print(f"Processing: {os.path.basename(image_path)}")
            print("=" * 80)
        
        processing_time = {}
        
        # 1. 이미지 로드
        start_time = time.time()
        image = self.load_image(image_path)
        load_time = time.time() - start_time
        processing_time['load'] = load_time
        
        if verbose:
            print(f"\n[Step 0] Image loaded: {image.shape}")
            print(f"         Time: {load_time:.3f}s")
        
        # 2. YOLO-Pose
        start_time = time.time()
        persons = self.pose_estimator.detect(image)
        pose_time = time.time() - start_time
        processing_time['pose'] = pose_time
        
        if verbose:
            print(f"\n[Step 1] YOLO-Pose completed")
            print(f"         Persons detected: {len(persons)}")
            print(f"         Time: {pose_time:.3f}s")
        
        if len(persons) == 0:
            if verbose:
                print("\n[WARNING] No persons detected!")
            return {
                'image_path': image_path,
                'image_shape': image.shape,
                'person_info': None,
                'labeled_masks': [],
                'statistics': {},
                'processing_time': processing_time
            }
        
        # 첫 번째 사람 선택 (다중 인물은 향후 지원)
        person = persons[0]
        
        if verbose:
            print(f"         Confidence: {person['confidence']:.3f}")
            print(f"         Valid keypoints: {np.sum(person['keypoints'][:, 2] >= 0.3)}/17")
        
        # 2. Person Segmentation (새로 추가!)
        start_time = time.time()
        person_seg_results = self.person_segmenter.segment_persons(image)
        seg_time = time.time() - start_time
        processing_time['person_seg'] = seg_time
        
        # 사람 마스크 생성
        person_mask = None
        if len(person_seg_results) > 0:
            person_mask = self.person_segmenter.create_person_mask(person_seg_results, image.shape)
            
            if verbose:
                print(f"\n[Step 1.5] Person Segmentation completed")
                print(f"           Persons segmented: {len(person_seg_results)}")
                print(f"           Mask area: {np.sum(person_mask)} px ({np.sum(person_mask)/(image.shape[0]*image.shape[1])*100:.1f}%)")
                print(f"           Time: {seg_time:.3f}s")
        
        # 3. SAM-HQ (person_mask 사용!)
        start_time = time.time()
        masks = self.sam_segmenter.process(
            image,
            person_mask=person_mask,  # 사람 마스크 전달!
            filter_masks=True,
            calculate_centroids=True
        )
        sam_time = time.time() - start_time
        processing_time['sam'] = sam_time
        
        if verbose:
            print(f"\n[Step 2] SAM-HQ completed")
            print(f"         Masks generated: {len(masks)}")
            print(f"         Time: {sam_time:.3f}s")
        
        # 4. Person Filter
        start_time = time.time()
        filtered_masks = self.person_filter.filter_by_person(
            masks,
            person['bbox']
        )
        filter_time = time.time() - start_time
        processing_time['filter'] = filter_time
        
        if verbose:
            print(f"\n[Step 3] Person Filter completed")
            print(f"         Filtered masks: {len(filtered_masks)}")
            print(f"         Time: {filter_time:.3f}s")
        
        # 5. Mask Mapper
        start_time = time.time()
        labeled_masks = self.mask_mapper.map_masks_to_labels(
            filtered_masks,
            person['keypoints'],
            person['body_regions']
        )
        mapper_time = time.time() - start_time
        processing_time['mapper'] = mapper_time
        
        if verbose:
            print(f"\n[Step 4] Mask Mapper completed")
            print(f"         Labeled masks: {len(labeled_masks)}")
            print(f"         Time: {mapper_time:.3f}s")
        
        # 6. 통계 생성
        statistics = self.get_statistics(labeled_masks)
        
        # 총 처리 시간
        total_time = sum(processing_time.values())
        processing_time['total'] = total_time
        
        if verbose:
            print(f"\n[Summary]")
            print(f"         Total time: {total_time:.3f}s")
            print(f"         Label distribution:")
            for label, count in statistics['by_label'].items():
                print(f"           - {label}: {count}")
            print("=" * 80)
        
        # 결과 반환
        return {
            'image_path': image_path,
            'image_shape': image.shape,
            'person_info': person,
            'labeled_masks': labeled_masks,
            'statistics': statistics,
            'processing_time': processing_time
        }
    
    def process_batch(self, image_paths, verbose=True):
        """
        배치 처리
        
        Args:
            image_paths (list): 이미지 경로 리스트
            verbose (bool): 상세 로그 출력 여부
            
        Returns:
            list: 결과 리스트
        """
        results = []
        
        for i, path in enumerate(image_paths):
            if verbose:
                print(f"\n{'='*80}")
                print(f"Processing {i+1}/{len(image_paths)}")
                print(f"{'='*80}")
            
            try:
                result = self.process_image(path, verbose=verbose)
                results.append(result)
            except Exception as e:
                print(f"[ERROR] Failed to process {path}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'image_path': path,
                    'error': str(e)
                })
        
        return results
    
    def get_statistics(self, labeled_masks):
        """
        라벨 통계 생성
        
        Returns:
            dict: 통계 정보
        """
        stats = {
            'total_masks': len(labeled_masks),
            'by_label': {},
            'avg_confidence': {},
            'total_area': {}
        }
        
        # 라벨별 집계
        for mask in labeled_masks:
            label = mask.get('label', 'unknown')
            
            # 개수
            stats['by_label'][label] = stats['by_label'].get(label, 0) + 1
            
            # 신뢰도
            if label not in stats['avg_confidence']:
                stats['avg_confidence'][label] = []
            stats['avg_confidence'][label].append(mask.get('label_confidence', 0))
            
            # 면적
            stats['total_area'][label] = stats['total_area'].get(label, 0) + mask.get('area', 0)
        
        # 평균 신뢰도 계산
        for label in stats['avg_confidence']:
            stats['avg_confidence'][label] = np.mean(stats['avg_confidence'][label])
        
        return stats
    
    def get_vram_usage(self):
        """VRAM 사용량 반환"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            return {
                'allocated': allocated,
                'reserved': reserved
            }
        return None
