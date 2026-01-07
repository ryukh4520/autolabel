"""
Person Segmenter Module using YOLO-Seg

이 모듈은 YOLOv8-seg를 사용하여:
1. 이미지에서 사람을 검출
2. 사람의 세그멘테이션 마스크 추출
3. 마스크를 사용하여 배경 제거
"""

import numpy as np
from ultralytics import YOLO
import cv2


class PersonSegmenter:
    """
    YOLO-Seg 기반 사람 세그멘테이션 클래스
    """
    
    def __init__(self, model_path, device='cuda', conf_threshold=0.5):
        """
        Args:
            model_path (str): YOLO-Seg 모델 경로
            device (str): 'cuda' or 'cpu'
            conf_threshold (float): Person detection 신뢰도 임계값
        """
        self.device = device
        self.conf_threshold = conf_threshold
        
        # YOLO-Seg 모델 로드
        print(f"[PersonSegmenter] Loading YOLO-Seg model from {model_path}...")
        self.model = YOLO(model_path)
        self.model.to(device)
        print(f"[PersonSegmenter] Model loaded on {device}")
        
        # COCO 클래스에서 person은 0번
        self.person_class_id = 0
    
    def segment_persons(self, image):
        """
        이미지에서 사람 세그멘테이션
        
        Args:
            image (np.ndarray): RGB 이미지
            
        Returns:
            list of dict: 검출된 각 사람의 정보
                {
                    'mask': np.array (bool mask),
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'area': int
                }
        """
        # YOLO-Seg 추론
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        persons = []
        
        for result in results:
            # 검출된 객체가 없으면 스킵
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            # 마스크가 없으면 스킵
            if result.masks is None:
                continue
            
            # 각 검출된 객체에 대해
            for i in range(len(result.boxes)):
                box = result.boxes[i]
                
                # Person 클래스만 선택
                class_id = int(box.cls[0])
                if class_id != self.person_class_id:
                    continue
                
                # Bounding box
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                confidence = float(box.conf[0])
                
                # Segmentation mask
                mask_data = result.masks[i].data[0].cpu().numpy()  # [H, W]
                
                # 마스크를 원본 이미지 크기로 리사이즈
                if mask_data.shape != image.shape[:2]:
                    mask_resized = cv2.resize(mask_data, 
                                            (image.shape[1], image.shape[0]), 
                                            interpolation=cv2.INTER_LINEAR)
                    mask_bool = mask_resized > 0.5
                else:
                    mask_bool = mask_data > 0.5
                
                # 면적 계산
                area = np.sum(mask_bool)
                
                person_info = {
                    'mask': mask_bool,
                    'bbox': bbox.tolist(),
                    'confidence': confidence,
                    'area': int(area)
                }
                
                persons.append(person_info)
        
        print(f"[PersonSegmenter] Detected {len(persons)} person(s)")
        
        return persons
    
    def create_person_mask(self, persons, image_shape):
        """
        여러 사람의 마스크를 하나로 합침
        
        Args:
            persons (list): 사람 정보 리스트
            image_shape (tuple): (height, width, channels)
            
        Returns:
            np.ndarray: 통합된 사람 마스크 (bool)
        """
        height, width = image_shape[:2]
        combined_mask = np.zeros((height, width), dtype=bool)
        
        for person in persons:
            combined_mask = np.logical_or(combined_mask, person['mask'])
        
        return combined_mask
    
    def apply_mask_to_image(self, image, mask):
        """
        마스크를 이미지에 적용 (배경 제거)
        
        Args:
            image (np.ndarray): RGB 이미지
            mask (np.ndarray): bool 마스크
            
        Returns:
            np.ndarray: 마스크가 적용된 이미지 (배경은 검은색)
        """
        masked_image = image.copy()
        masked_image[~mask] = 0  # 마스크 밖은 검은색
        
        return masked_image
    
    def get_masked_region(self, image, mask, padding=20):
        """
        마스크 영역만 크롭 (패딩 포함)
        
        Args:
            image (np.ndarray): RGB 이미지
            mask (np.ndarray): bool 마스크
            padding (int): 크롭 시 추가할 패딩 (픽셀)
            
        Returns:
            tuple: (cropped_image, crop_bbox)
                crop_bbox: [x1, y1, x2, y2]
        """
        # 마스크의 bounding box 찾기
        y_coords, x_coords = np.where(mask)
        
        if len(y_coords) == 0:
            return None, None
        
        x1 = max(0, np.min(x_coords) - padding)
        y1 = max(0, np.min(y_coords) - padding)
        x2 = min(image.shape[1], np.max(x_coords) + padding)
        y2 = min(image.shape[0], np.max(y_coords) + padding)
        
        # 크롭
        cropped = image[y1:y2, x1:x2]
        
        return cropped, [x1, y1, x2, y2]
