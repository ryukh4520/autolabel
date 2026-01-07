
import cv2
import numpy as np
import os

class LabelConverter:
    """
    마스크 데이터를 학습용 라벨 포맷으로 변환하는 클래스
    """
    
    def __init__(self, epsilon_factor=0.005):
        """
        Args:
            epsilon_factor (float): 폴리곤 단순화 강도 (값이 클수록 점 개수가 줄어듦)
        """
        self.epsilon_factor = epsilon_factor

    def mask_to_polygons(self, mask):
        """
        이진 마스크를 폴리곤 좌표 리스트로 변환
        
        Args:
            mask (np.ndarray): 2D 이진 마스크 (0 or 255/1)
            
        Returns:
            list: 정규화되지 않은 폴리곤 좌표 리스트 [[x1, y1, x2, y2, ...], ...]
        """
        # 윤곽선 찾기
        mask = mask.astype(np.uint8)
        if mask.max() == 1:
            mask = mask * 255
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            # 너무 작은 노이즈 제거 (면적 기준)
            if cv2.contourArea(contour) < 20:
                continue
                
            # 폴리곤 단순화 (점 개수 최적화)
            epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 점이 3개 미만이면 폴리곤이 아님
            if len(approx) < 3:
                continue
            
            # (N, 1, 2) -> (N, 2) -> flatten
            flattened = approx.flatten().tolist()
            polygons.append(flattened)
            
        return polygons

    def normalize_polygons(self, polygons, img_width, img_height):
        """
        폴리곤 좌표를 0~1 사이로 정규화
        """
        normalized_polygons = []
        
        for poly in polygons:
            normalized = []
            for i in range(0, len(poly), 2):
                x = min(max(poly[i] / img_width, 0.0), 1.0)
                y = min(max(poly[i+1] / img_height, 0.0), 1.0)
                normalized.extend([x, y])
            normalized_polygons.append(normalized)
            
        return normalized_polygons

    def save_yolo_label(self, labeled_masks, output_path, img_width, img_height):
        """
        라벨링된 마스크들을 YOLO Segmentation 포맷(.txt)으로 저장
        
        Format:
        <class_id> <x1> <y1> <x2> <y2> ...
        """
        lines = []
        
        for mask_data in labeled_masks:
            label_id = mask_data.get('label_id', -1)
            if label_id < 0:
                continue
                
            segmentation = mask_data.get('segmentation')
            if segmentation is None:
                continue
                
            # 마스크 -> 폴리곤
            raw_polygons = self.mask_to_polygons(segmentation)
            
            # 정규화
            norm_polygons = self.normalize_polygons(raw_polygons, img_width, img_height)
            
            for poly in norm_polygons:
                # 라인 생성
                coords_str = " ".join([f"{x:.6f}" for x in poly])
                line = f"{label_id} {coords_str}"
                lines.append(line)
        
        # 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
            
        return len(lines)
