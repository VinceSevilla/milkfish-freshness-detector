"""
Gills Detection Module
Detects and extracts gills from fish images
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class GillsDetector:
    """Detects and extracts gills from fish images"""
    
    def __init__(self):
        pass
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to enhance gill detection"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def detect_gills(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Simple gill detection: find red areas"""
        try:
            original_h, original_w = image.shape[:2]
            
            print(f"[GILL] Original image size: {original_w}x{original_h}")
            
            # PERFORMANCE FIX: Downsample large images for detection
            # Target max dimension of 1280 for fast detection
            max_detection_size = 1280
            if max(original_w, original_h) > max_detection_size:
                scale_down = max_detection_size / max(original_w, original_h)
                detect_w = int(original_w * scale_down)
                detect_h = int(original_h * scale_down)
                image_detect = cv2.resize(image, (detect_w, detect_h), interpolation=cv2.INTER_AREA)
                print(f"[GILL] Downsampled to {detect_w}x{detect_h} for detection (scale={scale_down:.3f})")
            else:
                image_detect = image
                detect_w, detect_h = original_w, original_h
                scale_down = 1.0
                print(f"[GILL] No downsampling needed")
            
            b, g, r = cv2.split(image_detect)
            h, w = image_detect.shape[:2]
            
            # Find red areas: high R, low B/G
            red_mask = (r.astype(int) - b.astype(int) > 30) & (r.astype(int) - g.astype(int) > 30)
            red_mask = red_mask.astype(np.uint8) * 255
            
            # Morphology to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            
            print(f"[GILL] Red pixels found: {np.sum(red_mask) // 255}")
            
            contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            
            # Scale area thresholds based on downsampled size
            min_area = 500
            max_area = h * w * 0.8
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:  # Reasonable gill size
                    continue
                
                x, y, cw, ch = cv2.boundingRect(contour)
                candidates.append((x, y, cw, ch, area))
            
            print(f"[GILL] Found {len(candidates)} candidates")
            
            if not candidates:
                return None
            
            candidates.sort(key=lambda c: c[4], reverse=True)
            best = candidates[0]
            
            # Scale coordinates back to original image size
            scale_up = 1.0 / scale_down
            bbox_x_orig = int(best[0] * scale_up)
            bbox_y_orig = int(best[1] * scale_up)
            bbox_w_orig = int(best[2] * scale_up)
            bbox_h_orig = int(best[3] * scale_up)
            
            print(f"[GILL] Scaled back to original: bbox=({bbox_x_orig}, {bbox_y_orig}, {bbox_w_orig}, {bbox_h_orig})")
            return (bbox_x_orig, bbox_y_orig, bbox_w_orig, bbox_h_orig)
        except Exception as e:
            print(f"[GILL] ERROR: {str(e)}")
            return None
    
    def extract_gill_roi(self, image: np.ndarray, gill_rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract gill region of interest"""
        x, y, w, h = gill_rect
        
        # Add padding
        padding = int(max(w, h) * 0.1)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # Resize to 224x224
        roi_resized = cv2.resize(roi, (224, 224))
        return roi_resized
    
    def detect_and_extract(self, image: np.ndarray) -> dict:
        """Detect and extract gill region"""
        try:
            result = {
                'gill': None,
                'gill_bbox': None,
                'gill_detected': False
            }
            
            print(f"\n[GILLS DETECTOR] ===== NEW DETECTION REQUEST =====")
            
            # Detect gills
            gill_rect = self.detect_gills(image)
            if gill_rect is not None:
                print(f"[GILLS DETECTOR] Gill detected!")
                result['gill_bbox'] = gill_rect
                result['gill'] = self.extract_gill_roi(image, gill_rect)
                result['gill_detected'] = result['gill'] is not None
            else:
                print(f"[GILLS DETECTOR] No gill detected")
            
            print(f"[GILLS DETECTOR] Final: gill={result['gill_detected']}")
            return result
        except Exception as e:
            print(f"[GILLS DETECTOR] EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'gill': None,
                'gill_bbox': None,
                'gill_detected': False
            }
