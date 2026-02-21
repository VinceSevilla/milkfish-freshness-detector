"""
Eye Detection Module
Detects and extracts eyes from fish images
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class EyeDetector:
    """Detects and extracts eyes from fish images"""
    
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to enhance eye detection"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def detect_eyes(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect fish eye by finding the darkest circle (pupil)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            original_h, original_w = gray.shape
            
            print(f"[EYE] Original image size: {original_w}x{original_h}")
            
            # PERFORMANCE FIX: Downsample large images for detection
            # Target max dimension of 1280 for fast detection
            max_detection_size = 1280
            if max(original_w, original_h) > max_detection_size:
                scale_down = max_detection_size / max(original_w, original_h)
                detect_w = int(original_w * scale_down)
                detect_h = int(original_h * scale_down)
                gray_detect = cv2.resize(gray, (detect_w, detect_h), interpolation=cv2.INTER_AREA)
                print(f"[EYE] Downsampled to {detect_w}x{detect_h} for detection (scale={scale_down:.3f})")
            else:
                gray_detect = gray
                detect_w, detect_h = original_w, original_h
                scale_down = 1.0
                print(f"[EYE] No downsampling needed")
            
            # Add buffer for searching: expand beyond head region in case fish is rotated
            search_h = int(detect_h * 0.7)  # Search top 70% of height
            search_w = int(detect_w * 1.0)  # Search full width
            search_roi = gray_detect[:search_h, :search_w]
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(search_roi)
            
            # Use binary threshold to isolate dark regions (pupils)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = cv2.bitwise_not(binary)  # Invert so pupils are white
            
            # Clean up noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Apply gaussian blur for better circle detection
            blurred = cv2.GaussianBlur(binary, (5, 5), 1)
            
            # Fixed detection parameters (working on downsampled image)
            # Look for smaller circles (just the pupil) with more lenient params
            min_dist = 100  # Reduced from 150
            min_radius = 8   # Reduced from 15 - fish pupils can be small
            max_radius = 80  # Reduced from 200 - focus on pupil size
            
            print(f"[EYE] Detection params: minDist={min_dist}, minRadius={min_radius}, maxRadius={max_radius}")
            
            # Detect dark circles - relax parameters to find more candidates
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.0,      # Changed from 1.2 for better accuracy
                minDist=min_dist,
                param1=50,   # Reduced from 80 - more lenient
                param2=30,   # Reduced from 40 - more lenient
                minRadius=min_radius,
                maxRadius=max_radius
            )
            
            candidates = []
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                print(f"[EYE] Found {len(circles[0])} dark circles (pupils)")
                
                for circle in circles[0]:
                    cx, cy, pupil_radius = circle
                    
                    # Loose boundary checks
                    if cy < pupil_radius*0.5 or cy + pupil_radius >= search_h or cx < pupil_radius*0.5 or cx + pupil_radius >= search_w:
                        continue
                    
                    # Dark center check - but very lenient (on downsampled image)
                    center_val = gray_detect[int(cy), int(cx)]
                    if center_val > 150:  # Relaxed from 120 to 150
                        continue
                    
                    # Check for surrounding bright area (eye white/sclera)
                    surrounding_radius = int(pupil_radius * 2)
                    y1 = max(0, int(cy - surrounding_radius))
                    y2 = min(search_h, int(cy + surrounding_radius))
                    x1 = max(0, int(cx - surrounding_radius))
                    x2 = min(search_w, int(cx + surrounding_radius))
                    surrounding_region = gray_detect[y1:y2, x1:x2]
                    surrounding_mean = np.mean(surrounding_region)
                    
                    # Eye should have bright surrounding (sclera) around dark pupil
                    contrast_score = surrounding_mean - center_val
                    
                    # Favor upper-left quadrant where fish eyes typically are
                    # Fish head is usually on the left side of the image
                    horizontal_position_score = (detect_w - cx) / detect_w * 30  # Favor left side
                    vertical_position_score = (search_h - cy) / search_h * 30    # Favor upper position
                    
                    # Score by: darkness + contrast (most important) + size + position
                    darkness_score = 255 - center_val  # Darker = higher score
                    size_score = pupil_radius * 0.5  # Larger = better (but less weight)
                    
                    # Contrast is most important for distinguishing real eyes
                    total_score = (darkness_score * 1.5 + contrast_score * 3.0 + 
                                   size_score + horizontal_position_score + vertical_position_score)
                    
                    # Expand to cover just the eye region tightly
                    # Fish eyes are typically 2.5-3x the pupil size
                    eye_radius = int(pupil_radius * 2.5)
                    bbox_x = max(0, int(cx - eye_radius))
                    bbox_y = max(0, int(cy - eye_radius))
                    bbox_w = min(search_w - bbox_x, eye_radius * 2)
                    bbox_h = min(search_h - bbox_y, eye_radius * 2)
                    
                    candidates.append((bbox_x, bbox_y, bbox_w, bbox_h, total_score, cx, cy, pupil_radius, center_val, contrast_score))
                    print(
                        f"[EYE] Candidate at ({cx:4d}, {cy:4d}) r={pupil_radius:3d} darkness={center_val:3d} "
                        f"contrast={contrast_score:.1f} score={total_score:.1f}"
                    )
            
            print(f"[EYE] Found {len(candidates)} eye candidates")
            
            if not candidates:
                print("[EYE] No candidates found - returning empty")
                return []
            
            # Sort by score and pick best
            candidates.sort(key=lambda c: c[4], reverse=True)
            best = candidates[0]
            print(f"[EYE] Selected best (downsampled coords): center=({best[5]}, {best[6]}) radius={best[7]} "
                  f"darkness={best[8]} contrast={best[9]:.1f} score={best[4]:.1f}")
            
            # Scale coordinates back to original image size
            scale_up = 1.0 / scale_down
            bbox_x_orig = int(best[0] * scale_up)
            bbox_y_orig = int(best[1] * scale_up)
            bbox_w_orig = int(best[2] * scale_up)
            bbox_h_orig = int(best[3] * scale_up)
            
            print(f"[EYE] Scaled back to original: bbox=({bbox_x_orig}, {bbox_y_orig}, {bbox_w_orig}, {bbox_h_orig})")
            return [(bbox_x_orig, bbox_y_orig, bbox_w_orig, bbox_h_orig)]
        except Exception as e:
            print(f"[EYE] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def extract_eye_roi(self, image: np.ndarray, eye_rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract eye region of interest"""
        x, y, w, h = eye_rect
        
        # Add minimal padding (5% instead of 20%)
        padding = int(max(w, h) * 0.05)
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
        """Detect and extract eye region"""
        try:
            result = {
                'eye': None,
                'eye_bbox': None,
                'eye_detected': False
            }
            
            print(f"\n[EYE DETECTOR] ===== NEW DETECTION REQUEST =====")
            
            # Detect eyes
            eyes = self.detect_eyes(image)
            if len(eyes) > 0:
                print(f"[EYE DETECTOR] Eye detected!")
                eye = max(eyes, key=lambda e: e[2] * e[3])
                result['eye_bbox'] = tuple(eye)
                result['eye'] = self.extract_eye_roi(image, eye)
                result['eye_detected'] = result['eye'] is not None
            else:
                print(f"[EYE DETECTOR] No eye detected")
            
            print(f"[EYE DETECTOR] Final: eye={result['eye_detected']}")
            return result
        except Exception as e:
            print(f"[EYE DETECTOR] EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'eye': None,
                'eye_bbox': None,
                'eye_detected': False
            }
