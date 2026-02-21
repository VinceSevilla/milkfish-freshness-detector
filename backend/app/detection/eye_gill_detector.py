

"""
Eye and Gill Detection Module
Detects and extracts eyes and gills from fish images
Supports freshness-specific detection parameters
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List


class EyeGillDetector:
    """Detects and extracts eyes and gills from fish images"""

    TEMPLATE_MAX = 40  # Use reasonable number of templates
    TEMPLATE_SCALES = [0.75, 1.0, 1.25]  # 3 scales
    TEMPLATE_ROTATIONS = [0, 90, 180]  # 3 rotations
    TEMPLATE_EYE_MIN_SCORE = 0.45  # Lower threshold for degraded fish
    TEMPLATE_GILL_MIN_SCORE = 0.40  # Lower threshold for degraded fish
    
    # Freshness-specific detection parameters
    DETECTION_PARAMS = {
        'fresh': {
            'eye': {
                'hough_dp': 1.0,
                'hough_param1': 40,
                'hough_param2': 8,
                'hough_mindist': 3,
                'hough_min_radius': 2,
                'hough_max_radius_ratio': 0.30,
                'canny_low': 20,
                'canny_high': 80,
                'circularity_min': 0.15,
                'area_min_ratio': 0.0001,
                'area_max_ratio': 0.15,
            },
            'gill': {
                'hsv_sat_min': 20,
                'hsv_sat_max': 255,
                'hsv_val_min': 10,
                'hsv_hue_red1_low': [0, 20, 10],
                'hsv_hue_red1_high': [25, 255, 255],
                'hsv_hue_red2_low': [155, 20, 10],
                'hsv_hue_red2_high': [180, 255, 255],
                'lab_threshold': 120,
                'bgr_red_diff': 3,
                'area_min_ratio': 0.001,
                'area_max_ratio': 0.9,
                'aspect_min': 0.1,
                'aspect_max': 7.0,
            }
        },
        'less_fresh': {
            'eye': {
                'hough_dp': 1.2,
                'hough_param1': 50,
                'hough_param2': 10,
                'hough_mindist': 5,
                'hough_min_radius': 3,
                'hough_max_radius_ratio': 0.25,
                'canny_low': 25,
                'canny_high': 90,
                'circularity_min': 0.25,
                'area_min_ratio': 0.0003,
                'area_max_ratio': 0.10,
            },
            'gill': {
                'hsv_sat_min': 30,
                'hsv_sat_max': 255,
                'hsv_val_min': 20,
                'hsv_hue_red1_low': [0, 30, 20],
                'hsv_hue_red1_high': [18, 255, 255],
                'hsv_hue_red2_low': [162, 30, 20],
                'hsv_hue_red2_high': [180, 255, 255],
                'lab_threshold': 135,
                'bgr_red_diff': 8,
                'area_min_ratio': 0.002,
                'area_max_ratio': 0.8,
                'aspect_min': 0.12,
                'aspect_max': 5.5,
            }
        },
        'starting_to_rot': {
            'eye': {
                'hough_dp': 1.3,
                'hough_param1': 40,
                'hough_param2': 8,
                'hough_mindist': 6,
                'hough_min_radius': 2,
                'hough_max_radius_ratio': 0.30,
                'canny_low': 20,
                'canny_high': 80,
                'circularity_min': 0.20,
                'area_min_ratio': 0.0002,
                'area_max_ratio': 0.12,
            },
            'gill': {
                'hsv_sat_min': 35,
                'hsv_sat_max': 255,
                'hsv_val_min': 25,
                'hsv_hue_red1_low': [0, 35, 25],
                'hsv_hue_red1_high': [25, 255, 255],
                'hsv_hue_red2_low': [155, 35, 25],
                'hsv_hue_red2_high': [180, 255, 255],
                'lab_threshold': 140,
                'bgr_red_diff': 12,
                'area_min_ratio': 0.0015,
                'area_max_ratio': 0.8,
                'aspect_min': 0.15,
                'aspect_max': 5.0,
            }
        },
        'rotten': {
            'eye': {
                'hough_dp': 1.4,
                'hough_param1': 30,
                'hough_param2': 6,
                'hough_mindist': 8,
                'hough_min_radius': 2,
                'hough_max_radius_ratio': 0.35,
                'canny_low': 15,
                'canny_high': 70,
                'circularity_min': 0.15,
                'area_min_ratio': 0.0001,
                'area_max_ratio': 0.15,
            },
            'gill': {
                'hsv_sat_min': 40,
                'hsv_sat_max': 255,
                'hsv_val_min': 30,
                'hsv_hue_red1_low': [0, 40, 30],
                'hsv_hue_red1_high': [30, 255, 255],
                'hsv_hue_red2_low': [150, 40, 30],
                'hsv_hue_red2_high': [180, 255, 255],
                'lab_threshold': 145,
                'bgr_red_diff': 15,
                'area_min_ratio': 0.001,
                'area_max_ratio': 0.8,
                'aspect_min': 0.2,
                'aspect_max': 4.5,
            }
        }
    }
    
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # Load templates from FULL FISH IMAGES (raw/eyes_and_gills)
        # NOT from cropped datasets - they won't match in full images
        print("[INIT] Loading templates from full fish images (eyes_and_gills)...")
        all_templates = self._load_templates("data/raw/eyes_and_gills")
        
        # Split templates for eye and gill detection
        # We'll use the same templates but with different search regions
        self.eye_templates = all_templates
        self.gill_templates = all_templates
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to enhance eye and gill detection"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced

    def _load_templates(self, rel_dir: str) -> List[np.ndarray]:
        """Load and precompute edge templates for matching"""
        root = Path(__file__).resolve().parents[3]
        template_dir = root / rel_dir
        if not template_dir.exists():
            print(f"[TEMPLATE] Missing template dir: {template_dir}")
            return []

        files = sorted(
            [p for p in template_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        files = files[: self.TEMPLATE_MAX]
        templates: List[np.ndarray] = []

        for path in files:
            img = cv2.imread(str(path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            edges = cv2.Canny(gray, 60, 180)

            for angle in self.TEMPLATE_ROTATIONS:
                if angle == 0:
                    templ = edges
                elif angle == 90:
                    templ = cv2.rotate(edges, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    templ = cv2.rotate(edges, cv2.ROTATE_180)
                else:
                    templ = cv2.rotate(edges, cv2.ROTATE_90_COUNTERCLOCKWISE)
                templates.append(templ)

        print(f"[TEMPLATE] Loaded {len(templates)} templates from {template_dir}")
        return templates

    def _match_templates(
        self,
        gray: np.ndarray,
        templates: List[np.ndarray],
        min_score: float,
        label: str
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return best template match as bbox if score passes threshold"""
        if not templates:
            return None

        edges = cv2.Canny(gray, 60, 180)
        h, w = gray.shape
        best_score = -1.0
        best_bbox: Optional[Tuple[int, int, int, int]] = None

        for templ in templates:
            th, tw = templ.shape[:2]
            for scale in self.TEMPLATE_SCALES:
                rw = int(tw * scale)
                rh = int(th * scale)
                if rw < 12 or rh < 12:
                    continue
                if rw >= w or rh >= h:
                    continue

                resized = cv2.resize(templ, (rw, rh), interpolation=cv2.INTER_AREA)
                result = cv2.matchTemplate(edges, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    best_bbox = (max_loc[0], max_loc[1], rw, rh)

        if best_bbox is None or best_score < min_score:
            print(f"[{label}] Template match failed: score={best_score:.3f}")
            return None

        print(f"[{label}] Template match: score={best_score:.3f} bbox={best_bbox}")
        return best_bbox
    
    def detect_eyes(self, image: np.ndarray, freshness_class: str = 'fresh') -> list:
        """Multi-strategy eye detection optimized for full fish images"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            print(f"[EYE] Image size: {w}x{h}")
            
            all_candidates = []
            
            # Eyes are typically in the HEAD region (upper 40% of image, left/right side)
            head_h = int(h * 0.4)
            head_region = gray[:head_h, :]
            
            print(f"[EYE] Searching in head region: {head_region.shape}")
            
            # Strategy 1: Template Matching - PRIMARY for full fish images
            template_bbox = self._match_templates(
                head_region,
                self.eye_templates,
                self.TEMPLATE_EYE_MIN_SCORE,
                "EYE_TEMPLATE"
            )
            if template_bbox:
                x, y, tw, th = template_bbox
                all_candidates.append((x, y, tw, th, 120.0, 'template'))  # Higher weight
                print(f"[EYE] Template match found: {template_bbox}")
            
            # Strategy 2: Circular Hough on head region ONLY
            hough_candidates = self._detect_eyes_by_circles_optimized(head_region)
            all_candidates.extend(hough_candidates)
            
            # Strategy 3: Edge-based Detection on head region
            edge_candidates = self._detect_eyes_by_edges_optimized(head_region)
            all_candidates.extend(edge_candidates)
            
            # Strategy 4: Dark spots (pupils) - very specific for eyes
            pupil_candidates = self._detect_eyes_by_pupils(head_region)
            all_candidates.extend(pupil_candidates)
            
            if not all_candidates:
                print("[EYE] No candidates from any strategy")
                return []
            
            # Merge overlapping detections and score
            final_candidates = self._merge_and_score_detections(all_candidates, head_region.shape)
            
            if final_candidates:
                best = final_candidates[0]
                print(f"[EYE] Best candidate: bbox={best[:4]}, score={best[4]:.1f}, method={best[5]}")
                return [best[:4]]
            
            return []
        except Exception as e:
            print(f"[EYE] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def detect_gills(self, image: np.ndarray, freshness_class: str = 'fresh') -> Optional[Tuple[int, int, int, int]]:
        """Multi-strategy gill detection optimized for full fish images"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = image.shape[:2]
            print(f"[GILL] Image size: {w}x{h}")
            
            all_candidates = []
            
            # Gills are typically BEHIND the head, in middle-left region
            # Search region: from 10% to 65% horizontally, 25% to 95% vertically
            gill_x_start = int(w * 0.1)
            gill_x_end = int(w * 0.65)
            gill_y_start = int(h * 0.25)
            gill_y_end = int(h * 0.95)
            
            gill_search_region = gray[gill_y_start:gill_y_end, gill_x_start:gill_x_end]
            print(f"[GILL] Gill search region: {gill_search_region.shape} at ({gill_x_start}, {gill_y_start})")
            
            # Strategy 1: Template Matching - PRIMARY
            template_bbox = self._match_templates(
                gill_search_region,
                self.gill_templates,
                self.TEMPLATE_GILL_MIN_SCORE,
                "GILL_TEMPLATE"
            )
            if template_bbox:
                x, y, tw, th = template_bbox
                # Adjust coordinates back to full image
                all_candidates.append((x + gill_x_start, y + gill_y_start, tw, th, 120.0, 'template'))
                print(f"[GILL] Template match found at adjusted coords: ({x + gill_x_start}, {y + gill_y_start})")
            
            # Strategy 2: Texture-based (BEST for degraded gills)
            texture_candidates = self._detect_gills_by_texture_improved(image, gill_x_start, gill_x_end, gill_y_start, gill_y_end)
            all_candidates.extend(texture_candidates)
            
            # Strategy 3: Color-based (only works for fresh/less fresh)
            color_candidates = self._detect_gills_by_color(image)
            all_candidates.extend(color_candidates)
            
            # Strategy 4: Lamella patterns (gill-specific edge patterns)
            lamella_candidates = self._detect_gills_by_lamella(gray, gill_x_start, gill_x_end, gill_y_start, gill_y_end)
            all_candidates.extend(lamella_candidates)
            
            if not all_candidates:
                print("[GILL] No candidates from any strategy")
                return None
            
            # Merge and score
            final_candidates = self._merge_and_score_detections(all_candidates, (h, w))
            
            if final_candidates:
                best = final_candidates[0]
                print(f"[GILL] Best candidate: bbox={best[:4]}, score={best[4]:.1f}, method={best[5]}")
                return best[:4]
            
            return None
        except Exception as e:
            print(f"[GILL] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _detect_eyes_by_edges(self, gray: np.ndarray) -> list:
        """Detect eyes using edge detection - works for all freshness levels"""
        candidates = []
        h, w = gray.shape
        search_region = gray[:int(h*0.6), :]  # Eyes in upper 60%
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(search_region)
        
        # Edge detection
        edges = cv2.Canny(enhanced, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200 or area > h * w * 0.2:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect = cw / ch if ch > 0 else 0
            
            # Eyes tend to be roughly circular
            if 0.5 < aspect < 2.0:
                # Ensure bounds
                x = max(0, min(x, w - 1))
                y = max(0, min(y, int(h*0.6) - 1))
                cw = min(cw, w - x)
                ch = min(ch, int(h*0.6) - y)
                if cw > 0 and ch > 0:
                    # Position score - favor upper-left region
                    pos_score = (h*0.6 - y) / (h*0.6) * 30
                    size_score = np.sqrt(area)
                    score = pos_score + size_score
                    candidates.append((x, y, cw, ch, score, 'edge'))
        
        return candidates
    
    def _detect_eyes_by_circles(self, gray: np.ndarray) -> list:
        """Detect eyes using circular Hough transform"""
        candidates = []
        h, w = gray.shape
        search_region = gray[:int(h*0.6), :]
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(search_region)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 1)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=60,
            param2=35,
            minRadius=10,
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0][:5]:  # Top 5 circles
                cx, cy, radius = circle
                bbox_size = int(radius * 3.5)
                bbox_size = min(bbox_size, min(w, int(h*0.6)))  # Cap to search region
                x = max(0, min(cx - bbox_size // 2, w - bbox_size))
                y = max(0, min(cy - bbox_size // 2, int(h*0.6) - bbox_size))
                if x + bbox_size > w:
                    bbox_size = w - x
                if y + bbox_size > int(h*0.6):
                    bbox_size = int(h*0.6) - y
                if bbox_size > 0:
                    score = 50 + radius  # Favor larger circles
                    candidates.append((x, y, bbox_size, bbox_size, score, 'hough'))
        
        return candidates
    
    def _detect_eyes_by_contours(self, gray: np.ndarray) -> list:
        """Detect eyes using contour shape analysis"""
        candidates = []
        h, w = gray.shape
        search_region = gray[:int(h*0.6), :]
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            search_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300 or area > h * w * 0.15:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue
            
            # Circularity: 4π * area / perimeter²
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            
            if circularity > 0.3:  # Somewhat circular
                # Ensure bounds
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                cw = min(cw, w - x)
                ch = min(ch, h - y)
                if cw > 0 and ch > 0:
                    score = circularity * 50 + np.sqrt(area)
                    candidates.append((x, y, cw, ch, score, 'contour'))
        
        return candidates
    
    def _detect_gills_by_texture(self, image: np.ndarray) -> list:
        """Detect gills using texture analysis - works for all freshness levels"""
        candidates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Gills have distinct texture patterns
        # Use variance of Laplacian to detect textured regions
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_abs = np.absolute(laplacian)
        
        # Threshold to find high-texture areas
        _, texture_mask = cv2.threshold(lap_abs.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 800 or area > h * w * 0.5:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Gills are typically in left-center region, elongated vertically
            if x < w * 0.7:  # Left part of fish
                aspect = ch / cw if cw > 0 else 0
                if aspect > 0.5:  # Some vertical extent
                    # Ensure bounds
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))
                    cw = min(cw, w - x)
                    ch = min(ch, h - y)
                    
                    if cw > 0 and ch > 0:
                        # Position score - favor center-left
                        pos_score = (w*0.5 - x) / (w*0.5) * 30 if x < w*0.5 else 0
                        texture_score = area / 100
                        score = pos_score + texture_score + 30
                        candidates.append((x, y, cw, ch, score, 'texture'))
        
        return candidates
    
    def _detect_gills_by_color(self, image: np.ndarray) -> list:
        """Detect gills using color (only effective for fresh/less fresh)"""
        candidates = []
        b, g, r = cv2.split(image)
        h, w = image.shape[:2]
        
        # Find reddish areas
        red_mask = ((r.astype(int) - b.astype(int) > 20) & 
                    (r.astype(int) - g.astype(int) > 10) &
                    (r > 70)).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Cap maximum size - gills shouldn't be more than 40% of image
            if area < 600 or area > h * w * 0.4:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Ensure box stays in bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            cw = min(cw, w - x)
            ch = min(ch, h - y)
            
            if cw > 0 and ch > 0:
                score = area / 50 + 20
                candidates.append((x, y, cw, ch, score, 'color'))
        
        return candidates
    
    def _detect_gills_by_edges(self, gray: np.ndarray) -> list:
        """Detect gills using edge patterns"""
        candidates = []
        h, w = gray.shape
        
        # Gills have many fine edges
        edges = cv2.Canny(gray, 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 700 or area > h * w * 0.5:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            if x < w * 0.65:  # Favor left side
                aspect = ch / cw if cw > 0 else 0
                if aspect > 0.4:
                    # Ensure bounds
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))
                    cw = min(cw, w - x)
                    ch = min(ch, h - y)
                    if cw > 0 and ch > 0:
                        score = np.sqrt(area) + 25
                        candidates.append((x, y, cw, ch, score, 'gill_edge'))
        
        return candidates
    
    def _detect_eyes_by_circles_optimized(self, head_region: np.ndarray) -> list:
        """Detect eyes using Hough circles optimized for head region"""
        candidates = []
        h, w = head_region.shape
        
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(head_region)
        blurred = cv2.GaussianBlur(enhanced, (7, 7), 1.5)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=80,
            param1=70,
            param2=30,
            minRadius=12,
            maxRadius=90
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0][:3]:  # Top 3 circles
                cx, cy, radius = circle
                if radius < 5:
                    continue
                bbox_size = int(radius * 3.0)
                # Cap to absolute size - eyes are typically 200-400px
                bbox_size = min(bbox_size, 400, min(w, h))
                x = max(0, min(cx - bbox_size // 2, w - bbox_size))
                y = max(0, min(cy - bbox_size // 2, h - bbox_size))
                if x + bbox_size > w:
                    bbox_size = w - x
                if y + bbox_size > h:
                    bbox_size = h - y
                if bbox_size > 0:
                    score = 60 + radius + (h - cy) / h * 20  # Bonus for upper position
                    candidates.append((x, y, bbox_size, bbox_size, score, 'hough_opt'))
        
        return candidates
    
    def _detect_eyes_by_edges_optimized(self, head_region: np.ndarray) -> list:
        """Detect eyes using edge detection optimized for head region"""
        candidates = []
        h, w = head_region.shape
        
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(head_region)
        
        edges = cv2.Canny(enhanced, 25, 85)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 250 or area > h * w * 0.25:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue
            
            # Circularity test
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            if circularity > 0.4:
                # Ensure bounds
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                cw = min(cw, w - x)
                ch = min(ch, h - y)
                if cw > 0 and ch > 0:
                    pos_score = (h - y) / h * 30
                    size_score = np.sqrt(area)
                    score = pos_score + size_score + circularity * 50
                    candidates.append((x, y, cw, ch, score, 'edge_opt'))
        
        return candidates
    
    def _detect_eyes_by_pupils(self, head_region: np.ndarray) -> list:
        """Detect eyes by locating dark pupils"""
        candidates = []
        h, w = head_region.shape
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(head_region)
        
        # Invert to find dark areas
        inverted = cv2.bitwise_not(enhanced)
        blurred = cv2.GaussianBlur(inverted, (5, 5), 1)
        
        # Threshold to find very dark areas (pupils)
        _, dark_areas = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_areas = cv2.morphologyEx(dark_areas, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(dark_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100 or area > h * w * 0.1:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Expand to eye region (pupils are small, eyes are bigger)
            # Cap expansion to absolute reasonable limits
            eye_scale = 2.5  # Reduced expansion
            eye_size = int(max(cw, ch) * eye_scale)
            # Cap at absolute size - eyes are typically 200-400px in high-res images
            eye_size = min(eye_size, 400)
            
            eye_x = max(0, min(x + cw//2 - eye_size//2, w - eye_size))
            eye_y = max(0, min(y + ch//2 - eye_size//2, h - eye_size))
            
            darkness = blurred[int(y + ch/2), int(x + cw/2)]
            score = (255 - darkness) + 40  # Higher for darker centers
            
            candidates.append((eye_x, eye_y, eye_size, eye_size, score, 'pupil'))
        
        return candidates
    
    def _detect_gills_by_texture_improved(self, image: np.ndarray, x_start: int, x_end: int, y_start: int, y_end: int) -> list:
        """Detect gills using texture analysis - WORKS FOR ALL FRESHNESS LEVELS"""
        candidates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        
        # Clip region to bounds
        x_start = max(0, min(x_start, w - 1))
        y_start = max(0, min(y_start, h - 1))
        x_end = max(x_start + 1, min(x_end, w))
        y_end = max(y_start + 1, min(y_end, h))
        
        # Extract search region
        gill_region = gray[y_start:y_end, x_start:x_end]
        
        # Multi-scale texture analysis
        # Use Laplacian for texture detection (works on all colors)
        laplacian = cv2.Laplacian(gill_region, cv2.CV_64F)
        lap_abs = np.absolute(laplacian)
        
        # Find high-texture areas (gills have lots of fine details)
        _, texture_mask = cv2.threshold(lap_abs.astype(np.uint8), 25, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, kernel)
        texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        
        contours, _ = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000 or area > (x_end - x_start) * (y_end - y_start) * 0.6:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Cap to absolute size (gills typically 300-800px)
            if cw > 1000 or ch > 1000:
                continue
            
            # Gills are typically larger than long
            aspect = ch / cw if cw > 0 else 0
            if aspect > 0.3:  # Some vertical extent
                # Score based on texture richness
                texture_score = (lap_abs[max(0, y):min(y+ch, gill_region.shape[0]), 
                                         max(0, x):min(x+cw, gill_region.shape[1])].mean())
                final_score = texture_score + area / 100
                
                # Adjust back to full image coordinates and validate
                final_x = x + x_start
                final_y = y + y_start
                final_x = max(0, min(final_x, w - 1))
                final_y = max(0, min(final_y, h - 1))
                final_cw = min(cw, w - final_x)
                final_ch = min(ch, h - final_y)
                
                if final_cw > 0 and final_ch > 0:
                    candidates.append((final_x, final_y, final_cw, final_ch, final_score, 'texture_improved'))
        
        return candidates
    
    def _detect_gills_by_lamella(self, gray: np.ndarray, x_start: int, x_end: int, y_start: int, y_end: int) -> list:
        """Detect gills by finding lamella patterns (fine parallel lines typical of gills)"""
        candidates = []
        h, w = gray.shape
        
        # Clip region to bounds
        x_start = max(0, min(x_start, w - 1))
        y_start = max(0, min(y_start, h - 1))
        x_end = max(x_start + 1, min(x_end, w))
        y_end = max(y_start + 1, min(y_end, h))
        
        # Extract search region  
        gill_region = gray[y_start:y_end, x_start:x_end]
        h_gr, w_gr = gill_region.shape
        
        if h_gr <= 0 or w_gr <= 0:
            return candidates
        
        # Detect vertical edges (lamella run vertically)
        sobelx = cv2.Sobel(gill_region, cv2.CV_64F, 1, 0, ksize=3)
        sobelx_abs = np.absolute(sobelx)
        
        # High vertical edge response = lamella pattern
        _, lamella_mask = cv2.threshold(sobelx_abs.astype(np.uint8), 40, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        lamella_mask = cv2.morphologyEx(lamella_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(lamella_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 800 or area > w_gr * h_gr * 0.5:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Cap to absolute size (gills typically 300-800px)
            if cw > 1000 or ch > 1000:
                continue
            
            # Lamella patterns create vertical structures
            if ch > cw * 0.8:  # Tall relative to width
                edge_density = sobelx_abs[max(0, y):min(y+ch, h_gr), 
                                         max(0, x):min(x+cw, w_gr)].mean()
                score = edge_density * 2 + area / 80
                
                # Adjust to full image coordinates and validate
                final_x = x + x_start
                final_y = y + y_start
                final_x = max(0, min(final_x, w - 1))
                final_y = max(0, min(final_y, h - 1))
                final_cw = min(cw, w - final_x)
                final_ch = min(ch, h - final_y)
                
                if final_cw > 0 and final_ch > 0:
                    candidates.append((final_x, final_y, final_cw, final_ch, score, 'lamella'))
        
        return candidates
    
    def _merge_and_score_detections(self, candidates: list, img_shape: Tuple) -> list:
        """Merge overlapping detections and compute final scores"""
        if not candidates:
            return []
        
        h, w = img_shape
        
        # Group overlapping boxes
        def iou(box1, box2):
            x1, y1, w1, h1 = box1[:4]
            x2, y2, w2, h2 = box2[:4]
            
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)
            
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = w1 * h1
            box2_area = w2 * h2
            union_area = box1_area + box2_area - inter_area
            
            return inter_area / union_area if union_area > 0 else 0
        
        merged = []
        used = set()
        
        for i, cand1 in enumerate(candidates):
            if i in used:
                continue
            
            group = [cand1]
            for j, cand2 in enumerate(candidates[i+1:], start=i+1):
                if j in used:
                    continue
                if iou(cand1, cand2) > 0.3:  # 30% overlap
                    group.append(cand2)
                    used.add(j)
            
            # Average the group
            if len(group) > 1:
                avg_x = int(np.mean([c[0] for c in group]))
                avg_y = int(np.mean([c[1] for c in group]))
                avg_w = int(np.mean([c[2] for c in group]))
                avg_h = int(np.mean([c[3] for c in group]))
                # Boost score if multiple methods agree
                avg_score = np.mean([c[4] for c in group]) + len(group) * 20
                methods = '+'.join(set(c[5] for c in group))
                merged.append((avg_x, avg_y, avg_w, avg_h, avg_score, methods))
            else:
                merged.append(group[0])
        
        # Sort by score
        merged.sort(key=lambda x: x[4], reverse=True)
        return merged
    
    def extract_eye_roi(self, image: np.ndarray, eye_rect: Tuple) -> Optional[np.ndarray]:
        """Extract eye region of interest"""
        try:
            x, y, w, h = eye_rect
            h_img, w_img = image.shape[:2]
            
            # Validate dimensions
            if w <= 0 or h <= 0:
                print(f"[EXTRACT_EYE] Invalid dimensions: w={w}, h={h}")
                return None
            
            # Add padding
            padding = int(max(w, h) * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_img, x + w + padding)
            y2 = min(h_img, y + h + padding)
            
            # Revalidate after clipping
            if x1 >= x2 or y1 >= y2:
                print(f"[EXTRACT_EYE] Clipped region invalid: ({x1},{y1}) to ({x2},{y2})")
                return None
            
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                print(f"[EXTRACT_EYE] Empty ROI after slicing")
                return None
            
            # Resize to 224x224
            roi_resized = cv2.resize(roi, (224, 224))
            return roi_resized
        except Exception as e:
            print(f"[EXTRACT_EYE] Exception: {str(e)}")
            return None
    
    def extract_gill_roi(self, image: np.ndarray, gill_rect: Tuple) -> Optional[np.ndarray]:
        """Extract gill region of interest"""
        try:
            x, y, w, h = gill_rect
            h_img, w_img = image.shape[:2]
            
            # Validate dimensions
            if w <= 0 or h <= 0:
                print(f"[EXTRACT_GILL] Invalid dimensions: w={w}, h={h}")
                return None
            
            # Add padding
            padding = int(max(w, h) * 0.1)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_img, x + w + padding)
            y2 = min(h_img, y + h + padding)
            
            # Revalidate after clipping
            if x1 >= x2 or y1 >= y2:
                print(f"[EXTRACT_GILL] Clipped region invalid: ({x1},{y1}) to ({x2},{y2})")
                return None
            
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                print(f"[EXTRACT_GILL] Empty ROI after slicing")
                return None
            
            # Resize to 224x224
            roi_resized = cv2.resize(roi, (224, 224))
            return roi_resized
        except Exception as e:
            print(f"[EXTRACT_GILL] Exception: {str(e)}")
            return None
    
    def detect_and_extract_regions(self, image: np.ndarray, freshness_class: str = None) -> dict:
        """Detect and extract both eye and gill regions"""
        try:
            h_img, w_img = image.shape[:2]
            
            result = {
                'eye': None,
                'gill': None,
                'eye_bbox': None,
                'gill_bbox': None,
                'full_image': image,
                'eye_detected': False,
                'gill_detected': False,
                'freshness_class_used': 'simple'
            }
            
            print(f"\n[DETECTOR] ===== NEW DETECTION REQUEST =====")
            
            # Detect eyes
            eyes = self.detect_eyes(image)
            if len(eyes) > 0:
                print(f"[DETECTOR] Eye detected!")
                eye = max(eyes, key=lambda e: e[2] * e[3])
                
                # Clip to image bounds
                x, y, w, h = eye
                x = max(0, min(x, w_img - 1))
                y = max(0, min(y, h_img - 1))
                w = min(w, w_img - x)
                h = min(h, h_img - y)
                
                if w > 0 and h > 0:
                    result['eye_bbox'] = (x, y, w, h)
                    result['eye'] = self.extract_eye_roi(image, (x, y, w, h))
                    result['eye_detected'] = result['eye'] is not None
                    print(f"[DETECTOR] Eye bbox clipped: ({x}, {y}, {w}, {h})")
            else:
                print(f"[DETECTOR] No eye detected")
            
            # Detect gills
            gill_rect = self.detect_gills(image)
            if gill_rect is not None:
                print(f"[DETECTOR] Gill detected!")
                
                # Clip to image bounds
                x, y, w, h = gill_rect
                x = max(0, min(x, w_img - 1))
                y = max(0, min(y, h_img - 1))
                w = min(w, w_img - x)
                h = min(h, h_img - y)
                
                if w > 0 and h > 0:
                    result['gill_bbox'] = (x, y, w, h)
                    result['gill'] = self.extract_gill_roi(image, (x, y, w, h))
                    result['gill_detected'] = result['gill'] is not None
                    print(f"[DETECTOR] Gill bbox clipped: ({x}, {y}, {w}, {h})")
            else:
                print(f"[DETECTOR] No gill detected")
            
            print(f"[DETECTOR] Final: eye={result['eye_detected']}, gill={result['gill_detected']}")
            return result
        except Exception as e:
            print(f"[DETECTOR] EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'eye': None,
                'gill': None,
                'eye_bbox': None,
                'gill_bbox': None,
                'full_image': image,
                'eye_detected': False,
                'gill_detected': False,
                'freshness_class_used': 'error'
            }
