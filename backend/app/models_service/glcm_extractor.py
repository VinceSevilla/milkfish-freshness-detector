"""
GLCM (Gray Level Co-occurrence Matrix) Feature Extractor
Extracts texture features from ROI images for diagnostics and analysis
"""

import numpy as np
import cv2
from typing import Dict, List, Optional


class GLCMExtractor:
    """Extracts GLCM texture features from images using OpenCV"""
    
    # GLCM properties to compute
    GLCM_PROPERTIES = [
        'contrast',
        'dissimilarity',
        'homogeneity',
        'energy',
        'correlation',
        'ASM'
    ]
    
    @staticmethod
    def _compute_glcm_opencv(gray: np.ndarray, distance: int = 1, angle: float = 0) -> np.ndarray:
        """Compute GLCM matrix using OpenCV operations - OPTIMIZED"""
        # Reduce gray levels to 16 for MUCH faster computation (was 64)
        levels = 16
        gray_scaled = (gray / 256.0 * levels).astype(np.uint8)
        
        h, w = gray_scaled.shape
        glcm = np.zeros((levels, levels), dtype=np.float32)
        
        # Calculate offset based on angle
        if angle == 0:  # Horizontal
            dx, dy = distance, 0
        elif angle == 45:  # Diagonal up-right
            dx, dy = distance, -distance
        elif angle == 90:  # Vertical
            dx, dy = 0, distance
        else:  # 135 degrees - Diagonal down-right
            dx, dy = distance, distance
        
        # Build GLCM - vectorized for speed
        valid_i_start = max(0, -dy)
        valid_i_end = min(h, h - dy)
        valid_j_start = max(0, -dx)
        valid_j_end = min(w, w - dx)
        
        ref_vals = gray_scaled[valid_i_start:valid_i_end, valid_j_start:valid_j_end]
        neighbor_vals = gray_scaled[valid_i_start+dy:valid_i_end+dy, valid_j_start+dx:valid_j_end+dx]
        
        # Use numpy bincount for fast histogram
        combined = ref_vals.ravel() * levels + neighbor_vals.ravel()
        counts = np.bincount(combined, minlength=levels*levels)
        glcm = counts.reshape((levels, levels)).astype(np.float32)
        
        # Normalize
        if glcm.sum() > 0:
            glcm = glcm / glcm.sum()
        
        return glcm
    
    @staticmethod
    def _compute_glcm_properties(glcm: np.ndarray) -> Dict[str, float]:
        """Compute texture properties from GLCM matrix"""
        levels = glcm.shape[0]
        i, j = np.ogrid[0:levels, 0:levels]
        
        # Contrast: sum of (i-j)^2 * P(i,j)
        contrast = float(np.sum((i - j) ** 2 * glcm))
        
        # Dissimilarity: sum of |i-j| * P(i,j)
        dissimilarity = float(np.sum(np.abs(i - j) * glcm))
        
        # Homogeneity: sum of P(i,j) / (1 + (i-j)^2)
        homogeneity = float(np.sum(glcm / (1 + (i - j) ** 2)))
        
        # Energy: sum of P(i,j)^2
        energy = float(np.sum(glcm ** 2))
        
        # ASM (Angular Second Moment) - same as energy
        asm = energy
        
        # Correlation
        # Calculate means and stds
        mu_i = np.sum(i * glcm)
        mu_j = np.sum(j * glcm)
        sigma_i = np.sqrt(np.sum(((i - mu_i) ** 2) * glcm))
        sigma_j = np.sqrt(np.sum(((j - mu_j) ** 2) * glcm))
        
        if sigma_i > 0 and sigma_j > 0:
            correlation = float(np.sum((i - mu_i) * (j - mu_j) * glcm) / (sigma_i * sigma_j))
        else:
            correlation = 0.0
        
        return {
            'contrast': contrast,
            'dissimilarity': dissimilarity,
            'homogeneity': homogeneity,
            'energy': energy,
            'correlation': correlation,
            'ASM': asm
        }
    
    @staticmethod
    def extract_glcm_features(image: np.ndarray, distances: List[int] = None, angles: List[float] = None) -> Dict[str, float]:
        """
        Extract GLCM features from image using OpenCV - OPTIMIZED.
        """
        if distances is None:
            distances = [1]
        if angles is None:
            # Use only 2 angles for speed (was 4)
            angles = [0, 90]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Compute GLCM for first distance and average over angles
        all_props = []
        for angle in angles:
            glcm = GLCMExtractor._compute_glcm_opencv(gray, distances[0], angle)
            props = GLCMExtractor._compute_glcm_properties(glcm)
            all_props.append(props)
        
        # Average properties across angles
        avg_props = {}
        for key in GLCMExtractor.GLCM_PROPERTIES:
            avg_props[key] = float(np.mean([p[key] for p in all_props]))
        
        return avg_props
    
    @staticmethod
    def extract_multi_scale_glcm(image: np.ndarray, scales: List[int] = None) -> Dict[str, Dict[str, float]]:
        """Extract GLCM at multiple scales - OPTIMIZED"""
        if scales is None:
            # Keep 3 scales to maintain 29-feature structure
            scales = [1, 2, 3]
        
        result = {}
        for scale in scales:
            result[f'scale_{scale}'] = GLCMExtractor.extract_glcm_features(image, distances=[scale])
        return result
    
    @staticmethod
    def extract_directional_glcm(image: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Extract GLCM for different directions - OPTIMIZED"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Use only 2 directions for speed (was 4)
        directions = {'0°': 0, '90°': 90}
        result = {}
        for name, angle in directions.items():
            glcm = GLCMExtractor._compute_glcm_opencv(gray, 1, angle)
            result[name] = GLCMExtractor._compute_glcm_properties(glcm)
        return result
    
    @staticmethod
    def compute_glcm_summary(image: np.ndarray) -> Dict:
        """Compute GLCM summary (simplified placeholder)"""
        return {
            'basic': GLCMExtractor.extract_glcm_features(image),
            'multi_scale': GLCMExtractor.extract_multi_scale_glcm(image),
            'directional': GLCMExtractor.extract_directional_glcm(image)
        }
