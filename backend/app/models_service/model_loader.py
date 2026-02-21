"""
Model Loader Service
Loads and manages the trained models
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from app.models_service.glcm_extractor import GLCMExtractor
import traceback


class ModelLoader:
    """Loads and manages fish freshness models"""
    
    FRESHNESS_CLASSES = ['Fresh', 'Less Fresh', 'Starting to Rot', 'Rotten']
    GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    
    def __init__(self, models_dir: str):
        """
        Initialize model loader
        
        Args:
            models_dir: Path to directory containing .h5 model files
        """
        self.models_dir = Path(models_dir)
        self.eye_model: Optional[tf.keras.Model] = None
        self.gill_model: Optional[tf.keras.Model] = None
        self.eyes_gills_model: Optional[tf.keras.Model] = None
        
        # Create ResNet50 feature extractor (2048 features)
        self.resnet_model = tf.keras.applications.ResNet50(
            weights='imagenet', 
            include_top=False, 
            pooling='avg'
        )
        
        # Create MobileNetV1 feature extractor (1024 features)
        self.mobilenet_model = tf.keras.applications.MobileNet(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        
        self._load_models()
    
    def _load_models(self):
        """Load all three models"""
        try:
            eye_path = self.models_dir / 'best_model_eyes.h5'
            if eye_path.exists():
                self.eye_model = load_model(str(eye_path), compile=False)
                print(f"✓ Loaded eye model")
            else:
                print(f"⚠ Eye model not found: {eye_path}")
            
            gill_path = self.models_dir / 'best_model_gills.h5'
            if gill_path.exists():
                self.gill_model = load_model(str(gill_path), compile=False)
                print(f"✓ Loaded gill model")
            else:
                print(f"⚠ Gill model not found: {gill_path}")
            
            eyes_gills_path = self.models_dir / 'best_model_eyes_and_gills.h5'
            if eyes_gills_path.exists():
                self.eyes_gills_model = load_model(str(eyes_gills_path), compile=False)
                print(f"✓ Loaded eyes and gills model")
            else:
                print(f"⚠ Eyes and gills model not found: {eyes_gills_path}")
        
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def _flatten_glcm_features(self, glcm_dict: dict) -> np.ndarray:
        """Flatten GLCM feature dict into a 1D vector of exactly 29 features"""
        features = []
        
        # Extract 6 basic features (averaged across angles)
        if 'basic' in glcm_dict and glcm_dict['basic']:
            basic = glcm_dict['basic']
            features.extend([
                basic.get('contrast', 0.0),
                basic.get('dissimilarity', 0.0),
                basic.get('homogeneity', 0.0),
                basic.get('energy', 0.0),
                basic.get('correlation', 0.0),
                basic.get('ASM', 0.0)
            ])
        else:
            features.extend([0.0] * 6)
        
        # Extract 18 multi-scale features (3 scales × 6 properties)
        if 'multi_scale' in glcm_dict and glcm_dict['multi_scale']:
            for scale_key in ['scale_1', 'scale_2', 'scale_3']:
                if scale_key in glcm_dict['multi_scale']:
                    scale_data = glcm_dict['multi_scale'][scale_key]
                    features.extend([
                        scale_data.get('contrast', 0.0),
                        scale_data.get('dissimilarity', 0.0),
                        scale_data.get('homogeneity', 0.0),
                        scale_data.get('energy', 0.0),
                        scale_data.get('correlation', 0.0),
                        scale_data.get('ASM', 0.0)
                    ])
                else:
                    features.extend([0.0] * 6)
        else:
            features.extend([0.0] * 18)
        
        # Extract 5 directional variance features (variance across 4 directions)
        if 'directional' in glcm_dict and glcm_dict['directional']:
            dir_data = glcm_dict['directional']
            # Calculate variance of key properties across directions
            for prop in ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity']:
                values = [dir_data[d].get(prop, 0.0) for d in ['0°', '45°', '90°', '135°'] if d in dir_data]
                if values:
                    features.append(float(np.var(values)))
                else:
                    features.append(0.0)
        else:
            features.extend([0.0] * 5)
        
        # Ensure exactly 29 features
        features = features[:29]
        while len(features) < 29:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def preprocess_image_resnet(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ResNet50"""
        if len(image.shape) == 2:
            # Convert grayscale to BGR
            image = np.stack([image] * 3, axis=2)
        
        # Resize to 224x224
        if image.shape[:2] != (224, 224):
            import cv2
            image = cv2.resize(image, (224, 224))
        
        # Convert to float32 and apply ResNet50 preprocessing
        image = image.astype(np.float32)
        image = preprocess_input(image)
        
        return image
    
    def preprocess_image_mobilenet(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for MobileNetV1"""
        if len(image.shape) == 2:
            # Convert grayscale to BGR
            image = np.stack([image] * 3, axis=2)
        
        # Resize to 224x224
        if image.shape[:2] != (224, 224):
            import cv2
            image = cv2.resize(image, (224, 224))
        
        # Convert to float32 and apply MobileNet preprocessing
        image = image.astype(np.float32)
        image = mobilenet_preprocess(image)
        
        return image
    
    def predict_eye(self, eye_image: np.ndarray, include_glcm: bool = False) -> Optional[dict]:
        """
        Predict freshness from eye image
        
        Args:
            eye_image: Eye ROI image
            include_glcm: If True, include GLCM texture features in output
        
        Returns: dict with 'class', 'confidence', 'probabilities', and optionally 'glcm_features'
        """
        if self.eye_model is None:
            return None
        
        try:
            # Resize to 224×224 for speed
            import cv2
            if eye_image.shape[:2] != (224, 224):
                eye_image_resized = cv2.resize(eye_image, (224, 224))
            else:
                eye_image_resized = eye_image
            
            # Extract ResNet50 features (2048-dim)
            processed_resnet = self.preprocess_image_resnet(eye_image_resized)
            batch_resnet = np.expand_dims(processed_resnet, axis=0)
            resnet_features = self.resnet_model.predict(batch_resnet, verbose=0)
            print(f"[MODEL] Eye ResNet50 features - min: {resnet_features.min():.3f}, max: {resnet_features.max():.3f}, mean: {resnet_features.mean():.3f}")
            
            # Extract MobileNetV1 features (1024-dim)
            processed_mobile = self.preprocess_image_mobilenet(eye_image_resized)
            batch_mobile = np.expand_dims(processed_mobile, axis=0)
            mobilenet_features = self.mobilenet_model.predict(batch_mobile, verbose=0)
            print(f"[MODEL] Eye MobileNetV1 features - min: {mobilenet_features.min():.3f}, max: {mobilenet_features.max():.3f}, mean: {mobilenet_features.mean():.3f}")
            
            # Combine CNN features (2048 + 1024 = 3072)
            cnn_features = np.concatenate([resnet_features, mobilenet_features], axis=1)
            
            # Get GLCM features from resized image
            glcm_dict = GLCMExtractor.compute_glcm_summary(eye_image_resized)
            glcm_features = self._flatten_glcm_features(glcm_dict)
            print(f"[MODEL] Eye GLCM features - min: {glcm_features.min():.3f}, max: {glcm_features.max():.3f}, mean: {glcm_features.mean():.3f}, non-zero: {np.count_nonzero(glcm_features)}/29")
            glcm_batch = np.expand_dims(glcm_features, axis=0)
            
            # For new retrained eyes model: concatenate features for single input
            combined_features = np.concatenate([cnn_features, glcm_batch], axis=1)
            predictions = self.eye_model.predict(combined_features, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            
            result = {
                'class': self.FRESHNESS_CLASSES[class_idx],
                'confidence': confidence,
                'probabilities': {
                    self.FRESHNESS_CLASSES[i]: float(predictions[0][i])
                    for i in range(len(self.FRESHNESS_CLASSES))
                }
            }
            
            # Add GLCM features if requested
            if include_glcm:
                result['glcm_features'] = GLCMExtractor.compute_glcm_summary(eye_image)
            
            return result
        except Exception as e:
            print(f"[ERROR] Error predicting eye: {e}")
            print(f"[ERROR] Full traceback:")
            traceback.print_exc()
            return None
    
    def normalize_gill_lighting(self, image: np.ndarray) -> np.ndarray:
        """
        Data-driven gill normalization based on training data analysis.
        
        Training data characteristics:
        - Fresh: brightness=144.7, R-G=20.3, Hue=17°, Saturation=72.1
        - Less Fresh: brightness=118.8, R-G=23.6, Hue=62°, Saturation=89.2
        - Starting to Rot: brightness=79.9, R-G=7.8, Hue=55°, Saturation=50.6
        - Rotten: brightness=71.7, R-G=9.1, Hue=59°, Saturation=62.7
        
        CRITICAL ISSUE: Training data has blue color cast (B=131-132 for rotten)
        This is NOT natural. Real rotten gills should be brown/dark red.
        Need to correct blue bias before feature extraction.
        """
        import cv2
        
        # Debug: Log original colors
        bgr_orig = image.astype(np.float32)
        b_orig, g_orig, r_orig = cv2.split(bgr_orig)
        print(f"[GILL PREPROCESS] Original RGB: R={r_orig.mean():.1f}, G={g_orig.mean():.1f}, B={b_orig.mean():.1f}")
        
        # STEP 1: White balance correction to remove blue bias
        # Training data shows unnatural high blue values in rotten class
        bgr = image.astype(np.float32)
        b, g, r = cv2.split(bgr)
        
        # Calculate gray world assumption correction
        r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()
        gray_mean = (r_mean + g_mean + b_mean) / 3
        
        # Apply correction - reduce blue if it's dominant
        r = r * (gray_mean / (r_mean + 1e-6))
        g = g * (gray_mean / (g_mean + 1e-6))
        b = b * (gray_mean / (b_mean + 1e-6))
        
        # Clip and merge
        bgr_balanced = cv2.merge([
            np.clip(b, 0, 255),
            np.clip(g, 0, 255),
            np.clip(r, 0, 255)
        ]).astype(np.uint8)
        
        # Debug: Log after white balance
        b_bal, g_bal, r_bal = cv2.split(bgr_balanced.astype(np.float32))
        print(f"[GILL PREPROCESS] After white balance: R={r_bal.mean():.1f}, G={g_bal.mean():.1f}, B={b_bal.mean():.1f}")
        
        # STEP 2: Brightness normalization using LAB
        lab = cv2.cvtColor(bgr_balanced, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        
        print(f"[GILL PREPROCESS] Brightness before CLAHE: {l.mean():.1f}")
        
        # Apply CLAHE to L channel (brightness) - critical for normalizing dark images
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_normalized = clahe.apply(l)
        
        print(f"[GILL PREPROCESS] Brightness after CLAHE: {l_normalized.mean():.1f}")
        
        # Merge back to LAB and convert to BGR
        lab_normalized = cv2.merge([l_normalized, a, b_ch])
        bgr_final = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
        
        # Debug: Log final colors
        b_fin, g_fin, r_fin = cv2.split(bgr_final.astype(np.float32))
        print(f"[GILL PREPROCESS] Final RGB: R={r_fin.mean():.1f}, G={g_fin.mean():.1f}, B={b_fin.mean():.1f}")
        
        return bgr_final
    
    def predict_gill(self, gill_image: np.ndarray, include_glcm: bool = False) -> Optional[dict]:
        """
        Predict freshness from gill image
        
        Args:
            gill_image: Gill ROI image
            include_glcm: If True, include GLCM texture features in output
        
        Returns: dict with 'class', 'confidence', 'probabilities', and optionally 'glcm_features'
        """
        if self.gill_model is None:
            return None
        
        try:
            # Resize to 224×224 for speed
            import cv2
            if gill_image.shape[:2] != (224, 224):
                gill_image_resized = cv2.resize(gill_image, (224, 224))
            else:
                gill_image_resized = gill_image
            
            # Normalize lighting for gills to handle shadows and brightness variations
            gill_image_resized = self.normalize_gill_lighting(gill_image_resized)
            
            # Extract ResNet50 features (2048-dim)
            processed_resnet = self.preprocess_image_resnet(gill_image_resized)
            batch_resnet = np.expand_dims(processed_resnet, axis=0)
            resnet_features = self.resnet_model.predict(batch_resnet, verbose=0)
            print(f"[MODEL] Gill ResNet50 features - min: {resnet_features.min():.3f}, max: {resnet_features.max():.3f}, mean: {resnet_features.mean():.3f}")
            
            # Extract MobileNetV1 features (1024-dim)
            processed_mobile = self.preprocess_image_mobilenet(gill_image_resized)
            batch_mobile = np.expand_dims(processed_mobile, axis=0)
            mobilenet_features = self.mobilenet_model.predict(batch_mobile, verbose=0)
            print(f"[MODEL] Gill MobileNetV1 features - min: {mobilenet_features.min():.3f}, max: {mobilenet_features.max():.3f}, mean: {mobilenet_features.mean():.3f}")
            
            # Combine CNN features (2048 + 1024 = 3072)
            cnn_features = np.concatenate([resnet_features, mobilenet_features], axis=1)
            
            # Get GLCM features from resized image
            glcm_dict = GLCMExtractor.compute_glcm_summary(gill_image_resized)
            glcm_features = self._flatten_glcm_features(glcm_dict)
            print(f"[MODEL] Gill GLCM features - min: {glcm_features.min():.3f}, max: {glcm_features.max():.3f}, mean: {glcm_features.mean():.3f}, non-zero: {np.count_nonzero(glcm_features)}/29")
            glcm_batch = np.expand_dims(glcm_features, axis=0)
            
            # Concatenate CNN + GLCM features (3072 + 29 = 3101) - retrained model expects single input
            combined_features = np.concatenate([cnn_features, glcm_batch], axis=1)
            
            # Predict
            predictions = self.gill_model.predict(combined_features, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            
            result = {
                'class': self.FRESHNESS_CLASSES[class_idx],
                'confidence': confidence,
                'probabilities': {
                    self.FRESHNESS_CLASSES[i]: float(predictions[0][i])
                    for i in range(len(self.FRESHNESS_CLASSES))
                }
            }
            
            # Add GLCM features if requested
            if include_glcm:
                result['glcm_features'] = GLCMExtractor.compute_glcm_summary(gill_image)
            
            return result
        except Exception as e:
            print(f"[ERROR] Error predicting gill: {e}")
            print(f"[ERROR] Full traceback:")
            traceback.print_exc()
            return None
    
    def predict_eyes_gills(self, full_image: np.ndarray, include_glcm: bool = False) -> Optional[dict]:
        """
        Predict freshness from full fish image (eyes and gills integrated)
        
        Args:
            full_image: Full fish image or integrated ROI
            include_glcm: If True, include GLCM texture features in output
        
        Returns: dict with 'class', 'confidence', 'probabilities', and optionally 'glcm_features'
        """
        if self.eyes_gills_model is None:
            return None
        
        try:
            # Resize to 224×224 for speed
            import cv2
            if full_image.shape[:2] != (224, 224):
                full_image_resized = cv2.resize(full_image, (224, 224))
            else:
                full_image_resized = full_image
            
            # Extract ResNet50 features (2048-dim)
            processed_resnet = self.preprocess_image_resnet(full_image_resized)
            batch_resnet = np.expand_dims(processed_resnet, axis=0)
            resnet_features = self.resnet_model.predict(batch_resnet, verbose=0)
            print(f"[MODEL] Eyes+Gills ResNet50 features - min: {resnet_features.min():.3f}, max: {resnet_features.max():.3f}, mean: {resnet_features.mean():.3f}")
            
            # Extract MobileNetV1 features (1024-dim)
            processed_mobile = self.preprocess_image_mobilenet(full_image_resized)
            batch_mobile = np.expand_dims(processed_mobile, axis=0)
            mobilenet_features = self.mobilenet_model.predict(batch_mobile, verbose=0)
            print(f"[MODEL] Eyes+Gills MobileNetV1 features - min: {mobilenet_features.min():.3f}, max: {mobilenet_features.max():.3f}, mean: {mobilenet_features.mean():.3f}")
            
            # Combine CNN features (2048 + 1024 = 3072)
            cnn_features = np.concatenate([resnet_features, mobilenet_features], axis=1)
            
            # Get GLCM features from resized image
            glcm_dict = GLCMExtractor.compute_glcm_summary(full_image_resized)
            glcm_features = self._flatten_glcm_features(glcm_dict)
            glcm_batch = np.expand_dims(glcm_features, axis=0)
            
            # Model expects [cnn_features, glcm] inputs
            predictions = self.eyes_gills_model.predict([cnn_features, glcm_batch], verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            
            result = {
                'class': self.FRESHNESS_CLASSES[class_idx],
                'confidence': confidence,
                'probabilities': {
                    self.FRESHNESS_CLASSES[i]: float(predictions[0][i])
                    for i in range(len(self.FRESHNESS_CLASSES))
                }
            }
            
            # Add GLCM features if requested
            if include_glcm:
                result['glcm_features'] = GLCMExtractor.compute_glcm_summary(full_image)
            
            return result
        except Exception as e:
            print(f"[ERROR] Error predicting eyes/gills: {e}")
            print(f"[ERROR] Full traceback:")
            traceback.print_exc()
            return None