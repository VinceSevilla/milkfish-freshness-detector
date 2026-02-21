"""
Retrain ONLY the eyes model with white-balance corrected data
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import cv2
from datetime import datetime
import imgaug.augmenters as iaa
from tqdm import tqdm

# Import GLCM extractor from backend
sys.path.insert(0, str(Path(__file__).parent / 'backend'))
from app.models_service.glcm_extractor import GLCMExtractor

class EyesModelRetrainer:
    """Retrain eyes model with corrected data"""
    
    FRESHNESS_CLASSES = ['fresh', 'less_fresh', 'starting_to_rot', 'rotten']
    CLASS_NAMES = ['Fresh', 'Less Fresh', 'Starting to Rot', 'Rotten']
    
    def __init__(self):
        self.output_dir = Path('results')
        self.output_dir.mkdir(exist_ok=True)
        
        # Load ResNet50 for feature extraction (2048 features)
        self.resnet_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        self.resnet_model.trainable = False
        print("✓ Loaded ResNet50 (frozen)")
        
        # Load MobileNetV1 for feature extraction (1024 features)
        self.mobilenet_model = tf.keras.applications.MobileNet(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        self.mobilenet_model.trainable = False
        print("✓ Loaded MobileNetV1 (frozen)")
        
        # GLCM extractor (29 features)
        self.glcm_extractor = GLCMExtractor()
        
        # Data augmentation
        self.augmenter = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.3),
            iaa.Affine(rotate=(-20, 20)),
            iaa.Affine(scale=(0.9, 1.1)),
            iaa.Multiply((0.8, 1.2)),
            iaa.GaussianBlur(sigma=(0.0, 1.0))
        ])

    def apply_white_balance(self, img):
        """Apply white balance correction to remove blue cast"""
        bgr = img.astype(np.float32)
        b, g, r = cv2.split(bgr)
        r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()
        gray_mean = (r_mean + g_mean + b_mean) / 3
        r_corrected = r * (gray_mean / (r_mean + 1e-6))
        g_corrected = g * (gray_mean / (g_mean + 1e-6))
        b_corrected = b * (gray_mean / (b_mean + 1e-6))
        bgr_corrected = cv2.merge([
            np.clip(b_corrected, 0, 255),
            np.clip(g_corrected, 0, 255),
            np.clip(r_corrected, 0, 255)
        ]).astype(np.uint8)
        return bgr_corrected

    def load_images(self):
        """Load eyes images from both raw and processed folders with white balance correction"""
        print("\n[LOADER] Loading eyes images from raw and processed folders with white balance correction...")
        X_images = []
        y_labels = []
        for class_idx, class_name in enumerate(self.FRESHNESS_CLASSES):
            # Load from raw
            raw_class_path = Path('data/raw/eyes') / class_name
            raw_images = list(raw_class_path.glob('*.jpg')) + list(raw_class_path.glob('*.png'))
            # Load from processed
            processed_class_path = Path('data/processed/eyes') / class_name
            processed_images = list(processed_class_path.glob('*.jpg')) + list(processed_class_path.glob('*.png'))
            images = raw_images + processed_images
            print(f"  {class_name}: {len(images)} images (raw: {len(raw_images)}, processed: {len(processed_images)})")
            for img_path in tqdm(images, desc=f"  Loading {class_name}", leave=False):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                # Apply white balance correction
                img_corrected = self.apply_white_balance(img)
                # Resize
                img_resized = cv2.resize(img_corrected, (224, 224))
                X_images.append(img_resized)
                y_labels.append(class_idx)
        X = np.array(X_images)
        y = np.array(y_labels)
        print(f"✓ Loaded {len(X)} images (white-balance corrected)")
        return X, y

    def extract_hybrid_features(self, images, use_augmentation=False):
        """Extract ResNet50 + MobileNetV1 + GLCM features"""
        if use_augmentation:
            images = self.augmenter(images=images)
        # ResNet50 features (expects RGB)
        X_resnet = self.resnet_model.predict(
            preprocess_input(images.astype(np.float32)),
            verbose=0,
            batch_size=32
        )
        # MobileNetV1 features (expects RGB scaled to [-1, 1])
        X_mobilenet = self.mobilenet_model.predict(
            mobilenet_preprocess(images.astype(np.float32)),
            verbose=0,
            batch_size=32
        )
        # GLCM features (29 features from multi-scale)
        X_glcm = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            glcm_multi = self.glcm_extractor.extract_multi_scale_glcm(gray)
            features = []
            for scale in sorted(glcm_multi.keys()):
                props = glcm_multi[scale]
                features.extend([props[k] for k in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']])
            features = features[:18] + [0.0]*(18-len(features))
            features += [0.0]*(29-len(features))
            X_glcm.append(features)
        X_glcm = np.array(X_glcm)
        # Concatenate all features
        X_combined = np.concatenate([X_resnet, X_mobilenet, X_glcm], axis=1)
        return X_combined

    def build_classifier(self, input_dim):
        """Build classification head"""
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(4, activation='softmax')
        ])
        return model

    def enable_fine_tuning(self):
        """Enable fine-tuning of last layers"""
        # Unfreeze last 30 layers of ResNet50
        self.resnet_model.trainable = True
        for layer in self.resnet_model.layers[:-30]:
            layer.trainable = False
        # Unfreeze last 15 layers of MobileNetV1
        self.mobilenet_model.trainable = True
        for layer in self.mobilenet_model.layers[:-15]:
            layer.trainable = False
        print("✓ Enabled fine-tuning (last 30 ResNet50 + 15 MobileNetV1 layers)")

    def train(self):
        """Train eyes model"""
        print("\n" + "="*60)
        print("RETRAINING EYES MODEL WITH WHITE BALANCE CORRECTION")
        print("="*60)
        # Load data
        X_images, y = self.load_images()
        # Split data
        X_train_img, X_test_img, y_train, y_test = train_test_split(
            X_images, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\n[SPLIT] Train: {len(X_train_img)}, Test: {len(X_test_img)}")
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        print(f"[WEIGHTS] Class weights: {class_weight_dict}")
        # Extract features
        print("\n[FEATURES] Extracting training features...")
        X_train = self.extract_hybrid_features(X_train_img, use_augmentation=True)
        print(f"✓ Training features shape: {X_train.shape}")
        print("\n[FEATURES] Extracting test features...")
        X_test = self.extract_hybrid_features(X_test_img, use_augmentation=False)
        print(f"✓ Test features shape: {X_test.shape}")
        # Build and compile model
        print("\n[MODEL] Building classifier...")
        model = self.build_classifier(X_train.shape[1])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        # Stage 1: Train with frozen feature extractors
        print("\n[TRAIN] Stage 1: Training with frozen feature extractors (30 epochs)...")
        history1 = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=30,
            batch_size=32,
            class_weight=class_weight_dict,
            verbose=1
        )
        # Stage 2: Fine-tune
        print("\n[TRAIN] Stage 2: Fine-tuning (70 epochs)...")
        self.enable_fine_tuning()
        # Re-extract features with trainable layers
        print("\n[FEATURES] Re-extracting features with trainable layers...")
        X_train = self.extract_hybrid_features(X_train_img, use_augmentation=True)
        X_test = self.extract_hybrid_features(X_test_img, use_augmentation=False)
        # Lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        history2 = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=70,
            batch_size=32,
            class_weight=class_weight_dict,
            verbose=1
        )
        # Evaluate
        print("\n[EVALUATION] Testing model...")
        y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
        test_acc = (y_pred == y_test).mean() * 100
        print(f"\n✓ Test Accuracy: {test_acc:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.CLASS_NAMES))
        # Save model
        output_path = self.output_dir / 'best_model_eyes.h5'
        model.save(output_path)
        print(f"\n✓ Saved model: {output_path}")
        return model, test_acc

if __name__ == '__main__':
    retrainer = EyesModelRetrainer()
    retrainer.train()
    print("\n" + "="*60)
    print("COMPLETE! Restart backend to use new model:")
    print("  cd backend")
    print("  python -m uvicorn app.main:app --host 0.0.0.0 --port 8001")
    print("="*60)
