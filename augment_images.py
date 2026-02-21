"""
Augment raw images from eyes and gills classes and save to processed folders.
10 augmentations per image. Classes: fresh, less_fresh, starting_to_rot, rotten.
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

def augment_image(img):
    """Apply all requested augmentations to an image."""
    aug_img = img.copy()
    h, w = aug_img.shape[:2]
    # 1. Random Rotation (10°–30°)
    angle = random.uniform(10, 30) * random.choice([-1, 1])
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    aug_img = cv2.warpAffine(aug_img, M, (w, h))
    # 2. Horizontal Flip
    if random.random() < 0.5:
        aug_img = cv2.flip(aug_img, 1)
    # 3. Random Cropping (with bounding box preservation)
    crop_size = random.uniform(0.8, 1.0)
    ch, cw = int(h * crop_size), int(w * crop_size)
    y = random.randint(0, h - ch) if h > ch else 0
    x = random.randint(0, w - cw) if w > cw else 0
    aug_img = aug_img[y:y+ch, x:x+cw]
    aug_img = cv2.resize(aug_img, (w, h))
    # 4. Brightness Adjustment
    hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV)
    brightness = random.randint(-40, 40)
    hsv[:,:,2] = np.clip(hsv[:,:,2] + brightness, 0, 255)
    aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # 5. Contrast Adjustment
    alpha = random.uniform(0.8, 1.2)
    aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=0)
    # 6. Hue & Saturation Shift (Color Jitter)
    hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV)
    hue_shift = random.randint(-10, 10)
    sat_shift = random.randint(-30, 30)
    hsv[:,:,0] = np.clip(hsv[:,:,0] + hue_shift, 0, 179)
    hsv[:,:,1] = np.clip(hsv[:,:,1] + sat_shift, 0, 255)
    aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # 7. Gaussian Noise Addition
    noise = np.random.normal(0, 10, aug_img.shape).astype(np.uint8)
    aug_img = cv2.add(aug_img, noise)
    # 8. Blur Augmentation (Gaussian/Motion Blur)
    if random.random() < 0.5:
        ksize = random.choice([3,5])
        aug_img = cv2.GaussianBlur(aug_img, (ksize, ksize), 0)
    else:
        # Motion blur
        ksize = random.choice([3,5])
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize-1)/2), :] = np.ones(ksize)
        kernel = kernel / ksize
        aug_img = cv2.filter2D(aug_img, -1, kernel)
    # 9. Random Scaling (Zoom In/Out)
    scale = random.uniform(0.8, 1.2)
    aug_img = cv2.resize(aug_img, None, fx=scale, fy=scale)
    aug_img = cv2.resize(aug_img, (w, h))
    # 10. CLAHE (Adaptive Histogram Equalization)
    lab = cv2.cvtColor(aug_img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    aug_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return aug_img

def augment_class(src_dir, dst_dir, class_name):
    src_path = src_dir / class_name
    dst_path = dst_dir / class_name
    dst_path.mkdir(parents=True, exist_ok=True)
    images = list(src_path.glob('*.jpg')) + list(src_path.glob('*.png'))
    print(f"\nAugmenting {class_name}: {len(images)} images")
    for img_path in tqdm(images, desc=f"  {class_name}"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for i in range(10):
            aug_img = augment_image(img)
            aug_name = f"{img_path.stem}_aug{i+1}{img_path.suffix}"
            cv2.imwrite(str(dst_path / aug_name), aug_img)

def main():
    classes = ['fresh', 'less_fresh', 'starting_to_rot', 'rotten']
    for region in ['eyes', 'gills']:
        src_dir = Path(f'data/raw/{region}')
        dst_dir = Path(f'data/processed/{region}')
        for class_name in classes:
            augment_class(src_dir, dst_dir, class_name)
    print("\nAugmentation complete. Augmented images saved to data/processed/.")

if __name__ == '__main__':
    main()
