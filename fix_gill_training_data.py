"""
Fix blue color cast in gill training images using white balance correction.
This will create corrected versions in data/processed/gills/
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def apply_white_balance(image_path):
    """Apply white balance correction to remove blue cast"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Convert to float for processing
    bgr = img.astype(np.float32)
    b, g, r = cv2.split(bgr)
    
    # Gray world white balance
    r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()
    gray_mean = (r_mean + g_mean + b_mean) / 3
    
    # Apply correction
    r_corrected = r * (gray_mean / (r_mean + 1e-6))
    g_corrected = g * (gray_mean / (g_mean + 1e-6))
    b_corrected = b * (gray_mean / (b_mean + 1e-6))
    
    # Merge and clip
    bgr_corrected = cv2.merge([
        np.clip(b_corrected, 0, 255),
        np.clip(g_corrected, 0, 255),
        np.clip(r_corrected, 0, 255)
    ]).astype(np.uint8)
    
    return bgr_corrected

def process_class(source_dir, dest_dir, class_name):
    """Process all images in a freshness class"""
    source_path = source_dir / class_name
    dest_path = dest_dir / class_name
    
    # Create destination directory
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    images = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))
    
    print(f"\nProcessing {class_name}: {len(images)} images")
    
    success_count = 0
    for img_path in tqdm(images, desc=f"  {class_name}"):
        corrected = apply_white_balance(img_path)
        if corrected is not None:
            # Save corrected image
            dest_file = dest_path / img_path.name
            cv2.imwrite(str(dest_file), corrected)
            success_count += 1
    
    print(f"  ✓ Saved {success_count} corrected images to {dest_path}")
    return success_count

def main():
    source_dir = Path('data/raw/gills')
    dest_dir = Path('data/processed/gills')
    
    print("="*60)
    print("FIXING GILL TRAINING DATA - WHITE BALANCE CORRECTION")
    print("="*60)
    print(f"\nSource: {source_dir}")
    print(f"Destination: {dest_dir}")
    
    classes = ['fresh', 'less_fresh', 'starting_to_rot', 'rotten']
    total = 0
    
    for class_name in classes:
        count = process_class(source_dir, dest_dir, class_name)
        total += count
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: Processed {total} images")
    print(f"{'='*60}")
    print(f"\nNext step:")
    print(f"  python train_models.py --region gills")
    print(f"\nThis will retrain ONLY the gills model with corrected data.")

if __name__ == '__main__':
    main()
