"""Quick viewer to check what training images actually look like"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def show_samples():
    """Show sample images from each freshness class"""
    base_path = Path('data/raw/gills')
    classes = ['fresh', 'less_fresh', 'starting_to_rot', 'rotten']
    
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    fig.suptitle('Training Data - Visual Inspection', fontsize=16, fontweight='bold')
    
    for row, class_name in enumerate(classes):
        class_path = base_path / class_name
        images = list(class_path.glob('*.jpg'))[:3]
        
        for col, img_path in enumerate(images):
            # Read in RGB
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Calculate color stats
            r, g, b = cv2.split(img_rgb)
            r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()
            brightness = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].mean()
            
            ax = axes[row, col]
            ax.imshow(img_rgb)
            ax.set_title(f'{class_name.upper()}\n'
                        f'RGB: {r_mean:.0f},{g_mean:.0f},{b_mean:.0f}\n'
                        f'Bright: {brightness:.0f}',
                        fontsize=9)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('training_visual_check.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visual check to: training_visual_check.png")
    print("\nNOTE: Check if 'starting_to_rot' images look similar to fresh (bright red)")
    print("      or if they're actually darker/brownish as expected.")
    plt.show()

if __name__ == '__main__':
    show_samples()
