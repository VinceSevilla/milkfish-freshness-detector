"""
Analyze gill colors and textures across freshness classes
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_image_colors(image_path):
    """Analyze color properties of a gill image"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Convert to different color spaces
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate statistics
    stats = {
        'filename': image_path.name,
        # RGB statistics
        'r_mean': np.mean(rgb[:, :, 0]),
        'g_mean': np.mean(rgb[:, :, 1]),
        'b_mean': np.mean(rgb[:, :, 2]),
        'r_std': np.std(rgb[:, :, 0]),
        'g_std': np.std(rgb[:, :, 1]),
        'b_std': np.std(rgb[:, :, 2]),
        # HSV statistics (Hue, Saturation, Value)
        'h_mean': np.mean(hsv[:, :, 0]),  # Hue (0-179 in OpenCV)
        's_mean': np.mean(hsv[:, :, 1]),  # Saturation
        'v_mean': np.mean(hsv[:, :, 2]),  # Value (brightness)
        'h_std': np.std(hsv[:, :, 0]),
        's_std': np.std(hsv[:, :, 1]),
        'v_std': np.std(hsv[:, :, 2]),
        # LAB statistics (good for color difference perception)
        'l_mean': np.mean(lab[:, :, 0]),  # Lightness
        'a_mean': np.mean(lab[:, :, 1]),  # Green-Red
        'b_mean': np.mean(lab[:, :, 2]),  # Blue-Yellow
        # Texture
        'brightness': np.mean(gray),
        'contrast': np.std(gray),
        # Dominant color (red channel focus for gills)
        'red_dominance': np.mean(rgb[:, :, 0]) - np.mean(rgb[:, :, 1]),  # R - G
        'red_green_ratio': np.mean(rgb[:, :, 0]) / (np.mean(rgb[:, :, 1]) + 1e-5)
    }
    
    return stats, rgb

def analyze_class(class_path, class_name, max_samples=15):
    """Analyze all images in a freshness class"""
    print(f"\n{'='*60}")
    print(f"Analyzing {class_name.upper()} gills")
    print(f"{'='*60}")
    
    image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("No images found!")
        return None, []
    
    # Sample up to max_samples images
    sampled_files = image_files[:max_samples]
    
    all_stats = []
    sample_images = []
    
    for img_path in sampled_files:
        result = analyze_image_colors(img_path)
        if result:
            stats, rgb = result
            all_stats.append(stats)
            sample_images.append((img_path.name, rgb))
    
    if not all_stats:
        return None, []
    
    # Calculate aggregate statistics
    agg_stats = {}
    for key in all_stats[0].keys():
        if key == 'filename':
            continue
        values = [s[key] for s in all_stats]
        agg_stats[f'{key}_avg'] = np.mean(values)
        agg_stats[f'{key}_std'] = np.std(values)
        agg_stats[f'{key}_min'] = np.min(values)
        agg_stats[f'{key}_max'] = np.max(values)
    
    # Print key statistics
    print(f"\nCOLOR STATISTICS (averages across {len(all_stats)} samples):")
    print(f"  RGB: R={agg_stats['r_mean_avg']:.1f} G={agg_stats['g_mean_avg']:.1f} B={agg_stats['b_mean_avg']:.1f}")
    print(f"  HSV: H={agg_stats['h_mean_avg']:.1f}° S={agg_stats['s_mean_avg']:.1f} V={agg_stats['v_mean_avg']:.1f}")
    print(f"  Brightness: {agg_stats['brightness_avg']:.1f} ± {agg_stats['brightness_std']:.1f}")
    print(f"  Contrast: {agg_stats['contrast_avg']:.1f}")
    print(f"  Red Dominance (R-G): {agg_stats['red_dominance_avg']:.1f}")
    print(f"  Red/Green Ratio: {agg_stats['red_green_ratio_avg']:.2f}")
    
    # Identify key characteristics
    print(f"\nKEY CHARACTERISTICS:")
    if agg_stats['h_mean_avg'] < 30 or agg_stats['h_mean_avg'] > 150:
        print(f"  ✓ Red hue dominant (H={agg_stats['h_mean_avg']:.1f}°)")
    if agg_stats['s_mean_avg'] > 100:
        print(f"  ✓ High saturation (vivid colors) S={agg_stats['s_mean_avg']:.1f}")
    elif agg_stats['s_mean_avg'] < 50:
        print(f"  ✓ Low saturation (dull/grayish) S={agg_stats['s_mean_avg']:.1f}")
    if agg_stats['v_mean_avg'] > 150:
        print(f"  ✓ Bright (V={agg_stats['v_mean_avg']:.1f})")
    elif agg_stats['v_mean_avg'] < 100:
        print(f"  ✓ Dark (V={agg_stats['v_mean_avg']:.1f})")
    
    return agg_stats, sample_images

def plot_sample_images(class_data, output_path):
    """Plot sample images from each class"""
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Sample Gill Images - All Freshness Classes', fontsize=18, fontweight='bold')
    
    class_names = ['Fresh', 'Less Fresh', 'Starting to Rot', 'Rotten']
    
    for row, (class_name, (stats, samples)) in enumerate(class_data):
        for col, (filename, img) in enumerate(samples[:5]):
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_title(f'{class_name}\n{filename}', fontsize=8)
            ax.axis('off')
        
        # Hide empty subplots
        for col in range(len(samples), 5):
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved sample images to: {output_path}")
    plt.close()

def plot_color_comparison(stats_dict, output_path):
    """Plot color statistics comparison across classes"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Color Characteristics Across Freshness Classes', fontsize=16, fontweight='bold')
    
    classes = ['Fresh', 'Less Fresh', 'Starting to Rot', 'Rotten']
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    
    # RGB means
    ax = axes[0, 0]
    rgb_data = [[stats_dict[c]['r_mean_avg'], stats_dict[c]['g_mean_avg'], stats_dict[c]['b_mean_avg']] 
                for c in classes]
    x = np.arange(len(classes))
    width = 0.25
    ax.bar(x - width, [d[0] for d in rgb_data], width, label='Red', color='red', alpha=0.7)
    ax.bar(x, [d[1] for d in rgb_data], width, label='Green', color='green', alpha=0.7)
    ax.bar(x + width, [d[2] for d in rgb_data], width, label='Blue', color='blue', alpha=0.7)
    ax.set_xlabel('Freshness Class')
    ax.set_ylabel('Mean Value (0-255)')
    ax.set_title('RGB Mean Values')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Brightness (Value)
    ax = axes[0, 1]
    brightness_vals = [stats_dict[c]['v_mean_avg'] for c in classes]
    ax.bar(classes, brightness_vals, color=colors, alpha=0.7)
    ax.set_xlabel('Freshness Class')
    ax.set_ylabel('Brightness (V in HSV)')
    ax.set_title('Brightness Comparison')
    ax.set_xticklabels(classes, rotation=15)
    ax.grid(axis='y', alpha=0.3)
    
    # Saturation
    ax = axes[0, 2]
    saturation_vals = [stats_dict[c]['s_mean_avg'] for c in classes]
    ax.bar(classes, saturation_vals, color=colors, alpha=0.7)
    ax.set_xlabel('Freshness Class')
    ax.set_ylabel('Saturation (S in HSV)')
    ax.set_title('Color Saturation')
    ax.set_xticklabels(classes, rotation=15)
    ax.grid(axis='y', alpha=0.3)
    
    # Hue
    ax = axes[1, 0]
    hue_vals = [stats_dict[c]['h_mean_avg'] for c in classes]
    ax.bar(classes, hue_vals, color=colors, alpha=0.7)
    ax.set_xlabel('Freshness Class')
    ax.set_ylabel('Hue (H in HSV, degrees)')
    ax.set_title('Color Hue (0=Red, 60=Yellow, 120=Green)')
    ax.set_xticklabels(classes, rotation=15)
    ax.grid(axis='y', alpha=0.3)
    
    # Red Dominance (R - G)
    ax = axes[1, 1]
    red_dom_vals = [stats_dict[c]['red_dominance_avg'] for c in classes]
    ax.bar(classes, red_dom_vals, color=colors, alpha=0.7)
    ax.set_xlabel('Freshness Class')
    ax.set_ylabel('Red Dominance (R - G)')
    ax.set_title('Red Dominance')
    ax.set_xticklabels(classes, rotation=15)
    ax.grid(axis='y', alpha=0.3)
    
    # Contrast
    ax = axes[1, 2]
    contrast_vals = [stats_dict[c]['contrast_avg'] for c in classes]
    ax.bar(classes, contrast_vals, color=colors, alpha=0.7)
    ax.set_xlabel('Freshness Class')
    ax.set_ylabel('Contrast (std of grayscale)')
    ax.set_title('Texture Contrast')
    ax.set_xticklabels(classes, rotation=15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved color comparison to: {output_path}")
    plt.close()

def main():
    data_dir = Path('data/raw/gills')
    output_dir = Path('gill_color_analysis')
    output_dir.mkdir(exist_ok=True)
    
    classes = ['fresh', 'less_fresh', 'starting_to_rot', 'rotten']
    class_display_names = ['Fresh', 'Less Fresh', 'Starting to Rot', 'Rotten']
    
    all_stats = {}
    all_samples = []
    
    for class_name, display_name in zip(classes, class_display_names):
        class_path = data_dir / class_name
        if not class_path.exists():
            print(f"⚠ Class folder not found: {class_path}")
            continue
        
        stats, samples = analyze_class(class_path, display_name, max_samples=5)
        if stats:
            all_stats[display_name] = stats
            all_samples.append((display_name, (stats, samples)))
    
    if len(all_stats) < 4:
        print("\n⚠ Not all classes found!")
        return
    
    # Generate visualizations
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    plot_sample_images(all_samples, output_dir / 'sample_gills.png')
    plot_color_comparison(all_stats, output_dir / 'color_comparison.png')
    
    # Generate recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS FOR PREPROCESSING")
    print(f"{'='*60}")
    
    # Compare brightness ranges
    brightness_range = max([all_stats[c]['v_mean_avg'] for c in class_display_names]) - \
                       min([all_stats[c]['v_mean_avg'] for c in class_display_names])
    
    if brightness_range > 50:
        print(f"\n✓ BRIGHTNESS NORMALIZATION NEEDED")
        print(f"  Brightness range: {brightness_range:.1f}")
        print(f"  Apply CLAHE or histogram equalization to L channel (LAB)")
    
    # Check saturation
    fresh_sat = all_stats['Fresh']['s_mean_avg']
    rotten_sat = all_stats['Rotten']['s_mean_avg']
    
    if abs(fresh_sat - rotten_sat) > 30:
        print(f"\n✓ SATURATION IS KEY DISCRIMINATOR")
        print(f"  Fresh saturation: {fresh_sat:.1f}")
        print(f"  Rotten saturation: {rotten_sat:.1f}")
        print(f"  Consider HSV-based preprocessing or augmentation")
    
    # Check red dominance
    fresh_red = all_stats['Fresh']['red_dominance_avg']
    rotten_red = all_stats['Rotten']['red_dominance_avg']
    
    if fresh_red > rotten_red + 10:
        print(f"\n✓ RED DOMINANCE DIFFERENTIATES FRESH vs ROTTEN")
        print(f"  Fresh R-G: {fresh_red:.1f}")
        print(f"  Rotten R-G: {rotten_red:.1f}")
    
    print(f"\n✓ Analysis complete! Check {output_dir}/ for visualizations")

if __name__ == '__main__':
    main()
