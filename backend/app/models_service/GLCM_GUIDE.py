"""
GLCM Feature Extraction - Diagnostic and Analysis Guide
======================================================

GLCM (Gray Level Co-occurrence Matrix) is a texture analysis technique that computes 
statistical properties from the spatial relationships between pixel intensities in an image.

This is now fully integrated into the Fish Freshness Detection API for transparency 
and advanced diagnostics.
"""

# ============================================================================
# 1. WHAT ARE GLCM FEATURES?
# ============================================================================

GLCM_FEATURES = {
    'contrast': {
        'description': 'Measures local variation in the image. Freshness changes may affect texture contrast.',
        'range': '0 to high',
        'typical': 'Fresh fish eyes have higher contrast; rotten eyes may be duller'
    },
    'dissimilarity': {
        'description': 'Similar to contrast but weight-independent. Captures structural differences.',
        'range': '0 to high',
        'typical': 'Increased as freshness decreases'
    },
    'homogeneity': {
        'description': 'Measures local uniformity. Higher values indicate more uniform texture.',
        'range': '0 to 1',
        'typical': 'Fresh fish tissues are more homogeneous; degradation creates heterogeneous patterns'
    },
    'energy': {
        'description': 'Also called Angular Second Moment (ASM). Measures textural orderliness.',
        'range': '0 to 1',
        'typical': 'Fresh tissues have higher energy; degraded tissues become more disordered'
    },
    'correlation': {
        'description': 'Measures linear dependency between pixel intensities. Correlation of tones.',
        'range': '0 to 1',
        'typical': 'Reflects intensity patterns in biological tissues'
    },
    'asm': {
        'description': 'Angular Second Moment (same as energy). Alternative name for the same property.',
        'range': '0 to 1',
        'typical': 'Proxy for texture orderliness'
    }
}

# ============================================================================
# 2. API RESPONSE STRUCTURE WITH GLCM
# ============================================================================

EXAMPLE_RESPONSE = {
    "eye_prediction": {
        "class": "Fresh",
        "confidence": 0.87,
        "probabilities": {
            "Fresh": 0.87,
            "Less Fresh": 0.10,
            "Starting to Rot": 0.02,
            "Rotten": 0.01
        },
        "glcm_features": {
            "basic": {
                # Base glcm computed at distances [1,2,3] and all 4 directions
                "contrast": 0.156,
                "dissimilarity": 0.089,
                "homogeneity": 0.812,
                "energy": 0.643,
                "correlation": 0.721,
                "ASM": 0.643
            },
            "multi_scale": {
                # GLCM features at 3 zoom levels (1x, 2x, 4x downscaling)
                "scale_1": {
                    "contrast": 0.156,
                    "dissimilarity": 0.089,
                    "homogeneity": 0.812,
                    "energy": 0.643,
                    "correlation": 0.721,
                    "ASM": 0.643
                },
                "scale_2": {
                    "contrast": 0.142,
                    "dissimilarity": 0.079,
                    "homogeneity": 0.825,
                    "energy": 0.661,
                    "correlation": 0.735,
                    "ASM": 0.661
                },
                "scale_3": {
                    "contrast": 0.128,
                    "dissimilarity": 0.071,
                    "homogeneity": 0.837,
                    "energy": 0.677,
                    "correlation": 0.748,
                    "ASM": 0.677
                }
            },
            "directional": {
                # GLCM features extracted from 4 cardinal directions (0°, 45°, 90°, 135°)
                # Useful for detecting directional patterns (e.g., gill striations)
                "0°": {
                    "contrast": 0.162,
                    "dissimilarity": 0.093,
                    "homogeneity": 0.808,
                    "energy": 0.635,
                    "correlation": 0.715,
                    "ASM": 0.635
                },
                "45°": {
                    "contrast": 0.154,
                    "dissimilarity": 0.087,
                    "homogeneity": 0.815,
                    "energy": 0.648,
                    "correlation": 0.726,
                    "ASM": 0.648
                },
                "90°": {
                    "contrast": 0.159,
                    "dissimilarity": 0.091,
                    "homogeneity": 0.810,
                    "energy": 0.641,
                    "correlation": 0.720,
                    "ASM": 0.641
                },
                "135°": {
                    "contrast": 0.150,
                    "dissimilarity": 0.084,
                    "homogeneity": 0.819,
                    "energy": 0.654,
                    "correlation": 0.732,
                    "ASM": 0.654
                }
            }
        }
    }
}

# ============================================================================
# 3. INTERPRETING GLCM FOR FISH FRESHNESS
# ============================================================================

INTERPRETATION_GUIDE = """
EYES:
-----
- Fresh eyes typically have:
  • High contrast (clear iris/pupil patterns)
  • High energy (structured, orderly eye structure)
  • High correlation (consistent intensity patterns)
  
- Degrading eyes show:
  • Decreased contrast (cloudy appearance)
  • Decreased energy (breakdown of tissue structure)
  • Decreased homogeneity (uneven texture changes)

GILLS:
------
- Fresh gills show:
  • Structured directional patterns (especially 0° and 90° from gill lamellae)
  • High energy (organized tissue striation)
  • Clear directional dissimilarity (lamellae structure)
  
- Degrading gills exhibit:
  • Loss of directional patterns (tissue breakdown)
  • Decreased energy (loss of tissue organization)
  • Loss of directional dissimilarity (blurred lamella structure)
  • Increased homogeneity (tissue becomes more uniform/mushy)

MULTI-SCALE ANALYSIS:
---------------------
- scale_1 (no downscaling): Fine-grained texture details
- scale_2 (2x downscaling): Mid-level structural patterns
- scale_3 (4x downscaling): Coarse patterns/overall tissue organization

Fresh tissue typically shows consistent features across scales.
Degraded tissue may show inconsistency across scales (detail loss at fine scales).
"""

# ============================================================================
# 4. HOW TO USE GLCM FOR MODEL DEBUGGING
# ============================================================================

DEBUGGING_SCENARIOS = """
Scenario 1: Model predicts "Fresh" but confidence is low
-----
Check GLCM features:
- If homogeneity is very high (>0.85), tissue may be too uniform (possible degradation)
- If energy is low (<0.60), tissue structure may be disorganized
- If directional features are absent/similar, tissue may lack expected patterns

Action: May indicate lighting issues or unusual fish specimen. Compare with 
known "Fresh" GLCM baseline.

---

Scenario 2: Predictions inconsistent between runs on same fish
-----
Check multi-scale GLCM:
- Large differences between scale_1 and scale_3 may indicate:
  • Image quality issues (noise at fine scales)
  • Partial occlusion or lighting artifacts
  • Mixed freshness areas (e.g., half-degraded gill)

Action: Improve image capture (lighting, angle, focus) or trim ROI boundaries.

---

Scenario 3: False positives (predicting "Rotten" for clearly fresh fish)
-----
Check directional GLCM:
- If all directions are identical, may be artificial texture or background
- If only certain directions show high contrast, may be ROI alignment issue
- Compare 0° and 90°: fresh gills should show differences (anisotropic)

Action: Review ROI detection; may need to adjust bounding box or detection params.

---

Scenario 4: Train/Validate discrepancy
-----
Collect GLCM statistics:
- Extract GLCM from validation set
- Group by predicted class vs. true class
- Identify feature ranges for each class
- Fine-tune model using GLCM-augmented features if available

Action: Use GLCM statistics to validate model behavior or retrain with 
GLCM-aware labels.
"""

# ============================================================================
# 5. USING GLCM FOR EXTERNAL VALIDATION
# ============================================================================

VALIDATION_WORKFLOW = """
Step 1: Establish Baseline
-
For each class (Fresh, Less Fresh, Starting to Rot, Rotten):
  a) Analyze 20-30 confirmed specimens
  b) Extract and average GLCM features
  c) Record typical ranges for each feature:
  
     Fresh Eye Baseline:
     - contrast: 0.15 ± 0.03
     - energy: 0.64 ± 0.05
     - homogeneity: 0.81 ± 0.03
     - (etc.)

Step 2: Test New Specimen
-
  a) Run image through API
  b) Extract GLCM features from response
  c) Compare to baseline:
     - If all features within ±2σ: likely that class
     - If features differ: may be atypical specimen or lighting issue

Step 3: Use for External QA
-
  a) Use GLCM features as secondary validation metric
  b) Flag predictions where GLCM is anomalous
  c) Route flagged images for manual review

Step 4: Continuous Learning
-
  a) Periodic re-baseline from confirmed new specimens
  b) Track GLCM drift over time (seasonal changes, camera calibration)
  c) Alert if baseline shifts significantly
"""

# ============================================================================
# 6. PYTHON EXAMPLE: Using GLCM Features
# ============================================================================

PYTHON_EXAMPLE = '''
import requests
import json

# Upload image and get GLCM features
response = requests.post(
    'http://localhost:8000/predict/upload',
    files={'file': open('fish_image.jpg', 'rb')}
)

result = response.json()

# Extract GLCM features
eye_glcm = result['eye_prediction']['glcm_features']

# Analyze basic features
basic_energy = eye_glcm['basic']['energy']
basic_contrast = eye_glcm['basic']['contrast']

print(f"Eye Energy (orderliness): {basic_energy:.3f}")
print(f"Eye Contrast: {basic_contrast:.3f}")

# Check directional patterns (useful for gills)
directional = eye_glcm['directional']
h_contrast = directional['0°']['contrast']      # Horizontal
v_contrast = directional['90°']['contrast']     # Vertical

print(f"Horizontal contrast: {h_contrast:.3f}")
print(f"Vertical contrast: {v_contrast:.3f}")
print(f"Directional anisotropy: {abs(h_contrast - v_contrast):.3f}")

# Check multi-scale consistency
scale_1_energy = eye_glcm['multi_scale']['scale_1']['energy']
scale_3_energy = eye_glcm['multi_scale']['scale_3']['energy']

scale_consistency = abs(scale_1_energy - scale_3_energy) / scale_1_energy
print(f"Scale consistency (lower is better): {scale_consistency:.3f}")

# Alert if anomalous
if scale_consistency > 0.15:
    print("WARNING: Large difference across scales. Image quality issue?")
'''

# ============================================================================
# 7. TECHNICAL DETAILS
# ============================================================================

TECHNICAL_NOTES = """
GLCM Computation:
- Distances: [1, 2, 3] pixels for basic features
- Angles: [0°, 45°, 90°, 135°] for directional analysis
- Levels: 256 (for uint8 grayscale)
- Normalization: Symmetric GLCM (P[i,j] = P[j,i])

Feature Definitions (from scikit-image):
- Contrast = Σ(i-j)² * P[i,j]
- Dissimilarity = Σ|i-j| * P[i,j]
- Homogeneity = Σ P[i,j] / (1 + (i-j)²)
- Energy = Σ P[i,j]²
- Correlation = Σ((i-μ_i)(j-μ_j)*P[i,j]) / (σ_i * σ_j)
- ASM (Angular Second Moment) = Σ P[i,j]²

Grayscale Conversion:
- BGR → Grayscale using OpenCV (NTSC formula: 0.299R + 0.587G + 0.114B)
- Ensures equivalent texture analysis across color variations

Performance:
- GLCM extraction adds ~50-200ms per ROI (depends on image size)
- Features can be cached if same ROI analyzed multiple times
"""

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("GLCM FEATURE CATEGORIES")
    print("="*70)
    for feature, info in GLCM_FEATURES.items():
        print(f"\n{feature.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Range: {info['range']}")
        print(f"  Typical: {info['typical']}")
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print(INTERPRETATION_GUIDE)
    
    print("\n" + "="*70)
    print("DEBUGGING SCENARIOS")
    print("="*70)
    print(DEBUGGING_SCENARIOS)
    
    print("\n" + "="*70)
    print("VALIDATION WORKFLOW")
    print("="*70)
    print(VALIDATION_WORKFLOW)
    
    print("\n" + "="*70)
    print("PYTHON USAGE EXAMPLE")
    print("="*70)
    print(PYTHON_EXAMPLE)
    
    print("\n" + "="*70)
    print("TECHNICAL DETAILS")
    print("="*70)
    print(TECHNICAL_NOTES)
