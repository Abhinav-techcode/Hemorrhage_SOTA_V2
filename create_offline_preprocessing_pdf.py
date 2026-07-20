import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

def create_preprocessing_pdf(output_path="Offline_Preprocessing_Techniques.pdf"):
    with PdfPages(output_path) as pdf:
        # Title Page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        plt.text(0.5, 0.90, 'Offline Data Preprocessing Pipeline', 
                 fontsize=22, weight='bold', ha='center', va='top')
        plt.text(0.5, 0.85, 'Analysis of preprocessing.py', 
                 fontsize=14, ha='center', va='top', style='italic')
        plt.text(0.5, 0.82, 'Automated Research Report', 
                 fontsize=12, ha='center', va='top', color='gray')
        
        intro = (
            "This document provides a detailed breakdown of the offline preprocessing pipeline "
            "found in preprocessing/preprocessing.py. This pipeline standardizes raw DICOM/NIfTI "
            "scans across multiple datasets (BHSD, CQ500, PhysioNet) into a clean, uniform format "
            "ready for 3D Segmentation Models."
        )
        wrapped_intro = textwrap.fill(intro, width=80)
        plt.text(0.05, 0.70, wrapped_intro, fontsize=12, va='top', family='monospace')
        
        pdf.savefig(fig)
        plt.close(fig)
        
        # Phase 1: Spatial Standardization
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        plt.text(0.05, 0.95, 'Phase 1: Spatial & Geometric Standardization', fontsize=16, weight='bold')
        
        y = 0.90
        def add_technique(title, why, y_pos):
            plt.text(0.05, y_pos, f"• {title}", fontsize=12, weight='bold', color='darkblue')
            why_text = textwrap.fill(f"Why: {why}", width=85)
            plt.text(0.08, y_pos - 0.02, why_text, fontsize=11, family='monospace', va='top')
            return y_pos - 0.12
            
        y = add_technique("Orthonormalize Direction Cosines", 
                      "Some CT scanners save images with slightly skewed transformation matrices. This mathematically corrects the affine matrix to be perfectly orthogonal (90-degree angles). This is absolutely critical for 3D CNNs, otherwise the convolutions will distort the physical anatomy.", y)
        
        y = add_technique("Isotropic Resampling", 
                      "CT scans from different hospitals have different voxel spacings (e.g., 0.5mm in X/Y, but 5.0mm in Z). This resamples every patient to a standard physical resolution (e.g., 1x1x1 mm) using Linear interpolation for images and Nearest Neighbor for masks, ensuring a 3x3x3 CNN kernel always represents the exact same physical volume across all patients.", y)
                      
        y = add_technique("Z-Axis Depth Standardization (standardize_depth)", 
                      "Scans have a different number of slices (e.g., 30 vs 150). This pads or crops the Z-axis to a fixed depth (e.g., 128 slices). Crucially, if it crops, it calculates the centroid of the hemorrhage and ensures the lesion is perfectly centered in the crop, preventing the ground truth from being cut off.", y)
                      
        pdf.savefig(fig)
        plt.close(fig)

        # Phase 2: Intensity & Contrast
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        plt.text(0.05, 0.95, 'Phase 2: Intensity & Contrast Normalization', fontsize=16, weight='bold')
        y = 0.90
        
        y = add_technique("CT Windowing (apply_window)", 
                      "Raw Hounsfield Units (HU) range from -1000 to +3000, but blood is only visible in a very narrow band. This applies a 'Brain Window' (or multi-channel windows for brain, bone, and subdural) to clamp intensities, drastically boosting the contrast of the hemorrhage against gray/white matter.", y)
                      
        y = add_technique("Histogram Matching", 
                      "Different scanner manufacturers (GE, Siemens, Philips) have different baseline intensity profiles. This forces the intensity histogram of every scan to match a 'golden reference' scan, completely eliminating scanner-specific intensity bias.", y)
                      
        y = add_technique("Z-Score / Min-Max Normalization", 
                      "Neural networks struggle with large numbers. This normalizes the windowed intensities to either a strict [0, 1] range or a Gaussian distribution (mean=0, std=1), optimizing gradient flow during backpropagation.", y)
                      
        pdf.savefig(fig)
        plt.close(fig)
        
        # Phase 3: Mask Cleaning & QC
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        plt.text(0.05, 0.95, 'Phase 3: Label Cleaning & Quality Control', fontsize=16, weight='bold')
        y = 0.90
        
        y = add_technique("Mask Morphological Cleaning (clean_mask)", 
                      "Removes tiny, isolated false-positive pixels in the manual annotations using Connected Component Analysis. It also uses BinaryFillhole to fill any accidental gaps inside a hemorrhage annotation, ensuring the CNN learns solid, continuous lesion structures.", y)
                      
        y = add_technique("Automated Quality Control (QC) Generation", 
                      "Automatically generates a 3-panel visualization (middle slice, max-pathology slice with red overlay, and binary mask) for every processed patient. This allows researchers to rapidly scan hundreds of patients visually without loading massive 3D NIfTI files into viewers like 3D Slicer.", y)
                      
        y = add_technique("Reproducibility Hashing (build_metadata)", 
                      "Embeds the current Git commit, configuration SHA256 hash, and Python/Library versions directly into the JSON metadata of every processed scan. This ensures 100% traceability for FDA/CE medical device compliance.", y)
                      
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    create_preprocessing_pdf()
    print("PDF generated successfully: Offline_Preprocessing_Techniques.pdf")
