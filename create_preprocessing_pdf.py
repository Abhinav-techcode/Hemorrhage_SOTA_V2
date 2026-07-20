import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

def create_preprocessing_pdf(output_path="Preprocessing_Techniques_Report.pdf"):
    with PdfPages(output_path) as pdf:
        # Title Page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        plt.text(0.5, 0.90, 'Data Preprocessing & Augmentation Strategy', 
                 fontsize=22, weight='bold', ha='center', va='top')
        plt.text(0.5, 0.85, 'Analysis of 3D Brain Hemorrhage CT Pipeline', 
                 fontsize=14, ha='center', va='top', style='italic')
        plt.text(0.5, 0.82, 'Automated Research Report', 
                 fontsize=12, ha='center', va='top', color='gray')
        
        intro = (
            "This document provides a detailed breakdown of the MONAI-based dictionary transform "
            "pipeline used in datasets/transforms.py. The pipeline is split into two phases: "
            "Deterministic Preprocessing (to standardize the data) and Stochastic Augmentation "
            "(to improve generalization and prevent overfitting)."
        )
        wrapped_intro = textwrap.fill(intro, width=80)
        plt.text(0.05, 0.70, wrapped_intro, fontsize=12, va='top', family='monospace')
        
        pdf.savefig(fig)
        plt.close(fig)
        
        # Phase 1: Deterministic Preprocessing
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        plt.text(0.05, 0.95, 'Phase 1: Deterministic Preprocessing', fontsize=16, weight='bold')
        
        y = 0.90
        def add_technique(title, why, y_pos):
            plt.text(0.05, y_pos, f"• {title}", fontsize=12, weight='bold', color='darkblue')
            why_text = textwrap.fill(f"Why: {why}", width=85)
            plt.text(0.08, y_pos - 0.02, why_text, fontsize=11, family='monospace', va='top')
            return y_pos - 0.12
            
        y = add_technique("CropForegroundd", 
                      "CT scans contain vast amounts of empty space (air) around the head. This transform automatically calculates a bounding box around the patient's head and crops away the empty space. This forces the neural network to focus exclusively on anatomy and drastically reduces memory and computation requirements.", y)
        
        y = add_technique("SpatialPadd", 
                      "Standardizes the input shape. If a cropped brain is smaller than the required Region of Interest (ROI) size of (64, 256, 256), this pads the edges with zeros so the model always receives consistent tensor dimensions without distorting the aspect ratio.", y)
                      
        y = add_technique("SinglePosNegCropd (RandCropByPosNegLabeld)", 
                      "Brain hemorrhages often occupy less than 1% of the total brain volume. If we randomly cropped patches, the model would almost never see a hemorrhage and would collapse to predicting 'background' everywhere. This transform forces the dataloader to sample patches containing a hemorrhage (Positive) and patches without (Negative) at a controlled 1:1 ratio, solving class imbalance.", y)
                      
        pdf.savefig(fig)
        plt.close(fig)

        # Phase 2: Stochastic Augmentations (Spatial)
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        plt.text(0.05, 0.95, 'Phase 2: Stochastic Spatial Augmentations', fontsize=16, weight='bold')
        y = 0.90
        
        y = add_technique("RandFlipd", 
                      "Randomly flips the 3D volume. Brain anatomy is roughly symmetric, so flipping left-to-right creates perfectly valid new training samples, doubling the effective dataset size and preventing the model from memorizing hemorrhage locations.", y)
                      
        y = add_technique("RandAffined & RandZoomd", 
                      "Applies random rotations, scaling (zoom), and translations. Patients are rarely perfectly aligned in the CT scanner; their heads might be tilted or off-center. These transforms train the model to be robust against poor patient positioning and varying head sizes.", y)
                      
        y = add_technique("Rand3DElasticd", 
                      "Applies a non-linear elastic warping to the 3D grid. Brains naturally vary in shape across patients. More importantly, large hemorrhages cause 'mass effect' (midline shift and ventricular compression). Elastic deformation simulates these pathological anatomical distortions, making the model highly robust to severe trauma cases.", y)
                      
        pdf.savefig(fig)
        plt.close(fig)
        
        # Phase 3: Stochastic Augmentations (Intensity & Dropout)
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        plt.text(0.05, 0.95, 'Phase 3: Intensity & Artifact Augmentations', fontsize=16, weight='bold')
        y = 0.90
        
        y = add_technique("RandGaussianNoised & Smoothd", 
                      "Simulates different CT scanner hardware qualities. High noise mimics low-dose CT scans or older machines. Smoothing simulates different reconstruction kernels (e.g., 'soft' vs 'bone' windows). This ensures the model works across different hospital machines.", y)
                      
        y = add_technique("RandScaleIntensityd, ShiftIntensityd, AdjustContrastd", 
                      "Modifies the Hounsfield Unit (HU) distributions by scaling, shifting, and applying gamma correction. This simulates differences in scanner calibration, contrast agent timing, and varying beam hardening artifacts. It prevents the model from relying on strict, narrow HU values to identify blood.", y)
                      
        y = add_technique("RandCoarseDropoutd", 
                      "Randomly drops out (masks with zeros) coarse rectangular regions of the image. This prevents the model from relying on hyper-local features or specific artifacts to make a prediction. It forces the network to learn broad, global anatomical context (e.g., relying on the surrounding skull and ventricles) to infer the presence of a lesion.", y)
                      
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    create_preprocessing_pdf()
    print("PDF generated successfully: Preprocessing_Techniques_Report.pdf")
