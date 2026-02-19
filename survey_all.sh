#!/bin/bash
B="/Volumes/MHFCBCR/imaging_datasets"

survey() {
    local name="$1"
    local path="$B/$name"
    
    echo "=========================================="
    echo "DATASET: $name"
    echo "=========================================="
    
    if [ ! -d "$path" ]; then
        echo "  NOT FOUND"
        echo ""
        return
    fi
    
    local count=$(ls -d "$path"/*/ 2>/dev/null | wc -l | tr -d ' ')
    echo "  Subjects: $count"
    
    echo "  Top-level (first 15):"
    ls "$path" 2>/dev/null | head -15 | sed 's/^/    /'
    
    local nifti_count=$(find "$path" -maxdepth 4 -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
    echo "  NIfTI files found: $nifti_count"
    if [ "$nifti_count" -gt 0 ]; then
        find "$path" -maxdepth 4 -name "*.nii.gz" 2>/dev/null | head -3 | xargs -I{} basename {} | sed 's/^/    /'
    fi
    
    local dcm_count=$(find "$path" -maxdepth 5 -name "*.dcm" 2>/dev/null | wc -l | tr -d ' ')
    echo "  DICOM files found: $dcm_count"
    
    if [ "$dcm_count" -eq 0 ] && [ "$nifti_count" -eq 0 ]; then
        echo "  Other file types:"
        find "$path" -maxdepth 5 -type f 2>/dev/null | head -5 | xargs -I{} basename {} | sed 's/^/    /'
    fi
    
    echo "  Segmentation/mask/label files:"
    local seg_count=$(find "$path" -maxdepth 4 \( -iname "*seg*" -o -iname "*mask*" -o -iname "*label*" \) 2>/dev/null | wc -l | tr -d ' ')
    echo "    Count: $seg_count"
    find "$path" -maxdepth 4 \( -iname "*seg*" -o -iname "*mask*" -o -iname "*label*" \) 2>/dev/null | head -3 | xargs -I{} basename {} | sed 's/^/    /'
    
    local first_subj=$(ls "$path" 2>/dev/null | grep -v '\.' | head -1)
    if [ -n "$first_subj" ] && [ -d "$path/$first_subj" ]; then
        echo "  Sample subject ($first_subj) files:"
        find "$path/$first_subj" -type f 2>/dev/null | head -15 | xargs -I{} basename {} | sed 's/^/    /'
    fi
    
    echo ""
}

survey "ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training"
survey "ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation"
survey "ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
survey "BraTS2024-SSA-Challenge-ValidationData"
survey "BraTS2025-GLI-PRE-Challenge-TrainingData"
survey "BraTS2025-GLI-PRE-Challenge-ValidationData"
survey "MICCAI-LH-BraTS2025-MET-Challenge-Training"
survey "MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels"
survey "MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth"
survey "MICCAI2024-BraTS-GoAT-TrainingData-WithOut-GroundTruth"
survey "CPTAC-GBM"
survey "GLIS-RT"
survey "IvyGAP"
survey "LUMIERE"
survey "REMBRANDT"
survey "TCGA-GBM"
survey "TCGA-LGG"
