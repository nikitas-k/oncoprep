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
    
    # Count subjects (top-level dirs)
    local count=$(ls -d "$path"/*/ 2>/dev/null | wc -l | tr -d ' ')
    echo "  Subjects: $count"
    
    # Top-level entries
    echo "  Top-level (first 15):"
    ls "$path" 2>/dev/null | head -15 | sed 's/^/    /'
    
    # NIfTI check
    local nifti_count=$(find "$path" -maxdepth 4 -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
    echo "  NIfTI files found: $nifti_count"
    if [ "$nifti_count" -gt 0 ]; then
        find "$path" -maxdepth 4 -name "*.nii.gz" 2>/dev/null | head -3 | sed 's/^/    /'
    fi
    
    # DICOM check
    local dcm_count=$(find "$path" -maxdepth 5 -name "*.dcm" 2>/dev/null | wc -l | tr -d ' ')
    echo "  DICOM files found: $dcm_count"
    if [ "$dcm_count" -gt 0 ]; then
        find "$path" -maxdepth 5 -name "*.dcm" 2>/dev/null | head -3 | sed 's/^/    /'
    fi
    
    # Also check for DICOM without .dcm extension (numbered files)
    if [ "$dcm_count" -eq 0 ] && [ "$nifti_count" -eq 0 ]; then
        echo "  Checking for other files..."
        find "$path" -maxdepth 5 -type f 2>/dev/null | head -5 | sed 's/^/    /'
    fi
    
    # Segmentation check
    echo "  Segmentation/mask/label files:"
    find "$path" -maxdepth 4 \( -iname "*seg*" -o -iname "*mask*" -o -iname "*label*" \) 2>/dev/null | head -5 | sed 's/^/    /'
    
    # Sample subject contents
    local first_subj=$(ls "$path" 2>/dev/null | grep -v '\.zip$' | grep -v '\.txt$' | grep -v '\.csv$' | grep -v '\.json$' | head -1)
    if [ -n "$first_subj" ] && [ -d "$path/$first_subj" ]; then
        echo "  Sample subject ($first_subj):"
        find "$path/$first_subj" -type f 2>/dev/null | head -15 | while read f; do
            echo "    $(basename "$f")"
        done
    fi
    
    echo ""
}

# Group 1: BraTS challenge datasets
survey "ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training"
survey "ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation"
survey "ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
survey "BraTS2024-SSA-Challenge-ValidationData"
