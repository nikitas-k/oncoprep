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
    
    echo "  Top-level (first 10):"
    ls "$path" 2>/dev/null | head -10 | sed 's/^/    /'
    
    local nifti_count=$(find "$path" -maxdepth 3 -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
    echo "  NIfTI files: $nifti_count"
    
    local dcm_count=$(find "$path" -maxdepth 4 -name "*.dcm" 2>/dev/null | wc -l | tr -d ' ')
    echo "  DICOM files: $dcm_count"
    
    if [ "$dcm_count" -eq 0 ] && [ "$nifti_count" -eq 0 ]; then
        echo "  Other files:"
        find "$path" -maxdepth 5 -type f 2>/dev/null | head -5 | xargs -I{} basename {} | sed 's/^/    /'
    fi
    
    local seg_count=$(find "$path" -maxdepth 3 \( -iname "*seg*" -o -iname "*mask*" -o -iname "*label*" \) 2>/dev/null | wc -l | tr -d ' ')
    echo "  Seg/mask/label files: $seg_count"
    if [ "$seg_count" -gt 0 ]; then
        find "$path" -maxdepth 3 \( -iname "*seg*" -o -iname "*mask*" -o -iname "*label*" \) 2>/dev/null | head -2 | xargs -I{} basename {} | sed 's/^/    /'
    fi
    
    local first_subj=$(ls "$path" 2>/dev/null | grep -v '\.' | head -1)
    if [ -n "$first_subj" ] && [ -d "$path/$first_subj" ]; then
        echo "  Sample ($first_subj):"
        find "$path/$first_subj" -type f 2>/dev/null | head -15 | xargs -I{} basename {} | sed 's/^/    /'
    fi
    
    echo ""
}

echo "BATCH: $1"
shift
for ds in "$@"; do
    survey "$ds"
done
