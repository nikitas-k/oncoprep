#!/usr/bin/env bash
# pull_seg_models.sh — Download OncoPrep segmentation model SIF files.
#
# Run this on an HPC login node where singularity/apptainer is available.
# No container or pip install required.
#
# Usage:
#   bash pull_seg_models.sh /path/to/seg_cache
#   bash pull_seg_models.sh /path/to/seg_cache --cpu-only
#   bash pull_seg_models.sh /path/to/seg_cache --gpu-only
set -e

OUTDIR="${1:?Usage: $0 <output_dir> [--cpu-only|--gpu-only]}"
FILTER="${2:-all}"

# Detect singularity or apptainer
SING=""
for cmd in singularity apptainer; do
    if command -v "$cmd" &>/dev/null; then
        SING="$cmd"
        break
    fi
done
if [[ -z "$SING" ]]; then
    echo "ERROR: Neither singularity nor apptainer found on PATH." >&2
    echo "Load the module first:  module load singularity" >&2
    exit 1
fi

mkdir -p "$OUTDIR"
echo "Using $SING — saving SIF files to $OUTDIR"
echo ""

pull() {
    local key="$1" image="$2" type="$3"
    local safe; safe="${image//\//_}"; safe="${safe//:/_}"
    local sif="$OUTDIR/${safe}.sif"
    if [[ -f "$sif" ]]; then
        echo "  $key: already cached ($(du -sh "$sif" | cut -f1)) ✓"
    else
        echo "  $key: pulling docker://$image …"
        "$SING" pull --force "$sif" "docker://$image"
        echo "  $key: done ($(du -sh "$sif" | cut -f1)) ✓"
    fi
}

# GPU models
if [[ "$FILTER" != "--cpu-only" ]]; then
    echo "=== GPU models ==="
    pull mic-dkfz     fabianisensee/isen2018              gpu
    pull scan         "mckinleyscan/brats:v2"             gpu
    pull xfeng        xf4j/brats18                        gpu
    pull zyx_2019     jiaocha/zyxbrats                     gpu
    pull scan_2019    scan/brats2019                       gpu
    #pull isen-20      brats/isen-20                        gpu # skip for now, can't build on singularity-ce 3.11.3
    pull hnfnetv1-20  brats/hnfnetv1-20                    gpu
    pull yixinmpl-20  brats/yixinmpl-20                    gpu
    pull sanet0-20    brats/sanet0-20                      gpu
    pull scan-20      brats/scan-20                        gpu
    pull kaist-21     rixez/brats21nnunet                   gpu
    echo ""
fi

# CPU models
if [[ "$FILTER" != "--gpu-only" ]]; then
    echo "=== CPU models ==="
    pull econib      econib/brats-2018                    cpu
    pull lfb_rwth    leonweninger/brats18_segmentation    cpu
    pull gbmnet      nknuecht/gbmnet18                    cpu
    echo ""
fi

echo "Done. Use --seg-cache-dir $OUTDIR when running oncoprep."
