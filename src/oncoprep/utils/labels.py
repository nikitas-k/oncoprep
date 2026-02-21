
# Split multi-label seg into per-label binary masks so each region
# gets its own distinct contour instead of nested isocontour lines.
def split_seg_labels(seg_file):
    """Split multi-label segmentation into per-label binary masks.

    BraTS old labels: 1=NCR, 2=ED, 3=ET, 4=RC.
    Returns one binary NIfTI per label in a fixed order so that the
    colour mapping in ``TumorROIsPlot`` stays consistent.
    """
    import os
    import nibabel as nib
    import numpy as np

    img = nib.load(seg_file)
    data = img.get_fdata()
    expected_labels = [1, 2, 3, 4]
    mask_files = []
    for label in expected_labels:
        mask_data = (data == label).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask_data, img.affine, img.header)
        out_path = os.path.abspath(f'label_{label}.nii.gz')
        nib.save(mask_img, out_path)
        mask_files.append(out_path)
    return mask_files
