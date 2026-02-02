# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""FSL interface customizations and wrappers."""

from nipype.interfaces.base import traits
from nipype.interfaces.fsl.preprocess import FAST as _FAST
from nipype.interfaces.fsl.preprocess import FASTInputSpec


class _FixTraitFASTInputSpec(FASTInputSpec):
    """Input specification for FAST with corrected bias_iters range."""

    bias_iters = traits.Range(
        low=0,
        high=10,
        argstr='-I %d',
        desc='number of main-loop iterations during bias-field removal',
    )


class FAST(_FAST):
    """
    Custom FAST interface allowing bias_iters=0 to disable bias field correction.

    This replaces nipype.interfaces.fsl.preprocess.FAST to allow setting
    `bias_iters=0` to completely disable bias field correction, which the
    original implementation does not support due to Range constraint starting at 1.

    Examples
    --------
    >>> from oncoprep.interfaces import FAST
    >>> fast = FAST()
    >>> fast.inputs.in_files = 'sub-01_desc-warped_T1w.nii.gz'
    >>> fast.inputs.bias_iters = 0
    >>> 'fast -I 0' in fast.cmdline
    True

    >>> fast.inputs.bias_iters = 3
    >>> 'fast -I 3' in fast.cmdline
    True

    """

    input_spec = _FixTraitFASTInputSpec


__all__ = ['FAST']
