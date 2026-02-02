# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Connectome Workbench interface wrappers."""

from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, traits
from nipype.interfaces.workbench.base import WBCommand


class CreateSignedDistanceVolumeInputSpec(CommandLineInputSpec):
    """Input specification for CreateSignedDistanceVolume interface."""

    surf_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='Input surface GIFTI file (.surf.gii)',
    )
    ref_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=1,
        desc='NIfTI volume in the desired output space (dims, spacing, origin)',
    )
    out_file = File(
        name_source=['surf_file'],
        name_template='%s_distvol.nii.gz',
        argstr='%s',
        position=2,
        desc='Name of output volume containing signed distances',
    )
    out_mask = File(
        name_source=['surf_file'],
        name_template='%s_distmask.nii.gz',
        argstr='-roi-out %s',
        desc='Name of file to store a mask where the ``out_file`` has a computed value',
    )
    fill_value = traits.Float(
        0.0,
        mandatory=False,
        usedefault=True,
        argstr='-fill-value %f',
        desc="value to put in all voxels that don't get assigned a distance",
    )
    exact_limit = traits.Float(
        5.0,
        usedefault=True,
        argstr='-exact-limit %f',
        desc='distance for exact output in mm',
    )
    approx_limit = traits.Float(
        20.0,
        usedefault=True,
        argstr='-approx-limit %f',
        desc='distance for approximate output in mm',
    )
    approx_neighborhood = traits.Int(
        2,
        usedefault=True,
        argstr='-approx-neighborhood %d',
        desc='size of neighborhood cube measured from center to face in voxels, default 2 = 5x5x5',
    )
    winding_method = traits.Enum(
        'EVEN_ODD',
        'NEGATIVE',
        'NONZERO',
        'NORMALS',
        argstr='-winding %s',
        usedefault=True,
        desc='winding method for point inside surface test',
    )


class CreateSignedDistanceVolumeOutputSpec(TraitedSpec):
    """Output specification for CreateSignedDistanceVolume interface."""

    out_file = File(desc='Name of output volume containing signed distances')
    out_mask = File(
        desc='Name of file to store a mask where the ``out_file`` has a computed value'
    )


class CreateSignedDistanceVolume(WBCommand):
    """
    Create signed distance volume from surface.

    Computes the signed distance function of the surface. Exact distance is
    calculated by finding the closest point on any surface triangle to the
    center of the voxel. Approximate distance is calculated starting with
    these distances, using Dijkstra's method with a neighborhood of voxels.

    Winding methods:
    - EVEN_ODD (default): Counts entry/exit crossings, inside if total is odd
    - NEGATIVE: Inside if total negative crossings
    - NONZERO: Inside if total nonzero
    - NORMALS: Uses triangle normals (faster, requires closed surface)

    """

    input_spec = CreateSignedDistanceVolumeInputSpec
    output_spec = CreateSignedDistanceVolumeOutputSpec
    _cmd = 'wb_command -create-signed-distance-volume'


class SurfaceAffineRegressionInputSpec(CommandLineInputSpec):
    """Input specification for SurfaceAffineRegression interface."""

    in_surface = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='Surface to warp',
    )
    target_surface = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=1,
        desc='Surface to match the coordinates of',
    )
    out_affine = File(
        name_template='%s_xfm',
        name_source=['in_surface'],
        argstr='%s',
        position=2,
        desc='the output affine file',
    )


class SurfaceAffineRegressionOutputSpec(TraitedSpec):
    """Output specification for SurfaceAffineRegression interface."""

    out_affine = File(desc='The output affine file')


class SurfaceAffineRegression(WBCommand):
    """
    Regress the affine transform between surfaces on the same mesh.

    Uses linear regression to compute an affine that minimizes the sum of
    squares of coordinate differences between target and warped source surfaces.
    Note that this has a bias to shrink the surface being warped.

    Output is written as a NIFTI 'world' matrix (use -convert-affine for other formats).

    """

    input_spec = SurfaceAffineRegressionInputSpec
    output_spec = SurfaceAffineRegressionOutputSpec
    _cmd = 'wb_command -surface-affine-regression'


class SurfaceApplyAffineInputSpec(CommandLineInputSpec):
    """Input specification for SurfaceApplyAffine interface."""

    in_surface = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='the surface to transform',
    )
    in_affine = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=1,
        desc='the affine file',
    )
    out_surface = File(
        name_template='%s_xformed.surf.gii',
        name_source=['in_surface'],
        argstr='%s',
        position=2,
        desc='the output transformed surface',
    )
    flirt_source = File(
        exists=True,
        requires=['flirt_target'],
        argstr='-flirt %s',
        position=3,
        desc='Source volume (must be used if affine is a flirt affine)',
    )
    flirt_target = File(
        exists=True,
        requires=['flirt_source'],
        argstr='%s',
        position=4,
        desc='Target volume (must be used if affine is a flirt affine)',
    )


class SurfaceApplyAffineOutputSpec(TraitedSpec):
    """Output specification for SurfaceApplyAffine interface."""

    out_surface = File(desc='the output transformed surface')


class SurfaceApplyAffine(WBCommand):
    """
    Apply affine transform to surface file.

    For FLIRT matrices, must use the -flirt option, because FLIRT matrices
    are not a complete description of the coordinate transform they represent.
    If -flirt option is not present, the affine must be a NIFTI 'world' affine
    (obtainable with -convert-affine or aff_conv from 4dfp suite).

    """

    input_spec = SurfaceApplyAffineInputSpec
    output_spec = SurfaceApplyAffineOutputSpec
    _cmd = 'wb_command -surface-apply-affine'


class SurfaceApplyWarpfieldInputSpec(CommandLineInputSpec):
    """Input specification for SurfaceApplyWarpfield interface."""

    in_surface = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='the surface to transform',
    )
    warpfield = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=1,
        desc='the INVERSE warpfield',
    )
    out_surface = File(
        name_template='%s_warped.surf.gii',
        name_source=['in_surface'],
        argstr='%s',
        position=2,
        desc='the output transformed surface',
    )
    fnirt_forward_warp = File(
        exists=True,
        argstr='-fnirt %s',
        position=3,
        desc='the forward warpfield (must be used if fnirt warpfield)',
    )


class SurfaceApplyWarpfieldOutputSpec(TraitedSpec):
    """Output specification for SurfaceApplyWarpfield interface."""

    out_surface = File(desc='the output transformed surface')


class SurfaceApplyWarpfield(WBCommand):
    """
    Apply warpfield to surface file.

    NOTE: Warping a surface requires the INVERSE of the warpfield used to
    warp the volume it lines up with. The header of the forward warp is
    needed by the -fnirt option to correctly interpret the displacements.

    If -fnirt option is not present, the warpfield must be a NIFTI 'world'
    warpfield (obtainable with -convert-warpfield command).

    """

    input_spec = SurfaceApplyWarpfieldInputSpec
    output_spec = SurfaceApplyWarpfieldOutputSpec
    _cmd = 'wb_command -surface-apply-warpfield'


class SurfaceModifySphereInputSpec(CommandLineInputSpec):
    """Input specification for SurfaceModifySphere interface."""

    in_surface = File(
        exists=True,
        mandatory=True,
        position=0,
        argstr='%s',
        desc='the sphere to modify',
    )
    radius = traits.Int(
        mandatory=True,
        position=1,
        argstr='%d',
        desc='the radius the output sphere should have',
    )
    out_surface = File(
        name_template='%s_mod.surf.gii',
        name_source='in_surface',
        position=2,
        argstr='%s',
        desc='the modified sphere',
    )
    recenter = traits.Bool(
        False,
        position=3,
        argstr='-recenter',
        desc='recenter the sphere by means of the bounding box',
    )


class SurfaceModifySphereOutputSpec(TraitedSpec):
    """Output specification for SurfaceModifySphere interface."""

    out_surface = File(desc='the modified sphere')


class SurfaceModifySphere(WBCommand):
    """
    Change radius and optionally recenter a sphere.

    May be useful if you have used -surface-resample to resample a sphere,
    which can suffer from problems not present in -surface-sphere-project-unproject.

    If sphere should already be centered around origin, using -recenter may
    still shift it slightly before changing radius, which may be undesirable.

    If sphere is not close to spherical or not centered around origin and
    -recenter is not used, a warning is printed.

    """

    input_spec = SurfaceModifySphereInputSpec
    output_spec = SurfaceModifySphereOutputSpec
    _cmd = 'wb_command -surface-modify-sphere'


class SurfaceSphereProjectUnprojectInputSpec(TraitedSpec):
    """Input specification for SurfaceSphereProjectUnproject interface."""

    sphere_in = File(
        desc='a sphere with the desired output mesh',
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
    )
    sphere_project_to = File(
        desc='a sphere that aligns with sphere-in',
        exists=True,
        mandatory=True,
        argstr='%s',
        position=1,
    )
    sphere_unproject_from = File(
        desc='<sphere-project-to> deformed to the desired output space',
        exists=True,
        mandatory=True,
        argstr='%s',
        position=2,
    )
    sphere_out = traits.File(
        name_template='%s_unprojected.surf.gii',
        name_source=['sphere_in'],
        desc='the output sphere',
        argstr='%s',
        position=3,
    )


class SurfaceSphereProjectUnprojectOutputSpec(TraitedSpec):
    """Output specification for SurfaceSphereProjectUnproject interface."""

    sphere_out = File(desc='the output sphere')


class SurfaceSphereProjectUnproject(WBCommand):
    """
    Copy registration deformations to different sphere.

    This command applies deformations from one surface registration to a
    different sphere, useful for concatenating registrations or inversion.

    Example: To concatenate Human→Chimpanzee and Chimpanzee→Macaque registrations,
    use the Human sphere registered to Chimpanzee as sphere_in, the Chimpanzee
    standard sphere as project_to, and the Chimpanzee sphere registered to Macaque
    as unproject_from. The output will be the Human sphere in register with Macaque.

    """

    input_spec = SurfaceSphereProjectUnprojectInputSpec
    output_spec = SurfaceSphereProjectUnprojectOutputSpec
    _cmd = 'wb_command -surface-sphere-project-unproject'


class SurfaceResampleInputSpec(TraitedSpec):
    """Input specification for SurfaceResample interface."""

    surface_in = File(
        desc='the surface file to resample',
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
    )
    current_sphere = File(
        desc='a sphere surface with the mesh that the input surface is currently on',
        exists=True,
        mandatory=True,
        argstr='%s',
        position=1,
    )
    new_sphere = File(
        desc='a sphere surface that is in register with <current-sphere> and has the '
        'desired output mesh',
        exists=True,
        mandatory=True,
        argstr='%s',
        position=2,
    )
    method = traits.Enum(
        'ADAP_BARY_AREA',
        'BARYCENTRIC',
        desc='the method name',
        mandatory=True,
        argstr='%s',
        position=3,
    )
    surface_out = traits.File(
        name_template='%s_resampled.surf.gii',
        name_source=['surface_in'],
        keep_extension=False,
        desc='the output surface file',
        argstr='%s',
        position=4,
    )
    correction_source = traits.Enum(
        'area_surfs',
        'area_metrics',
        desc='specify surfaces or vertex area metrics to do vertex area correction based on',
        argstr='-%s',
        position=5,
    )
    current_area = File(
        desc='a relevant surface with <current-sphere> mesh',
        exists=True,
        argstr='%s',
        position=6,
        requires=['correction_source'],
    )
    new_area = File(
        desc='a relevant surface with <new-sphere> mesh',
        exists=True,
        argstr='%s',
        position=7,
        requires=['correction_source'],
    )


class SurfaceResampleOutputSpec(TraitedSpec):
    """Output specification for SurfaceResample interface."""

    surface_out = File(desc='the output surface file')


class SurfaceResample(WBCommand):
    """
    Resample a surface to a different mesh.

    Resamples a surface file, given two spherical surfaces that are in register.
    If ADAP_BARY_AREA is used, exactly one of -area-surfs or -area-metrics
    must be specified.

    Methods:
    - BARYCENTRIC (recommended): Generally recommended for anatomical surfaces
      to minimize smoothing
    - ADAP_BARY_AREA: Not recommended generally, provided for completeness

    For cut surfaces (flatmaps), use -surface-cut-resample. For spherical
    surfaces, -surface-sphere-project-unproject is recommended.

    """

    input_spec = SurfaceResampleInputSpec
    output_spec = SurfaceResampleOutputSpec
    _cmd = 'wb_command -surface-resample'


__all__ = [
    'CreateSignedDistanceVolume',
    'SurfaceAffineRegression',
    'SurfaceApplyAffine',
    'SurfaceApplyWarpfield',
    'SurfaceModifySphere',
    'SurfaceSphereProjectUnproject',
    'SurfaceResample',
]
