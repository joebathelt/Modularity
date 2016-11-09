from nipype.interfaces.base import BaseInterface
from nipype.interfaces.base import BaseInterfaceInputSpec
from nipype.interfaces.base import File
from nipype.interfaces.base import traits
from nipype.interfaces.base import TraitedSpec

# ======================================================================
# DWI preprocessing

class DWIPreprocInputSpec(BaseInterfaceInputSpec):
    bval = File(desc='bval file', mandatory=True)
    bvec = File(desc='bvec file', mandatory=True)
    dwi = File(desc='diffusion-weighted image', mandatory=True)
    subject_id = traits.String(desc='subject ID', mandatory=True)
    out_directory = File(
        desc='directory where to dwi should be directed', mandatory=True)

class DWIPreprocOutputSpec(TraitedSpec):
    AD = File(exist=True, desc='axial diffusivity image')
    FA = File(exist=True, desc='fractional anisotropy image')
    MD = File(exist=True, desc='mean diffusivity image')
    RD = File(exist=True, desc='radial diffusivity image')
    dwi = File(exist=True, desc='processing diffusion-weighted image')
    mask = File(exist=True, desc='brain mask')
    b0 = File(exist=True, desc='b0 volume')

class DWIPreproc(BaseInterface):
    input_spec = DWIPreprocInputSpec
    output_spec = DWIPreprocOutputSpec

    def _run_interface(self, runtime):

        # Loading required packages
        from additional_interfaces import AdditionalDTIMeasures
        from additional_interfaces import DipyDenoise
        import nipype.interfaces.fsl as fsl
        import nipype.interfaces.io as nio
        import nipype.pipeline.engine as pe
        import os

        # ==============================================================
        # Processing of diffusion-weighted data
        # Denoising
        dwi_denoise = pe.Node(interface=DipyDenoise(), name='dwi_denoise')
        dwi_denoise.inputs.in_file = self.inputs.dwi

        # Eddy-current and motion correction
        eddycorrect = pe.Node(interface=fsl.epi.EddyCorrect(), name='eddycorrect')
        eddycorrect.inputs.ref_num = 0

        # Extract b0 image
        fslroi = pe.Node(interface=fsl.ExtractROI(), name='extract_b0')
        fslroi.inputs.t_min = 0
        fslroi.inputs.t_size = 1

        # Create a brain mask
        bet = pe.Node(interface=fsl.BET(
            frac=0.3, robust=False, mask=True), name='bet')

        # Fitting the diffusion tensor model
        dtifit = pe.Node(interface=fsl.DTIFit(), name='dtifit')
        dtifit.inputs.base_name = self.inputs.subject_id
        dtifit.inputs.dwi = self.inputs.dwi
        dtifit.inputs.bvecs = self.inputs.bvec
        dtifit.inputs.bvals = self.inputs.bval

        # Getting AD and RD
        get_rd = pe.Node(interface=AdditionalDTIMeasures(), name='get_rd')

        # DataSink
        datasink = pe.Node(interface=nio.DataSink(), name='datasink')
        datasink.inputs.parameterization = False
        datasink.inputs.base_directory = self.inputs.out_directory
        datasink.inputs.container = self.inputs.subject_id

        # ==============================================================
        # Setting up the workflow
        dwi_preprocessing = pe.Workflow(name='dwi_preprocessing')

        # Diffusion data
        # Preprocessing
        dwi_preprocessing.connect(
            dwi_denoise, 'out_file', eddycorrect, 'in_file')
        dwi_preprocessing.connect(
            eddycorrect, 'eddy_corrected', fslroi, 'in_file')
        dwi_preprocessing.connect(fslroi, 'roi_file', bet, 'in_file')

        # Calculate diffusion measures
        dwi_preprocessing.connect(bet, 'mask_file', dtifit, 'mask')
        dwi_preprocessing.connect(dtifit, 'L1', get_rd, 'L1')
        dwi_preprocessing.connect(dtifit, 'L2', get_rd, 'L2')
        dwi_preprocessing.connect(dtifit, 'L3', get_rd, 'L3')

        # Connecting to the datasink
        dwi_preprocessing.connect(bet, 'out_file', datasink, 'dwi.@b0')
        dwi_preprocessing.connect(bet, 'out_file', datasink, 'dwi.@dwi')
        dwi_preprocessing.connect(bet, 'mask_file', datasink, 'dwi.@mask')
        dwi_preprocessing.connect(dtifit, 'FA', datasink, 'dwi.@FA')
        dwi_preprocessing.connect(dtifit, 'MD', datasink, 'dwi.@MD')
        dwi_preprocessing.connect(get_rd, 'AD', datasink, 'dwi.@AD')
        dwi_preprocessing.connect(get_rd, 'RD', datasink, 'dwi.@RD')

        # ==============================================================
        # Running the workflow
        dwi_preprocessing.base_dir = os.path.abspath(self.inputs.out_directory)
        dwi_preprocessing.write_graph()
        dwi_preprocessing.run()

        return runtime

    def _list_outputs(self):
        import os
        outputs = self._outputs().get()
        out_directory = self.inputs.out_directory
        subject_id = self.inputs.subject_id

        outputs["AD"] = os.path.abspath(out_directory + '/' + subject_id + '/dwi/' + subject_id + '_AD.nii.gz')
        outputs["b0"] = os.path.abspath(out_directory + '/' + subject_id + '/dwi/' + subject_id + '_b0.nii.gz')
        outputs["FA"] = os.path.abspath(out_directory + '/' + subject_id + '/dwi/' + subject_id + '_FA.nii.gz')
        outputs["MD"] = os.path.abspath(out_directory + '/' + subject_id + '/dwi/' + subject_id + '_MD.nii.gz')
        outputs["RD"] = os.path.abspath(out_directory + '/' + subject_id + '/dwi/' + subject_id + '_RD.nii.gz')
        outputs["dwi"] = os.path.abspath(out_directory + '/' + subject_id + '/dwi/' + subject_id + '_dwi.nii.gz')
        outputs["mask"] = os.path.abspath(out_directory + '/' + subject_id + '/dwi/' + subject_id + '_mask.nii.gz')

        return outputs

# ======================================================================
# T1 preprocessing & FreeSurfer reconstruction

class T1PreprocInputSpec(BaseInterfaceInputSpec):
    subject_id = traits.String(desc='subject ID')
    T1 = File(exist=True, desc='T1-weighted anatomical image')
    template_directory = File(
        exist=True, desc='directory where template files are stored')
    out_directory = File(
        exist=True, desc='directory where FreeSurfer output should be directed')


class T1PreprocOutputSpec(TraitedSpec):
    brainmask = File(exist=True, desc='brain mask generated by FreeSurfer')
    subjects_dir = File(exist=True, desc='FreeSufer subject directory')
    subject_id = traits.String(desc='subject ID')
    T1 = File(exist=True, desc='T1 file used by FreeSurfer')
    wm = File(exist=True, desc='segmented white matter volume generated by FreeSufer')


class T1Preproc(BaseInterface):
    input_spec = T1PreprocInputSpec
    output_spec = T1PreprocOutputSpec

    def _run_interface(self, runtime):
        from additional_interfaces import DipyDenoiseT1
        from additional_interfaces import FSRename
        from additional_interfaces import FS_Gyrification
        from nipype.interfaces.ants import N4BiasFieldCorrection
        from nipype.interfaces.ants.segmentation import BrainExtraction
        from nipype.interfaces.freesurfer import MRIConvert
        from nipype.interfaces.freesurfer import ReconAll
        import nipype.interfaces.fsl as fsl
        import nipype.pipeline.engine as pe
        import os

        subject_id = self.inputs.subject_id
        T1 = self.inputs.T1
        template_directory = self.inputs.template_directory
        out_directory = self.inputs.out_directory
        subjects_dir = out_directory + '/FreeSurfer/'

        if not os.path.isdir(subjects_dir):
            os.mkdir(subjects_dir)

        # Getting a better field of view
        robustfov = pe.Node(interface=fsl.RobustFOV(), name='robustfov')
        robustfov.inputs.in_file = T1

        # Denoising
        T1_denoise = pe.Node(interface=DipyDenoiseT1(), name='T1_denoise')

        # Bias field correction
        n4 = pe.Node(interface=N4BiasFieldCorrection(), name='n4')
        n4.inputs.dimension = 3
        n4.inputs.save_bias = True

        # Brain extraction
        brainextraction = pe.Node(
            interface=BrainExtraction(), name='brainextraction')
        brainextraction.inputs.dimension = 3
        brainextraction.inputs.brain_template = template_directory + '/T_template.nii.gz'
        brainextraction.inputs.brain_probability_mask = template_directory + \
            '/T_template_BrainCerebellumProbabilityMask.nii.gz'

        # Renaming files for FreeSurfer
        rename = pe.Node(FSRename(), name='rename')

        # Running FreeSurfer
        autorecon1 = pe.Node(interface=ReconAll(), name='autorecon1')
        autorecon1.inputs.subject_id = subject_id
        autorecon1.inputs.directive = 'autorecon1'
        autorecon1.inputs.args = '-noskullstrip'
        autorecon1.inputs.subjects_dir = subjects_dir

        autorecon2 = pe.Node(interface=ReconAll(), name='autorecon2')
        autorecon2.inputs.directive = 'autorecon2'

        autorecon3 = pe.Node(interface=ReconAll(), name='autorecon3')
        autorecon3.inputs.directive = 'autorecon3'

        gyrification = pe.Node(
            interface=FS_Gyrification(), name='gyrification')

        wm_convert = pe.Node(interface=MRIConvert(), name='wm_convert')
        wm_convert.inputs.out_file = subjects_dir + '/' + subject_id + '/mri/' + 'wm.nii'
        wm_convert.inputs.out_type = 'nii'

        T1_convert = pe.Node(interface=MRIConvert(), name='T1_convert')
        T1_convert.inputs.out_file = subjects_dir + '/' + subject_id + '/mri/' + 'T1.nii.gz'
        T1_convert.inputs.out_type = 'niigz'

        mask_convert = pe.Node(interface=MRIConvert(), name='mask_convert')
        mask_convert.inputs.out_file = subjects_dir + '/' + subject_id + '/mri/' + 'brainmask.nii.gz'
        mask_convert.inputs.out_type = 'niigz'

        # Connecting the pipeline
        T1_preprocessing = pe.Workflow(name='T1_preprocessing')

        T1_preprocessing.connect(robustfov, 'out_roi', T1_denoise, 'in_file')
        T1_preprocessing.connect(T1_denoise, 'out_file', n4, 'input_image')
        T1_preprocessing.connect(
            n4, 'output_image', brainextraction, 'anatomical_image')
        T1_preprocessing.connect(
            brainextraction, 'BrainExtractionBrain', autorecon1, 'T1_files')
        T1_preprocessing.connect(
            autorecon1, 'subject_id', autorecon2, 'subject_id')
        T1_preprocessing.connect(
            autorecon1, 'subjects_dir', autorecon2, 'subjects_dir')
        T1_preprocessing.connect(
            autorecon1, 'subject_id', rename, 'subject_id')
        T1_preprocessing.connect(
            autorecon1, 'subjects_dir', rename, 'subjects_dir')
        T1_preprocessing.connect(
            autorecon2, 'subject_id', autorecon3, 'subject_id')
        T1_preprocessing.connect(
            autorecon2, 'subjects_dir', autorecon3, 'subjects_dir')
        T1_preprocessing.connect(autorecon3, 'wm', wm_convert, 'in_file')
        T1_preprocessing.connect(autorecon3, 'T1', T1_convert, 'in_file')
        T1_preprocessing.connect(
            autorecon3, 'brainmask', mask_convert, 'in_file')
        T1_preprocessing.connect(
            autorecon3, 'subject_id', gyrification, 'subject_id')
        T1_preprocessing.connect(
            autorecon3, 'subjects_dir', gyrification, 'subjects_dir')

        # ==============================================================
        # Running the workflow
        T1_preprocessing.base_dir = os.path.abspath(self.inputs.out_directory)
        T1_preprocessing.run()

        return runtime

    def _list_outputs(self):
        import os

        outputs = self._outputs().get()
        directory = self.inputs.out_directory + '/FreeSurfer/' + self.inputs.subject_id
        outputs["brainmask"] = os.path.abspath(directory + '/mri/' + 'brainmask.nii.gz')
        outputs["subjects_dir"] = os.path.abspath(self.inputs.out_directory + '/FreeSurfer/')
        outputs["subject_id"] = self.inputs.subject_id
        outputs["T1"] = os.path.abspath(directory + '/mri/' + 'T1.nii.gz')
        outputs["wm"] = os.path.abspath(directory + '/mri/' + 'wm.nii')

        return outputs


# ======================================================================
# Parcellation

class SubjectSpaceParcellationInputSpec(BaseInterfaceInputSpec):
    subject_id = traits.String(desc='subject ID')
    subjects_dir = File(exist=True, desc='FreeSufer subject directory')
    source_subject = traits.String(desc='subject ID')
    source_annot_file = File(exist=True, desc='T1-weighted anatomical image')
    out_directory = File(
        exist=True, desc='directory where FreeSurfer dwi should be directed')
    wm = File(exit=True, desc='segmented white matter image')

class SubjectSpaceParcellationOutputSpec(TraitedSpec):
    subject_id = traits.String(desc='subject ID')
    subjects_dir = File(exist=True, desc='FreeSufer subject directory')
    cortical = File(exists=True, desc="cortical parcellation")
    cortical_consecutive = File(exists=True, desc="cortical parcellation with consecutive numbering")
    cortical_expanded = File(exists=True, desc="cortical parcellation expanded into WM")
    cortical_expanded_consecutive = File(exists=True, desc="cortical parcellation expanded into WM with consecutive numbering")
    leftHemisphere = File(exists=True, desc="left hemisphere parcellation")
    leftHemisphere_expanded = File(exists=True, desc="left hemisphere parcellation expanded into WM")
    orig = File(exists=True, desc="original parcellation image")
    renum = File(exists=True, desc="renumbered parcellation")
    renum_expanded = File(exists=True, desc="renumbered parcellation expanded into WM")
    renum_subMask = File(exists=True, desc="renumbered parcellation with subcortical regions masked out")
    rightHemisphere = File(exists=True, desc="parcellation of the right hemisphere")
    rightHemisphere_expanded = File(exists=True, desc="parcellation of the right hemisphere expanded into WM")
    subcortical = File(exists=True, desc="parcellation of subcortical regions")
    subcortical_expanded = File(exists=True, desc="parcellation of subcortical regions expanded into WM")
    whiteMatter = File(exists=True, desc="white matter partial volume")
    whiteMatter_expanded = File(exists=True, desc="white matter partial image after expansion of cortical parcellation into WM")
    boundary_lh_rh = File(exists=True, desc="boundary label between hemisphere")
    boundary_sub_lh = File(exists=True, desc="oundary label between cortical and subcortical")
    aparc = File(exists=True, desc="DK atlas")
    aparc_subMask = File(exists=True, desc="DK atlas with subcortical regions masked out")

class SubjectSpaceParcellation(BaseInterface):
    input_spec = SubjectSpaceParcellationInputSpec
    output_spec = SubjectSpaceParcellationOutputSpec

    def _run_interface(self, runtime):
        from additional_interfaces import Aparc2Aseg
        from additional_interfaces import ExpandParcels
        from additional_interfaces import SurfaceTransform
        from additional_interfaces import ReunumberParcels
        import nipype.pipeline.engine as pe
        import os

        subject_id = self.inputs.subject_id
        subjects_dir = self.inputs.subjects_dir
        source_subject = self.inputs.source_subject
        source_annot_file = self.inputs.source_annot_file
        out_directory = self.inputs.out_directory
        wm = self.inputs.wm

        # Moving subparcellation of the atlas to subject space
        sxfm = pe.Node(interface=SurfaceTransform(), name='sxfm')
        sxfm.inputs.subject_id = subject_id
        sxfm.inputs.target_subject = subject_id
        sxfm.inputs.source_annot_file = source_annot_file
        sxfm.inputs.source_subject = source_subject
        sxfm.inputs.subjects_dir = subjects_dir
        sxfm.iterables = ('hemi', ['lh', 'rh'])

        # Transforming surface parcellation to volume
        aparc2aseg = pe.Node(interface=Aparc2Aseg(), name='aparc2aseg')
        aparc2aseg.inputs.subjects_dir = subjects_dir
        aparc2aseg.inputs.annotation_file = source_annot_file
        aparc2aseg.inputs.hemi = 'lh'

        # Dilating parcellation into the white matter
        expand = pe.Node(interface=ExpandParcels(), name='expand')
        expand.inputs.white_matter_image = wm
        expand.inputs.subjects_dir = subjects_dir
        expand.inputs.parcellation_name = source_annot_file
        expand.inputs.dilatationVoxel = 2


        renum = pe.Node(interface=ReunumberParcels(), name='renum')
        renum.inputs.subjects_dir = subjects_dir
        renum.inputs.parcellation_name = source_annot_file

        # Connecting the pipeline
        subject_parcellation = pe.Workflow(name='subject_parcellation')

        subject_parcellation.connect(
            sxfm, 'subject_id', aparc2aseg, 'subject_id')
        subject_parcellation.connect(
            aparc2aseg, 'volume_parcellation', expand, 'parcellation_file')
        subject_parcellation.connect(
            sxfm, 'subject_id', expand, 'subject_id')
        subject_parcellation.connect(
                expand, 'subject_id', renum, 'subject_id')

        # ==============================================================
        # Running the workflow
        subject_parcellation.base_dir = os.path.abspath(self.inputs.out_directory)
        subject_parcellation.run()

        return runtime

    def _list_outputs(self):
        from nipype.utils.filemanip import split_filename
        import os

        outputs = self._outputs().get()
        path_subj = self.inputs.subjects_dir + '/' + self.inputs.subject_id + '/'
        parcellation_name = self.inputs.source_annot_file

        outputs["cortical"] = os.path.abspath(path_subj + 'parcellation/' + parcellation_name + '_cortical.nii.gz')
        outputs["cortical_consecutive"] = os.path.abspath(path_subj + 'parcellation/' + parcellation_name + '_cortical_consecutive.nii.gz')
        outputs["cortical_expanded"] = os.path.abspath(path_subj + 'parcellation/' + parcellation_name + '_cortical_expanded.nii.gz')
        outputs["cortical_expanded_consecutive"] = os.path.abspath(path_subj + 'parcellation/' + parcellation_name +  '_cortical_expanded_consecutive.nii.gz')
        outputs["leftHemisphere"] = os.path.abspath(path_subj + 'parcellation/' + parcellation_name + '_leftHemisphere.nii.gz')
        outputs["leftHemisphere_expanded"] = os.path.abspath(path_subj + 'parcellation/' + parcellation_name + '_leftHemisphere_expanded.nii.gz')
        outputs["orig"] = os.path.abspath(path_subj + 'parcellation/' + parcellation_name + '_orig.nii.gz')
        outputs["renum"] = os.path.abspath(path_subj + 'parcellation/' + parcellation_name + '_renum.nii.gz')
        outputs["renum_expanded"] = os.path.abspath(path_subj + 'parcellation/' + parcellation_name + '_renum_expanded.nii.gz')
        outputs["renum_subMask"] = os.path.abspath(path_subj + 'parcellation/' + parcellation_name + '_renum_subMask.nii.gz')
        outputs["rightHemisphere"] = os.path.abspath(path_subj + 'parcellation/' + parcellation_name + '_rightHemisphere.nii.gz')
        outputs["rightHemisphere_expanded"] = os.path.abspath(path_subj + 'parcellation/' + parcellation_name + '_rightHemisphere_expanded.nii.gz')
        outputs["subcortical"] = os.path.abspath(path_subj + 'parcellation/' +parcellation_name +  '_subcortical.nii.gz')
        outputs["subcortical_expanded"] = os.path.abspath(path_subj + 'parcellation/' + parcellation_name +  '_subcortical_expanded.nii.gz')
        outputs["whiteMatter"] = os.path.abspath(path_subj + 'parcellation/' + parcellation_name + '_whiteMatter.nii.gz')
        outputs["whiteMatter_expanded"] = os.path.abspath(path_subj + 'parcellation/' + parcellation_name + '_whiteMatter_expanded.nii.gz')
        outputs["boundary_lh_rh"] = os.path.abspath(path_subj + 'parcellation/' + 'boundary_lh_rh.txt')
        outputs["boundary_sub_lh"] = os.path.abspath(path_subj + 'parcellation/' + 'boundary_sub_lh.txt')
        return outputs
