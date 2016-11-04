#! /usr/bin/env python
import optparse
import os
import re
import sys


# ======================================================================

def main():
    p = optparse.OptionParser()

    p.add_option('--base_directory', '-b')
    p.add_option('--out_directory', '-o')
    p.add_option('--subject_list', '-s')

    options, arguments = p.parse_args()
    base_directory = options.base_directory
    out_directory = options.out_directory
    subject_list = options.subject_list
    subject_list = [subject for subject in subject_list.split(
        ',') if re.search('CBU', subject)]

    def connectome(subject_list, base_directory, out_directory):

        # ==================================================================
        # Loading required packages
        import nipype.pipeline.engine as pe
        import nipype.interfaces.utility as util
        import nipype.interfaces.fsl as fsl
        import nipype.interfaces.dipy as dipy
        import nipype.interfaces.mrtrix as mrt
        import nipype.interfaces.diffusion_toolkit as dtk
        import nipype.algorithms.misc as misc
        from additional_interfaces import FAconnectome
        from additional_interfaces import DipyDenoise
        from additional_pipelines import DWIPreproc
        from additional_pipelines import T1Preproc
        from additional_pipelines import SubjectSpaceParcellation
        from own_nipype import Extractb0 as extract_b0

        from nipype import SelectFiles
        import os
        registration_reference = os.environ[
            'FSLDIR'] + '/data/standard/MNI152_T1_1mm_brain.nii.gz'
        nodes = list()

        # ==================================================================
        # Defining the nodes for the workflow

        # Utility nodes
        gunzip = pe.Node(interface=misc.Gunzip(), name='gunzip')
        gunzip2 = pe.Node(interface=misc.Gunzip(), name='gunzip2')
        fsl2mrtrix = pe.Node(interface=mrt.FSL2MRTrix(
            invert_x=True), name='fsl2mrtrix')

        # Getting the subject ID
        infosource = pe.Node(interface=util.IdentityInterface(
            fields=['subject_id']), name='infosource')
        infosource.iterables = ('subject_id', subject_list)

        # Getting the relevant diffusion-weighted data
        templates = dict(T1='{subject_id}/anat/{subject_id}_T1w.nii.gz',
                         dwi='{subject_id}/dwi/{subject_id}_dwi.nii.gz',
                         bvec='{subject_id}/dwi/{subject_id}_dwi.bvec',
                         bval='{subject_id}/dwi/{subject_id}_dwi.bval')

        selectfiles = pe.Node(SelectFiles(templates),
                              name='selectfiles')
        selectfiles.inputs.base_directory = os.path.abspath(base_directory)

        # ==============================================================
        # T1 processing
        t1_preproc = pe.Node(interface=T1Preproc(), name='t1_preproc')

        # DWI processing
        dwi_preproc = pe.Node(interface=DWIPreproc(), name='dwi_preproc')
        dwi_preproc.inputs.out_directory = out_directory

        gunzip = pe.Node(interface=misc.Gunzip(), name='gunzip')
        fsl2mrtrix = pe.Node(interface=mrt.FSL2MRTrix(invert_x=True), name='fsl2mrtrix')

        # Eroding the brain mask
        erode_mask_firstpass = pe.Node(
            interface=mrt.Erode(), name='erode_mask_firstpass')
        erode_mask_secondpass = pe.Node(
            interface=mrt.Erode(), name='erode_mask_secondpass')
        MRmultiply = pe.Node(interface=mrt.MRMultiply(), name='MRmultiply')
        MRmult_merge = pe.Node(interface=util.Merge(2),
                               name='MRmultiply_merge')
        threshold_FA = pe.Node(interface=mrt.Threshold(
            absolute_threshold_value=0.7), name='threshold_FA')

        # White matter mask
        gen_WM_mask = pe.Node(
            interface=mrt.GenerateWhiteMatterMask(), name='gen_WM_mask')
        threshold_wmmask = pe.Node(interface=mrt.Threshold(
            absolute_threshold_value=0.4), name='threshold_wmmask')

        # CSD probabilistic tractography
        estimateresponse = pe.Node(interface=mrt.EstimateResponseForSH(
            maximum_harmonic_order=8), name='estimateresponse')
        csdeconv = pe.Node(interface=mrt.ConstrainedSphericalDeconvolution(
            maximum_harmonic_order=8), name='csdeconv')

        # Tracking
        probCSDstreamtrack = pe.Node(
            interface=mrt.ProbabilisticSphericallyDeconvolutedStreamlineTrack(), name='probCSDstreamtrack')
        probCSDstreamtrack.inputs.inputmodel = 'SD_PROB'
        probCSDstreamtrack.inputs.desired_number_of_tracks = 150000
        tck2trk = pe.Node(interface=mrt.MRTrix2TrackVis(), name='tck2trk')

        # smoothing the tracts
        smooth = pe.Node(interface=dtk.SplineFilter(
            step_length=0.5), name='smooth')

        # Moving to subject space
        subject_parcellation = pe.Node(SubjectSpaceParcellation(), name='subject_parcellation')
        subject_parcellation.inputs.source_subject = 'fsaverage'
        subject_parcellation.inputs.source_annot_file = 'aparc.a2009s'
        subject_parcellation.inputs.out_directory = out_directory

        # Co-registering FA and T1
        flt = pe.Node(interface=fsl.FLIRT(reference=registration_reference, dof=6, cost_func='mutualinfo'), name='flt')

        dwi_to_T1_flirt = pe.Node(interface=fsl.FLIRT(), name='dwi_to_T1_flirt')
        dwi_to_T1_flirt.inputs.cost_func = 'mutualinfo'
        dwi_to_T1_flirt.inputs.dof = 6
        dwi_to_T1_flirt.inputs.out_matrix_file = 'subjectDWI_to_T1.mat'

        dwi_to_t1 = pe.Node(interface=fsl.ApplyXfm(apply_xfm=True), name='dwi_to_t1')

        # calcuating the connectome matrix
        calc_matrix = pe.Node(interface=FAconnectome(), name='calc_matrix')

        # ==================================================================
        # Setting up the workflow
        connectome = pe.Workflow(name='connectome')

        # Reading in files
        connectome.connect(infosource, 'subject_id', selectfiles, 'subject_id')

        # DWI preprocessing
        connectome.connect(infosource, 'subject_id', dwi_preproc, 'subject_id')
        connectome.connect(selectfiles, 'dwi', dwi_preproc, 'dwi')
        connectome.connect(selectfiles, 'bval', dwi_preproc, 'bval')
        connectome.connect(selectfiles, 'bvec', dwi_preproc, 'bvec')

        # Thresholding to create a mask of single fibre voxels
        connectome.connect(dwi_preproc, 'FA', MRmult_merge, 'in1')
        connectome.connect(dwi_preproc, 'mask', erode_mask_firstpass, 'in_file')
        connectome.connect(erode_mask_firstpass, 'out_file',
                           erode_mask_secondpass, 'in_file')
        connectome.connect(erode_mask_secondpass, 'out_file', MRmult_merge, 'in2')
        connectome.connect(MRmult_merge, 'out', MRmultiply, 'in_files')
        connectome.connect(MRmultiply, 'out_file', threshold_FA, 'in_file')

        # Create seed mask
        connectome.connect(dwi_preproc, 'dwi', gen_WM_mask, 'in_file')
        connectome.connect(dwi_preproc, 'mask', gen_WM_mask, 'binary_mask')
        connectome.connect(fsl2mrtrix, 'encoding_file', gen_WM_mask, 'encoding_file')
        connectome.connect(gen_WM_mask, 'WMprobabilitymap', threshold_wmmask, 'in_file')

        # Estimate response
        connectome.connect(selectfiles, 'bval', fsl2mrtrix, 'bval_file')
        connectome.connect(selectfiles, 'bvec', fsl2mrtrix, 'bvec_file')
        connectome.connect(fsl2mrtrix, 'encoding_file', estimateresponse, 'encoding_file')
        connectome.connect(dwi_preproc, 'dwi', estimateresponse, 'in_file')
        connectome.connect(threshold_FA, 'out_file', estimateresponse, 'mask_image')

        # CSD calculation
        connectome.connect(dwi_preproc, 'dwi', csdeconv, 'in_file')
        connectome.connect(gen_WM_mask, 'WMprobabilitymap', csdeconv, 'mask_image')
        connectome.connect(estimateresponse, 'response', csdeconv, 'response_file')
        connectome.connect(fsl2mrtrix, 'encoding_file', csdeconv, 'encoding_file')

        # Running the tractography
        connectome.connect(threshold_wmmask, 'out_file', probCSDstreamtrack, 'seed_file')
        connectome.connect(csdeconv, 'spherical_harmonics_image', probCSDstreamtrack, 'in_file')
        connectome.connect(dwi_preproc, 'dwi', tck2trk, 'image_file')
        connectome.connect(probCSDstreamtrack, 'tracked', tck2trk, 'in_file')

        # Smoothing the trackfile
        connectome.connect(tck2trk, 'out_file', smooth, 'track_file')

        # Moving T1 to dwi space
        connectome.connect(dwi_preproc, 'b0', flt, 'reference')
        connectome.connect(selectfiles, 'T1', flt, 'in_file')
        connectome.connect(flt, 'out_file', t1_preproc, 'T1')
        connectome.connect(t1_preproc, 'wm', subject_parcellation, 'wm')
        connectome.connect(t1_preproc, 'subjects_dir', subject_parcellation, 'subjects_dir')
        connectome.connect(t1_preproc, 'subject_id', subject_parcellation, 'subject_id')

        # Calculating the FA connectome
        connectome.connect(smooth, 'smoothed_track_file', calc_matrix, 'trackfile')
        connectome.connect(dwi_preproc, 'FA', calc_matrix, 'FA_file')
        connectome.connect(subject_parcellation, 'cortical_expanded', calc_matrix, 'ROI_file')

        # ==================================================================
        # Running the workflow
        connectome.base_dir = os.path.abspath(out_directory)
        connectome.write_graph()
        connectome.run()

    os.chdir(out_directory)
    connectome(subject_list, base_directory, out_directory)

if __name__ == '__main__':
    # main should return 0 for success, something else (usually 1) for error.
    sys.exit(main())
