#! /usr/bin/env python
import optparse
import os
import re
import sys


# ======================================================================

def main():
    p = optparse.OptionParser()

    p.add_option('--base_directory', '-b')
    p.add_option('--subject_list', '-s')
    p.add_option('--template_directory', '-t')
    p.add_option('--out_directory', '-o')
    p.add_option('--parcellation_directory', '-p')
    p.add_option('--acquisition_parameters', '-a')
    p.add_option('--index_file', '-i')
    sys.path.append(os.path.realpath(__file__))

    options, arguments = p.parse_args()
    base_directory = options.base_directory
    out_directory = options.out_directory
    subject_list = options.subject_list
    subject_list = [subject for subject in subject_list.split(
        ',') if re.search('CBU', subject)]
    template_directory = options.template_directory
    parcellation_directory = options.parcellation_directory
    acquisition_parameters = options.acquisition_parameters
    index_file = options.index_file
    subjects_dir = out_directory + '/connectome/FreeSurfer/'
    os.environ['SUBJECTS_DIR'] = subjects_dir

    def connectome(subject_list, base_directory, out_directory):

        # ==================================================================
        # Loading required packages
        import nipype.pipeline.engine as pe
        import nipype.interfaces.utility as util
        from nipype.interfaces.freesurfer import ApplyVolTransform
        from nipype.interfaces.freesurfer import BBRegister
        import nipype.interfaces.fsl as fsl
        import nipype.interfaces.dipy as dipy
        import nipype.interfaces.diffusion_toolkit as dtk
        import nipype.algorithms.misc as misc
        from additional_interfaces import CSDdet
        from additional_interfaces import DipyDenoise
        from additional_pipelines import DWIPreproc
        from additional_interfaces import CalcMatrix
        from additional_pipelines import T1Preproc
        from additional_pipelines import SubjectSpaceParcellation
        from own_nipype import Extractb0 as extract_b0

        from nipype import SelectFiles
        import os

        # ==================================================================
        # Defining the nodes for the workflow

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
        t1_preproc.inputs.out_directory = out_directory + '/connectome/'
        t1_preproc.inputs.template_directory = template_directory

        # DWI processing
        dwi_preproc = pe.Node(interface=DWIPreproc(), name='dwi_preproc')
        dwi_preproc.inputs.out_directory = out_directory + '/connectome/'
        dwi_preproc.inputs.acqparams = acquisition_parameters
        dwi_preproc.inputs.index_file = index_file
        dwi_preproc.inputs.out_directory = out_directory + '/connectome/'

        # Eroding the brain mask
        erode_mask = pe.Node(interface=fsl.maths.ErodeImage(), name='erode_mask')

        # CSD deterministic tractography
        csd_det = pe.Node(interface=CSDdet(), name='csd_det')

        # smoothing the tracts
        smooth = pe.Node(interface=dtk.SplineFilter(
            step_length=0.5), name='smooth')

        # Moving to subject space
        subject_parcellation = pe.Node(interface=SubjectSpaceParcellation(), name='subject_parcellation')
        subject_parcellation.inputs.source_subject = 'fsaverage'
        subject_parcellation.inputs.source_annot_file = 'aparc'
        subject_parcellation.inputs.out_directory = out_directory + '/connectome/'
        subject_parcellation.inputs.parcellation_directory = parcellation_directory

        # Co-registering T1 and dwi
        bbreg = pe.Node(interface=BBRegister(), name='bbreg')
        bbreg.inputs.init='fsl'
        bbreg.inputs.contrast_type='t2'

        applyreg = pe.Node(interface=ApplyVolTransform(), name='applyreg')
        applyreg.inputs.interp = 'nearest'
        applyreg.inputs.inverse = True

        # calcuating the connectome matrix
        calc_matrix = pe.Node(interface=CalcMatrix(), name='calc_matrix')
        calc_matrix.inputs.threshold = 1

        # ==================================================================
        # Setting up the workflow
        connectome = pe.Workflow(name='connectome')

        # Reading in files
        connectome.connect(infosource, 'subject_id', selectfiles, 'subject_id')

        # DWI preprocessing
        connectome.connect(infosource, 'subject_id', dwi_preproc, 'subject_id')
        connectome.connect(selectfiles, 'dwi', dwi_preproc, 'dwi')
        connectome.connect(selectfiles, 'bval', dwi_preproc, 'bvals')
        connectome.connect(selectfiles, 'bvec', dwi_preproc, 'bvecs')

        # CSD model and streamline tracking
        connectome.connect(dwi_preproc, 'mask', erode_mask, 'in_file')

        connectome.connect(selectfiles, 'bvec', csd_det, 'bvec')
        connectome.connect(selectfiles, 'bval', csd_det, 'bval')
        connectome.connect(dwi_preproc, 'dwi', csd_det, 'in_file')
        connectome.connect(dwi_preproc, 'FA', csd_det, 'FA')
        connectome.connect(erode_mask, 'out_file', csd_det, 'brain_mask')

        # Smoothing the trackfile
        connectome.connect(csd_det, 'out_track', smooth, 'track_file')

        # Preprocessing the T1-weighted file
        connectome.connect(infosource, 'subject_id', t1_preproc, 'subject_id')
        connectome.connect(selectfiles, 'T1', t1_preproc, 'T1')
        connectome.connect(t1_preproc, 'wm', subject_parcellation, 'wm')
        connectome.connect(t1_preproc, 'subjects_dir', subject_parcellation, 'subjects_dir')
        connectome.connect(t1_preproc, 'subject_id', subject_parcellation, 'subject_id')

        # Getting the parcellation into diffusion space
        connectome.connect(t1_preproc, 'subject_id', bbreg, 'subject_id')
        connectome.connect(t1_preproc, 'subjects_dir', bbreg, 'subjects_dir')
        connectome.connect(dwi_preproc, 'b0', bbreg, 'source_file')

        connectome.connect(dwi_preproc, 'b0', applyreg, 'source_file')
        connectome.connect(bbreg, 'out_reg_file', applyreg, 'reg_file')
        connectome.connect(subject_parcellation, 'renum_expanded', applyreg, 'target_file')

        # Calculating the FA connectome
        connectome.connect(csd_det, 'out_file', calc_matrix, 'track_file')
        connectome.connect(dwi_preproc, 'FA', calc_matrix, 'scalar_file')
        connectome.connect(applyreg, 'transformed_file', calc_matrix, 'ROI_file')

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
