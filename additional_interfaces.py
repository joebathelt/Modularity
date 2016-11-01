from nipype.interfaces.base import BaseInterface
from nipype.interfaces.base import BaseInterfaceInputSpec
from nipype.interfaces.base import CommandLineInputSpec
from nipype.interfaces.base import CommandLine
from nipype.interfaces.base import File
from nipype.interfaces.base import TraitedSpec

# ==================================================================
"""
Denoising with non-local means
This function is based on the example in the Dipy preprocessing tutorial:
http://nipy.org/dipy/examples_built/denoise_nlmeans.html#example-denoise-nlmeans
"""

class DipyDenoiseInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True, desc='diffusion weighted volume for denoising', mandatory=True)


class DipyDenoiseOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="denoised diffusion-weighted volume")


class DipyDenoise(BaseInterface):
    input_spec = DipyDenoiseInputSpec
    output_spec = DipyDenoiseOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np
        from dipy.denoise.nlmeans import nlmeans
        from nipype.utils.filemanip import split_filename

        fname = self.inputs.in_file
        img = nib.load(fname)
        data = img.get_data()
        affine = img.get_affine()
        mask = data[..., 0] > 80
        a = data.shape

        denoised_data = np.ndarray(shape=data.shape)
        for image in range(0, a[3]):
            print(str(image + 1) + '/' + str(a[3] + 1))
            dat = data[..., image]
            # Calculating the standard deviation of the noise
            sigma = np.std(dat[~mask])
            den = nlmeans(dat, sigma=sigma, mask=mask)
            denoised_data[:, :, :, image] = den

        _, base, _ = split_filename(fname)
        nib.save(nib.Nifti1Image(denoised_data, affine),
                 base + '_denoised.nii')

        return runtime

    def _list_outputs(self):
        from nipype.utils.filemanip import split_filename
        import os
        outputs = self._outputs().get()
        fname = self.inputs.in_file
        _, base, _ = split_filename(fname)
        outputs["out_file"] = os.path.abspath(base + '_denoised.nii')
        return outputs

# ======================================================================
# Extract b0

class Extractb0InputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True, desc='diffusion-weighted image (4D)', mandatory=True)


class Extractb0OutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="First volume of the dwi file")


class Extractb0(BaseInterface):
    input_spec = Extractb0InputSpec
    output_spec = Extractb0OutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        img = nib.load(self.inputs.in_file)
        data = img.get_data()
        affine = img.get_affine()

        from nipype.utils.filemanip import split_filename
        import os
        outputs = self._outputs().get()
        fname = self.inputs.in_file
        _, base, _ = split_filename(fname)
        nib.save(nib.Nifti1Image(data[..., 0], affine),
                 os.path.abspath(base + '_b0.nii.gz'))
        return runtime

    def _list_outputs(self):
        from nipype.utils.filemanip import split_filename
        import os
        outputs = self._outputs().get()
        fname = self.inputs.in_file
        _, base, _ = split_filename(fname)
        outputs["out_file"] = os.path.abspath(base + '_b0.nii.gz')
        return outputs

# ======================================================================
# FA connectome construction


class FAconnectomeInputSpec(BaseInterfaceInputSpec):
    trackfile = File(
        exists=True, desc='whole-brain tractography in .trk format', mandatory=True)
    ROI_file = File(
        exists=True, desc='image containing the ROIs', mandatory=True)
    FA_file = File(
        exists=True, desc='fractional anisotropy map in the same soace as the track file', mandatory=True)
    output_file = File(
        "FA_matrix.txt", desc="Adjacency matrix of ROIs with FA as conenction weight", usedefault=True)


class FAconnectomeOutputSpec(TraitedSpec):
    out_file = File(
        exists=True, desc="connectivity matrix of FA between each pair of ROIs")


class FAconnectome(BaseInterface):
    input_spec = FAconnectomeInputSpec
    output_spec = FAconnectomeOutputSpec

    def _run_interface(self, runtime):
        # Loading the ROI file
        import nibabel as nib
        import numpy as np
        from dipy.tracking import utils

        img = nib.load(self.inputs.ROI_file)
        data = img.get_data()
        affine = img.get_affine()

        # Getting the FA file
        img = nib.load(self.inputs.FA_file)
        FA_data = img.get_data()
        FA_affine = img.get_affine()

        # Loading the streamlines
        from nibabel import trackvis
        streams, hdr = trackvis.read(
            self.inputs.trackfile, points_space='rasmm')
        streamlines = [s[0] for s in streams]
        streamlines_affine = trackvis.aff_from_hdr(hdr, atleast_v2=True)

        # Checking for negative values
        from dipy.tracking._utils import _mapping_to_voxel, _to_voxel_coordinates
        endpoints = [sl[0::len(sl) - 1] for sl in streamlines]
        lin_T, offset = _mapping_to_voxel(affine, (1., 1., 1.))
        inds = np.dot(endpoints, lin_T)
        inds += offset
        negative_values = np.where(inds < 0)[0]
        for negative_value in sorted(negative_values, reverse=True):
            del streamlines[negative_value]

        # Constructing the streamlines matrix
        matrix, mapping = utils.connectivity_matrix(
            streamlines=streamlines, label_volume=data, affine=streamlines_affine, symmetric=True, return_mapping=True, mapping_as_streamlines=True)
        matrix[matrix < 10] = 0

        # Constructing the FA matrix
        dimensions = matrix.shape
        FA_matrix = np.empty(shape=dimensions)

        for i in range(0, dimensions[0]):
            for j in range(0, dimensions[1]):
                if matrix[i, j]:
                    dm = utils.density_map(
                        mapping[i, j], FA_data.shape, affine=streamlines_affine)
                    FA_matrix[i, j] = np.mean(FA_data[dm > 5])
                else:
                    FA_matrix[i, j] = 0

        FA_matrix[np.tril_indices(n=len(FA_matrix))] = 0
        FA_matrix = FA_matrix.T + FA_matrix - np.diagonal(FA_matrix)

        from nipype.utils.filemanip import split_filename
        _, base, _ = split_filename(self.inputs.trackfile)
        np.savetxt(base + '_FA_matrix.txt', FA_matrix, delimiter='\t')
        return runtime

    def _list_outputs(self):
        from nipype.utils.filemanip import split_filename
        import os
        outputs = self._outputs().get()
        fname = self.inputs.trackfile
        _, base, _ = split_filename(fname)
        outputs["out_file"] = os.path.abspath(base + '_FA_matrix.txt')
        return outputs

