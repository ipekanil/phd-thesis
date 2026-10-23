import os
import torch
from torchvision import transforms
import numpy as np

class PermuteTransform:
    """
    A transformation class to permute the dimensions of an image tensor.
    """
    def __init__(self, order):
        self.order = order

    def __call__(self, img):
        """
        Assumes img is a Tensor.
        """
        return img.permute(self.order)

"""

class PercentileNormalize3D:
    def __init__(self, p_low=1.0, p_high=99.7, eps=1e-8, return_params=False):
        self.p_low = p_low
        self.p_high = p_high
        self.eps = eps
        self.return_params = return_params

    def __call__(self, vol):
        v = vol.float()
        lo = torch.quantile(v, self.p_low / 100.0)
        hi = torch.quantile(v, self.p_high / 100.0)
        v_norm = (v - lo) / (hi - lo + self.eps)
        v_norm = v_norm.clamp(0.0, 1.0)
        if self.return_params:
            return v_norm, lo, hi
        return v_norm
"""
class PercentileNormalize3D:
    def __init__(self, p_low=0.5, p_high=99.5, eps=1e-6, return_params=True):
        self.p_low = p_low
        self.p_high = p_high
        self.eps = eps
        self.return_params = return_params

    def __call__(self, v):
        v = v.float()

        flat_np = v.detach().cpu().numpy().reshape(-1)
        lo = np.percentile(flat_np, self.p_low)
        hi = np.percentile(flat_np, self.p_high)

        lo_t = torch.tensor(lo, dtype=v.dtype, device=v.device)
        hi_t = torch.tensor(hi, dtype=v.dtype, device=v.device)

        out = (v - lo_t) / (hi_t - lo_t + self.eps)
        out = out.clamp(0, 1)

        if self.return_params:
            return out, lo_t, hi_t
        return out


class Config:
    def __init__(self):

        # Optical system parameters
        self.optics_params = {
            "emission_wavelength": 633e-9,
            "objective": {
                "magnification": 60,
                "na": 1.1,
                "medium_index": 1.33,
                "focal_length": 3e-3,
                "distance_to_back_aperture": 180e-3,
                "pupil_diameter": None,
            },
            "tube_lens": {
                "focal_length": 180e-3
            },
            "relay_lens": {
                "focal_length": 300e-3
            },           
            "camera": {
                "sensor_size": None,
                "pixel_count": None,
                "pixel_size": 6.5e-6, #3.248707533234859e-06, #
                "distance_from_phasemask": 43.1e-3 #same as MLA focal_length
            },
            "microlens": {
                "array_size": None,
                "pitch": 2.12e-3,
                "focal_length": None,
                "f_number": None,
                "profile": 'circ',
                "fill_factor": 100,
                # Add physical diameter bounds for learnable lenslets:
                "min_diameter": 0.25 * 2.12e-3, #0.5 * 2.12e-3,  # e.g. 50% of pitch (1.06e-3 m)
                "max_diameter": 1.75 * 2.12e-3,   #1.5 * 2.12e-3,        # e.g. 150% of pitch 
                "sampling_AS": None,
                "sampling": None,
                "DOF_target": 60e-6, #30e-6, #+-15 micron volume learn  # 200e-6
                "f_min": None,
                "f_max": None
            },
            "meta_unit": {
                "cell_size": 225e-9,
            },
            "demagnification": None,
            "M": None, #number of pixels in the simulation
            "L": None, #side length of the diffuser(meters)
            "axial_resolution": None,
            "lateral_resolution": None,
            "M_T": None,
            "blur_kernel_size_as": None,
            "blur_kernel_size_fs": None,
            "blur_kernel_size_fp": None,    
            "blur_kernel_size_sp": None,
            "padding_fp": None,
            "psf_sampling_fs": None,
            "psf_sampling_sp": None,
            "object_pixel_size": None,
            "volume_depth_min": -30e-6, #-15e-6,  -100e-6
            "volume_depth_max": +30e-6, #+15e-6,  +100e-6
            "depth_resolution": 1.4e-6 #0.7e-6      2.5e-6      # Depth resolution at object plane (meters) 3X oversampling = (Axial resolution/data z resolution)
        }

        # Compute dependent optical pa5rameters
        self.compute_optics_params()

        depth_min = self.optics_params["volume_depth_min"]
        depth_max = self.optics_params["volume_depth_max"]
        dz = self.optics_params["depth_resolution"]

        n_slices = int(np.ceil((depth_max - depth_min) / dz))
        self.optics_params["n_slices"] = n_slices


        # Print computed optical5 parameters
        self.print_optics_params()

        

        # Reconstruction model parameters
        #scale_factor = 1
        mla_size = float(self.optics_params["microlens"]["array_size"])
        M = float(self.optics_params["M"])
        base_out = int(np.floor(M / mla_size))        
        out_hw = int((base_out / 2) ) #* scale_factor)# divided by 2 comes from PSF resampling for 6.5 µm sensor pixels
        
        self.recon_params = {
            "input_channels": 43,
            "n_slices": int(self.optics_params["n_slices"]),
            #"scale_factor": scale_factor,
            "output_size": [512, 512],
            "channels_interp": 64,
            "view_num": 1, 
            "patch_size": 128,
        }

        # Training parameters
        self.train_params = {
            # Master RNG seed. Set to None for non-deterministic runs.
            # Seeds python/numpy/torch/cuda in train_dp.py, and the f_rand
            # initialisation in F_Optics, so a run can be reproduced exactly.
            "seed": 1234,

            # Mixed Poisson-Gaussian sensor noise:
            #     y = [Pois(x*photons) + read_noise_e*N(0,1)] / photons
            # i.e. the old additive Gaussian with shot noise added on top.
            # `photons` = expected photons in the brightest pixel.
            "sensor_noise": {
                "photons": 100.0,                      # training / validation
                "read_noise_e": 1.0,                   # ELECTRONS — fixed across the sweep
                "photons_test": [10, 30, 100, 300, 1000],   # swept at test time
            },

            "batch_size": 1,
            "num_epochs": 20,  # Total number of epochs

            # Step-wise learning parameters
            # lr_digital was 2e-5. That is too small for the reconstruction net to
            # remove the coarse volume's background pedestal: F_Recon computes
            # out = x_in + 0.2*conv, so the conv branch has to emit a constant of
            # about -1.28 to bring the coarse median (~0.44) down to the target
            # median (~0.19). With Adam an individual parameter moves at most
            # ~lr per step, so the OUTPUT offset moves at most 0.2*lr per step.
            # At 2e-5 that is 4e-6 per step -- which is exactly the rate val/bias
            # was observed to fall (5e-4 per epoch over ~125 steps). The offset
            # was learning-rate limited, not loss limited.
            "step_1": {
                "epoch": 0,  # Start Step 1 at epoch 0
                "lr_optical": 1e-4,  # No learning for optics in Step 1
                "lr_digital": 2e-4,  # Learning rate for digital (reconstruction)
                # load pretrained models
                "optics_ckpt_path": "./pretrained/optics_model.pth",
                "recon_ckpt_path": "./pretrained/recon_model.pth",            },

            "step_2": {
                "epoch": 1,  # Switch to Step 2 at epoch 
                "lr_optical": 1e-4,  
                "lr_digital": 2e-4   # must match step_1, or epoch 1 undoes the change
            }
        }


        # Path configurations
        # Data and checkpoint locations. Override with environment variables:
        #   FLFM_DATA_DIR  folder with train/, val/ and test/ sub-folders of
        #                  HDF5 volumes (dataset key "volume", Z x Y x X)
        #   FLFM_CKPT_DIR  where trained optics / reconstruction models go
        _data = os.environ.get("FLFM_DATA_DIR", "./data")
        _ckpt = os.environ.get("FLFM_CKPT_DIR", "./checkpoints")
        self.paths = {
            "model_optics_path": os.path.join(_ckpt, "optics_model.pth"),
            "model_recon_path": os.path.join(_ckpt, "recon_model.pth"),
            "stack_dir": os.path.join(_data, "train"),
            "val_stack_dir": os.path.join(_data, "val"),
            "test_stack_dir": os.path.join(_data, "test"),
        }

        # Loss configurations
        self.loss = {
                # STAGE-3 weights, read by train_dp.get_recon_loss_weights().
                #
                # Rebalanced towards the two terms that pay for STRUCTURE, since
                # PSNR, SSIM and NRMSE are all bounded by the correlation rho
                # between reconstruction and target:
                #       NRMSE_min = sqrt(1 - rho^2)
                # Measured on the 50-epoch tikhonov run: rho = 0.365, and the
                # affine-corrected NRMSE came out 0.9311 against sqrt(1-rho^2) =
                # 0.9310. So the reconstruction is already AT the ceiling rho
                # allows -- nothing improves until rho does.
                #
                #   ssim 0.1 -> 0.4 : SSIM was the SMALLEST weight in the loss
                #                     while being one of the reported metrics.
                #   edge 0.2 -> 0.4 : the other term that pays for contrast.
                #                     MAE alone returns the conditional median,
                #                     i.e. a blurred estimate, which caps rho.
                #   axial 1.0 -> 0.35: it subtracts each slice's mean before
                #                     comparing, so it is indifferent to exactly
                #                     the intensity accuracy PSNR and NRMSE
                #                     measure, yet carried the joint-largest
                #                     weight.
                #   mae 1.0 unchanged: it anchors the intensity level, and the
                #                     val/bias fix depends on it.
                "Recon_loss": {
                'mae_loss': 1.0,
                'edge_loss': 0.4,
                'ssim_loss': 0.4,
                'axial_diff_matching_loss': 0.35
            }
        }

        # Image Transformations
        # Example: transform_3d could be similar or include frame-wise operations
        self.transform_3d = transforms.Compose([
            PercentileNormalize3D(p_low=1.0, p_high=99.7, return_params=True)
        ])



    def compute_optics_params(self):
        optics = self.optics_params
        optics["objective"]["focal_length"] = optics["tube_lens"]["focal_length"] / optics["objective"]["magnification"]
        optics["objective"]["pupil_diameter"] = 2 * optics["objective"]["na"] * optics["objective"]["focal_length"]
        optics["demagnification"] = optics["tube_lens"]["focal_length"] / optics["relay_lens"]["focal_length"]
        optics["camera"]["sensor_size"] = optics["objective"]["pupil_diameter"] / optics["demagnification"]
        optics["camera"]["pixel_count"] = optics["camera"]["sensor_size"] / optics["camera"]["pixel_size"]
        optics["microlens"]["focal_length"] = optics["camera"]["distance_from_phasemask"]
        optics["M_T"] = (optics["microlens"]["focal_length"] / optics["objective"]["focal_length"]) * (optics["tube_lens"]["focal_length"] / optics["relay_lens"]["focal_length"])
        optics["microlens"]["pitch"] = optics["microlens"]["pitch"]
        optics["microlens"]["array_size"] = 2 * optics["objective"]["focal_length"]  * optics["relay_lens"]["focal_length"] * optics["objective"]["na"] / (optics["tube_lens"]["focal_length"] * optics["microlens"]["pitch"])
        optics["microlens"]["f_number"] = optics["microlens"]["focal_length"] / optics["microlens"]["pitch"]
    

        # Sampling in different planes
        # Calculate sampling at aperture stop plane (x1,y1)
        microlens_sampling_as =  optics["emission_wavelength"] * optics["microlens"]["focal_length"] / (4 * optics["microlens"]["pitch"])
        #PSF sampling at Fourier plane (x3,y3)
        microlens_sampling_fp = microlens_sampling_as / optics["demagnification"]
        if microlens_sampling_fp % (optics["meta_unit"]["cell_size"])!=0:
            microlens_sampling_fp = (microlens_sampling_fp // optics["meta_unit"]["cell_size"]) * optics["meta_unit"]["cell_size"]
        optics["microlens"]["sampling_FP"] = microlens_sampling_fp
        print(f'Number of meta units per microlens sampling FP: {microlens_sampling_fp/optics["meta_unit"]["cell_size"]}')
        optics["microlens"]["sampling_AS"] = microlens_sampling_fp * optics["demagnification"]

        #Simlation grid
        L = optics["camera"]["sensor_size"] #side length of the diffuser(meters)
        optics["L"] = L
        optics["M"] = int(L/optics["microlens"]["sampling_FP"])  #number of pixels in the simulation


        # Calculate PSF sampling at Image plane I (field stop plane)
        L_AS = optics["objective"]["pupil_diameter"]   #side length of the diffuser(meters) in aperture stop plane
        M_AS = int(L_AS/optics["microlens"]["sampling_AS"] ) #M_AS - Number of pixels in aperture stop plane (x1,y1)
        optics["blur_kernel_size_as"] = M_AS
        blur_kernel_size_fs = M_AS + 0 #blur kernel size in image plane I, FS
        optics["blur_kernel_size_fs"] = blur_kernel_size_fs
        fx1= 1/(blur_kernel_size_fs*optics["microlens"]["sampling_AS"] )  #Fourier domain frequency sampling
        psf_sampling_fs = optics["emission_wavelength"] *  optics["tube_lens"]["focal_length"] * fx1
        optics["psf_sampling_fs"] = psf_sampling_fs

        # Calculate PSF sampling at Image plane II (sensor plane)
        L_S = L_AS / optics["demagnification"] #side length of the diffuser(meters)
        M_FP = int(L_S/optics["microlens"]["sampling_FP"] )  ##M_FP - Number of pixels in Fourier plane (x3,y3) number of pixels in the simulation
        blur_kernel_size_fp = M_FP + 0 # blur kernel size in Fourier plane
        optics["blur_kernel_size_fp"] = blur_kernel_size_fp
        fx2= 1/(blur_kernel_size_fp*psf_sampling_fs)  #Fourier domain frequency sampling

        #PSF sampling at Fourier plane 
        psf_sampling_fp = optics["emission_wavelength"] *   optics["relay_lens"]["focal_length"] * fx2
        optics["psf_sampling_fp"] = psf_sampling_fp
        optics["padding_fp"] = 1260
        blur_kernel_size_sp = M_FP + optics["padding_fp"] # blur kernel size in image plane II, sensor plane
        optics["blur_kernel_size_sp"] = blur_kernel_size_sp
        fx3 = 1/(psf_sampling_fp*blur_kernel_size_sp)  #Fourier domain frequency sampling

        
        # PSF sampling at Image plane II (sensor plane)
        #psf_sampling_sp = optics["emission_wavelength"] *  optics["microlens"]["focal_length"]* fx3        
        fx2_new= 1/(blur_kernel_size_sp*psf_sampling_fs)  #Fourier domain frequency sampling
        psf_sampling_sp = optics["emission_wavelength"] *   optics["relay_lens"]["focal_length"] * fx2_new

        optics["psf_sampling_sp"] = psf_sampling_sp
        object_pixel_size = psf_sampling_sp / optics["M_T"]
        optics["object_pixel_size"] = object_pixel_size


        optics["axial_resolution"] = (optics["microlens"]["array_size"]**2)/(optics["microlens"]["array_size"]-1) * (1.22*optics["emission_wavelength"] /(2*optics["objective"]["na"]**2) ) # from Liu et al., Fourier DiffuserScope, Opt. Express 28, 28969 (2020)
        optics["lateral_resolution"] = 0.61*optics["emission_wavelength"] / (optics["objective"]["na"] / optics["microlens"]["array_size"]) # Rayleigh criterion


        # ------------------------------------------------------------------
        # f_min / f_max : the range of lenslet focal lengths that brings the
        # edges of the target volume into focus on the sensor.
        #
        # The lenslet sits at the MASK plane, so what matters is the radius of
        # curvature of the wavefront THERE, not in the objective pupil:
        #
        #       R = n * f_obj^2 / (z * demag^2)
        #
        #   * n           : the object sits in the immersion medium
        #   * 1/demag^2   : the relay magnifies the pupil by 1/demag, and a
        #                   wavefront radius scales as (magnification)^2
        #
        # The previous version used zi = f_obj^2 / z, i.e. the pupil-side radius
        # with neither factor, which is too small by n/demag^2 = 3.69 and made
        # the learnable range ~3.7x too wide (a lenslet at the old f_min was
        # focused on z = +-111 um, far outside the +-30 um volume).
        # Verified by sweeping f and finding the sensor best focus for each z.
        # ------------------------------------------------------------------
        z_half = optics["microlens"]["DOF_target"] / 2
        R_mask = (optics["objective"]["medium_index"]
                  * optics["objective"]["focal_length"] ** 2
                  / (z_half * optics["demagnification"] ** 2))
        d_cam = optics["camera"]["distance_from_phasemask"]

        optics["microlens"]["f_min"] = ((1 / d_cam) + (1 / R_mask)) ** (-1)
        optics["microlens"]["f_max"] = ((1 / d_cam) - (1 / R_mask)) ** (-1)


    def print_optics_params(self):
        optics = self.optics_params
        print("Wavelength (m):", optics["emission_wavelength"])
        print("Number of lenslets (one line):", optics["microlens"]["array_size"])
        print("Lateral magnification M_T:", optics["M_T"])
        print("Lenslet pitch (um):", optics["microlens"]["pitch"]*1e6)
        print("Sampling rate MLA placed at aperture stop (nm):", optics["microlens"]["sampling_AS"]*1e9)
        print("Sampling rate MLA placed at Fourier plane:", optics["microlens"]["sampling_FP"])
        print("Camera Sensor size (mm):", optics["camera"]["sensor_size"]*1e3)
        print("Camera Pixel size:", optics["camera"]["pixel_size"])
        print("Camera Pixel count:", optics["camera"]["pixel_count"])
        print("Objective NA:", optics["objective"]["na"])
        print("Objective Magnification:", optics["objective"]["magnification"])
        print("Objective Focal length (mm):", optics["objective"]["focal_length"]*1e3)
        print("M - Number of pixels in the simulation:", optics["M"])	
        print("L - Side length of the diffuser (mm):", optics["L"]*1e3)
        print("Objective Demagnification:", optics["demagnification"])
        print("Object pixel size (um):", optics["object_pixel_size"]*1e6)
        print(f'Psf sampling rate at field stop plane (um):',  optics["psf_sampling_fs"]*1e6)
        print(f'Psf sampling rate at Fourier plane (um):',  optics["psf_sampling_fp"]*1e6)
        print(f'Psf sampling rate at sensor plane (um):',  optics["psf_sampling_sp"]*1e6)
        print(f'Blur kernel size at field stop plane:',  optics["blur_kernel_size_fs"])
        print(f'Blur kernel size at Fourier plane:',  optics["blur_kernel_size_fp"])
        print(f'Blur kernel size at sensor plane:',  optics["blur_kernel_size_sp"])
        print(f'Axial resolution (um):',  optics["axial_resolution"]*1e6)
        print(f'Lateral resolution (um):',  optics["lateral_resolution"]*1e6)
        print(f"Microlens focal length (mm):", optics["microlens"]["focal_length"]*1e3)
        print(f'f_min:',  optics["microlens"]["f_min"])
        print(f'f_max:',  optics["microlens"]["f_max"])
        print("volume_depth_min (um):", optics["volume_depth_min"]*1e6 )
        print("volume_depth_max (um):", optics["volume_depth_max"]*1e6 )
        print("--------------------------------------------------")

# Create an instance of the configuration to be imported by other modules
config = Config()
