import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
import torch
import os
import torch.utils.checkpoint as cp
import gc

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
gc.collect()
torch.cuda.empty_cache()

# Helper function for clean shape matching
def match_shape(x, ref):
    return F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=True)

def center_crop(tensor, target_h, target_w):
    """
    Crops the input tensor centrally to match the target height and width.
    tensor: shape [B, C, H, W]
    Returns: cropped tensor [B, C, target_h, target_w]
    """
    _, _, h, w = tensor.shape
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return tensor[:, :, start_h:start_h + target_h, start_w:start_w + target_w]


    
class FastSlidingConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, patch_size=256, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        patch_size = self.patch_size
        stride = patch_size - 8  # mimic your overlap logic

        # Step 1: Pad input to fit patches evenly
        pad_h = (stride - (H - patch_size) % stride) % stride
        pad_w = (stride - (W - patch_size) % stride) % stride
        x = F.pad(x, (0, pad_w, 0, pad_h))

        # Step 2: Unfold to extract patches
        patches = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        B, C, n_h, n_w, ph, pw = patches.shape
        patches = patches.contiguous().view(B * n_h * n_w, C, patch_size, patch_size)

        # Step 3: Apply conv in batch
        out_patches = self.conv(patches)

        # Step 4: Fold output back to full image
        out_patch_h, out_patch_w = out_patches.shape[2], out_patches.shape[3]
        out_patches = out_patches.view(B, n_h, n_w, -1, out_patch_h, out_patch_w)
        out_patches = out_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        output = out_patches.view(B, -1, n_h * out_patch_h, n_w * out_patch_w)

        # Step 5: Crop to original size if needed
        return output[:, :, :H, :W]


class SlidingWindowConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, patch_size=128):
        super(SlidingWindowConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=kernel_size // 2
        )
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        patch_size = min(self.patch_size, H, W)
        stride = max(patch_size - 8, 1)

        output = torch.zeros((B, self.conv.out_channels, H, W), device=x.device, dtype=x.dtype)
        count = torch.zeros((B, 1, H, W), device=x.device, dtype=x.dtype)

        ys = list(range(0, max(H - patch_size + 1, 1), stride))
        xs = list(range(0, max(W - patch_size + 1, 1), stride))

        last_y = H - patch_size
        last_x = W - patch_size

        if len(ys) == 0 or ys[-1] != last_y:
            ys.append(last_y)
        if len(xs) == 0 or xs[-1] != last_x:
            xs.append(last_x)

        for i in ys:
            for j in xs:
                patch = x[:, :, i:i + patch_size, j:j + patch_size]
                patch_out = self.conv(patch)

                h_out, w_out = patch_out.shape[2], patch_out.shape[3]
                output[:, :, i:i + h_out, j:j + w_out] += patch_out
                count[:, :, i:i + h_out, j:j + w_out] += 1

        output = output / torch.clamp(count, min=1.0)
        return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, activation=nn.LeakyReLU(0.01), is_in=True, patch_size=128):
        super(ConvBlock, self).__init__()
        #self.conv = SlidingWindowConv(in_channels, out_channels, kernel_size, stride, patch_size)
        #self.conv = FastSlidingConv(in_channels, out_channels, kernel_size, stride, patch_size, padding=padding)       
        self.conv = SlidingWindowConv(in_channels, out_channels, kernel_size, stride, patch_size)        
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)  # Apply patch-wise convolution
        x = self.activation(x)
        return x



class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))




class MultiResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, alpha=1.0):
        super(MultiResBlock, self).__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        filter_num = int(out_channels * alpha)
        n1_, n2_, n3_ = int(filter_num * 0.25), int(filter_num * 0.25), int(filter_num * 0.5)

        #Replace ConvBlock with DepthwiseSeparableConv
        self.short_cut = DepthwiseSeparableConv(in_channels, n1_ + n2_ + n3_, kernel_size=1)
        self.conv3x3 = DepthwiseSeparableConv(in_channels, n1_, kernel_size=3)
        self.conv5x5 = DepthwiseSeparableConv(n1_, n2_, kernel_size=3)
        self.conv7x7 = DepthwiseSeparableConv(n2_, n3_, kernel_size=3)

        self.norm1 = nn.InstanceNorm2d(n1_ + n2_ + n3_)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.relu_final = nn.LeakyReLU()

    def forward(self, x):

        torch.cuda.empty_cache()
        #The shortcut is supposed to be a simple projection of the original input so you can add it back at the end.
        short_cut = self.short_cut(x) #short_cut takes the same x but projects it to the final concatenated channel count so it can be added at the end.
        #print('short_cut')
        conv3x3 = self.conv3x3(x)
        #print('conv3x3')
        conv5x5 = self.conv5x5(conv3x3)
        #print('conv5x5')
        conv7x7 = self.conv7x7(conv5x5)
        #print('conv7x7')

        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        out = self.norm1(out)
        out = out + short_cut  # Skip connection

        # Free memory of intermediate tensors
        del conv3x3, conv5x5, conv7x7, short_cut
        gc.collect()

        # Ensure final output has the correct number of channels
        #if out.shape[1] != self.out_channels:
            #out = nn.Conv2d(out.shape[1], self.out_channels, kernel_size=1)(out)

        out = self.relu_final(out)
        out = self.norm2(out)

        return out

    
'''
class SubpixelConv2d(nn.Module):
    def __init__(self, in_channels, scale):
        """
        Subpixel Convolution for upscaling.
        Args:
            in_channels: Number of input channels
            scale: Upscaling factor (e.g., 2, 3, 4)
        """
        super(SubpixelConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x
    
'''
class PatchwiseSubpixelConv2d(nn.Module):
    def __init__(self, in_channels, scale, patch_size=128):
        """
        Patchwise Subpixel Convolution for upscaling.
        Args:
            in_channels: Number of input channels
            scale: Upscaling factor (e.g., 2, 3, 4)
            patch_size: Size of each processed patch to save memory
        """
        super(PatchwiseSubpixelConv2d, self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels, in_channels * (scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        B, C, H, W = x.shape
        patch_size = min(self.patch_size, H, W)  # Ensure patch fits tensor size
        stride = patch_size - 16  # Slight overlap to avoid artifacts

        # Initialize output tensor
        output_H, output_W = H * self.pixel_shuffle.upscale_factor, W * self.pixel_shuffle.upscale_factor
        output = torch.zeros((B, C, output_H, output_W), device=x.device)
        count = torch.zeros((B, 1, output_H, output_W), device=x.device)


        # Sliding window loop
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                patch = x[:, :, i:i+patch_size, j:j+patch_size]
                patch_out = self.conv(patch)
                patch_out = self.pixel_shuffle(patch_out)

                # Ensure patch_out matches expected region before storing
                h_out, w_out = patch_out.shape[2], patch_out.shape[3]
                output[:, :, i*h_out//patch_size:(i+patch_size)*h_out//patch_size, 
                              j*w_out//patch_size:(j+patch_size)*w_out//patch_size] += patch_out
                count[:, :, i*h_out//patch_size:(i+patch_size)*h_out//patch_size, 
                              j*w_out//patch_size:(j+patch_size)*w_out//patch_size] += 1

        # Normalize overlapping regions
        output /= (count + 1e-6)

        # Free memory of intermediate tensors
        del count
        gc.collect()

        return output





class AxialRefine3D(nn.Module):
    """Local 3D refinement of a [B, D, H, W] volume.

    Why this exists. The 2D path puts the 43 depths in the CHANNEL dimension,
    so the only coupling along z is a dense per-depth mixing: a separate weight
    for every (z_in, z_out) pair, with nothing shared between neighbouring
    depths. There is no translation-equivariant operator along z anywhere in
    the network. A 3x3x3 kernel is one -- the same weights see (z-1, z, z+1) at
    every depth -- which is the right shape for undoing a depth-dependent blur,
    and it is the one structural prior the 2D network is missing.

    Deliberately thin. The volume is 43 x 512 x 512 = 11.3M voxels, so one
    feature map is ~45 MB per channel and the cost is activations, not
    parameters: 8 channels is ~5.8k weights but ~360 MB of activation per
    layer. Hence the default checkpoint=True.

    The output layer is zero-initialised, so the module starts as the exact
    identity and epoch 0 reproduces the 2D-only network. Without that, a random
    3D branch corrupts the lift before it has learned anything and the run
    spends its first epochs recovering.
    """
    def __init__(self, channels=8, layers=3, checkpoint=True):
        super().__init__()
        self.checkpoint = checkpoint
        c = channels
        self.inp = nn.Conv3d(1, c, 3, padding=1, bias=False)
        self.mid = nn.ModuleList([nn.Conv3d(c, c, 3, padding=1, bias=False)
                                  for _ in range(max(layers - 2, 0))])
        self.out = nn.Conv3d(c, 1, 3, padding=1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def _body(self, v):
        h = self.act(self.inp(v))
        for m in self.mid:
            h = self.act(m(h))
        return self.out(h)

    def forward(self, x):                      # x: [B, D, H, W]
        v = x.unsqueeze(1)                     # [B, 1, D, H, W]
        if self.checkpoint and self.training and torch.is_grad_enabled():
            d = cp.checkpoint(self._body, v, use_reentrant=False)
        else:
            d = self._body(v)
        return x + d.squeeze(1)                # residual


class F_Recon(nn.Module):
    def __init__(self, input_channels, n_slices, output_size, channels_interp, view_num, patch_size,
                 normalize_mode='percentile', res_scale=0.4, pyramid_channels=(64, 96, 128),
                 axial='none', axial_channels=8, axial_layers=3, axial_checkpoint=True,
                 sensor_channels=0):
        super(F_Recon, self).__init__()
        # res_scale and pyramid_channels are constructor args so they can be
        # swept from train_recon_only.py without editing this file. Defaults
        # reproduce the configuration every reported result used.
        #
        # axial adds the 3D convolution branch (see AxialRefine3D):
        #   'none' -- 2D only. Identical to every reported result.
        #   'post' -- refine the network's output volume. Cheapest place to put
        #             it and the one to try first: it sees the full 43-depth
        #             volume at output resolution.
        #   'pre'  -- refine the lifted volume before the 2D stem, i.e. clean up
        #             the CG output's axial smear before the 2D path ever sees
        #             it. x_in in the residual add stays the RAW lift either way.
        #   'both' -- two independent modules.
        self.res_scale = res_scale
        # sensor_channels > 0 means forward() also takes the raw sensor image,
        # space-to-depth packed into this many 512x512 channels, concatenated
        # onto the lift before the stem.
        #
        # The point of it. Every network tested so far sees ONLY the lift, so
        # whatever the CG solve discarded is gone before the network starts --
        # which is consistent with the measured result that rho is pinned at
        # the lift's own 0.803 no matter what the network does (0.7968-0.8033
        # across four architectures). If the network can see the measurement
        # too, it can in principle recover what the lift dropped. If rho still
        # does not move, the lift has extracted everything available and the
        # network's job really is only calibration and lateral detail.
        #
        # x_in in the residual add stays the LIFT alone -- the sensor is an
        # extra input, not something to add to the output.
        self.sensor_channels = sensor_channels
        self.axial = axial
        self.axial_pre = (AxialRefine3D(axial_channels, axial_layers, axial_checkpoint)
                          if axial in ('pre', 'both') else None)
        self.axial_post = (AxialRefine3D(axial_channels, axial_layers, axial_checkpoint)
                           if axial in ('post', 'both') else None)
        self.input_channels = input_channels
        self.n_slices = n_slices
        self.output_size = output_size  # This should be [height, width], e.g., [160, 160]
        self.channels_interp = channels_interp
        self.view_num = view_num
        self.normalize_mode = normalize_mode
        self.patch_size = patch_size


        # coarse volume comes in as [B, D, H, W]
        # so first conv must accept input_channels = D
        self.sensor_stem = nn.Sequential(
            nn.Conv2d(input_channels + sensor_channels, channels_interp,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.01, inplace=True),
            nn.InstanceNorm2d(channels_interp),

            nn.Conv2d(channels_interp, channels_interp, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.01, inplace=True),
            nn.InstanceNorm2d(channels_interp),
        )

        # Dilated convolutions for ASPP
        dilation_rates = [1, 2, 4]
        self.aspp_blocks = nn.ModuleList([
            nn.Conv2d(channels_interp, channels_interp, kernel_size=3, padding=d, dilation=d)
            for d in dilation_rates
        ])

        # Convolution to reduce channels after concatenation
        self.pyramid_conv = nn.Sequential(
            nn.Conv2d(channels_interp * len(dilation_rates), channels_interp, kernel_size=1, padding=0, bias=False),
            nn.LeakyReLU(0.01, inplace=True)
        )


        self.patch_conv1 = nn.Conv2d(channels_interp, channels_interp, kernel_size=3, padding=1, bias=False)
        self.patch_conv2 = nn.Conv2d(channels_interp, channels_interp, kernel_size=3, padding=1, bias=False)



        # Encoder with MultiResBlocks
        pyramid_channels = list(pyramid_channels)
        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        current_channels = channels_interp

        for i, pc in enumerate(pyramid_channels):
            layers = [MultiResBlock(current_channels, pc, patch_size=self.patch_size)]
            
            self.encoder_blocks.append(nn.Sequential(*layers))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            current_channels = pc


        # Decoder layers (pyramid)
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.concat_conv_layers = nn.ModuleList()

        for i in range(len(pyramid_channels) - 1, -1, -1):
            input_channels = pyramid_channels[i] * 2
            output_channels = pyramid_channels[max(i - 1, 0)] if i != 0 else pyramid_channels[0]

            # Upsampling
            self.upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

            # Convolution after concatenation
            self.concat_conv_layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))
            self.decoder_blocks.append(nn.Sequential(
                nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.InstanceNorm2d(output_channels),
            ))

        # Final convolution to output slices
        self.final_conv = nn.Conv2d(pyramid_channels[0], self.n_slices, kernel_size=3, padding=1, bias=True)
  

    def forward(self, x, sensor=None):
        # x: [B, D, H, W].  sensor: [B, sensor_channels, H, W] or None.
        x_in = x
        if self.axial_pre is not None:
            x = self.axial_pre(x)
        if self.sensor_channels:
            if sensor is None:
                raise ValueError(
                    f"F_Recon was built with sensor_channels="
                    f"{self.sensor_channels} but forward() got sensor=None")
            x = torch.cat([x, sensor.to(x.dtype)], dim=1)
        x_combined = self.sensor_stem(x)

        aspp_features = [aspp(x_combined) for aspp in self.aspp_blocks]
        x_combined = torch.cat(aspp_features, dim=1)
        x_combined = self.pyramid_conv(x_combined)

        x_combined = F.leaky_relu(self.patch_conv1(x_combined), negative_slope=0.01, inplace=True)
        x_combined = F.leaky_relu(self.patch_conv2(x_combined), negative_slope=0.01, inplace=True)
        # Encoder
        skip_connections = []
        for enc_block, pool in zip(self.encoder_blocks, self.pool_layers):
            x_combined = enc_block(x_combined)
            skip_connections.append(x_combined)
            x_combined = pool(x_combined)
        # Decoder
        for idx, (up, dec_block, concat_conv) in enumerate(
            zip(self.upsample_layers, self.decoder_blocks, self.concat_conv_layers)
        ):
            x_combined = up(x_combined)
            skip = skip_connections[-(idx + 1)]

            if x_combined.shape[2:] != skip.shape[2:]:
                x_combined = match_shape(x_combined, skip)

            #skip = F.instance_norm(skip)
            x_combined = torch.cat([x_combined, skip], dim=1)

            assert x_combined.shape[1] == concat_conv.in_channels, "Channel mismatch before concat_conv"

            x_combined = concat_conv(x_combined)
            x_combined = dec_block(x_combined)

        x_combined = self.final_conv(x_combined)

        # force output to target object resolution
        x_combined = F.interpolate(
            x_combined,
            size=tuple(self.output_size),   # (512, 512)
            mode='bilinear',
            align_corners=False
        )
        # 3D refinement of the residual volume, at output resolution and with
        # all 43 depths present. Zero-initialised, so at epoch 0 this line is
        # a no-op and the network is exactly the 2D baseline.
        if self.axial_post is not None:
            x_combined = self.axial_post(x_combined)
        # Residual refinement of the lifted volume.
        #
        # res_scale multiplies the conv branch's contribution AND the gradient
        # reaching it, so it acts as a learning-rate multiplier for that branch,
        # not just an output gain.
        #
        # 0.4, measured at 5 epochs with the CG lift (cg_reg 3.0, cg_iters 10),
        # against an IDENTICAL lift so only the network differs:
        #
        #   sample 1 (group B)      rho     SSIM    PSNR
        #     lift alone          0.9191   0.4753   22.48
        #     res_scale 0.2       0.9009   0.4797   21.65
        #     res_scale 0.4       0.9168   0.5145   22.77
        #
        #   sample 8 (group A)      rho     SSIM    PSNR
        #     lift alone          0.4980   0.0851   17.13
        #     res_scale 0.2       0.4522   0.0845   16.66
        #     res_scale 0.4       0.4766   0.0911   17.24
        #
        # At 0.2 the branch is still undertrained after 5 epochs and the output
        # is WORSE than not running the network at all. At 0.4 it beats the lift
        # on SSIM and PSNR on both populations.
        #
        # Judge this on SSIM and PSNR, NOT on rho: rho is invariant to any
        # affine rescaling, so it cannot see the calibration the network fixes.
        # Every configuration here costs a little rho while gaining SSIM/PSNR.
        x_combined = x_in + self.res_scale * x_combined
        # Training: return the UNCLAMPED output, so every loss term sees the true
        # values and a real gradient (overshoot above 1 is penalised directly).
        # Eval / test: clamp to the valid [0, 1] range.
        if not self.training:
            x_combined = x_combined.clamp(0.0, 1.0)

        return x_combined.squeeze(0)   # [B, D, H, W]   



if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt

    input_channels = 1
    n_slices = 41
    batch_size = 1
    h = 512
    w = 512
    channels_interp = 64
    patch_size = 512

    # keep view_num only because your current __init__ still asks for it
    # for raw sensor mode it can just be 1
    view_num = 1
    output_size = (h, w)

    model_recons = F_Recon(
        input_channels=input_channels,
        n_slices=n_slices,
        output_size=output_size,
        channels_interp=channels_interp,
        view_num=view_num,
        patch_size=patch_size,
    )

    # raw sensor input: [B, 1, H, W]
    optics_size = 3385
    input_tensor = torch.randn(batch_size, 1, optics_size, optics_size)

    model_recons.eval()

    with torch.no_grad():
        output = model_recons(input_tensor)

    print("Input shape :", input_tensor.shape)   # [B, 1, H, W]
    print("Output shape:", output.shape)         # [D, H, W]

    # show raw sensor image
    plt.figure(figsize=(6, 6))
    plt.imshow(input_tensor[0, 0].cpu(), cmap='gray')
    plt.title("Raw sensor input")
    plt.axis('off')
    plt.show()

    # show one reconstructed depth slice
    z = n_slices // 2
    plt.figure(figsize=(6, 6))
    plt.imshow(output[z].cpu(), cmap='gray')
    plt.title(f"Output slice z={z}")
    plt.axis('off')
    plt.show()