import torch
import torch.nn as nn
import torch.nn.functional as F
from piqa import SSIM

# Define MSE Loss
def mse_loss(output, target):
    val = F.mse_loss(output, target, reduction='mean')
    print("MSE Loss:", val.item(), val.requires_grad)

    return val  # Returns tensor with grad    


# Define MAE Loss
def mae_loss(output, target):
    val = F.l1_loss(output, target, reduction='mean')
    print("MAE Loss:", val.item(), val.requires_grad)
    return val


#Define axial_diff_matching_loss
def axial_diff_matching_loss(v_out, v_gt, mode="huber", delta=0.01, anchor_weight=0.3):
    print(" Axial Diff Matching Loss Input Shapes:", v_out.shape, v_gt.shape)
    # Accept [D,H,W] or [B,D,H,W]
    if v_out.dim() == 3:
        D = v_out.shape[0]

        # Remove spatial mean per slice
        v_out_c = v_out - v_out.mean(dim=(1,2), keepdim=True)
        v_gt_c  = v_gt  - v_gt.mean(dim=(1,2), keepdim=True)

        dz_out = v_out_c[1:] - v_out_c[:-1]
        dz_gt  = v_gt_c[1:]  - v_gt_c[:-1]
        
        anchor = (
            F.l1_loss(v_out[0], v_gt[0]) +
            F.l1_loss(v_out[-1], v_gt[-1]) +
            F.l1_loss(v_out[D//2], v_gt[D//2])
        )

    elif v_out.dim() == 4:

        B, D, H, W = v_out.shape
        v_out_c = v_out - v_out.mean(dim=(2,3), keepdim=True)
        v_gt_c  = v_gt  - v_gt.mean(dim=(2,3), keepdim=True)

        dz_out = v_out_c[:,1:] - v_out_c[:,:-1]
        dz_gt  = v_gt_c[:,1:]  - v_gt_c[:,:-1]
        
        anchor = (
            F.l1_loss(v_out[:, 0], v_gt[:, 0]) +
            F.l1_loss(v_out[:, -1], v_gt[:, -1]) +
            F.l1_loss(v_out[:, D//2], v_gt[:, D//2])
        )
    else:
        raise ValueError(f"Unexpected shape: {v_out.shape}")

    if mode == "l1":
        loss = (dz_out - dz_gt).abs().mean()
        print("Axial L1 Loss:", (loss.item(), loss.requires_grad))

    elif mode == "huber":
        loss = F.smooth_l1_loss(dz_out, dz_gt, beta=delta)
        print("Axial Huber Loss:", loss.item(), loss.requires_grad)
    else:
        raise ValueError("mode must be 'l1' or 'huber'")

    return loss + anchor_weight * anchor


# Define Edge Loss
def sobel_edges(image_stack):
    """
    Applies 2D Sobel edge detection independently to each slice of a 3D tensor.
    
    Args:
        image_stack (Tensor): A 3D tensor of shape [D, H, W] (no batch/channel dims).
    
    Returns:
        Tensor: Edge magnitude for each slice, shape [D, H, W].
    """

    if image_stack.dim() != 3:
        raise ValueError("Input must be a 3D tensor of shape [D, H, W]")

    device = image_stack.device
    D, H, W = image_stack.shape

    # Define 2D Sobel filters
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=device).view(1, 1, 3, 3)

    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], device=device).view(1, 1, 3, 3)

    # Add channel dimension: [D, 1, H, W]
    image_stack = image_stack.unsqueeze(1).float()

    # Apply 2D Sobel filters independently on each slice
    edge_x = F.conv2d(image_stack, sobel_x, padding=1)
    edge_y = F.conv2d(image_stack, sobel_y, padding=1)

    # Compute edge magnitude: sqrt(dx^2 + dy^2)
    edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

    # Remove channel dimension → [D, H, W]
    return edge_magnitude.squeeze(1)


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Define kernels as buffers so they move with the model across devices
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)

        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, output, target):
        # Ensure 3D input: [D, H, W]
        if output.dim() != 3:
            raise ValueError(f"Expected 3D tensors [D, H, W], got: {output.shape}")

        output = output.unsqueeze(1)  # [D, 1, H, W]
        target = target.unsqueeze(1)

        # Ensure kernel precision matches input (autocast-safe)
        kernel_x = self.kernel_x.to(dtype=output.dtype, device=output.device)
        kernel_y = self.kernel_y.to(dtype=output.dtype, device=output.device)

        #edge_o = torch.sqrt(F.conv2d(output, kernel_x, padding=1)**2 + F.conv2d(output, kernel_y, padding=1)**2)
        #edge_t = torch.sqrt(F.conv2d(target, kernel_x, padding=1)**2 + F.conv2d(target, kernel_y, padding=1)**2)


        eps = 1e-6

        output_pad = F.pad(output, (1, 1, 1, 1), mode="replicate")
        target_pad = F.pad(target, (1, 1, 1, 1), mode="replicate")

        gx_o = F.conv2d(output_pad, kernel_x, padding=0)
        gy_o = F.conv2d(output_pad, kernel_y, padding=0)
        gx_t = F.conv2d(target_pad, kernel_x, padding=0)
        gy_t = F.conv2d(target_pad, kernel_y, padding=0)
        edge_o = torch.sqrt(gx_o * gx_o + gy_o * gy_o + eps)
        edge_t = torch.sqrt(gx_t * gx_t + gy_t * gy_t + eps)


        val = F.l1_loss(edge_o, edge_t)
        print("L1 Edge Loss:", val.item(), val.requires_grad)

        return val  # Returns tensor with grad

"""
#Define SSIM Loss for 3D volumes
#[D, H, W] input
class SSIM3DLoss(nn.Module):
    def __init__(self, C1=0.01**2, C2=0.03**2, kernel_size=11, sigma=1.5):
        super(SSIM3DLoss, self).__init__()
        self.C1 = C1
        self.C2 = C2
        self.kernel_size = kernel_size
        self.sigma = sigma

    def _gaussian_kernel_3d(self, device, dtype):
        coords = torch.arange(self.kernel_size, dtype=dtype, device=device) - self.kernel_size // 2
        g = torch.exp(-(coords**2) / (2 * self.sigma**2))
        g = g / g.sum()
        g3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
        g3d = g3d.unsqueeze(0).unsqueeze(0)  # shape: [1,1,K,K,K]
        return g3d

    def forward(self, x, y):
        if x.dim() != 3 or y.dim() != 3:
            raise ValueError("Expected input shape [D, H, W]")

        x = x.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        y = y.unsqueeze(0).unsqueeze(0)

        device = x.device
        dtype = x.dtype

        kernel = self._gaussian_kernel_3d(device, dtype)

        mu_x = F.conv3d(x, kernel, padding=self.kernel_size // 2)
        mu_y = F.conv3d(y, kernel, padding=self.kernel_size // 2)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv3d(x * x, kernel, padding=self.kernel_size // 2) - mu_x_sq
        sigma_y_sq = F.conv3d(y * y, kernel, padding=self.kernel_size // 2) - mu_y_sq
        sigma_xy = F.conv3d(x * y, kernel, padding=self.kernel_size // 2) - mu_xy

        numerator = (2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)
        denominator = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2)

        ssim_map = numerator / (denominator + 1e-6)
        loss = 1 - ssim_map.mean()

        print("SSIM 3D Loss:", loss.item(), loss.requires_grad)
        return loss

"""

class WrappedSSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM(n_channels=1, channel_avg=False)

    def forward(self, output, target):
        # Ensure input is [D, H, W] or [B, D, H, W]
        if output.dim() == 3:
            # [D, H, W] â†’ compute SSIM for each [H, W] slice
            ssim_vals = []
            for d in range(output.shape[0]):
                out_slice = output[d].unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
                tgt_slice = target[d].unsqueeze(0).unsqueeze(0)
                ssim_val = self.ssim(out_slice, tgt_slice)
                ssim_vals.append(ssim_val)
            ssim_mean = torch.stack(ssim_vals).mean()
            print('Average SSIM over D slices:', ssim_mean.item(), ssim_mean.requires_grad)
            
            
            #return 1 - ssim_mean #Return SSIM loss
            return ssim_mean #Return SSIM loss

        else:
            raise ValueError("Input must be 3D [D, H, W]")
