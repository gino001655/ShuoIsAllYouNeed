import torch


class ImageConditioning:
    """
    Placeholder for CLD2 image conditioning.

    Phase-1 default: disabled. Later we can implement:
    - encode whole image into VAE latents
    - build per-layer bbox masks in latent space
    - create residual conditioning to inject into denoiser input or early layers
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    @torch.no_grad()
    def build_residual(
        self,
        vae,
        whole_img_bchw: torch.Tensor,
        list_layer_box,
        n_layers: int,
        dtype: torch.dtype,
        device: torch.device,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Minimal, VRAM-friendly image conditioning (Phase-1):

        - Encode the whole input image into a single latent map `z_img` (same latent space as targets).
        - For each layer bbox, build a latent-space mask and take `z_img * mask` as a residual condition.
        - Return residual tensor shaped like multi-layer latents: [1, L, C_lat, H_lat, W_lat]

        Note:
        - This is intentionally simple. Later you can replace it with a learned adapter network.
        """
        if not self.enabled:
            raise RuntimeError("ImageConditioning is disabled but build_residual() was called.")

        # Encode image -> latent
        z = vae.encode(whole_img_bchw.to(vae.dtype)).latent_dist.sample()
        z = (z - vae.config.shift_factor) * vae.config.scaling_factor
        z = z.to(device=device, dtype=dtype)  # [1,C_lat,H_lat,W_lat]

        _, c_lat, h_lat, w_lat = z.shape

        # Build per-layer masked residuals
        residual = torch.zeros((1, n_layers, c_lat, h_lat, w_lat), device=device, dtype=dtype)
        for i, box in enumerate(list_layer_box):
            if box is None:
                continue
            x1, y1, x2, y2 = box
            x1_t, y1_t, x2_t, y2_t = x1 // 8, y1 // 8, x2 // 8, y2 // 8
            x1_t, y1_t = max(0, x1_t), max(0, y1_t)
            x2_t, y2_t = min(w_lat, x2_t), min(h_lat, y2_t)
            if x2_t <= x1_t or y2_t <= y1_t:
                continue
            residual[:, i, :, y1_t:y2_t, x1_t:x2_t] = z[:, :, y1_t:y2_t, x1_t:x2_t]

        return residual * float(scale)


