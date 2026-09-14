import torch

from social_vae import SocialVAE


class MultiGPUSocialVAE(torch.nn.Module):
    """SocialVAE wrapper that supports single-GPU, multi-GPU, and CPU.

    This class keeps the same forward/loss usage pattern as SocialVAE,
    while wrapping the base model with DataParallel when multiple GPUs
    are available.
    """

    def __init__(
        self,
        horizon,
        ob_radius,
        hidden_dim=256,
        device=None,
        use_data_parallel=True,
        device_ids=None,
        output_device=None,
    ):
        super().__init__()

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        base_model = SocialVAE(
            horizon=horizon,
            ob_radius=ob_radius,
            hidden_dim=hidden_dim,
        )

        base_model = base_model.to(device)

        self._is_parallel = False
        if (
            use_data_parallel
            and device.type == "cuda"
            and torch.cuda.device_count() > 1
        ):
            if device_ids is None:
                device_ids = list(range(torch.cuda.device_count()))
            if output_device is None:
                output_device = device_ids[0]
            self.model = torch.nn.DataParallel(
                base_model,
                device_ids=device_ids,
                output_device=output_device,
            )
            self._is_parallel = True
        else:
            self.model = base_model

    @property
    def module(self):
        """Return the underlying SocialVAE module."""
        if self._is_parallel:
            return self.model.module
        return self.model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def loss(self, *args, **kwargs):
        """Pass through custom SocialVAE loss for compatibility."""
        return self.module.loss(*args, **kwargs)
