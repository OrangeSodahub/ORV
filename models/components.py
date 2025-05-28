import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import nn
from einops import rearrange
from dataclasses import dataclass

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.video_processor import VideoProcessor as _VideoProcessor
from diffusers.image_processor import PipelineImageInput

from training.utils import CONSOLE


@dataclass
class Transformer3DModelTrajOutput(Transformer2DModelOutput):

    is_action_mask: torch.Tensor | None
    actions_recon: torch.Tensor | None


class ActionEmbed(nn.Module):

    def __init__(
        self,
        state_dim: int,
        hidden_size: int,
        dropout: float = 0.,
        compress_ratio: int = 1,
        patch_size_t: Optional[int] = None,
        mask: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.compress_ratio = compress_ratio
        self.patch_size_t = patch_size_t or 1
        self.mask = mask

        self.mlp = nn.Sequential(
            nn.Linear(in_features=state_dim * compress_ratio * self.patch_size_t, out_features=hidden_size * 4, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_size * 4, out_features=hidden_size, bias=True),
            nn.Dropout(dropout),
        )

        self.mask_embed = nn.Embedding(num_embeddings=1, embedding_dim=hidden_size)

    def forward(self, x):
        B, F, state_dim = x.shape
        if state_dim != self.state_dim:
            raise ValueError(f'Got mismatched {x.shape=} and {self.state_dim=}.')

        # pad the first frame
        x = torch.cat([
            torch.zeros_like(x[:, :1, ...]),
            x
        ], dim=1)

        if self.compress_ratio > 1:
            x = x.reshape(B, (F + 1) // self.compress_ratio, -1)

        if self.patch_size_t > 1:
            _, F, _ = x.shape
            x = x.reshape(B, F // self.patch_size_t, -1)

        x = self.mlp(x)

        is_mask = torch.rand(B, device=x.device) < 0.1
        if self.mask and is_mask.sum() > 0:
            x[is_mask] = self.mask_embed.weight[None, ...].repeat(is_mask.sum(), x.shape[1], 1)

        return x, is_mask


class ActionRecon(nn.Module):

    def __init__(
        self,
        state_dim: int,
        hidden_size: int,
        compress_ratio: int = 1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.compress_ratio = compress_ratio

        self.mlp = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size * 4, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(in_features=hidden_size * 4, out_features=state_dim * compress_ratio, bias=True),
        )

    def forward(self, x):
        B, F, _ = x.shape

        x = self.mlp(x)

        if self.compress_ratio > 1:
            state_dim = x.shape[-1]
            x = x.reshape(B, int(F * self.compress_ratio), state_dim // self.compress_ratio).contiguous()

        # remove paddings
        x = x[:, 1:, ...]

        return x


class FloatGroupNorm(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.to(self.bias.dtype)).type(x.dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Fuser(nn.Module):
    def __init__(self, action_in_channel=128, out_channels=1152):
        super().__init__()
        self.out_channels = out_channels
        self.gamma_spatial = nn.Linear(action_in_channel, self.out_channels // 4)
        self.gamma_temporal = zero_module(
            nn.Linear(
                self.out_channels // 4,
                self.out_channels,
            )
        )
        self.beta_spatial = nn.Linear(action_in_channel, self.out_channels // 4)
        self.beta_temporal = zero_module(
            nn.Linear(
                self.out_channels // 4,
                self.out_channels,
            )
        )
        self.traj_cond_norm = FloatGroupNorm(32, self.out_channels)

    def forward(self, h, action_hidden_states: Optional[torch.Tensor] = None):

        B, F, P, D = action_hidden_states.shape

        traj = action_hidden_states

        if traj is not None:
            traj = rearrange(traj, 'b f p d -> (b f) p d')
            gamma_traj = self.gamma_spatial(traj)
            beta_traj = self.beta_spatial(traj)

            gamma_traj = rearrange(gamma_traj, '(b f) p d -> (b p) f d', f=F)
            beta_traj = rearrange(beta_traj, '(b f) p d -> (b p) f d', f=F)
            gamma_traj = self.gamma_temporal(gamma_traj)
            beta_traj = self.beta_temporal(beta_traj)

            gamma_traj = rearrange(gamma_traj, '(b p) f d -> b f p d', p=P)
            beta_traj = rearrange(beta_traj, '(b p) f d -> b f p d', p=P)
            h = h + self.traj_cond_norm(
                h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * gamma_traj + beta_traj

        return h


class ConditioningEmbedding(nn.Module):

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


import numpy as np
import PIL
import warnings
from diffusers.image_processor import is_valid_image
def is_valid_image_imagelist(images):
    r"""
    Checks if the input is a valid image or list of images.

    The input can be one of the following formats:
    - A 4D tensor or numpy array (batch of images).
    - A valid single image: `PIL.Image.Image`, 2D `np.ndarray` or `torch.Tensor` (grayscale image), 3D `np.ndarray` or
      `torch.Tensor`.
    - A list of valid images.

    Args:
        images (`Union[np.ndarray, torch.Tensor, PIL.Image.Image, List]`):
            The image(s) to check. Can be a batch of images (4D tensor/array), a single image, or a list of valid
            images.

    Returns:
        `bool`:
            `True` if the input is valid, `False` otherwise.
    """
    if isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 4:  # [B, C, H, W]
        return True
    elif isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 5:  # [B, F, C, H, W]
        return True
    elif is_valid_image(images):
        return True
    elif isinstance(images, list):
        return all(is_valid_image(image) for image in images)
    return False


class VideoProcessor(_VideoProcessor):

    def preprocess(
        self,
        image: PipelineImageInput,
        height: Optional[int] = None,
        width: Optional[int] = None,
        resize_mode: str = "default",  # "default", "fill", "crop"
        crops_coords: Optional[Tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        """
        Preprocess the image input.

        Args:
            image (`PipelineImageInput`):
                The image input, accepted formats are PIL images, NumPy arrays, PyTorch tensors; Also accept list of
                supported formats.
            height (`int`, *optional*):
                The height in preprocessed image. If `None`, will use the `get_default_height_width()` to get default
                height.
            width (`int`, *optional*):
                The width in preprocessed. If `None`, will use get_default_height_width()` to get the default width.
            resize_mode (`str`, *optional*, defaults to `default`):
                The resize mode, can be one of `default` or `fill`. If `default`, will resize the image to fit within
                the specified width and height, and it may not maintaining the original aspect ratio. If `fill`, will
                resize the image to fit within the specified width and height, maintaining the aspect ratio, and then
                center the image within the dimensions, filling empty with data from image. If `crop`, will resize the
                image to fit within the specified width and height, maintaining the aspect ratio, and then center the
                image within the dimensions, cropping the excess. Note that resize_mode `fill` and `crop` are only
                supported for PIL image input.
            crops_coords (`List[Tuple[int, int, int, int]]`, *optional*, defaults to `None`):
                The crop coordinates for each image in the batch. If `None`, will not crop the image.

        Returns:
            `torch.Tensor`:
                The preprocessed image.
        """
        supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)

        # Expand the missing dimension for 3-dimensional pytorch tensor or numpy array that represents grayscale image
        if self.config.do_convert_grayscale and isinstance(image, (torch.Tensor, np.ndarray)) and image.ndim == 3:
            if isinstance(image, torch.Tensor):
                # if image is a pytorch tensor could have 2 possible shapes:
                #    1. batch x height x width: we should insert the channel dimension at position 1
                #    2. channel x height x width: we should insert batch dimension at position 0,
                #       however, since both channel and batch dimension has same size 1, it is same to insert at position 1
                #    for simplicity, we insert a dimension of size 1 at position 1 for both cases
                image = image.unsqueeze(1)
            else:
                # if it is a numpy array, it could have 2 possible shapes:
                #   1. batch x height x width: insert channel dimension on last position
                #   2. height x width x channel: insert batch dimension on first position
                if image.shape[-1] == 1:
                    image = np.expand_dims(image, axis=0)
                else:
                    image = np.expand_dims(image, axis=-1)

        if isinstance(image, list) and isinstance(image[0], np.ndarray) and image[0].ndim == 4:
            warnings.warn(
                "Passing `image` as a list of 4d np.ndarray is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 4d np.ndarray",
                FutureWarning,
            )
            image = np.concatenate(image, axis=0)
        if isinstance(image, list) and isinstance(image[0], torch.Tensor) and image[0].ndim == 4:
            warnings.warn(
                "Passing `image` as a list of 4d torch.Tensor is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 4d torch.Tensor",
                FutureWarning,
            )
            image = torch.cat(image, axis=0)

        # we add support for 5D input image tensors
        if not is_valid_image_imagelist(image):
            raise ValueError(
                f"Input is in incorrect format. Currently, we only support {', '.join(str(x) for x in supported_formats)}"
            )
        if not isinstance(image, list):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            if crops_coords is not None:
                image = [i.crop(crops_coords) for i in image]
            if self.config.do_resize:
                height, width = self.get_default_height_width(image[0], height, width)
                image = [self.resize(i, height, width, resize_mode=resize_mode) for i in image]
            if self.config.do_convert_rgb:
                image = [self.convert_to_rgb(i) for i in image]
            elif self.config.do_convert_grayscale:
                image = [self.convert_to_grayscale(i) for i in image]
            image = self.pil_to_numpy(image)  # to np
            image = self.numpy_to_pt(image)  # to pt

        elif isinstance(image[0], np.ndarray):
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)

            image = self.numpy_to_pt(image)

            height, width = self.get_default_height_width(image, height, width)
            if self.config.do_resize:
                image = self.resize(image, height, width)

        elif isinstance(image[0], torch.Tensor):

            # We add support for 5D input image tenors:
            # Originally, diffusers assume that the input images represent a batch of
            # single-frame reference images, and it will add F=1 to image tensors in
            # `prepare_latents` in pipeline.__call__ function.
            # Here, we add an option when input images have dimension equal to 5.
            # Then, we check the channels to determine if they are latents or exactly
            # RGB images (channel=3) or Gray images (channel=1).
            # Note the 'F' is actually n_view * n_frame, the logic is defined in dataset.
            # 4: [B, C, H, W]; 5: [B, C, F, H, W]
            if image[0].ndim == 4 or image[0].ndim == 5:
                image = torch.cat(image, axis=0)
            else:  # ndim=3 or ndim=2
                image = torch.stack(image, axis=0)

            if self.config.do_convert_grayscale and image.ndim == 3:
                image = image.unsqueeze(1)

            channel = image.shape[1]
            # don't need any preprocess if the image is latents
            # we add support latents channels equal to vae_latent_channels or double of it.
            # 1. input channel == vae_latent_channels: the inputs are exactly latents;
            # 2. input channel == vae_latent_channels * 2:
            #                        the inputs are not sampled yet!
            if (channel == self.config.vae_latent_channels
                or
                channel == self.config.vae_latent_channels * 2
            ):
                return image

            height, width = self.get_default_height_width(image, height, width)
            if self.config.do_resize:
                image = self.resize(image, height, width)

        # expected range [0,1], normalize to [-1,1]
        do_normalize = self.config.do_normalize
        if do_normalize and image.min() < 0:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False
        if do_normalize:
            image = self.normalize(image)

        if self.config.do_binarize:
            image = self.binarize(image)

        return image
