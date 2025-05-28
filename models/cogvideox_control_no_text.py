from typing import Any, Dict, Optional, Tuple, Union, List, Callable

import traceback
import torch, os, math
import torch.nn.functional as F
from torch import nn
from PIL import Image
from einops import rearrange

from transformers import T5EncoderModel, T5Tokenizer
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock as _CogVideoXBlock
# from diffusers.models.embeddings import CogVideoXPatchEmbed as _CogVideoXPatchEmbed
from diffusers.models.embeddings import CogVideoXPatchEmbed
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline, CogVideoXPipelineOutput
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from diffusers.pipelines.cogvideo.pipeline_cogvideox_video2video import CogVideoXVideoToVideoPipeline
from diffusers.models.normalization import (
    AdaLayerNorm as _AdaLayerNorm,
    CogVideoXLayerNormZero as _CogVideoXLayerNormZero,
)
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from diffusers.pipelines import DiffusionPipeline   
from diffusers.models.attention_processor import Attention, CogVideoXAttnProcessor2_0
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler

from training.utils import CONSOLE


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ActionEmbed(nn.Module):

    def __init__(
            self,
            state_dim: int,
            hidden_size: int,
            dropout: float = 0.,
            compress_ratio: int = 1,
            mask: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.compress_ratio = compress_ratio
        self.mask = mask

        self.mlp = nn.Sequential(
            nn.Linear(in_features=state_dim * compress_ratio, out_features=hidden_size * compress_ratio * 4, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_size * compress_ratio * 4, out_features=hidden_size, bias=True),
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

        x = self.mlp(x)

        if self.mask and (is_mask := torch.rand(B, device=x.device) < 0.1).sum() > 0:
            x[is_mask] = self.mask_embed.weight[None, ...].repeat(is_mask.sum(), x.shape[1], 1)

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


class ControlNetConditioningEmbedding(nn.Module):

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


class CogVideoXLayerNormZero(_CogVideoXLayerNormZero):

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=-1)

        if temb.ndim > 2:  # [N, D]

            num_frames = temb.size(1)
            num_patches = hidden_states.shape[1] // num_frames

            scale = scale.repeat_interleave(repeats=num_patches, dim=1)
            shift = shift.repeat_interleave(repeats=num_patches, dim=1)
            gate = gate.repeat_interleave(repeats=num_patches, dim=1)
            hidden_states = self.norm(hidden_states) * (1 + scale) + shift

            enc_scale = enc_scale[:, 0:1, :]
            enc_shift = enc_shift[:, 0:1, :]
            enc_gate = enc_gate[:, 0:1, :]
            encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale) + enc_shift

        else:

            hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
            encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]

            gate = gate[:, None, :]
            enc_gate = enc_gate[:, None, :]

        return hidden_states, encoder_hidden_states, gate, enc_gate


class CogVideoXLayerNormZeroNoText(_CogVideoXLayerNormZero):

    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__(conditioning_dim, embedding_dim, elementwise_affine, eps=eps, bias=bias)

        self.linear = nn.Linear(conditioning_dim, 3 * embedding_dim, bias=bias)

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        shift, scale, gate = self.linear(self.silu(temb)).chunk(3, dim=-1)

        if temb.ndim > 2:  # [N, D]

            num_frames = temb.size(1)
            num_patches = hidden_states.shape[1] // num_frames

            scale = scale.repeat_interleave(repeats=num_patches, dim=1)
            shift = shift.repeat_interleave(repeats=num_patches, dim=1)
            hidden_states = self.norm(hidden_states) * (1 + scale) + shift

            gate = gate.repeat_interleave(repeats=num_patches, dim=1)

        else:

            hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]

            gate = gate[:, None, :]

        return hidden_states, gate


class AdaLayerNorm(_AdaLayerNorm):

    def forward(
        self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if self.emb is not None:
            temb = self.emb(timestep)

        temb = self.linear(self.silu(temb))

        if self.chunk_dim == 1 and temb.ndim == 2:
            # CONSOLE.log(f'{temb.ndim=}')

            # This is a bit weird why we have the order of "shift, scale" here and "scale, shift" in the
            # other if-branch. This branch is specific to CogVideoX for now.
            shift, scale = temb.chunk(2, dim=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]

        elif self.chunk_dim == 1 and temb.ndim == 3:
            # CONSOLE.log(f'{temb.ndim=}')
            # This is specific to ours

            shift, scale = temb.chunk(2, dim=2)

            num_frames = temb.size(1)
            num_patches = x.shape[1] // num_frames

            scale = scale.repeat_interleave(repeats=num_patches, dim=1)
            shift = shift.repeat_interleave(repeats=num_patches, dim=1)

        else:
            scale, shift = temb.chunk(2, dim=0)

        x = self.norm(x) * (1 + scale) + shift
        return x


class CogVideoXAttnProcessorNoText(CogVideoXAttnProcessor2_0):

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: None = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if encoder_hidden_states is not None:
            raise RuntimeError(f'Got invalid `encoder_hidden_states` which should be None!')

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class CogVideoXBlock(_CogVideoXBlock):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_out_bias: bool = True,
        empty_prompt: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(dim=dim,
                         num_attention_heads=num_attention_heads,
                         attention_head_dim=attention_head_dim,
                         time_embed_dim=time_embed_dim,
                         attention_bias=attention_bias,
                         qk_norm=qk_norm,
                         norm_elementwise_affine=norm_elementwise_affine,
                         norm_eps=norm_eps,
                         attention_out_bias=attention_out_bias,
                         **kwargs)
        self.empty_prompt = empty_prompt

        norm_init_args = (
            time_embed_dim, dim, norm_elementwise_affine, norm_eps
        )
        norm_init_kwargs = dict(bias=True)

        self.norm1 = (
            CogVideoXLayerNormZero(*norm_init_args, **norm_init_kwargs)
            if not empty_prompt else
            CogVideoXLayerNormZeroNoText(*norm_init_args, **norm_init_kwargs)
        )

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=(
                CogVideoXAttnProcessor2_0()
                if not empty_prompt else
                CogVideoXAttnProcessorNoText()
            ),
        )

        self.norm2 = (
            CogVideoXLayerNormZero(*norm_init_args, **norm_init_kwargs)
            if not empty_prompt else
            CogVideoXLayerNormZeroNoText(*norm_init_args, **norm_init_kwargs)
        )

    # def __init__(
    #     self,
    #     dim: int,
    #     time_embed_dim: int,
    #     norm_elementwise_affine: bool = True,
    #     norm_eps: float = 1e-5,
    #     empty_prompt: Optional[bool] = False,
    #     **kwargs,
    # ):
    #     super().__init__(dim=dim, time_embed_dim=time_embed_dim, norm_elementwise_affine=norm_elementwise_affine, norm_eps=norm_eps, **kwargs)

    #     self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
    #     self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
    #     self.empty_prompt = empty_prompt

    #     # self.fuser = Fuser(action_in_channel=time_embed_dim, out_channels=dim)

    def _forward_no_text(
        self,
        hidden_states: torch.Tensor,
        # Note here we keep the arguments list unchanged to align with the different calls
        encoder_hidden_states: None,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        action_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if encoder_hidden_states is not None:
            raise RuntimeError(f'Got invalid `encoder_hidden_states` which should be None!')

        # norm & modulate
        norm_hidden_states, gate_msa = self.norm1(hidden_states, temb)

        # attention
        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states

        norm_hidden_states, gate_ff = self.norm2(hidden_states, temb)

        # feed-forward
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output

        return hidden_states, None

    def _forward_with_text(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        action_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        text_seq_length = encoder_hidden_states.size(1)

        # inputs
        # hidden_states: [batch, num_frames * height * width, channels]
        # encoder_hidden_states: [batch, seq_len, chaneels]
        # temb: [batch, time_embed_size]

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # action_hidden_states: [batch, num_frames, temb_dim]
        # if action_hidden_states is not None:
        #     F = action_hidden_states.size(1)
        #     P = norm_hidden_states.size(1) // F
        #     action_hidden_states = action_hidden_states[:, :, None, ...].repeat(1, 1, P, 1)
        #     h = rearrange(norm_hidden_states, 'b (f p) c -> b f p c', f=F)
        #     h = self.fuser(h, action_hidden_states)
        #     norm_hidden_states = rearrange(h, 'b f p c ->  b (f p) c', f=F)

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states

    def forward(self, *args, **kwargs) -> torch.Tensor:

        if self.empty_prompt:
            return self._forward_no_text(*args, **kwargs)

        else:
            return self._forward_with_text(*args, **kwargs)


# class CogVideoXPatchEmbed(_CogVideoXPatchEmbed):
#     def __init__(self, empty_prompt: Optional[bool] = True, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.empty_prompt = empty_prompt

    # def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
    #     r"""
    #     Args:
    #         text_embeds (`torch.Tensor`):
    #             Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
    #         image_embeds (`torch.Tensor`):
    #             Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
    #     """

    #     batch_size, num_frames, channels, height, width = image_embeds.shape

    #     if self.patch_size_t is None:
    #         image_embeds = image_embeds.reshape(-1, channels, height, width)
    #         image_embeds = self.proj(image_embeds)
    #         image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
    #         image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
    #         image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
    #     else:
    #         p = self.patch_size
    #         p_t = self.patch_size_t

    #         image_embeds = image_embeds.permute(0, 1, 3, 4, 2)
    #         image_embeds = image_embeds.reshape(
    #             batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, channels
    #         )
    #         image_embeds = image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
    #         image_embeds = self.proj(image_embeds)

    #     if not self.empty_prompt:
    #         text_embeds = self.text_proj(text_embeds)
    #         embeds = torch.cat(
    #             [text_embeds, image_embeds], dim=1
    #         ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]
    #     else:
    #         embeds = image_embeds.contiguous()  # [batch, num_frames x height x width, channels]

    #     if self.use_positional_embeddings or self.use_learned_positional_embeddings:
    #         if self.use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != height):
    #             raise ValueError(
    #                 "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
    #                 "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
    #             )

    #         pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

    #         if (
    #             self.sample_height != height
    #             or self.sample_width != width
    #             or self.sample_frames != pre_time_compression_frames
    #         ):
    #             pos_embedding = self._get_positional_embeddings(
    #                 height, width, pre_time_compression_frames, device=embeds.device
    #             )
    #         else:
    #             pos_embedding = self.pos_embedding

    #         if self.empty_prompt:
    #             pos_embedding = pos_embedding[:, self.max_text_seq_length:, :]

    #         pos_embedding = pos_embedding.to(dtype=embeds.dtype)
    #         embeds = embeds + pos_embedding

    #     return embeds


class CogVideoXTransformer3DModelTraj(CogVideoXTransformer3DModel, ModelMixin):

    patch_embed: CogVideoXPatchEmbed

    # TODO: register!!!
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: Optional[int] = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        # Customized arguments
        empty_prompt: Optional[bool] = False,
        **kwargs
    ):

        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embed_dim=time_embed_dim,
            ofs_embed_dim=ofs_embed_dim,
            text_embed_dim=text_embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            attention_bias=attention_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            activation_fn=activation_fn,
            timestep_activation_fn=timestep_activation_fn,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_rotary_positional_embeddings=use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
            patch_bias=patch_bias,
            **kwargs
        )
        self.empty_prompt = empty_prompt

        inner_dim = num_attention_heads * attention_head_dim

        # Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
            # empty_prompt=empty_prompt,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    empty_prompt=empty_prompt,
                )
                for _ in range(num_layers)
            ]
        )

        # Embedding for states
        self.action_embed = ActionEmbed(state_dim=7, hidden_size=time_embed_dim, compress_ratio=4, mask=self.training)

        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        self.set_trainable_parameters()

    def set_trainable_parameters(self):

        for param in self.parameters():
            param.requires_grad_(True)

        # Freeze all parameters
        # for param in self.parameters():
        #     param.requires_grad_(False)

        # # Unfreeze parameters that need to be trained
        # for param in self.action_embed.parameters():
        #     param.requires_grad_(True)

        # for block in self.transformer_blocks:
        #     for param in block.norm1.parameters():
        #         param.requires_grad_(True)
        #     for param in block.norm2.parameters():
        #         param.requires_grad_(True)

        # for param in self.norm_out.parameters():
        #     param.requires_grad_(True)

        # for block in self.transformer_blocks:
        #     for param in block.fuser.parameters():
        #         param.requires_grad_(True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controls: Dict[str, torch.Tensor],
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        # CONSOLE.log(f"{emb.shape=}")

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        # CONSOLE.log(f"{encoder_hidden_states.shape=}, {hidden_states.shape=}")
        *_, height, width = hidden_states.shape
        if self.patch_embed.use_learned_positional_embeddings:
            if self.patch_embed.sample_width != width:
                CONSOLE.log(f"[on red]Input image embeddings have {width=}, while sample_height={self.patch_embed.sample_width}.")
            if self.patch_embed.sample_height != height:
                CONSOLE.log(f"[on red]Input image embeddings have {height=}, while sample_width={self.patch_embed.sample_height}.")
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        # CONSOLE.log(f"Patchify hidden_states: {hidden_states.shape=}")
        hidden_states = self.embedding_dropout(hidden_states)

        # Process trajectory controls
        action_hidden_states = None
        if controls.get('actions', None) is not None:
            actions = controls['actions']
            if (actions.size(1) + 1) % 4 != 0:
                pad = actions.new_zeros((actions.shape[0], 4 - (actions.size(1) + 1) % 4, actions.shape[2]))
                actions = torch.cat([pad, actions], dim=1)
            action_hidden_states = self.action_embed(actions)  # [n_batch, n_frame, hidden_dim]
            # CONSOLE.log(f"Encoded action_hidden_states (B F D): {action_hidden_states.shape}")

            # Add trajectory embeddings
            emb = emb[:, None, ...] + action_hidden_states  # [batch_size, embed_dim] -> [batch_size, num_frames, embed_dim]

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]  # [batch, num_frames x height x width, channels]

        # Reshape input image embeddings
        # channels = hidden_states.shape[-1]
        # hidden_states = hidden_states.reshape(batch_size * num_frames, -1, channels)
        # CONSOLE.log(f"Input hidden_states: {hidden_states.shape=}")

        if self.empty_prompt:
            encoder_hidden_states = None

        # Process transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:
                # Gradient checkpointing logic for hidden states
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    action_hidden_states,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    action_hidden_states=action_hidden_states,
                )

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        try:
            model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            CONSOLE.log(f"[blue] Loaded {cls.__class__.__name__} from {pretrained_model_name_or_path} checkpoint directly.")

            model.set_trainable_parameters()

            return model

        except Exception as e:
            CONSOLE.log(f"[bold yellow]Failed to load {pretrained_model_name_or_path} to {cls.__class__.__name__}: {e}")
            CONSOLE.log(f"Trace: {traceback.format_exc()}")
            CONSOLE.log("[bold yellow]Attempting to load as CogVideoXTransformer3DModel and convert...")

            base_model = CogVideoXTransformer3DModel.from_pretrained(pretrained_model_name_or_path, **kwargs)

            config = dict(base_model.config)

            model = cls(**config)
            model.load_state_dict(base_model.state_dict(), strict=False)

            model.set_trainable_parameters()
            
            return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size: Union[int, str] = "5GB",
        push_to_hub: bool = False,
        **kwargs,
    ):
        super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            save_function=save_function,
            safe_serialization=safe_serialization,
            variant=variant,
            max_shard_size=max_shard_size,
            push_to_hub=push_to_hub,
            **kwargs,
        )
        
        if is_main_process:
            config_dict = dict(self.config)
            config_dict.pop("_name_or_path", None)
            config_dict.pop("_use_default_values", None)
            config_dict["_class_name"] = "CogVideoXTransformer3DModelTraj"
            
            os.makedirs(save_directory, exist_ok=True)
            with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
                import json
                json.dump(config_dict, f, indent=2)


class CogVideoXTransformer3DModelTrajControl(CogVideoXTransformer3DModelTraj, ModelMixin):
    """
    Add control maps to the CogVideoX transformer model.

    Parameters:
        num_control_blocks (`int`, defaults to `18`):
            The number of control blocks to use. Must be less than or equal to num_layers.
    """

    patch_embed: CogVideoXPatchEmbed

    def __init__(
        self,
        num_control_blocks: Optional[int] = 18,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        **kwargs
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embed_dim=time_embed_dim,
            text_embed_dim=text_embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            attention_bias=attention_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            activation_fn=activation_fn,
            timestep_activation_fn=timestep_activation_fn,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_rotary_positional_embeddings=use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
            patch_bias=patch_bias,
            **kwargs
        )

        inner_dim = num_attention_heads * attention_head_dim
        self.num_control_blocks = num_control_blocks

        # Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )

        # Ensure num_tracking_blocks is not greater than num_layers
        if num_control_blocks > num_layers:
            raise ValueError("num_tracking_blocks must be less than or equal to num_layers")

        # Create linear layers for combining hidden states and controls
        self.combine_linears = nn.ModuleList(
            [nn.Linear(inner_dim, inner_dim) for _ in range(num_control_blocks)]
        )

        # Initialize weights of combine_linears to zero
        for linear in self.combine_linears:
            linear.weight.data.zero_()
            linear.bias.data.zero_()

        # Create transformer blocks for processing controls
        self.transformer_blocks_copy = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    time_embed_dim=self.config.time_embed_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                )
                for _ in range(num_control_blocks)
            ]
        )

        # For initial combination of hidden states and controls
        self.initial_combine_linear = nn.Linear(inner_dim, inner_dim)
        self.initial_combine_linear.weight.data.zero_()
        self.initial_combine_linear.bias.data.zero_()

        # Embedding for states
        self.action_embed = ActionEmbed(state_dim=7, hidden_size=inner_dim, compress_ratio=4, mask=self.training)

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze parameters that need to be trained
        for linear in self.combine_linears:
            for param in linear.parameters():
                param.requires_grad = True
        
        for block in self.transformer_blocks_copy:
            for param in block.parameters():
                param.requires_grad = True
        
        for param in self.initial_combine_linear.parameters():
            param.requires_grad = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controls: Dict[str, torch.Tensor],
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        *_, height, width = hidden_states.shape
        if self.patch_embed.use_learned_positional_embeddings:
            if self.patch_embed.sample_width != width:
                CONSOLE.log(f"[on red]Input image embeddings have {width=}, while sample_height={self.patch_embed.sample_width}.")
            if self.patch_embed.sample_height != height:
                CONSOLE.log(f"[on red]Input image embeddings have {height=}, while sample_width={self.patch_embed.sample_height}.")
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        # Process controls
        actions = controls['actions']
        action_hidden_states = self.action_embed(actions)  # [n_batch, n_frame, hidden_dim]

        depth_maps = controls['depths']
        depth_maps_hidden_states = self.patch_embed(encoder_hidden_states.clone(), depth_maps)
        depth_maps_hidden_states = self.embedding_dropout(depth_maps_hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        depth_maps = depth_maps_hidden_states[:, text_seq_length:]

        # Combine hidden states and controls initially
        depth_maps = self.initial_combine_linear(
            hidden_states + depth_maps
        )

        # Process transformer blocks
        for i in range(len(self.transformer_blocks)):
            if self.training and self.gradient_checkpointing:
                # Gradient checkpointing logic for hidden states
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.transformer_blocks[i]),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = self.transformer_blocks[i](
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

            if i < len(self.transformer_blocks_copy):
                if self.training and self.gradient_checkpointing:
                    # Gradient checkpointing logic for tracking maps
                    depth_maps, _ = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.transformer_blocks_copy[i]),
                        depth_maps,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    depth_maps, _ = self.transformer_blocks_copy[i](
                        hidden_states=depth_maps,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                    )

                # Combine hidden states and tracking maps
                depth_maps = self.combine_linears[i](depth_maps)
                hidden_states = hidden_states + depth_maps

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        try:
            model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            CONSOLE.log("[blue] Loaded DiffusionAsShader checkpoint directly.")

            for param in model.parameters():
                param.requires_grad = False

            for linear in model.combine_linears:
                for param in linear.parameters():
                    param.requires_grad = True

            for block in model.transformer_blocks_copy:
                for param in block.parameters():
                    param.requires_grad = True

            for param in model.initial_combine_linear.parameters():
                param.requires_grad = True

            return model

        except Exception as e:
            CONSOLE.log(f"[bold yellow]Failed to load {pretrained_model_name_or_path} to {cls.__class__.__name__}: {e}")
            CONSOLE.log(f"Trace: {traceback.format_exc()}")
            CONSOLE.log("[bold yellow]Attempting to load as CogVideoXTransformer3DModel and convert...")

            base_model = CogVideoXTransformer3DModelTraj.from_pretrained(pretrained_model_name_or_path, **kwargs)

            config = dict(base_model.config)
            config["num_control_blocks"] = kwargs.pop("num_control_blocks", 18)

            model = cls(**config)
            model.load_state_dict(base_model.state_dict(), strict=False)

            model.initial_combine_linear.weight.data.zero_()
            model.initial_combine_linear.bias.data.zero_()
            
            for linear in model.combine_linears:
                linear.weight.data.zero_()
                linear.bias.data.zero_()
            
            for i in range(model.num_control_blocks):
                model.transformer_blocks_copy[i].load_state_dict(model.transformer_blocks[i].state_dict())

            for param in model.parameters():
                param.requires_grad = False
            
            for linear in model.combine_linears:
                for param in linear.parameters():
                    param.requires_grad = True
                
            for block in model.transformer_blocks_copy:
                for param in block.parameters():
                    param.requires_grad = True
                
            for param in model.initial_combine_linear.parameters():
                param.requires_grad = True
            
            return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size: Union[int, str] = "5GB",
        push_to_hub: bool = False,
        **kwargs,
    ):
        super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            save_function=save_function,
            safe_serialization=safe_serialization,
            variant=variant,
            max_shard_size=max_shard_size,
            push_to_hub=push_to_hub,
            **kwargs,
        )
        
        if is_main_process:
            config_dict = dict(self.config)
            config_dict.pop("_name_or_path", None)
            config_dict.pop("_use_default_values", None)
            config_dict["_class_name"] = "CogVideoXTransformer3DModelControl"
            config_dict["num_control_blocks"] = self.num_control_blocks
            
            os.makedirs(save_directory, exist_ok=True)
            with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
                import json
                json.dump(config_dict, f, indent=2)


class CogVideoXImageToVideoPipelineTraj(CogVideoXImageToVideoPipeline, DiffusionPipeline):

    transformer: CogVideoXTransformer3DModelTraj

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModelTraj,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)
        
        if not isinstance(self.transformer, CogVideoXTransformer3DModelTraj):
            raise ValueError("The transformer in this pipeline must be of type CogVideoXTransformer3DModelTracking")
            
        CONSOLE.log(f"[bold yellow] Number of transformer blocks: {len(self.transformer.transformer_blocks)}")
        # self.transformer = torch.compile(self.transformer)

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, Image.Image],
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        control: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        # Most of the implementation remains the same as the parent class
        # We will modify the parts that need to handle tracking_maps

        # 1. Check inputs and set default values
        self.check_inputs(
            image,
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            del negative_prompt_embeds

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )

        if self.transformer.config.in_channels != 16:
            latent_channels = self.transformer.config.in_channels // 2
        else:
            latent_channels = self.transformer.config.in_channels
        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        del image

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop

        # Create ofs embeds if required
        ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
                del latent_image_input

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # Predict noise
                self.transformer.to(dtype=latent_model_input.dtype)
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    controls=control,
                    return_dict=False,
                )[0]
                del latent_model_input
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    del noise_pred_uncond, noise_pred_text

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                del noise_pred
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 9. Post-processing
        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)


class CogVideoXImageToVideoPipelineTrajControl(CogVideoXImageToVideoPipeline, DiffusionPipeline):

    transformer: CogVideoXTransformer3DModelTrajControl

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModelTrajControl,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)
        
        if not isinstance(self.transformer, CogVideoXTransformer3DModelTrajControl):
            raise ValueError("The transformer in this pipeline must be of type CogVideoXTransformer3DModelTracking")
            
        CONSOLE.log(f"[bold yellow] Number of transformer blocks: {len(self.transformer.transformer_blocks)}")
        CONSOLE.log(f"[bold yellow] Number of control transformer blocks: {len(self.transformer.transformer_blocks_copy)}")
        self.transformer = torch.compile(self.transformer)

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, Image.Image],
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        tracking_maps: Optional[torch.Tensor] = None,
        tracking_image: Optional[torch.Tensor] = None,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        # Most of the implementation remains the same as the parent class
        # We will modify the parts that need to handle tracking_maps

        # 1. Check inputs and set default values
        self.check_inputs(
            image,
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            del negative_prompt_embeds

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )

        tracking_image = self.video_processor.preprocess(tracking_image, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )
        if self.transformer.config.in_channels != 16:
            latent_channels = self.transformer.config.in_channels // 2
        else:
            latent_channels = self.transformer.config.in_channels
        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        del image
        
        _, tracking_image_latents = self.prepare_latents(
            tracking_image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents=None,
        )
        del tracking_image

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
                del latent_image_input

                # Handle tracking maps
                if tracking_maps is not None:
                    latents_tracking_image = torch.cat([tracking_image_latents] * 2) if do_classifier_free_guidance else tracking_image_latents
                    tracking_maps_input = torch.cat([tracking_maps] * 2) if do_classifier_free_guidance else tracking_maps
                    tracking_maps_input = torch.cat([tracking_maps_input, latents_tracking_image], dim=2)
                    del latents_tracking_image
                else:
                    tracking_maps_input = None

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # Predict noise
                self.transformer.to(dtype=latent_model_input.dtype)
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    tracking_maps=tracking_maps_input,
                    return_dict=False,
                )[0]
                del latent_model_input
                if tracking_maps_input is not None:
                    del tracking_maps_input
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    del noise_pred_uncond, noise_pred_text

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                del noise_pred
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 9. Post-processing
        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)
