import traceback
import torch, os, math
import torch.nn.functional as F
import fnmatch
from torch import nn
from PIL import Image
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Sequence, Tuple, Union, List, Callable

from transformers.models.t5 import T5EncoderModel, T5Tokenizer
from diffusers.utils.constants import USE_PEFT_BACKEND
from diffusers.utils.import_utils import is_torch_version, logging
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import randn_tensor
from diffusers.configuration_utils import register_to_config
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock as _CogVideoXBlock
from diffusers.models.embeddings import CogVideoXPatchEmbed, get_3d_sincos_pos_embed
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.attention_processor import Attention, CogVideoXAttnProcessor2_0 as _CogVideoXAttnProcessor2_0
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.models.normalization import (
    AdaLayerNorm as _AdaLayerNorm,
    CogVideoXLayerNormZero as _CogVideoXLayerNormZero,
)
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_ddim_cogvideox import CogVideoXDDIMScheduler
from diffusers.schedulers.scheduling_dpm_cogvideox import CogVideoXDPMScheduler

from orv.utils import CONSOLE
from orv.models.components import Transformer3DModelTrajOutput, ActionEmbed, ActionRecon, VideoProcessor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CogVideoXLayerNormZero(_CogVideoXLayerNormZero):

    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        modulate_encoder_hidden_states: Optional[bool] = False,
    ) -> None:
        super().__init__(conditioning_dim, embedding_dim, elementwise_affine, eps, bias)

        # CogVideoX predicts scale, shift, gate vectors for both images and texts
        # Here we choose to or not to modulate the text features
        self.modulate_encoder_hidden_states = modulate_encoder_hidden_states
        if not modulate_encoder_hidden_states:
            self.linear = nn.Linear(conditioning_dim, 3 * embedding_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        action_emb: Optional[torch.Tensor] = None,
    ) -> Sequence[torch.Tensor | None]:

        gate = enc_gate = None

        if not self.modulate_encoder_hidden_states:

            if action_emb is None:

                shift, scale, gate = self.linear(self.silu(temb)).chunk(3, dim=-1)

                hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
                encoder_hidden_states = self.norm(encoder_hidden_states)

                gate = gate[:, None, :]

            # Given action embedding's shape [N, F, D] with frame-level modulations,
            # we discard the modulations on `encoder_hidden_states`.
            if action_emb is not None:

                # Add action embeddings to temb: [N, D] -> [N, F, D]
                temb = temb[:, None, :] + action_emb

                shift, scale, gate = self.linear(self.silu(temb)).chunk(3, dim=-1)

                num_frames = action_emb.size(1)
                num_patches = hidden_states.shape[1] // num_frames

                scale = scale.repeat_interleave(repeats=num_patches, dim=1)
                shift = shift.repeat_interleave(repeats=num_patches, dim=1)
                hidden_states = self.norm(hidden_states) * (1 + scale) + shift

                gate = gate.repeat_interleave(repeats=num_patches, dim=1)

                encoder_hidden_states = self.norm(encoder_hidden_states)

        elif self.modulate_encoder_hidden_states:

            # Same default settings as the original CogVideoX
            if action_emb is None:

                shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=-1)

                hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
                encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]

                gate = gate[:, None, :]
                enc_gate = enc_gate[:, None, :]

            # We modulate images through time embeddings + action embeddings;
            # and modulate texts through only the time embeddings.
            # For efficiency, we partially forward `self.linear` twice!
            if action_emb is not None:

                embedding_dim = hidden_states.shape[-1]  # hidden_size

                shift, scale, gate = torch.nn.functional.linear(
                    self.silu(temb[:, None, :] + action_emb),
                    self.linear.weight[: 3 * embedding_dim],
                    self.linear.bias[: 3 * embedding_dim],
                ).chunk(3, dim=-1)
                enc_shift, enc_scale, enc_gate = torch.nn.functional.linear(
                    self.silu(temb),
                    self.linear.weight[3 * embedding_dim :],
                    self.linear.bias[3 * embedding_dim :],
                ).chunk(3, dim=-1)

                # modulate images
                num_frames = action_emb.size(1)
                num_patches = hidden_states.size(1) // num_frames

                scale = scale.repeat_interleave(repeats=num_patches, dim=1)
                shift = shift.repeat_interleave(repeats=num_patches, dim=1)
                hidden_states = self.norm(hidden_states) * (1 + scale) + shift

                gate = gate.repeat_interleave(repeats=num_patches, dim=1)

                # modulate texts
                encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]

                enc_gate = enc_gate[:, None, :]

        else:
            raise RuntimeError(f'Invalid inputs: {self.modulate_encoder_hidden_states=} and {temb.shape=}.')

        return hidden_states, encoder_hidden_states, gate, enc_gate


class AdaLayerNorm(_AdaLayerNorm):

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        action_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self.emb is not None:
            temb = self.emb(timestep)

        # Add action embeddings to temb: [N, D] -> [N, F, D]
        if action_emb is not None and temb is not None:
            temb = temb[:, None, :] + action_emb

        temb = self.linear(self.silu(temb))

        # Same as default settings of CogVideoX
        if self.chunk_dim == 1 and action_emb is None:

            # This is a bit weird why we have the order of "shift, scale" here and "scale, shift" in the
            # other if-branch. This branch is specific to CogVideoX for now.
            shift, scale = temb.chunk(2, dim=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]

        # With action frame-level modulation
        elif self.chunk_dim == 1 and action_emb is not None:

            shift, scale = temb.chunk(2, dim=2)

            num_frames = action_emb.size(1)
            num_patches = x.shape[1] // num_frames

            scale = scale.repeat_interleave(repeats=num_patches, dim=1)
            shift = shift.repeat_interleave(repeats=num_patches, dim=1)

        else:
            scale, shift = temb.chunk(2, dim=0)

        x = self.norm(x) * (1 + scale) + shift

        return x


class CogVideoXAttnProcessor2_0(_CogVideoXAttnProcessor2_0):

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if attn.to_k is None:
            raise RuntimeError('attn.to_k cannot be None!')
        if attn.to_v is None:
            raise RuntimeError('attn.to_v cannot be None!')
        if attn.to_out is None:
            raise RuntimeError('attn.to_out cannot be None!')

        text_seq_length = 0
        if encoder_hidden_states is not None:
            text_seq_length = encoder_hidden_states.size(1)

            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

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

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


class MVBlock(nn.Module):

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
        modulate_encoder_hidden_states: Optional[bool] = False,
    ) -> None:
        super().__init__()

        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True,
                                            modulate_encoder_hidden_states=modulate_encoder_hidden_states)
        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm='layer_norm' if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )
        self.modulate_encoder_hidden_states = modulate_encoder_hidden_states

        self.cam_encoder = nn.Linear(12, dim)
        self.proj_out = nn.Linear(dim, dim)

        # Zero initialization
        self.cam_encoder.weight.data.zero_()
        self.cam_encoder.bias.data.zero_()
        self.proj_out.weight.data.zero_()
        self.proj_out.bias.data.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb_view: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_view: Optional[int] = None,
        n_frame: Optional[int] = None,
    ) -> torch.Tensor:

        norm_hidden_states, norm_encoder_hidden_states, gate_msa, _ = self.norm1(
            hidden_states, encoder_hidden_states, temb,
        )

        # view attention
        norm_hidden_states = rearrange(norm_hidden_states, '(b v) (f s) d -> (b f) (v s) d', f=n_frame, v=n_view)
        if self.modulate_encoder_hidden_states:
            norm_encoder_hidden_states = rearrange(norm_encoder_hidden_states, '(b v) n d -> b (v n) d', v=n_view)
            norm_encoder_hidden_states = repeat(norm_encoder_hidden_states, 'b n d -> (b f) n d', f=n_frame)

        attn_hidden_states, _ = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=(
                norm_encoder_hidden_states
                if self.modulate_encoder_hidden_states
                else None
            ),
            image_rotary_emb=image_rotary_emb_view,
        )

        # project back with residual connection
        attn_hidden_states = self.proj_out(attn_hidden_states)
        attn_hidden_states = rearrange(attn_hidden_states, '(b f) (v s) d -> (b v) (f s) d', f=n_frame, v=n_view)
        hidden_states = hidden_states + gate_msa * attn_hidden_states

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
        modulate_encoder_hidden_states: Optional[bool] = False,
        **kwargs,
    ) -> None:
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

        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True,
                                            modulate_encoder_hidden_states=modulate_encoder_hidden_states)
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True,
                                            modulate_encoder_hidden_states=modulate_encoder_hidden_states)
        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )
        self.modulate_encoder_hidden_states = modulate_encoder_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        action_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb, action_emb=action_emb,
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=(
                norm_encoder_hidden_states
                if self.modulate_encoder_hidden_states
                else None
            ),
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        if self.modulate_encoder_hidden_states:
            encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb, action_emb=action_emb,
        )

        # feed-forward
        if not self.modulate_encoder_hidden_states:

            ff_output = self.ff(norm_hidden_states)

            hidden_states = hidden_states + gate_ff * ff_output

        else:  # Same as default CogVideoX

            text_seq_length = encoder_hidden_states.size(1)

            norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
            ff_output = self.ff(norm_hidden_states)

            hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
            encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class CogVideoXTransformer3DModelTraj(CogVideoXTransformer3DModel, ModelMixin):

    patch_embed: CogVideoXPatchEmbed

    @register_to_config
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
        # additional arguments
        loaded_pretrained_model_name_or_path: Optional[str] = None,
        modulate_encoder_hidden_states: bool = False,
        num_control_blocks: int = 12,
        recon_action: bool = False,
        visual_guidance: bool = False,
        num_control_keys: int = 2,  # ['depth', 'label']
        multiview: bool = False,
        max_n_view: int = 3,
        from_t2v: bool = False,
        **kwargs,
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
        )

        if fnmatch.fnmatch(str(loaded_pretrained_model_name_or_path), 'THUDM*CogVideoX*'):
            if not modulate_encoder_hidden_states:
                raise RuntimeError(f"You're trying to load {loaded_pretrained_model_name_or_path} but"
                                   "set modulate_encoder_hidden_states to False!")

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
                    modulate_encoder_hidden_states=modulate_encoder_hidden_states,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        # Embedding for actions of robot arm
        self.action_embed = ActionEmbed(state_dim=7, hidden_size=time_embed_dim, compress_ratio=4,
                                        patch_size_t=patch_size_t, mask=self.training)
        self.action_recon = None
        if recon_action:
            self.action_recon = ActionRecon(state_dim=7, hidden_size=time_embed_dim, compress_ratio=4)

        # Additional spatial-aligned guidance singal
        if visual_guidance:

            # Ensure num_tracking_blocks is not greater than num_layers
            if num_control_blocks > num_layers:
                raise ValueError("num_tracking_blocks must be less than or equal to num_layers")

            self.num_control_keys = num_control_keys

            # For initial combination of hidden states and tracking maps
            self.initial_combine_linear = nn.Linear(inner_dim * self.num_control_keys, inner_dim)

        # Multiview module
        if multiview:

            # positional embedding
            pos_embedding_v = self._get_positional_embeddings_v(sample_views=max_n_view)
            self.register_buffer('pos_embedding_v', pos_embedding_v, persistent=False)

            self.mv_blocks = nn.ModuleList(
                [
                    MVBlock(
                        dim=inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        time_embed_dim=time_embed_dim,
                        attention_bias=attention_bias,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        modulate_encoder_hidden_states=modulate_encoder_hidden_states,
                    )
                    for _ in range(num_layers)
                ]
            )

        self._set_zeros()
        self._set_trainable_parameters()

    def _set_zeros(self):

        # when extending t2v to i2v, we zero-
        # initialize the latter 16 channels.
        if self.config.from_t2v:
            self.patch_embed.proj.weight.data[:, -16:, ...].zero_()

        if hasattr(self, 'initial_combine_linear'):
            self.initial_combine_linear.weight.data.zero_()
            self.initial_combine_linear.bias.data.zero_()

        if hasattr(self, 'combine_linears'):
            for linear in self.combine_linears:
                linear.weight.data.zero_()
                linear.bias.data.zero_()

    def _set_trainable_parameters(self):

        # finetune multiview version
        if self.config.multiview:

            for param in self.parameters():
                param.requires_grad_(False)

            for param in self.mv_blocks.parameters():
                param.requires_grad_(True)

        # base model
        else:

            for param in self.parameters():
                param.requires_grad_(True)

    # Adapted from `CogVideoXPatchEmbed`
    def _get_positional_embeddings_v(
        self, sample_views: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        post_patch_height = self.patch_embed.sample_height // self.patch_embed.patch_size
        post_patch_width = self.patch_embed.sample_width // self.patch_embed.patch_size
        # -> num_patches = post_patch_height * post_patch_width * sample_views

        pos_embedding = get_3d_sincos_pos_embed(
            self.patch_embed.embed_dim,
            (post_patch_width, post_patch_height),
            sample_views,
            self.patch_embed.spatial_interpolation_scale,
            1.0,
            device=device,
            output_type="pt",
        )  # [T, H*W, D]
        pos_embedding = pos_embedding.flatten(0, 1).unsqueeze(0)  # -> [1, num_patches, embed_dim]

        return pos_embedding

    def _get_sample_pos_embed_v(self, n_view):
        if n_view == self.config.max_n_view:
            pos_embedding_v = self.pos_embedding_v
        else:
            _, _, embed_dim = self.pos_embedding_v.shape
            pos_embedding_v = self.pos_embedding_v.reshape(
                1, self.config.max_n_view, -1, embed_dim 
            )
            pos_embedding_v = pos_embedding_v[:, :n_view, ...].flatten(1, 2)
        return pos_embedding_v

    @staticmethod
    def compute_action_loss(
        x: torch.Tensor,
        x_recon: torch.Tensor,
        loss_weight: dict,
        mask: Optional[torch.Tensor]=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if mask is None:
            mask = torch.ones((x.size(0),), device=x.device).bool()

        rot_loss = 1 - torch.cos(
            x_recon[mask, ..., 3:6] - x[mask, ..., 3:6]
        ).mean()

        x_recon[..., -1] = F.sigmoid(x_recon[..., -1])
        pos_loss = F.smooth_l1_loss(x_recon[mask, ..., :3], x[mask, ..., :3])
        grip_loss = F.smooth_l1_loss(x_recon[mask, ..., -1], x[mask, ..., -1])

        rot_loss = rot_loss * loss_weight['rot_loss']
        pos_loss = pos_loss * loss_weight['pos_loss']
        grip_loss = grip_loss * loss_weight['grip_loss']

        return rot_loss, pos_loss, grip_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controls_or_guidances: Dict[str, torch.Tensor],
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        num_views: int = 1,
        image_rotary_emb_view: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple | Transformer3DModelTrajOutput:
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

        # Check multiview
        if self.config.multiview:
            if num_views <= 1:
                CONSOLE.log(f"[bold yellow]You're tring multiview mode but no multiview inputs!")

        # Gradient checkpointing logic for hidden states
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

        if num_views > 1:
            hidden_states = rearrange(hidden_states, 'b (v f) c h w -> (b v) f c h w', v=num_views)
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=num_views, dim=0)
        batch_size, num_frames, _, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        temb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None and self.ofs_proj is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            temb = temb + ofs_emb

        # multiviews share the same noise level
        if num_views > 1:
            temb = temb.repeat_interleave(repeats=num_views, dim=0)

        # 2. Patch embedding
        *_, height, width = hidden_states.shape
        if self.patch_embed.use_learned_positional_embeddings:
            if self.patch_embed.sample_width != width:
                CONSOLE.log(f"[on red]Input image embeddings have {width=}, while sample_height={self.patch_embed.sample_width}.")
            if self.patch_embed.sample_height != height:
                CONSOLE.log(f"[on red]Input image embeddings have {height=}, while sample_width={self.patch_embed.sample_height}.")
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)
        encoder_hidden_states_clone = encoder_hidden_states.clone()

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]  # [batch, n_frames x height x width, channels]

        # 2.1. Multiview positional embedding if possible
        if num_views > 1:
            hidden_states = rearrange(hidden_states, '(b v) (f s) d -> (b f) (v s) d', v=num_views, f=num_frames)
            hidden_states = hidden_states + self._get_sample_pos_embed_v(n_view=num_views).to(dtype=hidden_states.dtype)
            hidden_states = rearrange(hidden_states, '(b f) (v s) d -> (b v) (f s) d', v=num_views, f=num_frames)

        # Process action controls
        action_hidden_states = is_action_mask = action_emb = actions_recon = None
        pad_frames = 0
        if controls_or_guidances.get('actions', None) is not None:
            actions = controls_or_guidances['actions']
            res_frames = (actions.size(1) + 1) % 4
            if res_frames > 0:
                pad_frames = 4 - res_frames
                pad = actions.new_zeros((actions.shape[0], pad_frames, actions.shape[2]))
                actions = torch.cat([pad, actions], dim=1)
            action_hidden_states, is_action_mask = self.action_embed(actions)  # [n_batch, n_frame, hidden_dim]

            # multivews share the same actions
            if num_views > 1:
                action_hidden_states = action_hidden_states.repeat_interleave(repeats=num_views, dim=0)

            # Will add action embeddings to `temb` BUT we do not add it here
            """>>> temb = temb[:, None, ...] + action_hidden_states"""
            action_emb = action_hidden_states  # [n_batch, n_frame, embed_dim]

            if self.training and self.config.recon_action and self.action_recon is not None:
                actions_recon = self.action_recon(action_hidden_states)  # [n_batch, n_frame, state_dim]
                if pad_frames > 0:
                    actions_recon = actions_recon[:, pad_frames:]

        # Process additional visual controls
        controls_hidden_states = []
        if (depths := controls_or_guidances.get('depths', None)) is not None and self.config.visual_guidance:

            if num_views > 1:
                depths = rearrange(depths, 'b (v f) c h w -> (b v) f c h w', v=num_views)
            depths_hidden_states = self.patch_embed(encoder_hidden_states_clone, depths)
            depths_hidden_states = self.embedding_dropout(depths_hidden_states)
            depths_hidden_states = depths_hidden_states[:, text_seq_length:]  # [batch, n_frames x height x width, channels]
            controls_hidden_states.append(depths_hidden_states)

        if (labels := controls_or_guidances.get('labels', None)) is not None and self.config.visual_guidance:

            if num_views > 1:
                labels = rearrange(labels, 'b (v f) c h w -> (b v) f c h w', v=num_views)
            labels_hidden_states = self.patch_embed(encoder_hidden_states_clone, labels)
            labels_hidden_states = self.embedding_dropout(labels_hidden_states)
            labels_hidden_states = labels_hidden_states[:, text_seq_length:]  # [batch, n_frames x height x width, channels]
            controls_hidden_states.append(labels_hidden_states)

        if controls_hidden_states:

            assert len(controls_hidden_states) == self.num_control_keys, f'Mismatched number of controls: {len(controls_hidden_states)=} but {self.num_control_keys=}.'
            controls_hidden_states = torch.cat(controls_hidden_states, dim=-1)

            # Combine hidden states and controls initially
            controls_hidden_states = self.initial_combine_linear(
                hidden_states.repeat(1, 1, self.num_control_keys) + controls_hidden_states
            )

            # initially combine with visual controls
            hidden_states = hidden_states + controls_hidden_states

        # Process sequential DiT blocks
        for i in range(self.config.num_layers):

            if self.config.multiview:

                block3d_v = self.mv_blocks[i]

                if self.training and self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block3d_v),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb_view,
                        num_views,
                        num_frames,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = block3d_v(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb_view=image_rotary_emb_view,
                        n_view=num_views,
                        n_frame=num_frames,
                    )

            block3d_t = self.transformer_blocks[i]

            if self.training and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block3d_t),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    action_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block3d_t(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    action_emb=action_emb,
                )

        if fnmatch.fnmatch(str(self.config.loaded_pretrained_model_name_or_path), '*CogVideoX*-5b*'):
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]
        else:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=temb, action_emb=action_emb)
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

        output = rearrange(output, '(b v) f c h w -> b (v f) c h w', v=num_views)

        if not return_dict:
            return (output, is_action_mask, actions_recon)
        return Transformer3DModelTrajOutput(
            sample=output, is_action_mask=is_action_mask, actions_recon=actions_recon,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        model: CogVideoXTransformer3DModelTraj

        try:
            model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            CONSOLE.log(f"[bold blue]Loaded {cls.__name__} from {pretrained_model_name_or_path} checkpoint directly.")

        except Exception as e:
            CONSOLE.log(f"[bold yellow]Failed to load {pretrained_model_name_or_path} to {cls.__name__}: {e}")
            CONSOLE.log(f"Trace: {traceback.format_exc()}")

            pretrained_model_config = CogVideoXTransformer3DModelTraj.load_config(
                pretrained_model_name_or_path, subfolder='transformer')

            if pretrained_model_config['_class_name'] == cls.__name__:
                # The pretrained model if from the same class.
                # Be careful if it causes infinite cycle calls.
                CONSOLE.log(f"[bold yellow]Attempting to load as {cls.__name__} and convert...")
                base_model = CogVideoXTransformer3DModelTraj.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder=kwargs.pop('subfolder'),
                    torch_dtype=kwargs.pop('torch_dtype'),
                    revision=kwargs.pop('revision'),
                    variant=kwargs.pop('variant'),  # do not put all kwargs here
                )
            else:
                # The pretrained model if from the original CogVideoX (2b or 5b).
                CONSOLE.log("[bold yellow]Attempting to load as CogVideoXTransformer3DModel and convert...")
                base_model = CogVideoXTransformer3DModel.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder=kwargs.pop('subfolder'),
                    torch_dtype=kwargs.pop('torch_dtype'),
                    revision=kwargs.pop('revision'),
                    variant=kwargs.pop('variant'),  # do not put all kwargs here
                )

            CONSOLE.log(f"Creating scratch model ...")
            model_config = dict(base_model.config)

            # replace the sample parameters to setup the sin_cos positional embeddings.
            # we actually no need to do this since `kwargs` will be used in __init__() anyway.
            # but we still do it explicitly and give a kind reminder.
            if fnmatch.fnmatch(pretrained_model_name_or_path, 'THUDM*CogVideoX*'):
                CONSOLE.log(f"Sample parameters will be modified as:\n"
                            f"sample_height: from {model_config['sample_height']} to {kwargs['sample_height']}\n"
                            f"sample_width: from {model_config['sample_width']} to {kwargs['sample_width']}\n"
                            f"sample_frames: from {model_config['sample_frames']} to {kwargs['sample_frames']}")
                model_config['sample_height'] = kwargs.pop('sample_height')
                model_config['sample_width'] = kwargs.pop('sample_width')
                model_config['sample_frames'] = kwargs.pop('sample_frames')

            # if pretrained model is CogVideoX 2b (T2V), we need to double
            # the input channels to support the image condition input.
            if fnmatch.fnmatch(pretrained_model_name_or_path, 'THUDM*CogVideoX*-2b*'):
                assert model_config['in_channels'] == 16, f'Wrong `in_channels` in config of {pretrained_model_name_or_path}!'
                model_config['in_channels'] = 32

                # `kwargs` contains additional arguments! If there exists repeated configs
                # compared to `base_model.config`, they will be supassed!
                model = cls.from_config(model_config, from_t2v=True, **kwargs)

                CONSOLE.log(f"Loading parameters ...")
                pretrain_state_dict = base_model.state_dict()
                # load other paramters first
                input_weight = pretrain_state_dict.pop('patch_embed.proj.weight')
                model.load_state_dict(pretrain_state_dict, strict=False)
                # load input channels
                model.patch_embed.proj.weight.data[:, :16, ...].copy_(input_weight)

            # for non-special case, just do as expected.
            else:

                # `kwargs` contains additional arguments! If there exists repeated configs
                # compared to `base_model.config`, they will be supassed!
                model = cls.from_config(model_config, **kwargs)

                CONSOLE.log(f"Loading parameters from base model ...")
                model.load_state_dict(base_model.state_dict(), strict=False)

            # copy parameters from 3d attention to multiview attention
            if model.config.multiview:
                # if the pretrained model is a multiview model, we then do not
                # copy mv parameters from the 3d attention!
                if isinstance(pretrained_model_name_or_path, str) and not ('multiview' in pretrained_model_name_or_path or base_model.config.multiview):
                    for i in range(len(model.mv_blocks)):
                        model.mv_blocks[i].load_state_dict(
                            model.transformer_blocks[i].state_dict(), strict=False
                        )

        model._set_trainable_parameters()
        
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
            raise ValueError("The transformer in this pipeline must be of type CogVideoXTransformer3DModelTraj")
            
        CONSOLE.log(f"[bold yellow] Number of transformer blocks: {len(self.transformer.transformer_blocks)}")
        # self.transformer = torch.compile(self.transformer)

        self.video_processor = VideoProcessor(
            vae_latent_channels=self.vae.config.latent_channels,
            vae_scale_factor=self.vae_scale_factor_spatial,
        )

    def prepare_latents(
        self,
        image: torch.Tensor,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        num_frames: int = 13,
        num_views: int = 1,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        """Added support for 5D input images/latents"""

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_views * num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        # For CogVideoX1.5, the latent should add 1 for padding (Not use)
        if self.transformer.config.patch_size_t is not None:
            shape = shape[:1] + (shape[1] + shape[1] % self.transformer.config.patch_size_t,) + shape[2:]

        if image.ndim == 4:  # [B, C, H, W]
            # encode rgb tensors to latents
            from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import retrieve_latents
            if image.size(1) != 3:
                raise RuntimeError(f'Invalid input channels {image.shape=}!')

            image = rearrange(image, '(b v f) c h w -> (b v) f c h w', b=batch_size, v=num_views)  # note here 'f' (reference images) is not num_frames
            image = image.permute(0, 2, 1, 3, 4)  # -> [B, C, F, H, W]

            if isinstance(generator, list):
                image_latents = [
                    retrieve_latents(self.vae.encode(image[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
                ]
            else:
                image_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in image]

            image_latents = torch.cat(image_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # -> [B, F, C, H, W]
            image_latents = rearrange(image_latents, '(b v) f c h w -> b (v f) c h w', v=num_views)

        elif image.ndim == 5:  # [B, C, F, H, W]
            # already encoded as latents

            input_channel = image.size(1)
            if input_channel == num_channels_latents * 2:
                # need to sample latents from distributions
                latent_dist = DiagonalGaussianDistribution(image)
                image_latents = latent_dist.sample(generator)
                image_latents = image_latents.permute(0, 2, 1, 3, 4)  # -> [B, F, C, H, W]
            elif input_channel == num_channels_latents:
                image_latents = image.permute(0, 2, 1, 3, 4)  # -> [B, F, C, H, W]
            else:

                raise RuntimeError(f'Invalid input channels {image.shape=} while {num_channels_latents=}!')
        else:
            raise RuntimeError(f'Invalid dimensions of image input: {image.shape=}')

        if not self.vae.config.invert_scale_latents:
            image_latents = self.vae_scaling_factor_image * image_latents
        else:
            # This is awkward but required because the CogVideoX team forgot to multiply the
            # scaling factor during training :)
            image_latents = 1 / self.vae_scaling_factor_image * image_latents

        # ! padding image conditions
        image_latents = rearrange(image_latents, 'b (v f) c h w -> b v f c h w', v=num_views)

        image_frames = image_latents.size(2)
        if image_frames > num_frames:
            raise RuntimeError(f'Invalid input {image_frames=} while {num_frames=}!')
        padding_shape = (
            batch_size,
            num_views,
            num_frames - image_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        latent_padding = torch.zeros(padding_shape, device=device, dtype=dtype)
        image_latents = torch.cat([image_latents, latent_padding], dim=2)

        # Select the first frame along the second dimension
        if self.transformer.config.patch_size_t is not None:
            first_frame = image_latents[:, :, : image_latents.size(1) % self.transformer.config.patch_size_t, ...]
            image_latents = torch.cat([first_frame, image_latents], dim=2)

        image_latents = rearrange(image_latents, 'b v f c h w -> b (v f) c h w')

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents, image_latents

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, Image.Image],
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_views: int = 1,
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
        controls_or_guidances: Dict[str, torch.Tensor] = {},
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """
        Args:
            image: [[[Image, Image], [Imgage, Image]], [[Image, Image], [Image, Image]], ...], n_batch -> n_view -> n_frame
        """

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

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)
        assert timesteps is not None, f'Wrong! timesteps cannot be None!'

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        if self.transformer.config.in_channels != 16:
            latent_channels = self.transformer.config.in_channels // 2
        else:
            latent_channels = self.transformer.config.in_channels

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

            # Also pad the actions if needed
            if (actions := controls_or_guidances.get('actions', None)) is not None:
                actions = torch.cat([
                    actions,
                    torch.zeros((actions.size(0), additional_frames * self.vae_scale_factor_temporal, actions.size(2)),
                                 dtype=actions.dtype, device=device),
                ], dim=1)
                controls_or_guidances['actions'] = actions

        # ! Handle the depth controls
        if (depths := controls_or_guidances.get('depths', None)) is not None:

            if depths.ndim == 5 and depths.size(1) == latent_channels * 2:  # [B, C, F, H, W]

                depth_latent_dist = DiagonalGaussianDistribution(depths)
                depth_latents = depth_latent_dist.sample()
                if not self.vae.config.invert_scale_latents:
                    depth_latents = self.vae_scaling_factor_image * depth_latents
                else:
                    # This is awkward but required because the CogVideoX team forgot to multiply the
                    # scaling factor during training :)
                    depth_latents = 1 / self.vae_scaling_factor_image * depth_latents
                depth_latents = depth_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

                depth_input = torch.cat([depth_latents, depth_latents], dim=2)
                controls_or_guidances['depths'] = depth_input

        if (labels := controls_or_guidances.get('labels', None)) is not None:

            if labels.ndim == 5 and labels.size(1) == latent_channels * 2:

                label_latent_dist = DiagonalGaussianDistribution(labels)
                label_latents = label_latent_dist.sample()
                if not self.vae.config.invert_scale_latents:
                    label_latents = self.vae_scaling_factor_image * label_latents
                else:
                    # This is awkward but required because the CogVideoX team forgot to multiply the
                    # scaling factor during training :)
                    label_latents = 1 / self.vae_scaling_factor_image * label_latents
                label_latents = label_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

                label_input = torch.cat([label_latents, label_latents], dim=2)
                controls_or_guidances['labels'] = label_input

        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )

        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            num_views,
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
                    controls_or_guidances=controls_or_guidances,
                    return_dict=False,
                    num_views=num_views,
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
        latents = rearrange(latents, 'b (v f) c h w -> (b v) f c h w', v=num_views, f=latent_frames)
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
