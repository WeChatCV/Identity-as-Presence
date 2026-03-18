import os
import re
import torch
import torch.nn as nn
from safetensors.torch import load_file
from .iap_base_model import WanLayerNorm, WanRMSNorm, rope_apply, flash_attention, WanTransformer3DModel

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version, logging

from ..dist import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    usp_attn_forward, xFuserLongContextAttention)

from typing import Any, Dict, Optional, Union


def load_sharded_safetensors(folder_path, device="cpu"):

    from safetensors.torch import load_file, safe_open
    from collections import OrderedDict
    
    """
    Load all sharded .safetensors files from a specified folder and merge them into a single state_dict
    
    Args:
        folder_path (str): Path to the folder containing sharded .safetensors files
        device (str): Device to load the tensors to, e.g., "cpu", "cuda", "cuda:0"
    
    Returns:
        OrderedDict: Merged state_dict
    """
    # Get all files matching the sharding naming pattern
    shard_files = []
    for file in os.listdir(folder_path):
        if file.startswith("diffusion_pytorch_model-") and file.endswith(".safetensors"):
            shard_files.append(file)
    
    if not shard_files:
        raise FileNotFoundError(f"No diffusion_pytorch_model-*.safetensors files found in {folder_path}")

    # Sort files by shard number: ensure 00001, 00002, 00003 order is correct
    shard_files = sorted(
        shard_files,
        key=lambda x: int(re.search(r"-(\d+)-of-", x).group(1))
    )
    print(f"Found {len(shard_files)} shard files:")
    for f in shard_files:
        print(f"  - {f}")

    state_dict = OrderedDict()

    # Load shards one by one
    for shard_file in shard_files:
        shard_path = os.path.join(folder_path, shard_file)
        print(f"Loading: {shard_path}")
        
        # Load single shard
        shard = load_file(shard_path, device=device)
        
        # Merge into total state_dict
        for key, value in shard.items():
            if key in state_dict:
                print(f"Warning: Key '{key}' appears in multiple shards, will be overwritten")
            state_dict[key] = value

    return state_dict
    

def load_fusion_checkpoint_id(model, checkpoint_path, from_meta=False):
    if checkpoint_path and os.path.exists(checkpoint_path):
        if checkpoint_path.endswith(".safetensors"): 
            df = load_file(checkpoint_path, device="cpu")
        elif checkpoint_path.endswith(".pt"):
            try:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                df = df['module'] if 'module' in df else df
            except Exception as e:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                df = df['app']['model']
        elif os.path.isdir(checkpoint_path):
            try:
                df = load_sharded_safetensors(checkpoint_path)
            except:
                raise RuntimeError("We only support .safetensors and .pt checkpoints in the provided folder")
        else: 
            raise RuntimeError("We only support .safetensors and .pt checkpoints")

        df_new = {}
        for k, v in df.items():
            if 'modulation.modulation' in k:
                df_new[k.replace('modulation.modulation', 'modulation', 1)] = v
            else:
                df_new[k] = v

        for k, v in model.state_dict().items():
            if k not in df_new:
                if '_prime' in k and '_img' not in k:
                    print(f"Warning: Key '{k}' not in model state_dict, will be initialized w/o key: _prime")
                    df_new[k] = df_new[k.replace('_prime', '', 1)].clone()
                else:
                    print(f"Warning: Key '{k}' not in model state_dict, maintain default initialization")
        
        for k in list(df_new.keys()):
            if k not in model.state_dict():
                print(f"Warning: Key '{k}' not in model state_dict, will be removed")
                del df_new[k]
                
        m, u = model.load_state_dict(df_new, strict=False, assign=from_meta)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

        del df
        import gc
        gc.collect()
        print(f"Successfully loaded fusion checkpoint from {checkpoint_path}")
    else: 
        raise RuntimeError(f"{checkpoint_path} does not exists'")


class FusionModelIDAV(ModelMixin, ConfigMixin, FromOriginalModelMixin):

    def __init__(self, video_config=None, audio_config=None):
        super().__init__()

        self.update_prime = True
        self.enable_prime_cross_attn = False
        self.share_ref_emb = True
        
        has_video = True 
        has_audio = True
        if video_config is not None:
            self.video_model = WanTransformer3DModel(**video_config)
        else:
            has_video = False
            self.video_model = None
            print("Warning: No video model is provided!")
        
        if audio_config is not None:
            self.audio_model = WanTransformer3DModel(**audio_config)
        else:
            has_audio = False
            self.audio_model = None
            print("Warning: No audio model is provided!")

        if has_video and has_audio:
            assert len(self.video_model.blocks) == len(self.audio_model.blocks)
            self.num_blocks = len(self.video_model.blocks)
            self.inject_cross_attention_kv_projections()

        self.init_weights()

        self.gradient_checkpointing = False
        self.sp_world_size = 1
        self.sp_world_rank = 0

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        self.video_model.enable_gradient_checkpointing()
        self.audio_model.enable_gradient_checkpointing()

    def enable_multi_gpus_inference(self):
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather
        self.video_model.enable_multi_gpus_inference()
        self.audio_model.enable_multi_gpus_inference()

    def inject_cross_attention_kv_projections(self):
        for vid_block in self.video_model.blocks:
            vid_block.cross_attn.k_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.v_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(vid_block.dim, elementwise_affine=True)
            vid_block.cross_attn.norm_k_fusion = WanRMSNorm(vid_block.dim, eps=1e-6) if vid_block.qk_norm else nn.Identity()

        for audio_block in self.audio_model.blocks:
            audio_block.cross_attn.k_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.v_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(audio_block.dim, elementwise_affine=True)
            audio_block.cross_attn.norm_k_fusion = WanRMSNorm(audio_block.dim, eps=1e-6) if audio_block.qk_norm else nn.Identity()


    def merge_kwargs(self, vid_kwargs, audio_kwargs):
        """
        keys in each kwarg:
        e
        seq_lens
        grid_sizes
        freqs
        context
        context_lens
        """
        merged_kwargs = {}
        for key in vid_kwargs:
            merged_kwargs[f"vid_{key}"] = vid_kwargs[key]
        for key in audio_kwargs:
            merged_kwargs[f"audio_{key}"] = audio_kwargs[key]
        return merged_kwargs


    def single_fusion_cross_attention_forward(self,
                                            cross_attn_block,
                                            src_seq,
                                            src_grid_sizes,
                                            src_freqs,
                                            target_seq,
                                            target_seq_lens,
                                            target_grid_sizes,
                                            target_freqs,
                                            context,
                                            context_lens
                                            ):
        b, n, d = src_seq.size(0), cross_attn_block.num_heads, cross_attn_block.head_dim
        if hasattr(cross_attn_block, "k_img"):
            # means i2v block
            q, k, v, k_img, v_img = cross_attn_block.qkv_fn(src_seq, context)
        else:
            # means t2v block
            q, k, v = cross_attn_block.qkv_fn(src_seq, context)
            k_img = v_img = None

        x = flash_attention(q, k, v, k_lens=context_lens)

        if k_img is not None:
            img_x = flash_attention(q, k_img, v_img, k_lens=None)
            x = x + img_x

        is_vid = src_grid_sizes.shape[1] > 1
        # compute target attention
        target_seq = cross_attn_block.pre_attn_norm_fusion(target_seq)
        k_target = cross_attn_block.norm_k_fusion(cross_attn_block.k_fusion(target_seq)).view(b, -1, n, d)
        v_target = cross_attn_block.v_fusion(target_seq).view(b, -1, n, d)
        
        q = rope_apply(q, src_grid_sizes, src_freqs)
        k_target = rope_apply(k_target, target_grid_sizes, target_freqs)
        
        target_x = flash_attention(q, k_target, v_target, k_lens=target_seq_lens)
        
        x = x + target_x

        x = x.flatten(2)

        x = cross_attn_block.o(x)
        return x

    def single_fusion_cross_attention_ffn_forward(self,
                                            attn_block,
                                            src_seq,
                                            src_grid_sizes,
                                            src_freqs,
                                            target_seq,
                                            target_seq_lens,
                                            target_grid_sizes,
                                            target_freqs,
                                            context,
                                            context_lens,
                                            src_e):
        
        src_seq = src_seq + self.single_fusion_cross_attention_forward(attn_block.cross_attn,
                                                                       attn_block.norm3(src_seq),
                                                                       src_grid_sizes=src_grid_sizes,
                                                                       src_freqs=src_freqs,
                                                                       target_seq=target_seq,
                                                                       target_seq_lens=target_seq_lens,
                                                                       target_grid_sizes=target_grid_sizes,
                                                                       target_freqs=target_freqs,
                                                                       context=context,
                                                                       context_lens=context_lens
                                                                       )
        y = attn_block.ffn(attn_block.norm2(src_seq) * (1 + src_e[4].squeeze(2)) + src_e[3].squeeze(2))

        src_seq = src_seq + y * src_e[5].squeeze(2)
        return src_seq
        
    def single_fusion_block_forward(self,
                                    vid_block,
                                    audio_block,
                                    vid,
                                    audio,
                                    vid_e,
                                    vid_seq_lens,
                                    vid_grid_sizes,
                                    vid_freqs,
                                    vid_context,
                                    vid_context_lens,
                                    vid_x_prime,
                                    vid_seq_lens_prime,
                                    vid_grid_sizes_prime,
                                    audio_e,
                                    audio_seq_lens,
                                    audio_grid_sizes,
                                    audio_freqs,
                                    audio_context,
                                    audio_context_lens,
                                    audio_x_prime,
                                    audio_seq_lens_prime,
                                    audio_grid_sizes_prime,
                                    vid_e_prime=None,
                                    audio_e_prime=None
                                    ):
        #######################
        # audio model start
        #######################

        # audio modulation
        dtype = audio.dtype
        assert len(audio_e.shape) == 4 and audio_e.size(2) == 6 and audio_e.shape[1] == audio.shape[1], f"{audio_e.shape}, {audio.shape}"
        
        if audio_x_prime is not None and audio_e_prime is None:
            audio_e_prime = [item[:audio_x_prime.shape[1],:] for item in audio_e]
            audio_e_prime = torch.stack(audio_e_prime)

        audio_e = (audio_block.modulation.unsqueeze(0) + audio_e).chunk(6, dim=2)
        audio_e = [audio_e.squeeze(2) for audio_e in audio_e]

        if audio_e_prime is not None:
            audio_e_prime = (audio_block.modulation.unsqueeze(0) + audio_e_prime).chunk(6, dim=2)
            audio_e_prime = [e_prime.squeeze(2) for e_prime in audio_e_prime]

        tmp_audio = audio_block.norm1(audio) * (1 + audio_e[1].squeeze(2)) + audio_e[0].squeeze(2)
        if audio_x_prime is not None:
            tmp_audio_x_prime = audio_block.norm1(audio_x_prime) * (1 + audio_e_prime[1].squeeze(2)) + audio_e_prime[0].squeeze(2)
        else:
            tmp_audio_x_prime = None

        # audio self-attention
        audio_y_results = audio_block.self_attn(
            tmp_audio, audio_seq_lens, audio_grid_sizes,
            audio_freqs, dtype=dtype, t=None, 
            x_prime=tmp_audio_x_prime, seq_lens_prime=audio_seq_lens_prime, grid_sizes_prime=audio_grid_sizes_prime)
        
        if len(audio_y_results) == 2:
            audio_y, audio_y_prime = audio_y_results
        else:
            audio_y = audio_y_results

        audio = audio + audio_y * audio_e[2].squeeze(2)

        if audio_x_prime is not None:
            audio_x_prime = audio_x_prime + audio_y_prime * audio_e_prime[2].squeeze(2)

        #######################
        # video model start
        #######################
        
        # video modulation
        dtype = vid.dtype
        assert len(vid_e.shape) == 4 and vid_e.size(2) == 6 and vid_e.shape[1] == vid.shape[1], f"{vid_e.shape}, {vid.shape}"

        if vid_x_prime is not None and vid_e_prime is None:
            vid_e_prime = [item[:vid_x_prime.shape[1],:] for item in vid_e]
            vid_e_prime = torch.stack(vid_e_prime)

        vid_e = (vid_block.modulation.unsqueeze(0) + vid_e).chunk(6, dim=2)
        vid_e = [vid_e.squeeze(2) for vid_e in vid_e]

        if vid_e_prime is not None:
            vid_e_prime = (vid_block.modulation_prime.unsqueeze(0) + vid_e_prime).chunk(6, dim=2)
            vid_e_prime = [e_prime.squeeze(2) for e_prime in vid_e_prime]

        # video self-attention
        # self-attention
        tmp_vid = vid_block.norm1(vid) * (1 + vid_e[1].squeeze(2)) + vid_e[0].squeeze(2)
        if vid_x_prime is not None:
            tmp_vid_x_prime = vid_block.norm1_prime(vid_x_prime) * (1 + vid_e_prime[1].squeeze(2)) + vid_e_prime[0].squeeze(2)
        else:
            tmp_vid_x_prime = None

        y_results = vid_block.self_attn(
            tmp_vid, vid_seq_lens, vid_grid_sizes,
            vid_freqs, dtype=dtype, t=None,
            x_prime=tmp_vid_x_prime, seq_lens_prime=vid_seq_lens_prime, grid_sizes_prime=vid_grid_sizes_prime)

        if len(y_results) == 2:
            vid_y, vid_y_prime = y_results
        else:
            vid_y = y_results
            
        vid = vid + vid_y * vid_e[2].squeeze(2)

        if vid_x_prime is not None:
            vid_x_prime = vid_x_prime + vid_y_prime * vid_e_prime[2].squeeze(2)

        #######################
        # video model end
        #######################

        og_audio = audio

        # audio cross-attention
        audio = self.single_fusion_cross_attention_ffn_forward(
            audio_block,
            audio,
            audio_grid_sizes,
            audio_freqs,
            vid,
            vid_seq_lens,
            vid_grid_sizes,
            vid_freqs,
            audio_context,
            audio_context_lens,
            audio_e
        )

        assert not torch.equal(og_audio, audio), "Audio should be changed after cross-attention!"

        # video cross-attention
        vid = self.single_fusion_cross_attention_ffn_forward(
            vid_block,
            vid,
            vid_grid_sizes,
            vid_freqs,
            og_audio,
            audio_seq_lens,
            audio_grid_sizes,
            audio_freqs,
            vid_context,
            vid_context_lens,
            vid_e
        )
        
        #######################
        # video model start
        #######################
        if self.enable_prime_cross_attn:
            vid_x_prime = vid_x_prime + vid_block.cross_attn_prime(vid_block.norm3_prime(vid_x_prime), vid_context, vid_context_lens, dtype, t=None)

        if vid_x_prime is not None:
            tmp_vid_x_prime = vid_block.norm2_prime(vid_x_prime) * (1 + vid_e_prime[4].squeeze(2)) + vid_e_prime[3].squeeze(2)
            tmp_vid_x_prime = tmp_vid_x_prime.to(dtype)

            vid_y_prime = vid_block.ffn_prime(tmp_vid_x_prime)
            vid_x_prime = vid_x_prime + vid_y_prime * vid_e_prime[5].squeeze(2)

        #######################
        # video model end
        #######################

        #######################
        # audio model end
        #######################
        if self.enable_prime_cross_attn:
            audio_x_prime = audio_x_prime + audio_block.cross_attn(audio_block.norm3(audio_x_prime), audio_context, audio_context_lens, dtype, t=None)

        if audio_x_prime is not None:
            tmp_audio_x_prime = audio_block.norm2(audio_x_prime) * (1 + audio_e_prime[4].squeeze(2)) + audio_e_prime[3].squeeze(2)
            tmp_audio_x_prime = tmp_audio_x_prime.to(dtype)

            audio_y_prime = audio_block.ffn(tmp_audio_x_prime)
            audio_x_prime = audio_x_prime + audio_y_prime * audio_e_prime[5].squeeze(2)

        #######################
        # audio model end
        #######################
        
        return vid, audio, vid_x_prime, audio_x_prime

    def forward(
        self,
        vid,
        audio,
        t,
        vid_context,
        audio_context,
        vid_seq_len,
        audio_seq_len,
        clip_fea=None,
        clip_fea_audio=None,
        y=None,
        first_frame_is_clean=False,
        slg_layer=False,
        id_images=None,
        id_seq_len=None,
        spkid=None, 
        spkid_seq_len=None
    ):  
        # assert clip_fea is None  # for face embeds
        assert y is None

        if vid is None or all([x is None for x in vid]):
            # assert clip_fea is None  # for face embeds
            assert vid_context is None
            assert vid_seq_len is None
            assert self.audio_model is not None

            return None, self.audio_model(x=audio, t=t, context=audio_context, seq_len=audio_seq_len, clip_fea=clip_fea_audio, y=None)
        
        if audio is None or all([x is None for x in audio]):
            assert clip_fea_audio is None
            assert audio_context is None
            assert audio_seq_len is None
            assert self.video_model is not None

            return self.video_model(x=vid, t=t, context=vid_context, seq_len=vid_seq_len, clip_fea=clip_fea, y=y, first_frame_is_clean=first_frame_is_clean), None
        
        vid, vid_e, vid_kwargs = self.video_model.prepare_transformer_block_kwargs(
            x=vid, t=t, context=vid_context, seq_len=vid_seq_len, clip_fea=clip_fea, y=y, first_frame_is_clean=first_frame_is_clean, id_images=id_images, id_seq_len=id_seq_len
        )

        audio, audio_e, audio_kwargs = self.audio_model.prepare_transformer_block_kwargs(
            x=audio, t=t, context=audio_context, seq_len=audio_seq_len, clip_fea=clip_fea_audio, y=None, first_frame_is_clean=False, spkid=spkid, spkid_seq_len=spkid_seq_len
        )

        kwargs = self.merge_kwargs(vid_kwargs, audio_kwargs)
        vid_x_prime = kwargs['vid_x_prime']
        audio_x_prime = kwargs['audio_x_prime']

        for i in range(self.num_blocks):
            """
            1 fusion block refers to 1 audio block with 1 video block.
            """
            if slg_layer > 0 and i == slg_layer:
                continue
            vid_block = self.video_model.blocks[i]
            audio_block = self.audio_model.blocks[i]

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                vid, audio, vid_x_prime, audio_x_prime = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.single_fusion_block_forward),
                    vid_block,
                    audio_block,
                    vid,
                    audio,
                    kwargs['vid_e'],
                    kwargs['vid_seq_lens'],
                    kwargs['vid_grid_sizes'],
                    kwargs['vid_freqs'],
                    kwargs['vid_context'],
                    kwargs['vid_context_lens'],
                    vid_x_prime,
                    kwargs['vid_seq_lens_prime'],
                    kwargs['vid_grid_sizes_prime'],
                    kwargs['audio_e'],
                    kwargs['audio_seq_lens'],
                    kwargs['audio_grid_sizes'],
                    kwargs['audio_freqs'],
                    kwargs['audio_context'],
                    kwargs['audio_context_lens'],
                    audio_x_prime,
                    kwargs['audio_seq_lens_prime'],
                    kwargs['audio_grid_sizes_prime'],
                    kwargs['vid_e_prime'] if 'vid_e_prime' in kwargs else None,
                    kwargs['audio_e_prime'] if 'audio_e_prime' in kwargs else None,
                    **ckpt_kwargs
                )
            else:
                vid, audio, vid_x_prime, audio_x_prime = self.single_fusion_block_forward(
                    vid_block=vid_block,
                    audio_block=audio_block,
                    vid=vid,
                    audio=audio,
                    vid_e=kwargs['vid_e'],
                    vid_seq_lens=kwargs['vid_seq_lens'],
                    vid_grid_sizes=kwargs['vid_grid_sizes'],
                    vid_freqs=kwargs['vid_freqs'],
                    vid_context=kwargs['vid_context'],
                    vid_context_lens=kwargs['vid_context_lens'],
                    vid_x_prime=vid_x_prime,
                    vid_seq_lens_prime=kwargs['vid_seq_lens_prime'],
                    vid_grid_sizes_prime=kwargs['vid_grid_sizes_prime'],
                    audio_e=kwargs['audio_e'],
                    audio_seq_lens=kwargs['audio_seq_lens'],
                    audio_grid_sizes=kwargs['audio_grid_sizes'],
                    audio_freqs=kwargs['audio_freqs'],
                    audio_context=kwargs['audio_context'],
                    audio_context_lens=kwargs['audio_context_lens'],
                    audio_x_prime=audio_x_prime,
                    audio_seq_lens_prime=kwargs['audio_seq_lens_prime'],
                    audio_grid_sizes_prime=kwargs['audio_grid_sizes_prime'],
                    vid_e_prime=kwargs['vid_e_prime'] if 'vid_e_prime' in kwargs else None,
                    audio_e_prime=kwargs['audio_e_prime'] if 'audio_e_prime' in kwargs else None,
                )

        vid = self.video_model.post_transformer_block_out(vid, vid_kwargs['grid_sizes'], vid_e)
        audio = self.audio_model.post_transformer_block_out(audio, audio_kwargs['grid_sizes'], audio_e)

        return vid, audio

    def init_weights(self):
        if self.audio_model is not None:
            self.audio_model.init_weights()

        if self.video_model is not None:
            self.video_model.init_weights()

        for name, mod in self.video_model.named_modules():
            if "fusion" in name and isinstance(mod, nn.Linear):
                with torch.no_grad():
                    mod.weight.div_(10.0)

    
    def set_rope_params(self):
        self.video_model.set_rope_params()
        self.audio_model.set_rope_params()