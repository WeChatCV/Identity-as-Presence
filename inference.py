#!/usr/bin/env python
# coding=utf-8
"""
T2V + Audio Inference Script with Multi-GPU Support via torchrun
- Supports: str / .txt / .csv prompts
- Multi-GPU: each GPU handles a subset of prompts
- Output saved to shared output_dir (no filename conflicts)
"""

import os
import sys
import ast
import csv
import random
import torch
import argparse
import re
import time
import datetime
from omegaconf import OmegaConf
from contextlib import nullcontext

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from transformers import AutoTokenizer

from identity_as_presence.models import AutoencoderKLWan3_8, WanT5EncoderModel, FusionModelIDAV
from identity_as_presence.models import load_fusion_checkpoint_id
from identity_as_presence.pipeline import IDAVPipeline
from identity_as_presence.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from identity_as_presence.models.mmaudio.features_utils import FeaturesUtils
from identity_as_presence.utils.utils import save_video
from identity_as_presence.models.resampler import Resampler

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def slugify(s, max_length=50):
    """Convert string to safe filename"""
    s = re.sub(r'[^\w\s-]', '', s).strip().lower()
    s = re.sub(r'[-\s]+', '_', s)
    return s[:max_length]
    

def load_prompts(prompt_input):
    """
    Load one or more prompts from:
      - string
      - .txt file (single prompt)
      - .csv file (multiple prompts, header: text_prompt, optional image_path, image_path2, ...)
    
    Returns: 
        list of tuples: (prompt, image_paths: list, video_paths: list)
        or list of strings (if only text)
    """
    if os.path.isfile(prompt_input) and prompt_input.endswith(".csv"):
        prompts = []
        with open(prompt_input, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            if 'text_prompt' not in reader.fieldnames:
                raise ValueError("CSV must have header: 'text_prompt'")

            image_pattern = re.compile(r"^image_path\d*$", re.IGNORECASE)
            video_pattern = re.compile(r"^video_path\d*$", re.IGNORECASE)
            audio_pattern = re.compile(r"^audio_path\d*$", re.IGNORECASE)

            for row_num, row in enumerate(reader, start=2):
                prompt = row['text_prompt']

                image_paths = []
                for key, value in row.items():
                    if image_pattern.match(key) and value is not None:
                        img_path = value
                        if not os.path.isabs(img_path):
                            img_path = os.path.join(os.path.dirname(prompt_input), img_path)
                        image_paths.append(img_path)

                video_paths = []
                for key, value in row.items():
                    if video_pattern.match(key) and value.strip():
                        vid_path = value
                        if not os.path.isabs(vid_path):
                            vid_path = os.path.join(os.path.dirname(prompt_input), vid_path)
                        video_paths.append(vid_path)

                audio_paths = []
                for key, value in row.items():
                    if audio_pattern.match(key) and value.strip():
                        aud_path = value
                        if not os.path.isabs(aud_path):
                            aud_path = os.path.join(os.path.dirname(prompt_input), aud_path)
                        audio_paths.append(aud_path)

                image_paths = list(dict.fromkeys(image_paths))
                video_paths = list(dict.fromkeys(video_paths))
                audio_paths = list(dict.fromkeys(audio_paths))
        
                prompts.append((prompt, image_paths, audio_paths, video_paths))

        if not prompts:
            raise ValueError("No valid prompts found in CSV.")

        return prompts

    if os.path.isfile(prompt_input) and prompt_input.endswith(".txt"):
        with open(prompt_input, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        if not prompt:
            raise ValueError("Prompt file is empty.")
        return [prompt]

    if prompt_input.strip():
        return [prompt_input.strip()]

    raise ValueError("Prompt input is empty or invalid.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Base model path")
    parser.add_argument("--ckpt", type=str, default=None,help="Checkpoint path (overrides fusion_model_path)")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt string, or path to .txt/.csv")
    parser.add_argument("--audio_vae_config", type=str, required=True, help="Path to audio_vae.json")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Output directory")
    parser.add_argument("--frames", type=int, default=121, help="Number of frames")
    parser.add_argument("--size", type=int, nargs=2, default=[512, 512], help="Height and width")
    parser.add_argument("--seed", type=int, default=103, help="Base random seed")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank (set by torchrun)")

    parser.add_argument("--infer_mode", type=str, default="t2av", help="Infer mode")

    parser.add_argument("--enable_face_parsing", action="store_true", help="Enable face parsing")
    parser.add_argument("--enable_audio_prompt", action="store_true", help="Enable audio prompt")

    return parser.parse_args()


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def setup_distributed():
    """Initialize distributed environment"""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        if world_size > 1:
            timeout = datetime.timedelta(seconds=36000)
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank,
                timeout=timeout
            )
        is_main_process = rank == 0
        print(f"[Rank {rank}/{world_size}] Initialized.")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        is_main_process = True
        print("[Single GPU] No distributed training detected.")

    return rank, world_size, local_rank, is_main_process


def main():
    args = parse_args()

    # Setup distributed
    rank, world_size, local_rank, is_main_process = setup_distributed()
    device = f"cuda:{local_rank}"
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    height, width = args.size

    # Load prompts
    prompts = load_prompts(args.prompt)
    total_prompts = len(prompts)
    print(f"[Rank {rank}] Total prompts loaded: {total_prompts}")

    per_gpu = (total_prompts + world_size - 1) // world_size
    start_idx = rank * per_gpu
    end_idx = min(start_idx + per_gpu, total_prompts)
    prompts_assigned = prompts[start_idx:end_idx]

    if len(prompts_assigned) == 0:
        print(f"[Rank {rank}] No prompts assigned.")
        if world_size > 1:
            torch.cuda.synchronize(device)
            torch.distributed.barrier()
        return

    print(f"[World_size {world_size}, Rank {rank}] Assigned {len(prompts_assigned)} prompts: {start_idx} to {end_idx-1}")

    # Create output dir (only main process needs to do this, but safe with exist_ok)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    config = OmegaConf.load(args.config_path)

    # Build pipeline
    print(f"[Rank {rank}] Loading models...")

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer'))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        torch_dtype=dtype,
    ).eval().to(device)

    vae = AutoencoderKLWan3_8.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).eval().to(device)

    with open(args.audio_vae_config, 'r') as f:
        import json
        audio_cfg = json.load(f)
    audio_encoder = FeaturesUtils(**audio_cfg).eval().to(device)

    ckpt_path = args.ckpt if args.ckpt is not None else config['fusion_model_path']
    
    if os.path.exists(ckpt_path):
        transformer = FusionModelIDAV(config['video_config'], config['audio_config'])
        load_fusion_checkpoint_id(transformer, checkpoint_path=ckpt_path, from_meta=True)
    else:
        raise ValueError
    
    transformer = transformer.eval().to(device, dtype=dtype)

    video_scheduler = FlowUniPCMultistepScheduler(
        **filter_kwargs(FlowUniPCMultistepScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    audio_scheduler = FlowUniPCMultistepScheduler(
        **filter_kwargs(FlowUniPCMultistepScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    pipe = IDAVPipeline(
        vae=vae,
        audio_vae=audio_encoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        video_scheduler=video_scheduler,
        audio_scheduler=audio_scheduler
    ).to(device)

    text_formatter = lambda text: re.sub(r"<AUDCAP>(.*?)<ENDAUDCAP>", r"Audio: \1", text, flags=re.S)

    # Generate
    for idx, val_prompt in enumerate(prompts_assigned):
        global_idx = start_idx + idx
        current_seed = args.seed + global_idx
        generator = torch.Generator(device).manual_seed(current_seed)

        if isinstance(val_prompt, str):
            prompt = val_prompt
            image_path = None
        else:
            prompt, image_path, audio_path, video_path = val_prompt

        prompt = text_formatter(prompt)
        safe_name = slugify(prompt)

        mp4_path = os.path.join(args.output_dir, f"rank{rank}_{global_idx}-{safe_name}.mp4")
        if os.path.exists(mp4_path):
            continue

        print(f"[Rank {rank}] [{idx+1}/{len(prompts_assigned)}] Generating: {prompt}")

        try:
            with torch.no_grad():
                sample = pipe(
                    prompt=prompt,
                    num_frames=args.frames,
                    height=height,
                    width=width,
                    video_guidance_scale=4.0,
                    audio_guidance_scale=3.0,
                    id_guidance_scale=1.0,
                    video_negative_prompt="jitter, bad hands, blur, distortion",
                    audio_negative_prompt="robotic, muffled, echo, distorted",
                    fps=24,
                    sample_rate=16000,
                    slg_layer=11,
                    generator=generator,
                    shift=5.0,
                    id_image_paths=image_path if args.enable_face_parsing else None,
                    audio_wav_paths=audio_path if args.enable_audio_prompt else None,
                    num_inference_steps=50,
                )

            video, audio = sample.videos, sample.audios

            video_np = video.squeeze(0).cpu().float().numpy()
            audio_np = audio.squeeze().cpu().float().numpy()
            mp4_path = os.path.join(args.output_dir, f"rank{rank}_{global_idx}-{safe_name}.mp4")
            save_video(mp4_path, video_np, audio_np, fps=24, sample_rate=16000)

            print(f"[Rank {rank}] Saved: {mp4_path}")

        except Exception as e:
            print(f"[Rank {rank}] Error generating video for prompt '{prompt}': {e}")
            continue

        del sample, video, audio
        torch.cuda.empty_cache()

    # Sync
    if world_size > 1:
        torch.distributed.barrier()

    if is_main_process:
        print(f"\n All done! {total_prompts} prompts processed. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()