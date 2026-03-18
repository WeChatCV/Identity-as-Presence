import inspect
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from .models import AutoencoderKLWan, AutoTokenizer, WanT5EncoderModel, WanTransformer3DModel
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from .models.mmaudio.features_utils import FeaturesUtils
from .models.resampler import Resampler
from .utils.utils import save_video

import random
import torchaudio
from torchaudio import transforms as T
import torchvision.utils as vutils

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class IDAVPipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        video (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    videos: torch.Tensor
    audios: torch.Tensor


class IDAVPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "video_latent_model_input", 
        "audio_latent_model_input",
        "prompt_embeds",
        "video_negative_prompt_embeds",
        "audio_negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        audio_vae: FeaturesUtils,
        transformer,
        video_scheduler: FlowUniPCMultistepScheduler,
        audio_scheduler: FlowUniPCMultistepScheduler
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, audio_vae=audio_vae, transformer=transformer, 
            video_scheduler=video_scheduler, audio_scheduler=audio_scheduler
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        video_negative_prompt: Optional[Union[str, List[str]]] = None,
        audio_negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        video_negative_prompt_embeds: Optional[torch.Tensor] = None,
        audio_negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `video_negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            video_negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, video_negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and video_negative_prompt_embeds is None:
            video_negative_prompt = video_negative_prompt or ""
            video_negative_prompt = batch_size * [video_negative_prompt] if isinstance(video_negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(video_negative_prompt):
                raise TypeError(
                    f"`video_negative_prompt` should be the same type to `prompt`, but got {type(video_negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(video_negative_prompt):
                raise ValueError(
                    f"`video_negative_prompt`: {video_negative_prompt} has batch size {len(video_negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `video_negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            video_negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=video_negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and audio_negative_prompt_embeds is None:
            audio_negative_prompt = audio_negative_prompt or ""
            audio_negative_prompt = batch_size * [audio_negative_prompt] if isinstance(audio_negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(audio_negative_prompt):
                raise TypeError(
                    f"`audio_negative_prompt` should be the same type to `prompt`, but got {type(audio_negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(audio_negative_prompt):
                raise ValueError(
                    f"`audio_negative_prompt`: {audio_negative_prompt} has batch size {len(audio_negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `audio_negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            audio_negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=audio_negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, video_negative_prompt_embeds, audio_negative_prompt_embeds

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.video_scheduler, "init_noise_sigma"):
            latents = latents * self.video_scheduler.init_noise_sigma
        return latents

    def prepare_latents_audio(
        self, batch_size, num_channels_latents, num_frames, fps, sample_rate, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            int((num_frames / fps * sample_rate) // 512),
            num_channels_latents
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.audio_scheduler, "init_noise_sigma"):
            latents = latents * self.audio_scheduler.init_noise_sigma
        return latents

    def encode_latents(self, id_image_paths, batch_size, height, width, dtype, device, generator, latents=None):
        if latents is None:
            id_image_paths = [id_image_paths] if isinstance(id_image_paths, str) else id_image_paths    

            id_transforms = transforms.Compose(
                [
                    transforms.Resize(min(height, width)),
                    transforms.CenterCrop((height, width)),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ]
            )

            latents = []

            for img_path in id_image_paths:
                id_image_pil = Image.open(img_path).convert("RGB")
                id_image = np.array(id_image_pil)

                id_image = np.repeat(id_image[np.newaxis, ...], batch_size, axis=0)
                id_image = torch.from_numpy(id_image).permute(0, 3, 1, 2).contiguous()
                id_image = id_image / 255.0
                id_image = id_transforms(id_image)

                id_image = id_image.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device=device, dtype=dtype)
                id_image = self.vae.encode(id_image).latent_dist.mean

                latents.append(id_image)

            latents = torch.cat(latents, dim=2)

        else:
            latents = latents.to(device)
        
        return latents

    def prepare_latents_audio_prompt(
        self, prompt_wav_path, device, sample_rate=16000, max_audio_length=5.0, enable_random_crop=False, enable_pad=False
    ):
        if prompt_wav_path is None:
            audio_tensor = torch.zeros(1, int(max_audio_length * sample_rate))
        else:
            if max_audio_length > 10.:
                max_audio_length = 10.0
            audio_tensor, original_sample_rate = torchaudio.load(prompt_wav_path)
            if original_sample_rate != sample_rate:
                resample_tf = T.Resample(original_sample_rate, sample_rate)
                audio_tensor = resample_tf(audio_tensor)

            channels, samples = audio_tensor.shape
            max_samples = int(max_audio_length * sample_rate)

            if samples > max_samples:
                if enable_random_crop:
                    start_idx = random.randint(0, samples - max_samples)
                    audio_tensor = audio_tensor[:, start_idx:start_idx + max_samples]
                else:
                    audio_tensor = audio_tensor[:, :max_samples]
            else:
                if enable_pad:
                    audio_tensor = torch.nn.functional.pad(audio_tensor, (0, max_samples - samples))
                else:
                    pass

        audio_latents = self.audio_vae.encode_audio(audio_tensor).sample().to(device)
        audio_latents = audio_latents.permute(0, 2, 1).contiguous()

        if audio_latents.shape[0] != 1:
            audio_latents = audio_latents.mean(dim=0, keepdim=True)
        return audio_latents
        
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        frames = frames.cpu().float().numpy()
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.video_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.video_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        video_negative_prompt,
        audio_negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        video_negative_prompt_embeds=None,
        audio_negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if video_negative_prompt is not None and video_negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `video_negative_prompt`: {video_negative_prompt} and `video_negative_prompt_embeds`:"
                f" {video_negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if video_negative_prompt is not None and video_negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `video_negative_prompt`: {video_negative_prompt} and `video_negative_prompt_embeds`:"
                f" {video_negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if audio_negative_prompt is not None and audio_negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `audio_negative_prompt`: {audio_negative_prompt} and `audio_negative_prompt_embeds`:"
                f" {audio_negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and video_negative_prompt_embeds is not None:
            if prompt_embeds.shape != video_negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `video_negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `video_negative_prompt_embeds`"
                    f" {video_negative_prompt_embeds.shape}."
                )

    @property
    def video_guidance_scale(self):
        return self._video_guidance_scale

    @property
    def audio_guidance_scale(self):
        return self._audio_guidance_scale

    @property
    def id_guidance_scale(self):
        return self._id_guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        video_negative_prompt: Optional[Union[str, List[str]]] = None,
        audio_negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        fps=24,
        sample_rate=16000,
        slg_layer=-1,
        num_inference_steps: int = 50,
        video_timesteps: Optional[List[int]] = None,
        audio_timesteps: Optional[List[int]] = None,
        audio_guidance_scale: float = 3,
        video_guidance_scale: float = 4,
        id_guidance_scale: float = 1,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        video_latents: Optional[torch.FloatTensor] = None,
        audio_latents: Optional[torch.FloatTensor] = None,
        id_images: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        video_negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        audio_negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        id_image_paths: Optional[str] = None,
        audio_wav_paths: Optional[str] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["video_latent_model_input", "audio_latent_model_input"],
        max_sequence_length: int = 512,
        comfyui_progressbar: bool = False,
        shift: int = 5,
    ) -> Union[IDAVPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.
        Args:

        Examples:

        Returns:

        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        self.check_inputs(
            prompt,
            height,
            width,
            video_negative_prompt,
            audio_negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            video_negative_prompt_embeds,
            audio_negative_prompt_embeds,
        )
        self._video_guidance_scale = video_guidance_scale
        self._audio_guidance_scale = audio_guidance_scale
        self._id_guidance_scale = id_guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        do_classifier_free_guidance = video_guidance_scale > 1.0

        prompt_embeds, video_negative_prompt_embeds, audio_negative_prompt_embeds = self.encode_prompt(
            prompt,
            video_negative_prompt,
            audio_negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            video_negative_prompt_embeds=video_negative_prompt_embeds,
            audio_negative_prompt_embeds=audio_negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        self.video_scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        video_timesteps = self.video_scheduler.timesteps

        self.audio_scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        audio_timesteps = self.audio_scheduler.timesteps

        self._num_timesteps = len(video_timesteps)
        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_inference_steps + 1)

        latent_channels = self.transformer.video_model.in_dim
        video_latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            video_latents,
        )
        audio_latent_channels = self.transformer.audio_model.in_dim
        audio_latents = self.prepare_latents_audio(
            batch_size * num_videos_per_prompt, 
            audio_latent_channels, 
            num_frames, 
            fps, 
            sample_rate, 
            weight_dtype,
            device,
            generator,
            audio_latents,
        )

        if id_image_paths is not None or id_images is not None:
            id_images = self.encode_latents(
                id_image_paths,
                batch_size * num_videos_per_prompt,
                512,
                512,
                weight_dtype,
                device,
                generator,
                id_images
            )

        if audio_wav_paths is not None:
            audio_latents_prompt = []
            
            for aidx, audio_wav_path in enumerate(audio_wav_paths):
                
                audio_prompt_latents = self.prepare_latents_audio_prompt(
                    audio_wav_path, 
                    device, 
                    max_audio_length=6.0 if len(audio_wav_paths) > 1 else float(audio_latents.size()[1]/31.25),
                    enable_pad=True if len(audio_wav_paths) > 1 else False
                )
                audio_latents_prompt.append(audio_prompt_latents)
            
            audio_latents_prompt = torch.cat(audio_latents_prompt, dim=0)
        else:
            audio_latents_prompt = None

        if comfyui_progressbar:
            pbar.update(1)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (self.vae.latent_channels, (num_frames - 1) // self.vae.temporal_compression_ratio + 1, width // self.vae.spatial_compression_ratio, height // self.vae.spatial_compression_ratio)
        video_seq_len = math.ceil((target_shape[2] * target_shape[3]) / (self.transformer.video_model.patch_size[1] * self.transformer.video_model.patch_size[2]) * target_shape[1]) 
        audio_seq_len = int((num_frames / fps * sample_rate) // 512)
        if id_image_paths is not None:
            id_image_paths = [id_image_paths] if isinstance(id_image_paths, str) else id_image_paths    
            id_seq_len = math.ceil((id_images.shape[3] * id_images.shape[4]) / (self.transformer.video_model.patch_size[1] * self.transformer.video_model.patch_size[2]) * id_images.shape[2])
        else:
            id_seq_len = None

        if audio_wav_paths is not None:
            audio_prompt_seq_len = int(audio_latents_prompt.size()[1])
        else:
            audio_prompt_seq_len = None
        
        num_warmup_steps = max(len(video_timesteps) - num_inference_steps * self.video_scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps
        with self.progress_bar(total=num_inference_steps) as progress_bar:

            video_latent_model_input = video_latents
            audio_latent_model_input = audio_latents

            for i, (t_v, t_a) in enumerate(zip(video_timesteps, audio_timesteps)):
                self.transformer.current_steps = i

                if self.interrupt:
                    continue

                timestep = t_v.expand(video_latent_model_input.shape[0])

                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                    video_noise_pred, audio_noise_pred = self.transformer(
                        vid=video_latent_model_input,
                        audio=audio_latent_model_input,
                        vid_context=prompt_embeds,
                        audio_context=prompt_embeds,
                        vid_seq_len=video_seq_len,
                        audio_seq_len=audio_seq_len,
                        first_frame_is_clean=False,
                        slg_layer=-1,
                        t=timestep,
                        id_images=id_images,
                        id_seq_len=id_seq_len,
                        spkid=audio_latents_prompt if audio_wav_paths is not None else None,
                        spkid_seq_len=audio_prompt_seq_len if audio_wav_paths is not None else None
                    )

                    if do_classifier_free_guidance:
                        video_noise_pred_uncond, audio_noise_pred_uncond = self.transformer(
                            vid=video_latent_model_input,
                            audio=audio_latent_model_input,
                            vid_context=video_negative_prompt_embeds,
                            audio_context=audio_negative_prompt_embeds,
                            vid_seq_len=video_seq_len,
                            audio_seq_len=audio_seq_len,
                            first_frame_is_clean=False,
                            slg_layer=slg_layer,
                            t=timestep,
                            id_images=id_images,
                            id_seq_len=id_seq_len,
                            spkid=None,
                            spkid_seq_len=None
                        )

                if do_classifier_free_guidance:
                    video_noise_pred = video_noise_pred_uncond + self.video_guidance_scale * (video_noise_pred - video_noise_pred_uncond)
                    audio_noise_pred = audio_noise_pred_uncond + self.audio_guidance_scale * (audio_noise_pred - audio_noise_pred_uncond)

                video_latent_model_input = self.video_scheduler.step(video_noise_pred, t_v, video_latent_model_input, **extra_step_kwargs, return_dict=False)[0]
                audio_latent_model_input = self.audio_scheduler.step(audio_noise_pred, t_a, audio_latent_model_input, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t_v, callback_kwargs)

                    video_latent_model_input = callback_outputs.pop("video_latent_model_input", video_latent_model_input)
                    audio_latent_model_input = callback_outputs.pop("audio_latent_model_input", audio_latent_model_input)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    video_negative_prompt_embeds = callback_outputs.pop("video_negative_prompt_embeds", video_negative_prompt_embeds)
                    audio_negative_prompt_embeds = callback_outputs.pop("audio_negative_prompt_embeds", audio_negative_prompt_embeds)

                if i == self.num_timesteps - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.video_scheduler.order == 0):
                    progress_bar.update()
                if comfyui_progressbar:
                    pbar.update(1)

        video = self.decode_latents(video_latent_model_input)
        audio_latents_for_vae = audio_latent_model_input.transpose(1, 2)
        audio = self.audio_vae.wrapped_decode(audio_latents_for_vae)
        audio = audio.cpu().float().numpy()

        self.maybe_free_model_hooks()

        if not return_dict:
            video = torch.from_numpy(video)
            audio = torch.from_numpy(audio)

        return IDAVPipelineOutput(videos=video, audios=audio)
