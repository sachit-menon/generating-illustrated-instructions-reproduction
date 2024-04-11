from typing import Optional

from dataclasses import dataclass, field
from diffusers.models import AutoencoderKL, UNet2DConditionModel


import torch
from torch import nn


from trainer.models.base_model import BaseModelConfig


from diffusers import AutoencoderKL, UNet2DConditionModel
from trainer.noise_schedulers.scheduling_ddpm_zerosnr import DDPMScheduler

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.training_utils import EMAModel

from diffusers.utils import logging

from diffusers.utils.hub_utils import PushToHubMixin

from diffusers.models.modeling_utils import ModelMixin

from diffusers.configuration_utils import ConfigMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# from hydra.utils import instantiate
from peft import get_peft_model

from layers import PositionalEncodingPermute1D
from einops import rearrange, repeat

from typing import Optional
from omegaconf import II


@dataclass
class LoraConfig:
    _target_: str = "peft.LoraConfig"
    r: int = 8
    lora_alpha: int =32
    target_modules: list = field(default_factory=lambda: ["to_q", "to_v", "query", "value"])
    lora_dropout: float =0.0
    bias: str ="none"


@dataclass
class SDModelConfig(BaseModelConfig):
    _target_: str = "trainer.models.sd_model.SDModel"
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    conditioning_dropout_prob: float = 0.05
    use_ema: bool = True
    concat_all_steps: bool = II("dataset.concat_all_steps")
    positional_encoding_type: Optional[str] = "sinusoidal"
    positional_encoding_length: Optional[int] = None
    image_positional_encoding_type: Optional[str] = None #"sinusoidal"
    image_positional_encoding_length: Optional[int] = None
    broadcast_positional_encoding: bool = True
    sequence_length: Optional[int] = II("dataset.sequence_length") # TODO consider changing interp on next line to this +1? 
    text_sequence_length: Optional[int] = II("dataset.text_sequence_length")
    use_lora: bool = False
    # lora_cfg: Any = LoraConfig()
    zero_snr: bool = True
    # seed: int = 42 # TODO: inherit from higher config
    # lora: LoraConfig = LoraConfig(
    #     )
    
    
class SDModel(ModelMixin, ConfigMixin, PushToHubMixin):
    def __init__(self, cfg: SDModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path, 
            subfolder="scheduler",
            zero_snr=self.cfg.zero_snr)
        
        
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder", 
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        
        self.vae = AutoencoderKL.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="unet"
        )
        
        in_channels = 8 # TODO make part of cfg
        out_channels = self.unet.conv_in.out_channels
        self.unet.register_to_config(in_channels=in_channels)

        with torch.no_grad(): 
            new_conv_in = nn.Conv2d(
                in_channels, out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride, self.unet.conv_in.padding
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight) # copy the pretrained weights, leave the rest as zero
            new_conv_in.bias.copy_(self.unet.conv_in.bias) # EXTREMELY IMPORTANT MODIFICATION FROM INITIAL DIFFUSERS CODE
            self.unet.conv_in = new_conv_in
            
        self.init_pos()
        self.init_image_pos()
        
        
        if self.cfg.use_lora:
            config = LoraConfig(
                    r=8,
                    lora_alpha=32,
                    target_modules=["to_q", "to_v", "query", "value"],
                    lora_dropout=0.0,
                    bias="none",
                )
            # config = self.cfg.lora_cfg
            # LoraConfig(
            #     r=self.cfg.lora.lora_r,
            #     lora_alpha=self.cfg.lora.lora_alpha,
            #     target_modules=self.cfg.lora.UNET_TARGET_MODULES,
            #     lora_dropout=self.cfg.lora.lora_dropout,
            #     bias=self.cfg.lora.lora_bias,
            # )
            self.unet = get_peft_model(self.unet, config)
            self.unet.conv_in.requires_grad_(True)  # NOTE: this makes the whole input conv trainable, not just the new parameters! consider if that's what you really want
            self.unet.print_trainable_parameters()
            print(self.unet)
            
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # use_ema = True
        # if use_ema:
        if self.cfg.use_ema:
            self.ema_unet = EMAModel(self.unet.parameters(), model_cls=UNet2DConditionModel, model_config=self.unet.config)
            
        self.generator = None 
        self.register_to_config(cfg=cfg)

    def init_pos(self):
        self.cfg.positional_encoding_length = self.cfg.text_sequence_length
        if not self.cfg.broadcast_positional_encoding:
            self.cfg.positional_encoding_length *= 77
        elif self.cfg.positional_encoding_type == 'sinusoidal':
            self.unet.pos = PositionalEncodingPermute1D(self.cfg.positional_encoding_length)
        elif self.cfg.positional_encoding_type is None or self.cfg.positional_encoding_type == 'None':
            self.unet.pos = nn.Identity()
        else:
            raise ValueError(f'Unknown positional encoding type {self.cfg.positional_encoding_type}')#torch.Generator(self.unet.device).manual_seed(42) # seed: int = 42 # TODO: inherit from higher config # device=self.unet.device
    
    def init_image_pos(self):
        self.cfg.image_positional_encoding_length = self.cfg.sequence_length
        if self.cfg.image_positional_encoding_type == 'sinusoidal':
            self.unet.image_pos = PositionalEncodingPermute1D(self.cfg.image_positional_encoding_length)
        elif self.cfg.image_positional_encoding_type is None:
            self.unet.image_pos = nn.Identity()
        else:
            raise ValueError(f'Unknown image positional encoding type {self.cfg.image_positional_encoding_type}')
        
    def tokenize_captions(self, captions):
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
            
    def forward(self, batch): # replace with input_ids, edited_pixel_values, original_pixel_values
        batch_size = batch["input_ids"].shape[0]
        condition_image = batch["original_pixel_values"]
        input_ids = batch["input_ids"].to(self.text_encoder.device)
        # We want to learn the denoising process w.r.t the edited images which
        # are conditioned on the original image (which was edited) and the edit instruction.
        # So, first, convert images to latent space.
        edited_images = batch["edited_pixel_values"]#.to(self.cfg.weight_dtype) #TODO check dtype thing
        output_seq_length = edited_images.shape[1]
        # edited_images = edited_images.flatten(0,1)
        edited_images = rearrange(edited_images, 'b s c h w -> (b s) c h w')
        
        latents = self.vae.encode(edited_images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        
        latents = rearrange(latents, '(b s) c h w -> b c (s h) w', s=output_seq_length)
        # latents = latents.unflatten(0,(batch_size,output_seq_length)).transpose(1,2).flatten(2,3) # TODO: change the (batch_size, 3) to (batch_size, output_seq_length)
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        if self.cfg.image_positional_encoding_type is not None:            
            latents = self.apply_image_positional_encoding(noisy_latents, output_seq_length)
        
        if len(input_ids.shape) == 2:
            input_ids = input_ids.unsqueeze(0)

        encoder_hidden_states = self.input_ids_to_text_condition(input_ids)
        if self.cfg.positional_encoding_type is not None:
            encoder_hidden_states = self.apply_step_positional_encoding(encoder_hidden_states)

        # Get the additional image embedding for conditioning.
        # Instead of getting a diagonal Gaussian here, we simply take the mode.
        original_image_embeds = self.vae.encode(condition_image).latent_dist.mode() #.to(self.cfg.weight_dtype)).latent_dist.mode() #TODO check dtype thing

        # Conditioning dropout to support classifier-free guidance during inference. For more details
        # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
        if self.cfg.conditioning_dropout_prob is not None:
            encoder_hidden_states, original_image_embeds = self.apply_conditioning_dropout(encoder_hidden_states, original_image_embeds)

        # original_image_embeds = original_image_embeds.repeat(1,1,2,1)
        # original_image_embeds = original_image_embeds.unsqueeze(2).expand(-1, -1, output_seq_length, -1, -1).reshape(batch_size, 4, 32*output_seq_length, 32)
        original_image_embeds = repeat(original_image_embeds, 'b c h w -> b c (s h) w', s=output_seq_length) # TODO unify with pipeline get_image_latents
        
        # Concatenate the `original_image_embeds` with the `noisy_latents`.
        concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

        target = self.get_loss_target(latents, noise, timesteps)

        # Predict the noise residual and compute loss
        model_pred = self.unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample
        return model_pred, target

    def get_loss_target(self, latents, noise, timesteps):
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        return target

    def apply_conditioning_dropout(self, encoder_hidden_states, original_image_embeds):
        bsz = original_image_embeds.shape[0] # changed from the comment in line 141 from latents, but should be same. TODO check
        random_p = torch.rand(bsz, device=encoder_hidden_states.device, generator=self.generator) # was originally latents.device, TODO check
            # Sample masks for the edit prompts.
        prompt_mask = random_p < 2 * self.cfg.conditioning_dropout_prob
        prompt_mask = prompt_mask.reshape(bsz, 1, 1)
            # Final text conditioning.
        null_conditioning = self.get_null_conditioning()
        encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

            # Sample masks for the original images.
        image_mask_dtype = original_image_embeds.dtype
        image_mask = 1 - (
                (random_p >= self.cfg.conditioning_dropout_prob).to(image_mask_dtype)
                * (random_p < 3 * self.cfg.conditioning_dropout_prob).to(image_mask_dtype)
            )
        image_mask = image_mask.reshape(bsz, 1, 1, 1)
        # Final image conditioning.
        original_image_embeds = image_mask * original_image_embeds
        return encoder_hidden_states,original_image_embeds

    def get_null_conditioning(self):
        null_token = self.tokenize_captions([""]).to(self.text_encoder.device)
        # null_conditioning = self.input_ids_to_text_condition(null_token) # would apply positional encoding twice
        null_conditioning = self.text_encoder(null_token)[0] # TODO fuse with input_ids_to_text_condition
        if not self.cfg.concat_all_steps:
            null_conditioning = repeat(null_conditioning, 'b t l -> b (s t) l', s=self.cfg.text_sequence_length)
        return null_conditioning

    def input_ids_to_text_condition(self, input_ids):
        # Get the text embedding for conditioning.
        if self.cfg.concat_all_steps:
            encoder_hidden_states = self.text_encoder(input_ids)[0] # text padded to 77 tokens; encoder_hidden_states.shape = (bsz, 77, 768)
        else:
            input_ids = rearrange(input_ids, 'b s t->(b s) t')
            encoder_hidden_states = self.text_encoder(input_ids)[0] # text padded to 77 tokens; encoder_hidden_states.shape = (bsz, 77, 768) # TODO check why this doesn't match concatenating the encodings of the three tokens; the ones that don't match are the 769-1535 dims of the feature, for tokens 15-76
            
            # if args.use_positional_encoding: # old way: added before concat which doesn't make sense
            #     encoder_hidden_states = pos(encoder_hidden_states) + encoder_hidden_states
            encoder_hidden_states = rearrange(encoder_hidden_states, '(b s) t d->b (s t) d', s=self.cfg.text_sequence_length)

        return encoder_hidden_states

    def apply_step_positional_encoding(self, encoder_hidden_states):
        positional_encoding = self.unet.pos(encoder_hidden_states)
        if self.cfg.broadcast_positional_encoding:
            positional_encoding = repeat(positional_encoding, 'b s d -> b (s t) d', t=77) # TODO check this
        encoder_hidden_states = positional_encoding + encoder_hidden_states
        return encoder_hidden_states
    
    def apply_image_positional_encoding(self, latents, output_seq_length):
        original_latents_shape = latents.shape
        h = original_latents_shape[2]//output_seq_length
        latents = rearrange(latents, 'b c (s h) w -> b s (c h w)', s=output_seq_length)
        image_pos = self.unet.image_pos(latents)
        latents = latents + image_pos
        latents = rearrange(latents, 'b s (c h w) -> b c (s h) w', s=output_seq_length, c=original_latents_shape[1], h=h, w=original_latents_shape[3]) # confirmed that without the pos addition in between, this reshaping brings it back to the original tensor
        return latents
    
    def instantiate_pipeline(self):
        pass
        

# @dataclass
# class ShowNotTellPipelineConfig:
#     _target_: str = "trainer.models.sd_model.ShowNotTellPipeline.from_pretrained"
#     cfg: SDModelConfig = SDModelConfig()

# class ShowNotTellPipeline(StableDiffusionInstructPix2PixPipeline):
#     # def __init__(self, cfg, vae: AutoencoderKL, 
#     #              text_encoder: CLIPTextModel, 
#     #              tokenizer: CLIPTokenizer, 
#     #              unet: UNet2DConditionModel, 
#     #              scheduler: KarrasDiffusionSchedulers, 
#     #              safety_checker: StableDiffusionSafetyChecker, 
#     #              feature_extractor: CLIPImageProcessor, 
#     #              requires_safety_checker: bool = True):
#     def __init__(self, cfg: SDModelConfig = None, 
#                  model: SDModel = None,
#                  vae: AutoencoderKL = None, 
#                  text_encoder: CLIPTextModel = None, 
#                  tokenizer: CLIPTokenizer = None, 
#                  unet: UNet2DConditionModel = None, 
#                  scheduler: KarrasDiffusionSchedulers = None, 
#                  safety_checker: StableDiffusionSafetyChecker = None, 
#                  feature_extractor: CLIPImageProcessor = None, 
#                  requires_safety_checker: bool = False,
#                  ):
#         # DiffusionPipeline.__init__(self) #, safety_checker, feature_extractor, requires_safety_checker
#         if model is not None and cfg is not None:
#             self.cfg = cfg
#             self.model = model
#             vae, text_encoder, tokenizer, unet, scheduler = model.vae, model.text_encoder, model.tokenizer, model.unet, model.noise_scheduler
#             self.vae, self.text_encoder, self.tokenizer, self.unet, self.scheduler = vae, text_encoder, tokenizer, unet, scheduler
#             self.register_modules(model=model)
#             self.register_to_config(cfg=dataclasses.asdict(cfg))
        
#         super().__init__(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False)
#         # unet, text_encoder, vae, tokenizer, noise_scheduler = self.load_modules(cfg.saved_model_path, cfg.pretrained_model_name_or_path)
#         # self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
#         # self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
#         # self.register_to_config(requires_safety_checker=requires_safety_checker)

    
#     def __call__(        
#         self,
#         prompts,
#         image,
#         num_inference_steps: int = 100,
#         guidance_scale: float = 7.5,
#         image_guidance_scale: float = 1.5,
#         negative_prompt: Optional[Union[str, List[str]]] = None,
#         num_images_per_prompt: Optional[int] = 1,
#         eta: float = 0.0,
#         generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
#         latents: Optional[torch.FloatTensor] = None,
#         prompt_embeds: Optional[torch.FloatTensor] = None,
#         negative_prompt_embeds: Optional[torch.FloatTensor] = None,
#         output_type: Optional[str] = "pil",
#         return_dict: bool = True,
#         callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
#         callback_steps: int = 1,):
        
#         if isinstance(prompts, str):
#             prompts = [prompts]
#         if isinstance(prompts, list):
#             input_ids = self.fancy_get_input_ids(prompts, self.text_encoder.device) # TODO see if reshaping needed to match train dataloader
#         else:
#             input_ids = prompts
        
#         if isinstance(image, PIL.Image.Image):
#             image = [image]
#         if isinstance(image, list):
#             preprocessed_images = self.image_processor.preprocess(image)
#         else:
#             preprocessed_images = image
            
#         batch_size = input_ids.shape[0]
        
#         # device = self._execution_device
#         device = self.text_encoder.device # TODO figure out execution device stuff
#         # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
#         # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
#         # corresponds to doing no classifier free guidance.
#         do_classifier_free_guidance = guidance_scale > 1.0 and image_guidance_scale >= 1.0
#         # check if scheduler is in sigmas space
#         scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")
        
        
#         prompt_embeds = self.encode_prompt_batch(input_ids, batch_size, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds, negative_prompt_embeds)
        
#         # 4. set timesteps
#         self.scheduler.set_timesteps(num_inference_steps, device=device)
#         timesteps = self.scheduler.timesteps
        
#         # 5. Prepare Image latents
#         image_latents = self.prepare_image_latents(
#             preprocessed_images,
#             batch_size,
#             num_images_per_prompt,
#             prompt_embeds.dtype,
#             device,
#             do_classifier_free_guidance,
#             generator,
#         )
        
#         height, width = image_latents.shape[-2:]
#         height = height * self.vae_scale_factor
#         width = width * self.vae_scale_factor
        
#         # 6. Prepare latent variables
#         num_channels_latents = self.vae.config.latent_channels
        
#         latents = self.prepare_latents(
#             batch_size * num_images_per_prompt,
#             num_channels_latents,
#             height,
#             width,
#             prompt_embeds.dtype,
#             device,
#             generator,
#             latents,
#         )
        
#         # 7. Check that shapes of latents and image match the UNet channels
#         num_channels_image = image_latents.shape[1]
#         if num_channels_latents + num_channels_image != self.unet.config.in_channels:
#             raise ValueError(
#                 f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
#                 f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
#                 f" `num_channels_image`: {num_channels_image} "
#                 f" = {num_channels_latents+num_channels_image}. Please verify the config of"
#                 " `pipeline.unet` or your `image` input."
#             )

#         # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
#         extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

#         # 9. Denoising loop
#         num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
#         with self.progress_bar(total=num_inference_steps) as progress_bar:
#             for i, t in enumerate(timesteps):
#                 # Expand the latents if we are doing classifier free guidance.
#                 # The latents are expanded 3 times because for pix2pix the guidance\
#                 # is applied for both the text and the input image.
#                 latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents

#                 # concat latents, image_latents in the channel dimension
#                 scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

#                 scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

#                 # predict the noise residual
#                 noise_pred = self.unet(
#                     scaled_latent_model_input, t, encoder_hidden_states=prompt_embeds, return_dict=False
#                 )[0]

#                 # Hack:
#                 # For karras style schedulers the model does classifer free guidance using the
#                 # predicted_original_sample instead of the noise_pred. So we need to compute the
#                 # predicted_original_sample here if we are using a karras style scheduler.
#                 if scheduler_is_in_sigma_space:
#                     step_index = (self.scheduler.timesteps == t).nonzero().item()
#                     sigma = self.scheduler.sigmas[step_index]
#                     noise_pred = latent_model_input - sigma * noise_pred

#                 # perform guidance
#                 if do_classifier_free_guidance:
#                     noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
#                     noise_pred = (
#                         noise_pred_uncond
#                         + guidance_scale * (noise_pred_text - noise_pred_image)
#                         + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
#                     )

#                 # Hack:
#                 # For karras style schedulers the model does classifer free guidance using the
#                 # predicted_original_sample instead of the noise_pred. But the scheduler.step function
#                 # expects the noise_pred and computes the predicted_original_sample internally. So we
#                 # need to overwrite the noise_pred here such that the value of the computed
#                 # predicted_original_sample is correct.
#                 if scheduler_is_in_sigma_space:
#                     noise_pred = (noise_pred - latents) / (-sigma)

#                 # compute the previous noisy sample x_t -> x_t-1
#                 latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

#                 # call the callback, if provided
#                 if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
#                     progress_bar.update()
#                     if callback is not None and i % callback_steps == 0:
#                         callback(i, t, latents)

#         if not output_type == "latent":
            
#             latents = rearrange(latents, 'b c (s h) w -> (b s) c h w', s=self.cfg.sequence_length) # these are image latents, so sequence_length instead of text_sequence_length
#             image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
#             # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
#         else:
#             image = latents

#         has_nsfw_concept = None
#         do_denormalize = [True] * image.shape[0]
#         # if has_nsfw_concept is None:
#         #     do_denormalize = [True] * image.shape[0]
#         # else:
#         #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

#         image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

#         # Offload last model to CPU
#         if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
#             self.final_offload_hook.offload()

#         if not return_dict:
#             return (image, has_nsfw_concept)

#         return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
        
    
#     def prepare_image_latents(self, image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None):
#         image_latents = super().prepare_image_latents(image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator)
#         return repeat(image_latents, 'b c h w -> b c (s h) w', s=self.cfg.sequence_length)

#     def fancy_get_input_ids(self, prompt, device):
#         # textual inversion: procecss multi-vector tokens if necessary
#         if isinstance(self, TextualInversionLoaderMixin):
#             prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

#         text_inputs = self.tokenizer(
#                 prompt,
#                 padding="max_length",
#                 max_length=self.tokenizer.model_max_length,
#                 truncation=True,
#                 return_tensors="pt",
#             )
#         text_input_ids = text_inputs.input_ids
#         untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

#         if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
#                 text_input_ids, untruncated_ids
#             ):
#             removed_text = self.tokenizer.batch_decode(
#                     untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
#                 )
#             logger.warning(
#                     "The following part of your input was truncated because CLIP can only handle sequences up to"
#                     f" {self.tokenizer.model_max_length} tokens: {removed_text}"
#                 )

#         if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
#             attention_mask = text_inputs.attention_mask.to(device)
#         else:
#             attention_mask = None
#         text_input_ids = text_input_ids
#         return text_input_ids,attention_mask
    
#     def encode_prompt_batch(self, 
#                             input_ids, 
#                             batch_size,
#                             device,
#                             num_images_per_prompt: int=1,
#                             do_classifier_free_guidance: bool=False,
#                             negative_prompt=None,
#                             prompt_embeds=None,
#                             negative_prompt_embeds=None,):
#         encoder_hidden_states = self.model.input_ids_to_text_condition(input_ids)
#         if self.cfg.positional_encoding_type is not None:
#             encoder_hidden_states = self.model.apply_step_positional_encoding(encoder_hidden_states)
#         prompt_embeds = encoder_hidden_states
#         prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

#         bs_embed, seq_len, _ = prompt_embeds.shape
#         # duplicate text embeddings for each generation per prompt, using mps friendly method
#         prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
#         prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        
#         if do_classifier_free_guidance:
#             if negative_prompt_embeds is None:
#                 negative_prompt_embeds = self.model.get_null_conditioning()
#                 negative_prompt_embeds = repeat(negative_prompt_embeds, 'o t l -> (b o) t l', b=batch_size) #, o=1
#             # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
#             seq_len = negative_prompt_embeds.shape[1]

#             negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

#             negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
#             negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

#             # For classifier free guidance, we need to do two forward passes.
#             # Here we concatenate the unconditional and text embeddings into a single batch
#             # to avoid doing two forward passes
#             # pix2pix has two  negative embeddings, and unlike in other pipelines latents are ordered [prompt_embeds, negative_prompt_embeds, negative_prompt_embeds]
#             prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds, negative_prompt_embeds])
#         return prompt_embeds
    
#     @staticmethod
#     def _get_signature_keys(obj):
#         parameters = inspect.signature(obj.__init__).parameters
#         required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
#         optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
#         expected_modules = set(required_parameters.keys()) - {"self"}
#         return optional_parameters, optional_parameters # because all params are optional here
    
#     # @property
#     # # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
#     # def _execution_device(self):
#     #     r"""
#     #     Returns the device on which the pipeline's models will be executed. After calling
#     #     `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
#     #     hooks.
#     #     """
#     #     if not hasattr(self.unet, "_hf_hook"):
#     #         return self.device
#     #     for module in self.unet.modules():
#     #         if (
#     #             hasattr(module, "_hf_hook")
#     #             and hasattr(module._hf_hook, "execution_device")
#     #             and module._hf_hook.execution_device is not None
#     #         ):
#     #             return torch.device(module._hf_hook.execution_device)
#     #     return self.device
    
# #     @staticmethod
# #     def load_modules(unet_path, pretrained_model_name_or_path):
# #         # Load scheduler, tokenizer and models.
# #         unet = UNet2DConditionModel.from_pretrained(
# #             unet_path, subfolder="unet", ignore_mismatched_sizes=True, low_cpu_mem_usage=False, # TODO address saving of .pos
# #         )
# #         text_encoder = CLIPTextModel.from_pretrained(
# #             pretrained_model_name_or_path, subfolder="text_encoder",
# #         )
# #         vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
# #         tokenizer = CLIPTokenizer.from_pretrained(
# #             pretrained_model_name_or_path, subfolder="tokenizer",
# #         )
# #         noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
# #         return unet, text_encoder, vae, tokenizer, noise_scheduler


# # @dataclass
# # class ClipModelConfig(BaseModelConfig):
# #     _target_: str = "trainer.models.clip_model.CLIPModel"
# #     pretrained_model_name_or_path: str = "openai/clip-vit-base-patch32"


# # class CLIPModel(nn.Module):
# #     def __init__(self, cfg: ClipModelConfig):
# #         super().__init__()
# #         self.model = HFCLIPModel.from_pretrained(cfg.pretrained_model_name_or_path)

# #     def get_text_features(self, *args, **kwargs):
# #         return self.model.get_text_features(*args, **kwargs)

# #     def get_image_features(self, *args, **kwargs):
# #         return self.model.get_image_features(*args, **kwargs)

# #     def forward(self, text_inputs=None, image_inputs=None):
# #         outputs = ()
# #         if text_inputs is not None:
# #             outputs += self.model.get_text_features(text_inputs),
# #         if image_inputs is not None:
# #             outputs += self.model.get_image_features(image_inputs),
# #         return outputs


# #     @property
# #     def logit_scale(self):
# #         return self.model.logit_scale

# #     def save(self, path):
# #         self.model.save_pretrained(path)

