import collections
from dataclasses import dataclass

import torch
from PIL import Image
from accelerate.logging import get_logger
from accelerate.utils import LoggerType, PrecisionType

from typing import Optional
from omegaconf import II
from transformers import AutoTokenizer

from trainer.accelerators.base_accelerator import BaseAccelerator
from trainer.tasks.base_task import BaseTaskConfig, BaseTask

from trainer.pipelines.snt_pipeline import ShowNotTellPipeline #TODO make this a choice



logger = get_logger(__name__)

from torch.nn import functional as F

from helpers import create_diagram

@dataclass
class SDTaskConfig(BaseTaskConfig):
    _target_: str = "trainer.tasks.sd_task.SDTask"
    pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")
    mixed_precision: Optional[PrecisionType] = II("accelerator.mixed_precision")
    num_val_images_per_prompt: int = 1
    model_target: str = II("model._target_")


class SDTask(BaseTask):
    def __init__(self, cfg: SDTaskConfig, accelerator: BaseAccelerator):
        super().__init__(cfg, accelerator)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer")
        self.cfg = cfg

    def train_step(self, model, batch):
        model_pred, target = model(batch)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss



    def run_inference(self, model, dataset, dataloader):
        eval_dict = collections.defaultdict(list)
        logger.info("Building pipeline...")
        create_pipeline_cfg = dict(
            pretrained_model_name_or_path=self.cfg.pretrained_model_name_or_path,
            # cfg=model.cfg,
            model=model
        )
        
        pipeline = ShowNotTellPipeline.from_pretrained(**create_pipeline_cfg)
            
        # pipeline = ShowNotTellPipeline.from_pretrained(**create_pipeline_cfg)
        
        logger.info("Running qualitative validation...")
        diagrams = {}
        for batch in dataloader:
            prompts = batch['input_ids']#.to(cfg.device)
            attention_masks = batch['attention_masks']
            condition_images = batch['original_pixel_values']#.to(cfg.device)
            # %%
            
            if self.cfg.mixed_precision == PrecisionType.BF16:
                dtype = torch.bfloat16
            elif self.cfg.mixed_precision == PrecisionType.FP16:
                dtype = torch.float16
            else:
                dtype = torch.float32
                
            
            for image_num in range(self.cfg.num_val_images_per_prompt):
                if len(prompts.shape) == 2:
                    prompts = prompts.unsqueeze(0)
                with torch.autocast(device_type="cuda", 
                                    dtype=dtype, 
                                    enabled= True):
                    output_images = pipeline(
                        prompts,
                        image=condition_images,
                        num_inference_steps=50,
                        image_guidance_scale=1.5,
                        guidance_scale=7,
                        generator=model.generator,
                        # attention_mask=attention_masks,
                        # num_images_per_prompt = cfg.num_validation_images
                    ).images
                prompt_text = dataset.decode_batch(prompts)
                condition_images_pil = pipeline.image_processor.postprocess(condition_images)
                

                for index in range(len(prompt_text)):
                    texts = prompt_text[index]
                    goal = str(texts[0])
                    steps = texts[1:]


                    diagram = create_diagram(
                                goal, 
                                condition_images_pil[index], 
                                steps, 
                                output_images[model.cfg.sequence_length*index:model.cfg.sequence_length*(index+1)], 
                                )
                    if goal in diagrams:
                        diagrams[goal].append(diagram)
                    else:
                        diagrams[goal] = [diagram]
                
        if LoggerType.WANDB == self.accelerator.cfg.log_with:
            table_name = "test_predictions"
            if dataset.cfg.use_vicuna:
                table_name = "vicuna_test_predictions"
            self.log_to_wandb(diagrams, table_name=table_name)
            
        
        # 1. Instantiate pipeline
        # 2. Infer on batch
        return {}
    

        
        
    # def run_inference(self, model, dataloader):
    #     eval_dict = collections.defaultdict(list)
    #     logger.info("Running clip score...")
    #     for batch in dataloader:
    #         image_0_probs, image_1_probs = self.valid_step(model, batch)
    #         agree_on_0 = (image_0_probs > image_1_probs) * batch[self.cfg.label_0_column_name]
    #         agree_on_1 = (image_0_probs < image_1_probs) * batch[self.cfg.label_1_column_name]
    #         is_correct = agree_on_0 + agree_on_1
    #         eval_dict["is_correct"] += is_correct.tolist()
    #         eval_dict["captions"] += self.tokenizer.batch_decode(
    #             batch[self.cfg.input_ids_column_name],
    #             skip_special_tokens=True
    #         )
    #         eval_dict["image_0"] += self.pixel_values_to_pil_images(batch[self.cfg.pixels_0_column_name])
    #         eval_dict["image_1"] += self.pixel_values_to_pil_images(batch[self.cfg.pixels_1_column_name])
    #         eval_dict["prob_0"] += image_0_probs.tolist()
    #         eval_dict["prob_1"] += image_1_probs.tolist()

    #         eval_dict["label_0"] += batch[self.cfg.label_0_column_name].tolist()
    #         eval_dict["label_1"] += batch[self.cfg.label_1_column_name].tolist()

    #     return eval_dict

    @torch.no_grad()
    def evaluate(self, model, dataset, dataloader):
        # logger.info("Eval not yet implemented")
        eval_dict = self.run_inference(model, dataset, dataloader)
        eval_dict = self.gather_dict(eval_dict)
        # metrics = {
        #     "accuracy": sum(eval_dict["is_correct"]) / len(eval_dict["is_correct"]),
        #     "num_samples": len(eval_dict["is_correct"])
        # }
        # if LoggerType.WANDB == self.accelerator.cfg.log_with:
        #     self.log_to_wandb(eval_dict)
        return eval_dict
        return {}
