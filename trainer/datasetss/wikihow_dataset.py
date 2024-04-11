from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)

from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import os

from random import randint

import torch
from PIL import Image
USE_ACCELERATE = False
if USE_ACCELERATE:
    from accelerate.logging import get_logger
else:
    from logging import getLogger as get_logger
# from datasets import load_from_disk, load_dataset, Dataset
from hydra.utils import instantiate
from omegaconf import II


from trainer.datasetss.base_dataset import BaseDataset, BaseDatasetConfig

import pathlib

from diffusers.image_processor import VaeImageProcessor
from torchvision import transforms

from einops import rearrange
import numpy as np

logger = get_logger(__name__)


from PIL import Image
from helpers import gaussian_noise_image_rescaled


def create_image_wb():
    # Create a new image with the specified dimensions
    img = Image.new('RGB', (256, 256), "white")
    
    # Drawing the bottom half as black
    for y in range(128, 256):
        for x in range(256):
            img.putpixel((x, y), (0, 0, 0))
            
    return img

def simple_collate(batch, column_name):
    return torch.cat([item[column_name] for item in batch], dim=0)


@dataclass
class ImageProcessorConfig: # TODO see if this works instead of torch transforms
    _target_: str = "diffusers.image_processor.VaeImageProcessor"
    do_resize: bool = True # TODO: change defaults, consider sizing issues
    vae_scale_factor: int = 8
    resample: str = "lanczos"
    do_normalize: bool = True
    do_convert_rgb: bool = False
    
@dataclass
class TokenizerConfig:
    _target_: str = "transformers.AutoTokenizer.from_pretrained"
    pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")
    subfolder: str = "tokenizer"

class Processor:
    def __init__(self, tokenizer, image_processor): # TODO see about cfging this
        self.tokenizer = tokenizer
        self.image_processor = image_processor

@dataclass
class ProcessorConfig:
    _target_: str = "trainer.datasetss.wikihow_dataset.Processor"
    tokenizer: TokenizerConfig = TokenizerConfig()
    image_processor: ImageProcessorConfig = ImageProcessorConfig()

@dataclass
class WikiHowDatasetConfig(BaseDatasetConfig):
    _target_: str = "trainer.datasetss.wikihow_dataset.WikiHowDataset"
    base: str = "/proj/vondrick2/datasets/wikihow/"
    train: bool = True
    max_samples: Optional[int] = None
    return_type: str = "tensors"
    resolution: int = 256
    sequence_length: int = 6
    text_sequence_length: Optional[int] = II("eval:1 + ${dataset.sequence_length}")
    processor: ProcessorConfig = ProcessorConfig()
    condition_image: str = 'gray'
    tile_with: str = 'gray'
    tile_text_with: str = 'gray'
    category_hierarchy_filter: Optional[str] = None #'Recipe'
    gold_set: bool = False
    heval_set: bool = False
    concat_all_steps: bool = False
    prepend_goal_to_steps: bool = False
    descr_usage: Optional[str] = None # options: None, 'append_text'
    also_return_originals: bool = False
    random_subset: bool = False
    return_goal_labels: bool = False
    override_data_source: Optional[str] = None
    use_rows_from_csv: Optional[str] = None
    append_method_to_goal: bool = True
    return_goal_method_id: bool = False
    # input_id_return_type: str = 'steps_only'
    number_tile_text: bool = False
    steps_from_one: bool = False
    black_from: Optional[int] = None
    white_until: Optional[int] = None
    use_vicuna: bool = False
    max_tokenizer_length: Optional[int] = 77
    sanity_check_wb: bool = False
    blip_descr: bool = False
    # start_from
    batch_size: int = 12 #II("accelerator.deepspeed.train_batch_size")
    duplicate_first: bool = False
    use_gpt: bool = False
    data_prop: Optional[float] = None
    no_goal: Optional[bool] = False
    
    
# @dataclass
# class CLIPWikiHowDatasetConfig(WikiHowDatasetConfig):
#     _target_: str = "trainer.datasetss.wikihow_dataset.CLIPWikiHowDataset"
    

import pandas as pd

class WikiHowDataset(BaseDataset):

    def __init__(self, cfg: WikiHowDatasetConfig, split: str = "train"):
        self.cfg = cfg
        
        if self.cfg.prepend_goal_to_steps: # TODO remove 
            self.cfg.text_sequence_length = self.cfg.sequence_length
        else:
            self.cfg.text_sequence_length = self.cfg.sequence_length + 1
            
        if self.cfg.random_subset and self.cfg.sequence_length > 1:
            raise NotImplementedError("Random subset only implemented for sequence length 1")
        
        if self.cfg.override_data_source and os.path.isfile(self.cfg.override_data_source + f'merged.csv'):
            logger.info(f"Overriding data source to {self.cfg.override_data_source}")
            self.root_dir = ''
            csv_file_path = self.cfg.override_data_source + f'merged.csv' #TODO change this
            self.df = pd.read_csv(csv_file_path)
            if 'sample_num' in self.df.columns:
                self.df = self.df[self.df['sample_num'] == 0]
                        
            if self.cfg.category_hierarchy_filter:
                self.df = self.df[self.df.category_hierarchy.str.contains(self.cfg.category_hierarchy_filter)]
            
        else:
            self.split = split
            logger.info(f"Loading {self.split} dataset")
            
            csv_file_path = self.cfg.base + f'splits/{self.split}.csv'
            if self.cfg.use_vicuna:
                csv_file_path = self.cfg.base + f'splits/vicuna_val_heval.csv'
            if self.cfg.use_gpt:
                csv_file_path = self.cfg.base + f'splits/gpt_val_heval.csv'
            
            self.root_dir = self.cfg.base + 'wiki_images/train/'
            # self.root_dir1 = self.cfg.base + 'wiki_images/test/'
            # self.root_dir2 = self.cfg.base + 'wiki_images/train/'
            # csv_file_path = f'/data/home/sachit/wikihow/splits/{self.split}.csv'
            # ,file_id,goal,goal_description,category_hierarchy,headline,description,img,img_license,step_id
            # Load data from csv_file_path using pandas
            cols_to_load = ['step_id', 'goal_method_id', 'goal', 'method_name', 'headline', 'step_number', 'description', 'category_hierarchy']
            if self.cfg.max_samples:
                self.df = pd.read_csv(csv_file_path, usecols=cols_to_load, nrows=self.cfg.max_samples)
            else:
                self.df = pd.read_csv(csv_file_path, usecols=cols_to_load)
                
            if self.cfg.use_vicuna:
                self.df['step_number']  = self.df['step_number'] - 1 # TODO remove, off by one error in training
            
            if self.cfg.override_data_source and os.path.isdir(self.cfg.override_data_source):
                logger.info(f"Overriding data source to {self.cfg.override_data_source}")
                self.root_dir = self.cfg.override_data_source
                fnames = pd.Series(os.listdir(self.root_dir)).str[:-4]
                self.df['step_id'] = '0-' + self.df['step_id']
                self.df = self.df[self.df['step_id'].isin(fnames)]
            elif self.cfg.override_data_source:
                raise Exception(f"Invalid override data source: {self.cfg.override_data_source} is not directory")
            
            if self.cfg.use_rows_from_csv and os.path.isfile(self.cfg.use_rows_from_csv + f'merged.csv'):
                otherdf = pd.read_csv(self.cfg.use_rows_from_csv + f'merged.csv') #TODO change this
                self.df = self.df[self.df['goal_method_id'].isin(otherdf['goal_method_id']) & (self.df['step_number'] <= otherdf['step_number'].max())]
            elif self.cfg.heval_set:
                otherdf = pd.read_csv(self.cfg.base + f'splits/vicuna_val_heval.csv')
                self.df = self.df[self.df['goal_method_id'].isin(otherdf['goal_method_id']) & (self.df['step_number'] <= otherdf['step_number'].max())]
            elif self.cfg.use_rows_from_csv and os.path.isdir(self.cfg.use_rows_from_csv):
                fnames = pd.Series(os.listdir(self.cfg.use_rows_from_csv)).str[2:-4]
                self.df = self.df[self.df['step_id'].isin(fnames)]
            
            if self.cfg.category_hierarchy_filter:
                self.df = self.df[self.df.category_hierarchy.str.contains(self.cfg.category_hierarchy_filter)]
            
            self.val_gold_set_goal_ids = ["3206490_1","1072580_0","1892824_0","9306203_1","2870746_2","5035888_0","3259393_0"]
            self.train_gold_set_goal_ids = ["1808379_1", "3784417_2", "69206_0", "1084835_0", "518621_1"]
            if self.split == 'train':
                self.gold_set_goal_ids = self.train_gold_set_goal_ids
            else:
                self.gold_set_goal_ids = self.val_gold_set_goal_ids
            if self.cfg.gold_set:
                self.df = self.df[self.df.goal_method_id.isin(self.gold_set_goal_ids)]
                

            
            logger.info(f"Loaded {len(self.df)} examples from {self.split} dataset")
            
            if self.cfg.train and self.cfg.blip_descr:     
                self.cfg.descr_usage = 'append_text'  
                # Read the CSV into a DataFrame new_vals_df
                new_vals_df = pd.read_csv(self.cfg.base + 'blip_descr.csv')  # Assuming the CSV has columns 'id' and 'val'

                # Merge the DataFrames based on 'id'
                merged_df = pd.merge(self.df, new_vals_df, on='step_id', how='left')

                # Replace the 'description' column with the new 'val' column
                merged_df['description'] = merged_df['val']

                # Drop the 'val' column as it's no longer needed
                merged_df.drop('val', axis=1, inplace=True)

                # Update the original df
                self.df = merged_df
            
            if not self.cfg.train and self.cfg.blip_descr:
                # Read the CSV into a DataFrame new_vals_df
                new_vals_df = pd.read_csv(self.cfg.base + 'vicuna_descr.csv')  # Assuming the CSV has columns 'id' and 'val'

                # Merge the DataFrames based on 'id'
                merged_df = pd.merge(self.df, new_vals_df, on='step_id', how='left')

                # Replace the 'description' column with the new 'val' column
                merged_df['description'] = merged_df['val']

                # Drop the 'val' column as it's no longer needed
                merged_df.drop('val', axis=1, inplace=True)

                # Update the original df
                self.df = merged_df
                
                
            
                
            # Add root_dir to the image paths
            self.df['step_id'] = self.df['step_id'] + '.png'

            
        
        self.groups = self.df.groupby('goal_method_id')
        self.ids = list(self.groups.groups.keys())
        
        
        if self.cfg.return_goal_labels:
            self.unique_goals = self.df.goal.unique().tolist()
            self.goal_to_label = dict(zip(self.unique_goals, torch.arange(len(self.unique_goals))))

        processor = instantiate(cfg.processor)
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor

        if (hasattr(self.cfg, 'data_prop') and (self.cfg.data_prop is not None) and (self.cfg.data_prop < 1.) and (self.split == "train")):
            # Sample self.cfg.data_prop percent of the self.ids
            # self.ids = self.ids[:int(self.cfg.data_prop*len(self.ids))]
            self.ids = pd.Series(self.ids).sample(frac=self.cfg.data_prop, random_state=42).tolist()
            self.df = self.df[self.df.goal_method_id.isin(self.ids)]
            self.groups = self.df.groupby('goal_method_id')
            
            # self.groups = self.groups.sample(frac=self.cfg.data_prop).groupby('goal_method_id')
            # self.ids = list(self.groups.groups.keys())
            # Filter the dataframe to only keep those samples
            logger.info(f"Sampled {self.cfg.data_prop} of the {self.split} groups to get {len(self.ids)} groups and {len(self.df)} examples")

    def tokenize(self, caption, max_length=None):
        if max_length is None:
            if self.cfg.max_tokenizer_length is not None:
                max_length = self.cfg.max_tokenizer_length
            else:
                max_length = self.tokenizer.model_max_length
        # caption = example[self.cfg.caption_column_name]
        tokenizer_output = self.tokenizer(
            caption,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        return input_ids, attention_mask
    
    def decode(self, input_ids):
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    def decode_batch(self, input_ids_with_sequences):
        if self.cfg.concat_all_steps:
            return self.decode(input_ids_with_sequences)
        ids = rearrange(input_ids_with_sequences, 'b s t -> (b s) t')
        decoded_ids = np.array(self.decode(ids))
        reshaped = rearrange(decoded_ids, '(b s) -> b s', s=self.cfg.text_sequence_length)
        return reshaped
        

    # def process_image(self, image):
    #     if isinstance(image, dict):
    #         image = image["bytes"]
    #     if isinstance(image, bytes):
    #         image = Image.open(BytesIO(image))
    #     image = image.convert("RGB")
    #     pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"]
    #     return pixel_values
    
    def __getitem__(self, idx):
        # load all rows with the same goal_method_id
        goal_method_id = self.ids[idx]
        rows = self.groups.get_group(goal_method_id)
        # return all rows
        step_numbers = rows['step_number'].tolist() # TODO refactor
        goals = rows['goal'].tolist()
        methods = rows['method_name'].tolist()
        step_texts = rows['headline'].tolist()
        img_paths = rows['step_id'].tolist()
        descr_texts = rows['description'].tolist()
        # make the above cleaner
        
        
        if self.cfg.return_type == 'paths':
            images = [self.root_dir + img_path for img_path in img_paths]
        else:
            if self.cfg.use_vicuna or self.cfg.use_gpt: 
                images = [Image.new('RGB', (self.cfg.resolution, self.cfg.resolution), color='black') for img_path in img_paths]
            else:
                images = [Image.open(self.root_dir + img_path).convert('RGB') for img_path in img_paths]
                
        
        
        if self.cfg.random_subset: # NOTE: sequence_length better be 1!
            step_index = randint(0, len(step_texts)-1)
            images = images[step_index:step_index+1]
            goals = goals[step_index:step_index+1]
            methods = methods[step_index:step_index+1]
            step_texts = step_texts[step_index:step_index+1]
            step_numbers = step_numbers[step_index:step_index+1]
            descr_texts = descr_texts[step_index:step_index+1]
        else:
            images = images[:self.cfg.sequence_length]
            goals = goals[:self.cfg.sequence_length]
            methods = methods[:self.cfg.sequence_length]
            step_texts = step_texts[:self.cfg.sequence_length]
            step_numbers = step_numbers[:self.cfg.sequence_length]
            descr_texts = descr_texts[:self.cfg.sequence_length]
        
        # edit_prompt = f"Goal: {goals[0]}. 1. {step_texts[0]} 2. {step_texts[1]}"
        edit_prompt = f"Goal: {goals[0]}"
        if self.cfg.no_goal:
            edit_prompt = ""
        if self.cfg.append_method_to_goal:
            edit_prompt += f", {methods[0]}."
        else:
            edit_prompt += "."
            
        if self.cfg.concat_all_steps:
            for i in range(0, len(step_texts)):
                if self.cfg.steps_from_one:
                    newstep = f"{step_numbers[i]+1}. {step_texts[i]}"
                else:
                    newstep = f"{step_numbers[i]}. {step_texts[i]}"
                edit_prompt += f" {newstep}"    
        else:
            edit_prompts = []
            if not self.cfg.prepend_goal_to_steps:
                edit_prompts.append(edit_prompt)
            for i in range(0, len(step_texts)):
                newstep = f"{step_numbers[i]}. {step_texts[i]}"
                if self.cfg.descr_usage == 'append_text':
                    newstep += f" {descr_texts[i]}" 
                    # newstep += f" {descr_texts[i]}" 
                if self.cfg.prepend_goal_to_steps:
                    to_append = edit_prompt + f" {newstep}"
                else:
                    to_append = f"{newstep}"
                edit_prompts.append(to_append)
                if self.cfg.descr_usage == 'append_encoding':
                    raise NotImplementedError
            edit_prompt = edit_prompts
            
        if self.cfg.condition_image == 'first':
            input_image = images[0]
        elif self.cfg.condition_image == 'last':
            input_image = images[-1]
        elif self.cfg.condition_image == 'black':
            input_image = Image.new('RGB', (self.cfg.resolution, self.cfg.resolution), color='black')
        elif self.cfg.condition_image == 'white':
            input_image = Image.new('RGB', (self.cfg.resolution, self.cfg.resolution), color='white')
        elif self.cfg.condition_image == 'gray':
            input_image = Image.new('RGB', (self.cfg.resolution, self.cfg.resolution), color='gray')
        elif self.cfg.condition_image == 'noise':
            input_image = gaussian_noise_image_rescaled(self.cfg.resolution, self.cfg.resolution, mean=0, std=1, mode='RGB')
            
        edited_image = images#[1:]
        if self.cfg.tile_with == 'first':
            tiler = images[0]
        elif self.cfg.tile_with == 'last':
            tiler = images[-1]
        elif self.cfg.tile_with == 'black':
            tiler = Image.new('RGB', (self.cfg.resolution, self.cfg.resolution), color='black')
        elif self.cfg.tile_with == 'white':
            tiler = Image.new('RGB', (self.cfg.resolution, self.cfg.resolution), color='white')
        elif self.cfg.tile_with == 'gray':
            tiler = Image.new('RGB', (self.cfg.resolution, self.cfg.resolution), color='gray')
        elif self.cfg.tile_with == 'noise':
            tiler = gaussian_noise_image_rescaled(self.cfg.resolution, self.cfg.resolution, mean=0, std=1, mode='RGB')
        
        if len(edited_image) < self.cfg.sequence_length:# (n-1): # TODO: look into/reconsider this
            edited_image = edited_image + [tiler] * ((self.cfg.sequence_length) - len(edited_image)) #((n-1) - len(edited_image)) # TODO check why padding with first
        
        if self.cfg.tile_text_with == 'first':
            text_tiler = edit_prompt[0]
        elif self.cfg.tile_text_with == 'last':
            text_tiler = edit_prompt[-1]
        elif self.cfg.tile_text_with == 'null':
            text_tiler = ''
        elif self.cfg.tile_text_with == 'black':
            text_tiler = '<BLACK>'
        elif self.cfg.tile_text_with == 'white':
            text_tiler = '<WHITE>'
        elif self.cfg.tile_text_with == 'gray':
            text_tiler = '<GRAY>'
        elif self.cfg.tile_text_with == 'noise':
            text_tiler = '<NOISE>'
        
        # TODO: check if this causes any issues; some of the text steps also don't have the same number 
        if not self.cfg.number_tile_text:
            text_tile_list = [text_tiler] * ((self.cfg.text_sequence_length) - len(edit_prompt)) #((n-1) - len(edited_image)) # pad with last 
        else:
            # TODO: add "if start index is 0"
            if self.cfg.steps_from_one:
                text_tile_list = [f"{j+1}. {text_tiler}" for j in range(len(edit_prompt), self.cfg.text_sequence_length)]
            else:
                text_tile_list = [f"{j}. {text_tiler}" for j in range(len(edit_prompt), self.cfg.text_sequence_length)]
            
        if len(edit_prompt) < self.cfg.text_sequence_length: # TODO check if this is off by one
            edit_prompt = edit_prompt + text_tile_list
            
        if self.cfg.black_from is not None:
            edited_image = edited_image[:self.cfg.black_from] + [Image.new('RGB', (self.cfg.resolution, self.cfg.resolution), color='black')] * (self.cfg.sequence_length - self.cfg.black_from)
            edit_prompt = edit_prompt[:self.cfg.black_from] + ['<BLACK>'] * (self.cfg.text_sequence_length - self.cfg.black_from)
        if self.cfg.white_until is not None:
            edited_image = [Image.new('RGB', (self.cfg.resolution, self.cfg.resolution), color='white')] * self.cfg.white_until + edited_image[self.cfg.white_until:]
            edit_prompt = ['<WHITE>'] * self.cfg.white_until + edit_prompt[self.cfg.white_until:]
        
        if self.cfg.sanity_check_wb:
            input_image = Image.new('RGB', (self.cfg.resolution, self.cfg.resolution), color='white')
            edited_image = [create_image_wb()]
            edit_prompt = ['NONE']*self.cfg.text_sequence_length
            
        if self.cfg.duplicate_first:
            edited_image = [edited_image[0]]*self.cfg.sequence_length
            edit_prompt = [edit_prompt[0]]*self.cfg.text_sequence_length
            
        
        if self.cfg.also_return_originals:
            out = {'input_image_og': input_image, 'edit_prompt_og': edit_prompt, 'edited_image_og': edited_image}
        else:
            out = {}
            
        if self.cfg.descr_usage == 'return_separate':
            out['descr_texts'] = descr_texts
            
        if self.cfg.return_type == 'tensors':
            edit_prompt, attention_mask = self.text_transform(edit_prompt)
            if self.cfg.concat_all_steps:
                attention_mask = attention_mask.squeeze(0) # consider changing the later stack to cat? TODO
            input_image = self.transform(input_image)
            edited_image = torch.stack([self.transform(img) for img in edited_image])
        
        
        out = {**out, 'input_image': input_image, 'edit_prompt': edit_prompt, 'attention_mask': attention_mask, 'edited_image': edited_image}
        
        if self.cfg.return_goal_labels:
            out['goal_label'] = self.goal_to_label[goals[0]]
            used_mask = torch.zeros(self.cfg.sequence_length)
            used_mask[:len(goals)] = 1
            out['goals_used_mask'] = used_mask
            if len(goals) < self.cfg.sequence_length:
                goals = goals + ['<SKIP>'] * (self.cfg.sequence_length - len(goals))
            out['goal'], out['goal_attention_mask'] = self.text_transform(goals)
        # out = {'input_image': input_image, 'edit_prompt': edit_prompt, 'edited_image': edited_image}
        if self.cfg.return_goal_method_id:
            out['goal_method_id'] = goal_method_id
            
                
        return out

    def collate_fn(self, examples):
        original_pixel_values = torch.stack([example["input_image"] for example in examples])
        original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format)
        edited_pixel_values = torch.stack([example["edited_image"] for example in examples])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format)
        
        input_ids = torch.stack([example["edit_prompt"] for example in examples]).squeeze(1)
        input_ids = input_ids.to(memory_format=torch.contiguous_format)
        attention_masks = torch.stack([example["attention_mask"] for example in examples]) # TODO check if needs squeeze
        attention_masks = attention_masks.to(memory_format=torch.contiguous_format)
        
        out = {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "input_ids": input_ids,
            "attention_masks": attention_masks,
        }
        if self.cfg.also_return_originals:
            for k,v in examples[0].items():
                if isinstance(v, Image.Image) or isinstance(v, str) or isinstance(v, list):
                    out[k] = [example[k] for example in examples]
            # out = {
            #     **out,
            #     "original_pixel_values_og": [example["input_image_og"] for example in examples],
            #     "edited_pixel_values_og": [example["edited_image_og"] for example in examples],
            #     "input_ids_og": [example["edit_prompt_og"] for example in examples],
            # }
        if self.cfg.return_goal_labels:
            out["goal_labels"] = torch.stack([example["goal_label"] for example in examples]).to(memory_format=torch.contiguous_format)
            out["goal"] = torch.cat([example["goal"] for example in examples], dim=0)
            out["goal_attention_masks"] = torch.cat([example["goal_attention_mask"] for example in examples], dim=0)
            out["goals_used_mask"] = torch.cat([example["goals_used_mask"] for example in examples], dim=0).to(memory_format=torch.contiguous_format)
        if self.cfg.return_goal_method_id:
            out["goal_method_id"] = [example["goal_method_id"] for example in examples]
        return out
        
    def transform(self, image):
        return self.transform_images(image, resolution=self.cfg.resolution)
    
    def text_transform(self, text):
        return self.tokenize(text)
    
    @staticmethod
    def transform_images(image, resolution=256, center_crop=False, random_flip=False):
        tfms = transforms.Compose(
            [
                transforms.Resize((resolution, resolution)),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(), # original: 2*(x/255) - 1
            ]
        )
        out = tfms(image)
        out = 2.*out - 1. # normalize to [-1, 1]
        return out

    def __len__(self):
        return len(self.ids)

class BlipWikiHowDataset(WikiHowDataset):

    def __init__(self, cfg: WikiHowDatasetConfig, split: str = "train"):
        super().__init__(cfg, split)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.root_dir + row['step_id']).convert('RGB')
        return image, row['step_id'][:-4]

    def collate_fn(self, examples):
        images = [example[0] for example in examples]
        ids = [example[1] for example in examples]
        return images, ids

    def __len__(self):
        return len(self.df)
    
    

class WikiHowTextDataset(BaseDataset):

    def __init__(self, cfg: WikiHowDatasetConfig, split: str = "train"):
        self.prompt = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

User: Please give instructions in the style of WikiHow for the goal "{goal}:{method}."

Assistant: """
        
        
        self.cfg = cfg
        
        if self.cfg.prepend_goal_to_steps: # TODO remove 
            self.cfg.text_sequence_length = self.cfg.sequence_length
        else:
            self.cfg.text_sequence_length = self.cfg.sequence_length + 1
            
        if self.cfg.random_subset and self.cfg.sequence_length > 1:
            raise NotImplementedError("Random subset only implemented for sequence length 1")
        
        if self.cfg.override_data_source:
            logger.info(f"Overriding data source to {self.cfg.override_data_source}")
            self.root_dir = ''
            csv_file_path = self.cfg.override_data_source + f'merged.csv' #TODO change this
            self.df = pd.read_csv(csv_file_path)
            if 'sample_num' in self.df.columns:
                self.df = self.df[self.df['sample_num'] == 0]
                        
            if self.cfg.category_hierarchy_filter:
                self.df = self.df[self.df.category_hierarchy.str.contains(self.cfg.category_hierarchy_filter)]
            
        else:
            self.split = split
            logger.info(f"Loading {self.split} dataset")
            
            csv_file_path = self.cfg.base + f'splits/{self.split}.csv'
            self.root_dir = self.cfg.base + 'wiki_images/train/'
            # self.root_dir1 = self.cfg.base + 'wiki_images/test/'
            # self.root_dir2 = self.cfg.base + 'wiki_images/train/'
            # csv_file_path = f'/data/home/sachit/wikihow/splits/{self.split}.csv'
            # ,file_id,goal,goal_description,category_hierarchy,headline,description,img,img_license,step_id
            # Load data from csv_file_path using pandas
            cols_to_load = ['step_id', 'goal_method_id', 'goal', 'method_name', 'headline', 'step_number', 'description', 'category_hierarchy']
            if self.cfg.max_samples:
                self.df = pd.read_csv(csv_file_path, usecols=cols_to_load, nrows=self.cfg.max_samples)
            else:
                self.df = pd.read_csv(csv_file_path, usecols=cols_to_load)
                
            if self.cfg.use_rows_from_csv:
                otherdf = pd.read_csv(self.cfg.use_rows_from_csv + f'merged.csv') #TODO change this
                self.df = self.df[self.df['goal_method_id'].isin(otherdf['goal_method_id']) & (self.df['step_number'] <= otherdf['step_number'].max())]
            
            if self.cfg.category_hierarchy_filter:
                self.df = self.df[self.df.category_hierarchy.str.contains(self.cfg.category_hierarchy_filter)]
            
            self.val_gold_set_goal_ids = ["3206490_1","1072580_0","1892824_0","9306203_1","2870746_2","5035888_0","3259393_0"]
            self.train_gold_set_goal_ids = ["1808379_1", "3784417_2", "69206_0", "1084835_0", "518621_1"]
            if self.split == 'train':
                self.gold_set_goal_ids = self.train_gold_set_goal_ids
            else:
                self.gold_set_goal_ids = self.val_gold_set_goal_ids
            if self.cfg.gold_set:
                self.df = self.df[self.df.goal_method_id.isin(self.gold_set_goal_ids)]
            
            logger.info(f"Loaded {len(self.df)} examples from {self.split} dataset")
            self.df = self.df[self.df.step_number == 0]


    def tokenize(self, caption):
        # caption = example[self.cfg.caption_column_name]
        input_ids = self.tokenizer(
            caption,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return input_ids
    
    def decode(self, input_ids):
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        goal, method = row['goal'], row['method_name']
        goal_method_id = row['goal_method_id']
        out = {}
        if self.cfg.return_type == 'tensors':
            out['prompt'] = self.text_transform(self.prompt.format(goal=goal, method=method))
            if self.cfg.also_return_originals:
                out['goal_og'], out['method_og'] = goal, method
        else:
            out['prompt'] = self.prompt.format(goal=goal, method=method)
        out['goal_method_id'] = goal_method_id
        return out

    def collate_fn(self, examples):
        if self.cfg.return_type == 'tensors':
            prompts = torch.stack([example["prompt"] for example in examples]).squeeze(1)
            prompts = prompts.to(memory_format=torch.contiguous_format)
        else:
            prompts = [example["prompt"] for example in examples]
        goal_method_ids = [example["goal_method_id"] for example in examples]
        return {
            "prompts": prompts,
            "goal_method_ids": goal_method_ids,
        }
        
    def text_transform(self, text):
        return self.tokenize(text)

    def __len__(self):
        return len(self.df)
