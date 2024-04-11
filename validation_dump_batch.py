# %%
print('starting')
import torch
# %%
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pathlib
import time

from easydict import EasyDict
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

# %%
from transformers import CLIPTextModel, CLIPTokenizer

from IPython.display import display

import re


from trainer.models.sd_model import SDModelConfig, SDModel
from trainer.pipelines.snt_pipeline import ShowNotTellPipeline
from trainer.datasetss.wikihow_dataset import WikiHowDataset, WikiHowDatasetConfig

import random
import numpy as np

from helpers import create_diagram
from einops import rearrange, repeat
from PIL import Image

from tqdm.auto import tqdm

# %%


print('finished imports')

def save_diagram(diagram, goal, diagram_dir, save_prefix=None, override_suffix=None):
    goal = goal.replace(" ", "_").replace('_:_', '_').replace(',','')
    if override_suffix is not None:
        suffix = override_suffix
    else:
        suffix = 999
    if save_prefix is not None:
        fname = f"{save_prefix}_{goal}_{suffix}.png"
    else:
        fname = f"{goal}_{suffix}.png"
    diagram_fname = os.path.join(diagram_dir, fname)
    diagram.save(diagram_fname)
    print(f"Saving diagram {suffix} to {diagram_fname}")


# %%

from hydra.experimental import compose, initialize

cfg = EasyDict()

cfg.use_train = False
split = 'train' if cfg.use_train else 'validation'
   
# cfg.resume_from_checkpoint = "latest"
# cfg.resume_from_checkpoint = "checkpoint-10000"
# cfg.output_dir = 'sin_pos_broadcast'
# cfg.output_dir = os.path.join("/data/home/sachit/proj/tracking", cfg.output_dir)
cfg.resume_from_checkpoint = "checkpoint-gstep70000"
cfg.output_dir = 'test2/default'

cfg.output_dir = os.path.join("./outputs/", cfg.output_dir)
cfg.weight_dtype = torch.bfloat16


# config_path = os.path.join(cfg.output_dir, cfg.resume_from_checkpoint)
config_path = cfg.output_dir
relative_config_path = os.path.relpath(config_path, start=os.getcwd())

with initialize(config_path=relative_config_path):
    loaded_cfg = compose(config_name="config.yaml")


# purely for type hinting
# cfg = EasyDict(loaded_cfg)
cfg.dataset = WikiHowDatasetConfig(**loaded_cfg.dataset)
cfg.model = SDModelConfig(**loaded_cfg.model)
cfg.model.sequence_length = cfg.dataset.sequence_length


# cfg = WikiHowDatasetConfig()
# cfg.total_batches = 1
cfg.total_batches = None
cfg.gpu_number = 2
cfg.device = f'cuda:{cfg.gpu_number}'
cfg.batch_size = 1
cfg.num_workers = 10

cfg.dataset.also_return_originals = False
cfg.dataset.return_goal_method_id = True

cfg.dataset.category_hierarchy_filter = 'Recipe'


# cfg.mode = 'random'
cfg.mode = 'random' # options: random, gold
cfg.dataset.gold_set = cfg.mode == 'gold'

cfg.viz_type = "dump" # options: diagrams, dump, display
# cfg.viz_type = "diagrams" # options: diagrams, dump, display
if cfg.use_train:
    cfg.diagram_dir = os.path.join(cfg.output_dir, f"_{split}_{cfg.viz_type}")
else:
    cfg.diagram_dir = os.path.join(cfg.output_dir, f"_{cfg.viz_type}")

cfg.dataset.use_vicuna = False
if cfg.dataset.use_vicuna: 
    cfg.diagram_dir = cfg.diagram_dir + '_vicuna'



cfg.n_to_save = 250

cfg.seed = 99

cfg.num_validation_images = 1

if cfg.dataset.descr_usage is not None:
    cfg.diagram_dir = cfg.diagram_dir + f'_descr_{cfg.dataset.descr_usage}'



# cfg.processor.tokenizer.pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5" # NOTE this only works bc uses the CLIP tokenizer


torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)
generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)



if cfg.resume_from_checkpoint == "latest":
    # Get the most recent checkpoint
    dirs = os.listdir(cfg.output_dir)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.replace("gstep","").split("-")[1]))
    path = dirs[-1] if len(dirs) > 0 else None
    print(f"Found checkpoint: {path}")
elif cfg.resume_from_checkpoint == "random":
    # Get a random checkpoint
    dirs = os.listdir(cfg.output_dir)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    path = random.choice(dirs) if len(dirs) > 0 else None
    print(f"Found checkpoint: {path}")
else:
    path = os.path.basename(cfg.resume_from_checkpoint)
if path is None:
    raise FileNotFoundError(f"No checkpoint dir found in {cfg.output_dir}")

if cfg.mode == 'gold':
    cfg.diagram_dir = cfg.diagram_dir + '_gold'

if cfg.viz_type != 'display':
    cfg.diagram_dir = os.path.join(cfg.diagram_dir, path)
    os.makedirs(cfg.diagram_dir, exist_ok=True)

print('loading modules')


model = SDModel(cfg.model)

unet_path = os.path.join(cfg.output_dir, path)
model.unet = UNet2DConditionModel.from_pretrained(
    unet_path, subfolder="unet", ignore_mismatched_sizes=True, low_cpu_mem_usage=False, # TODO address saving of .pos
)
model.init_pos()
model.init_image_pos()

# model.dtype = model.unet.dtype

# %%
# %%
create_pipeline_cfg = dict(
    pretrained_model_name_or_path=cfg.model.pretrained_model_name_or_path,
    cfg=cfg.model,
    model=model,
    safety_checker=None,
    feature_extractor=None,
)
# %%
pipeline = ShowNotTellPipeline.from_pretrained(**create_pipeline_cfg)

# %%
pipeline = pipeline.to(cfg.device)

# %%
print('loading dataset')


 
wikihow = WikiHowDataset(cfg.dataset, split)

dl = torch.utils.data.DataLoader(
    wikihow,
    shuffle=False,
    batch_size=cfg.batch_size,
    collate_fn=wikihow.collate_fn,
    num_workers=cfg.num_workers
)


# %%
# idl = iter(dl)
# o = next(idl)

# %%
# pipeline(o['input_ids'], o['original_pixel_values'])
# pipeline.model.cfg.sequence_length = pipeline.model.cfg.text_sequence_length - 1
for batch_number, batch in enumerate(tqdm(dl, desc="Running through batches", total=len(dl))):
    if (cfg.total_batches is not None) and (batch_number == cfg.total_batches):
        break
    prompts = batch['input_ids'].to(cfg.device)
    condition_images = batch['original_pixel_values'].to(cfg.device)
    curr_batch_size = prompts.shape[0]
    
    if cfg.viz_type == 'diagrams':
        prompt_text = wikihow.decode_batch(prompts)
        condition_images_pil = pipeline.image_processor.postprocess(condition_images)
    elif cfg.viz_type == 'dump':
        goal_method_ids = batch['goal_method_id']
    # %%
    # with torch.autocast(device_type=str(cfg.device).replace(f":{cfg.gpu_number}", ""), 
    #                     dtype=cfg.weight_dtype, 
    #                     enabled= True):
    #     output_images = pipeline(
    #         prompts,
    #         image=condition_images,
    #         num_inference_steps=50,
    #         image_guidance_scale=1.5,
    #         guidance_scale=7,
    #         generator=generator,
    #         # num_images_per_prompt = cfg.num_validation_images
    #     ).images
    # output_images[0]
    # %%
    for sample_num in range(cfg.num_validation_images):
        with torch.autocast(device_type=str(cfg.device).replace(f":{cfg.gpu_number}", ""), 
                            dtype=cfg.weight_dtype, 
                            enabled= True):
            output_images = pipeline(
                prompts,
                image=condition_images,
                num_inference_steps=50,
                image_guidance_scale=1.5,
                guidance_scale=7,
                generator=generator,
                # num_images_per_prompt = cfg.num_validation_images
            ).images



        # %%


        for index in range(curr_batch_size):
            if cfg.viz_type == 'diagrams':    
                texts = prompt_text[index]
                goal = texts[0]
                steps = texts[1:]


                diagram = create_diagram(
                            goal, 
                            condition_images_pil[index], 
                            steps, 
                            output_images[cfg.model.sequence_length*index:cfg.model.sequence_length*(index+1)], 
                            )
                save_diagram(diagram, goal, cfg.diagram_dir, index, sample_num) # TODO replace with outer loop var
            elif cfg.viz_type == 'dump':
                # texts = prompt_text[index]
                # goal = texts[0]
                # goal, method = goal.replace('Goal: ', '').split(', ')
                # steps = texts[1:]
                curr_output_images = output_images[cfg.model.sequence_length*index:cfg.model.sequence_length*(index+1)]
                
                for i in range(cfg.model.sequence_length):
                    curr_goal_method_id = goal_method_ids[index]
                    # pattern is samplenum-originalfname.png
                    # output_image_fname = f"{sample_num}-{curr_goal_method_id}_{i}.png"
                    output_image_fname = f"{sample_num}-{curr_goal_method_id}_{i}.png"
                    output_image_fname = os.path.join(cfg.diagram_dir, output_image_fname)
                    curr_output_images[i].save(output_image_fname)
                
                # output_image_fnames = [f"{index}_{i}_{sample_num}.png" for i in range(cfg.model.sequence_length)]
                # for i in range(cfg.model.sequence_length):
                #     output_images[cfg.model.sequence_length*index + i].save(os.path.join(cfg.diagram_dir, output_image_fnames[i]))
                # out_dict = dict(
                #     goal=goal,
                #     steps=steps,
                #     condition_images=condition_images_pil[index],
                #     output_images=output_images[cfg.model.sequence_length*index:cfg.model.sequence_length*(index+1)],
                # )
                
                # save_diagram(diagram, goal, cfg.diagram_dir, index, 999) # TODO replace with outer loop var
    # print(goal)
    # for i in range(cfg.model.sequence_length):
    #     print(steps[i])
    #     display(output_images[cfg.model.sequence_length*index + i])

    # %%


    # if cfg.mode == 'gold':
    #     print("Using gold set, so setting n_to_save to length of dataset")
    #     cfg.n_to_save = len(wikihow)
        
    # # %%
    # if cfg.use_vicuna:
    #     if use_golden_set:
    #         use_cached = True
    #         print("Using gold set, so skipping Vicuna loading.")
    #         from vicuna_steps_saved_proc import vcache
    #     else:
    #         use_cached = False
    #         from transformers import AutoTokenizer, AutoModelForCausalLM

    #         tokenizer = AutoTokenizer.from_pretrained("eachadea/vicuna-13b-1.1")

    #         model = AutoModelForCausalLM.from_pretrained("eachadea/vicuna-13b-1.1")

    # # %%
    # if not cfg.save: 
    #     cfg.n_to_save = 1

    # all_out = []
    # for num_saving in range(cfg.n_to_save):
    #     print(f"Saving {num_saving+1} of {cfg.n_to_save}")
    #     # try:
    #     if cfg.mode == 'random':
    #         index = random.randint(0, len(wikihow))
    #     else:
    #         index = num_saving
    #     # index = 1237
    #     # index = 5
    #     input_image = wikihow[index]['input_image']
    #     original_image = transform_images(input_image, 256)
    #     validation_prompt = wikihow[index]['edit_prompt']
        
    #     print(f"{validation_prompt} \n")
        
    #     if cfg.mode == 'gold':
    #         save_prefix = None
    #     else:
    #         save_prefix = num_saving
    #     out_curr = run_pipeline(validation_prompt, original_image, cfg, generator, pipeline, save_prefix=save_prefix, 
    #                 n_generations=cfg.num_validation_images,
    #                     # override_suffix=count
    #                 )
    #     all_out.extend(out_curr)
        
    # import csv
    # myFile = open('demo_file.csv', 'w')
    # writer = csv.DictWriter(myFile, fieldnames=list(all_out[0].keys()))
    # writer.writeheader()
    # writer.writerows(all_out)
    # myFile.close()
    # myFile = open('demo_file.csv', 'r')
    # print("The content of the csv file is:")
    # print(myFile.read())
    # myFile.close()
