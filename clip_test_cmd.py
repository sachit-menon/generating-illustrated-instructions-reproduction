 # %%

import torch
from trainer.datasetss.wikihow_dataset import WikiHowDataset, WikiHowDatasetConfig

from PIL import Image
import torchvision.transforms as transforms

from torch.nn import functional as F

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_CACHE'] = './models/hf_models'

from helpers import show_single_image, reshape_for_multiple_choice

from tqdm.auto import trange, tqdm
from einops import rearrange, repeat

import numpy as np
import pandas as pd

from easydict import EasyDict
import hydra

import random

def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(loaded_cfg):
    
    if hasattr(loaded_cfg, 'gpu_number'):
        gpu_number = loaded_cfg.gpu_number
    else:
        gpu_number = 0
        
    device = f'cuda:{gpu_number}'
    n_classes = 4
    
    if hasattr(loaded_cfg, 'metric'):
        metric_to_compute = loaded_cfg.metric
    else:
        metric_to_compute = 'goal' # options: 'goal', 'step'

    cfg = WikiHowDatasetConfig()
    cfg.batch_size = 100
    # if cfg.batch_size % n_classes != 0:
    #     raise ValueError('batch_size must be divisible by n_classes')
    # cfg.max_samples = 1000
    cfg.sequence_length = loaded_cfg.dataset.sequence_length
    cfg.random_subset = False
    cfg.return_goal_labels = metric_to_compute == 'goal'

    cfg.return_goal_method_id = True

    cfg.category_hierarchy_filter = 'Recipe'

    cfg.also_return_originals = True
    
    cfg.resume_from_checkpoint = "checkpoint-gstep70000"
    
    cfg.path = os.path.join('./', loaded_cfg.output_dir, "_dump", cfg.resume_from_checkpoint) + '/'
    
    # cfg.override_data_source = './outputs/1seq_rand_real/_dump/checkpoint-gstep34200/'
    # cfg.override_data_source = './outputs/4seq_real/_dump/checkpoint-gstep41100/'
    # cfg.use_rows_from_csv = './outputs/4seq_real/_dump/checkpoint-gstep41100/'
    # cfg.use_rows_from_csv = './outputs/4seq_real/_dump/checkpoint-gstep25000/'
    
    # check if it has the key first then if it's true
    if hasattr(loaded_cfg, 'gt') and loaded_cfg.gt:
        cfg.use_rows_from_csv = cfg.path
    else:
        cfg.override_data_source = cfg.path
    
    cfg.processor.tokenizer.pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5" # NOTE this only works bc uses the CLIP tokenizer
    cfg.append_method_to_goal = False



    # Shuffle the tensor
    torch.manual_seed(42)  # Set a seed for reproducibility
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # %%
    wh = WikiHowDataset(cfg, 'validation')
    wh.transform = _transform(224)
    
    results_fname = f'clip_{metric_to_compute}_metric_test'
    if cfg.override_data_source is not None:
        results_fname += '_generated'
    elif cfg.use_rows_from_csv is not None:
        results_fname += '_gt'




    import torchmetrics
    # n_classes = len(wh.unique_goals)
    # n_classes = cfg.batch_size
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes).to(device)
    acc2_metric = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes, top_k=2).to(device)

    # %%

    from transformers import CLIPProcessor, CLIPModel
    os.environ['TRANSFORMERS_CACHE'] = './models/hf_models'
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    import open_clip

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

    def tokenizer_dummy(text, **kwargs):
        out = EasyDict()
        out.input_ids = tokenizer(text)
        out.attention_mask = torch.ones_like(out.input_ids)
        return out

    wh.tokenizer = tokenizer_dummy

    dl = torch.utils.data.DataLoader(
        wh,
        shuffle=False,
        batch_size=cfg.batch_size,
        collate_fn=wh.collate_fn,
        num_workers=cfg.num_workers
    )


    idl = iter(dl)
    batch = next(idl)

    # %%

    model = model.to(device)
    model.eval()



    torch.set_grad_enabled(False)



    if metric_to_compute == 'goal':
        goal_tokens = wh.tokenize(wh.df.goal.unique().tolist())[0].to(device)
        goal_embeddings = model.encode_text(goal_tokens) # TODO check if needs normalization
        goal_embeddings = F.normalize(goal_embeddings, dim=1)
    else:
        labels = torch.arange(n_classes).to(device).tile(cfg.batch_size) 

    out_csv_fname = f'{results_fname}.csv'

    if cfg.override_data_source is not None:
        full_save_fname = os.path.join(cfg.override_data_source, out_csv_fname)
    elif cfg.use_rows_from_csv is not None:
        full_save_fname = os.path.join(cfg.use_rows_from_csv, out_csv_fname)

    # %%
    for batch_num, batch in enumerate(tqdm(dl)): # TODO loop over goals instead of goal_method?
    # for batch in dl: # TODO loop over goals instead of goal_method?
        pv = batch['edited_pixel_values'].to(device)
        
        if metric_to_compute == 'step':
            ii = batch['input_ids'][:,1:,:].to(device) # take only embeddings after the goal
            first_dim_indices = torch.arange(pv.shape[0]).unsqueeze(1).repeat(1, 4).flatten()
            # second_dim_indices = torch.randint(0,cfg.sequence_length, (pv.shape[0],4))
            # replace this with the first 4 indices
            # second_dim_indices = torch.arange(4).unsqueeze(0).repeat(pv.shape[0], 1)
            # replace this with 4 random unique indices 
            
            all_indices = []

            for _ in range(pv.shape[0]):
                indices = torch.randperm(cfg.sequence_length)[:4]
                all_indices.append(indices)

            second_dim_indices = torch.stack(all_indices, dim=0)
                
            # second_dim_indices = torch.randperm(cfg.sequence_length)[:4].unsqueeze(0).repeat(pv.shape[0], 1)
            indices = first_dim_indices.view(-1,4), second_dim_indices
            ii = ii[indices]
            pv = pv[indices]
        
        pv = rearrange(pv, 'b n c h w -> (b n) c h w')
        goal_method_id = np.array(batch['goal_method_id'])
        
        if metric_to_compute == 'step':
            repeat_n = n_classes
        else:
            repeat_n = cfg.sequence_length
            
        goal_method_id = repeat(goal_method_id, 'b -> (b n)', n=repeat_n)
        
        if metric_to_compute == 'goal':
            ii = batch['goal'].to(device)
            labels = batch['goal_labels'].to(device)
            labels = repeat(labels, 'b -> (b s)', s=cfg.sequence_length)
            ii = ii[batch['goals_used_mask'].bool()]
            labels = labels[batch['goals_used_mask'].bool()]
            pv = pv[batch['goals_used_mask'].bool()]
            goal_method_id = goal_method_id[batch['goals_used_mask'].bool()]
        
        
        
        if len(ii.shape) == 3:
            ii = rearrange(ii, 'b n t -> (b n) t')
            
        # out = model(ii, pv)['logits_per_text']
        # image_embeddings = model.get_image_features(pv)
        # text_embeddings = model.get_text_features(ii)
        
        
        image_embeddings = model.encode_image(pv)
        text_embeddings = model.encode_text(ii)
        
        
        image_embeddings = F.normalize(image_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        # similarity_matrix = (100.0 * image_embeddings @ text_embeddings.T)
        if metric_to_compute == 'goal':
            original_index, gie, gte, goal_method_id = reshape_for_multiple_choice(labels, image_embeddings, text_embeddings, goal_method_id)
            labels = torch.arange(n_classes).to(device).tile(gie.shape[0]).to(gie.device)
        else:
            if ii.shape[0] % n_classes != 0:
                print(f'batch {batch_num} not divisible by n_classes, skipping')
                continue
            
            if ii.shape[0] != labels.shape[0]:
                print(f'batch {batch_num} has {ii.shape[0]} inputs but {labels.shape[0]} labels, skipping')
                continue
            
            gie = rearrange(image_embeddings, '(b n) d -> b n d', n=n_classes)
            gte = rearrange(text_embeddings, '(b n) d -> b n d', n=n_classes)
        allsim = torch.einsum('b m d, b n d -> b m n', gie, gte)
        out = allsim.softmax(dim=-2).reshape(-1, n_classes) # use dim=-1 for 'pick text given image', dim=-2 for image given text
        # similarity_matrix = (100.0 * image_embeddings @ goal_embeddings.T)
        # out = similarity_matrix.softmax(dim=-1)
        # out = torch.softmax(out, dim=-1)
        
        preds = out.argmax(dim=-1)
        correct = (preds == labels)
        
        
        # break
        acc_metric.update(out, labels)
        acc2_metric.update(out, labels)
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'goal_method_id': goal_method_id,
            'preds': preds.cpu().numpy(),
            'labels': labels.cpu().numpy(),
            'correct': correct.cpu().numpy(),
        })
        
        # If first iteration, write with a header, otherwise append without writing the header.
        if batch_num == 0:
            df.to_csv(full_save_fname, index=False)
            print(f"Writing results to {full_save_fname}")
        else:
            df.to_csv(full_save_fname, mode='a', header=False, index=False)
        
        # if batch_num == 1:
        #     break
        # break

    print(acc_metric.compute().item())
    
if __name__ == '__main__':
    main()