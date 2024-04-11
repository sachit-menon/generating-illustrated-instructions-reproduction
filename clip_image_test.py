# %%

import torch
from trainer.datasetss.wikihow_dataset import WikiHowDataset, WikiHowDatasetConfig

from PIL import Image
import torchvision.transforms as transforms

from torch.nn import functional as F

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from helpers import show_single_image

from tqdm.auto import trange, tqdm
from einops import rearrange, repeat

import numpy as np
import pandas as pd


def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
def _dino_transform(n_px):
    return transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

def extract_lower_triangle(tensor):
    """
    Extracts the lower triangle of each matrix in a tensor of shape (b, n, n) 
    and returns a tensor of shape (b, m) where m is the number of elements in the 
    lower triangle (excluding the diagonal).
    """
    b, n, _ = tensor.shape
    indices = torch.tril_indices(n, n, offset=-1)
    lower_triangle = tensor[:, indices[0], indices[1]]
    return lower_triangle

device = 'cuda:5'
n_classes = 4

cfg = WikiHowDatasetConfig()
cfg.batch_size = 100
# if cfg.batch_size % n_classes != 0:
#     raise ValueError('batch_size must be divisible by n_classes')
# cfg.max_samples = 1000
cfg.sequence_length = 6
cfg.random_subset = False
cfg.return_goal_labels = False

cfg.return_goal_method_id = True
cfg.category_hierarchy_filter = 'Recipe'

cfg.override_data_source = './outputs/test2/default/_dump/checkpoint-gstep70000/'
# cfg.use_rows_from_csv = './outputs/4seq_real/_dump/checkpoint-gstep25000/'

cfg.processor.tokenizer.pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5" # NOTE this only works bc uses the CLIP tokenizer
cfg.append_method_to_goal = False



# Shuffle the tensor
torch.manual_seed(42)  # Set a seed for reproducibility

# %%
wh = WikiHowDataset(cfg, 'validation')
wh.transform = _dino_transform(224)


# %%


results_fname = 'clip_step_test'
if cfg.override_data_source is not None:
    results_fname += '_generated'
elif cfg.use_rows_from_csv is not None:
    results_fname += '_gt'


dl = torch.utils.data.DataLoader(
    wh,
    shuffle=False,
    batch_size=cfg.batch_size,
    collate_fn=wh.collate_fn,
    num_workers=cfg.num_workers
)

import torchmetrics
# n_classes = len(wh.unique_goals)
# n_classes = cfg.batch_size
acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes).to(device)
acc2_metric = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes, top_k=2).to(device)

# %%

# from transformers import CLIPProcessor, CLIPModel
# os.environ['TRANSFORMERS_CACHE'] = './models/hf_models'
# # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# import open_clip

# model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
# tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')



import os
os.environ['TRANSFORMERS_CACHE'] = './models/hf_models'
# Load model directly
# from transformers import AutoFeatureExtractor, AutoModel

# extractor = AutoFeatureExtractor.from_pretrained("facebook/dino-vitb8")

# model = AutoModel.from_pretrained("facebook/dino-vitb8")

# extractor = extractor.to(device)
# extractor.eval()

model = model.to(device)
model.eval()



torch.set_grad_enabled(False)




# labels = torch.arange(n_classes).to(device).tile(cfg.batch_size) 
# goal_tokens = processor(wh.df.goal.unique().tolist(), padding='max_length', max_length=77, return_tensors="pt").to(device)
# goal_embeddings = model.get_text_features(**goal_tokens) # TODO check if needs normalization
# goal_embeddings = F.normalize(goal_embeddings, dim=1)

out_csv_fname = f'{results_fname}.csv'

if cfg.override_data_source is not None:
    full_save_fname = os.path.join(cfg.override_data_source, out_csv_fname)
elif cfg.use_rows_from_csv is not None:
    full_save_fname = os.path.join(cfg.use_rows_from_csv, out_csv_fname)


means = []
for batch_num, batch in enumerate(tqdm(dl)): # TODO loop over goals instead of goal_method?
# for batch in dl: # TODO loop over goals instead of goal_method?
    # labels = batch['goal_labels'].to(device)
    ii = batch['input_ids'][:,1:,:].to(device) # take only embeddings after the goal
    pv = batch['edited_pixel_values'].to(device)
    goal_method_id = np.array(batch['goal_method_id'])
    
    
    goal_method_id = repeat(goal_method_id, 'b -> (b n)', n=ii.shape[1])
    # ii = rearrange(ii, 'b n t -> (b n) t')
    pv = rearrange(pv, 'b n c h w -> (b n) c h w')
    
    if pv.shape[0] % n_classes != 0:
        print(f'batch {batch_num} not divisible by n_classes, skipping')
        continue
    
    # if pv.shape[0] != labels.shape[0]:
    #     print(f'batch {batch_num} has {pv.shape[0]} inputs but {labels.shape[0]} labels, skipping')
    #     continue
        
        
    # out = model(ii, pv)['logits_per_text']
    # image_embeddings = model.get_image_features(pv)
    # text_embeddings = model.get_text_features(ii)
    image_embeddings = model(pv)
    # image_embeddings = extractor(pv)['pixel_values']
    # image_embeddings = model.encode_image(pv)
    # text_embeddings = model.encode_text(ii)
    
    
    # image_embeddings = F.normalize(image_embeddings, dim=1)
    # # text_embeddings = F.normalize(text_embeddings, dim=1)
    # # similarity_matrix = (100.0 * image_embeddings @ text_embeddings.T)
    
    gie = rearrange(image_embeddings, '(b n) d -> b n d', n=n_classes)
    # Compute pairwise distances

    gie_expanded_1 = gie.unsqueeze(2)  # shape: b x n x 1 x d
    gie_expanded_2 = gie.unsqueeze(1)  # shape: b x 1 x n x d

    squared_diff = (gie_expanded_1 - gie_expanded_2)**2
    allsiml2 = torch.sqrt(squared_diff.sum(-1))
    
    lt = extract_lower_triangle(allsiml2)
    
    # # gte = rearrange(text_embeddings, '(b n) d -> b n d', n=n_classes)
    # allsim = torch.einsum('b m d, b n d -> b m n', gie, gie)
    
    
    # L2 norm instead
     
    
    means.append(lt.mean())
    # means.append(allsiml2.mean())
    # break
    # out = allsim.softmax(dim=-2).reshape(-1, n_classes) # use dim=-1 for 'pick text given image', dim=-2 for image given text
    # # similarity_matrix = (100.0 * image_embeddings @ goal_embeddings.T)
    # # out = similarity_matrix.softmax(dim=-1)
    # # out = torch.softmax(out, dim=-1)
    
    # preds = out.argmax(dim=-1)
    # correct = (preds == labels)
    
    
    # # break
    # acc_metric.update(out, labels)
    # acc2_metric.update(out, labels)
    
    # # Convert to DataFrame
    # df = pd.DataFrame({
    #     'goal_method_id': goal_method_id,
    #     'preds': preds.cpu().numpy(),
    #     'labels': labels.cpu().numpy(),
    #     'correct': correct.cpu().numpy(),
    # })
    
    # # If first iteration, write with a header, otherwise append without writing the header.
    # if batch_num == 0:
    #     df.to_csv(full_save_fname, index=False)
    #     print(f"Writing results to {full_save_fname}")
    # else:
    #     df.to_csv(full_save_fname, mode='a', header=False, index=False)
    
    # if batch_num == 1:
    #     break
    # break

print(torch.mean(torch.tensor(means)))

# print(acc_metric.compute().item())