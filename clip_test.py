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

def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

device = 'cuda:0'
n_classes = 4

cfg = WikiHowDatasetConfig()
cfg.batch_size = 100
if cfg.batch_size % n_classes != 0:
    raise ValueError('batch_size must be divisible by n_classes')
# cfg.max_samples = 1000
cfg.sequence_length = 1
cfg.random_subset = True
cfg.return_goal_labels = False
cfg.override_data_source = './'
# cfg.use_rows_from_csv = './'

cfg.processor.tokenizer.pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5" # NOTE this only works bc uses the CLIP tokenizer
cfg.append_method_to_goal = False

# Shuffle the tensor
torch.manual_seed(42)  # Set a seed for reproducibility

# %%
wh = WikiHowDataset(cfg, 'validation')
wh.transform = _transform(224)



dl = torch.utils.data.DataLoader(
    wh,
    shuffle=True,
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
from transformers import CLIPProcessor, CLIPModel
os.environ['TRANSFORMERS_CACHE'] = './models/hf_models'
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model = model.to(device)
model.eval()
torch.set_grad_enabled(False)


processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
labels = torch.arange(n_classes).to(device).tile(cfg.batch_size // n_classes) 
# goal_tokens = processor(wh.df.goal.unique().tolist(), padding='max_length', max_length=77, return_tensors="pt").to(device)
# goal_embeddings = model.get_text_features(**goal_tokens) # TODO check if needs normalization
# goal_embeddings = F.normalize(goal_embeddings, dim=1)


for batch_num, batch in enumerate(tqdm(dl)): # TODO loop over goals instead of goal_method?
# for batch in dl: # TODO loop over goals instead of goal_method?
    # labels = batch['goal_labels'].to(device)
    ii = batch['input_ids'][:,0,:].squeeze(1).to(device)
    pv = batch['edited_pixel_values'].squeeze(1).to(device)
    if ii.shape[0] % n_classes != 0:
        print(f'batch {batch_num} not divisible by n_classes, skipping')
        continue
        
    out = model(ii, pv)['logits_per_text']
    image_embeddings = F.normalize(model.get_image_features(pv), dim=1)
    text_embeddings = F.normalize(model.get_text_features(ii), dim=1)
    # similarity_matrix = (100.0 * image_embeddings @ text_embeddings.T)
    
    gie = rearrange(image_embeddings, '(b n) d -> b n d', n=n_classes)
    gte = rearrange(text_embeddings, '(b n) d -> b n d', n=n_classes)
    allsim = torch.einsum('b m d, b n d -> b m n', gie, gte)
    out = allsim.softmax(dim=-2).reshape(-1, 4) # use dim=-1 for 'pick text given image', dim=-2 for image given text
    # similarity_matrix = (100.0 * image_embeddings @ goal_embeddings.T)
    # out = similarity_matrix.softmax(dim=-1)
    # out = torch.softmax(out, dim=-1)
    acc_metric.update(out, labels)
    acc2_metric.update(out, labels)
    if batch_num == 10:
        break
    # break

print(acc_metric.compute().item())