import argparse
import os
import PIL
import requests
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None, # /data/home/sachit/.cache/huggingface/datasets/generator/default-eda69c73494a41c4/0.0.0/
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--positional_encoding_type",
        type=str,
        default='sinusoidal',
        help=(
            "Type of positional encoding. Default is sinusoidal."
        ),
    )
    parser.add_argument(
        "--broadcast_positional_encoding",
        action="store_true",
        default=True,
        help="Whether to use positional encoding repeated across tokens in same step",
    )
    parser.add_argument(
        "--prepend_goal_to_steps",
        action="store_true",
        default=False,
        help="Prepend goal text to all steps or not",
    )
    # parser.add_argument(
    #     "--broadcast_positional_encoding",
    #     action="store_true",
    #     default=False,
    #     help="Whether to use positional encoding repeated across tokens in same step",
    # )
    parser.add_argument(
        "--category_hierarchy_filter",
        type=str,
        default=None,
        help="Whether to filter dataset on any category, eg 'Recipe'",
    )
    parser.add_argument(
        "--condition_image",
        type=str,
        default="first",
        help="Which image to condition on; options are 'first', 'last', or 'black'",
    )
    parser.add_argument(
        "--tile_with",
        type=str,
        default="last",
        help="Which image to tile for missing; options are 'first', 'last', or 'black'",
    )
    parser.add_argument(
        "--tile_text_with",
        type=str,
        default="last",
        help="Which image to tile for missing; options are 'first', 'last', or 'null'",
    )
    parser.add_argument(
        "--descr_usage",
        type=str,
        default=None,
        help="How to use the WikiHow descriptions if at all; choices are 'append_text', 'append_encoding', and None",
    )
    parser.add_argument(
        "--original_image_column",
        type=str,
        default="input_image",
        help="The column of the dataset containing the original image on which edits where made.",
    )
    parser.add_argument(
        "--edited_image_column",
        type=str,
        default="edited_image",
        help="The column of the dataset containing the edited image.",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default="edit_prompt",
        help="The column of the dataset containing the edit instruction.",
    )
    parser.add_argument(
        "--val_image_url",
        type=str,
        default=None,
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="instruct-pix2pix-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=3,
        help="Number of images in the sequence to be generated (including the first condition, if applicable)."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    
    # lora args
    parser.add_argument("--use_lora", action="store_true", help="Whether to use Lora for parameter efficient tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="Lora rank, only used if use_lora is True")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Lora alpha, only used if use_lora is True")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Lora dropout, only used if use_lora is True")
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        help="Bias type for Lora. Can be 'none', 'all' or 'lora_only', only used if use_lora is True",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


from PIL import Image, ImageDraw, ImageFont
def create_diagram(goal, condition, output_texts, output_images):
    if isinstance(output_texts, str):
        print("output_texts in diagram creation was str instead of list. Converting, but check.")
        output_texts = [output_texts]
    # account for mismatch in number of output texts and output images
    if len(output_texts) > len(output_images):
        print(f"There are more output texts than output images. The last {len(output_texts) - len(output_images)} output texts will not be shown.")
        output_images += [Image.new('RGB', (1, 1), 'white') for _ in range(len(output_texts) - len(output_images))]
    elif len(output_texts) < len(output_images):
        print(f"There are more output images than output texts. The last {len(output_images) - len(output_texts)} output images will not be shown.")
        output_texts = list(output_texts)
        output_texts += ['' for _ in range(len(output_images) - len(output_texts))]
        
    
    x_offset = 40
    y_offset = 30
    x_margin = 20
    y_margin = 20
    # Calculate the dimensions of the final image
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    total_width = max(condition.width, max([img.width + x_offset for img in output_images]))
    # Get the text width of all texts including the goal
    text_widths = [ImageDraw.Draw(Image.new('RGB', (1, 1), 'white')).textsize(output_text, font)[0] for output_text in output_texts]
    text_widths.append(ImageDraw.Draw(Image.new('RGB', (1, 1), 'white')).textsize(goal, font)[0])
    # Calculate the max width including the images and the texts
    total_width = max(total_width, max(text_widths)) + x_margin
    
    total_height = condition.height + sum([img.height for img in output_images]) + ((len(output_texts)+1) * y_offset) + y_margin

    # Create a white canvas with the calculated dimensions
    result = Image.new("RGB", (total_width, total_height), color="white")
    
    # Get a drawing context
    draw = ImageDraw.Draw(result)

    # Load the default font
    font.size *= 1.4
    # Draw the goal text
    draw.text((0, 0), goal, font=font, fill="black")


    # Paste the condition image
    result.paste(condition, (x_offset, y_offset))
    y_offset += condition.height

    # Draw output texts and output images
    for output_img, output_txt in zip(output_images, output_texts):
        draw.text((0, y_offset), output_txt, font=font, fill="black")
        y_offset += 30
        result.paste(output_img, (x_offset, y_offset))
        y_offset += output_img.height

    return result

import re
def split_text(text):
    # Using a regular expression to find natural split points
    split_points = re.compile(r'(\d+\. ?|Goal: ?)', re.I)
    parts = split_points.split(text)

    # Grouping the result into a list
    steps = ["".join(part) for part in zip(parts[1::2], parts[2::2])]
    
    return steps


# Viz
import torch
import matplotlib.pyplot as plt
# stats = ((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225)) 
stats = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

def denormalize(images, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225)):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means

def show_single_image(image, save_path=None, size='small'):
    if size == 'small':
        figsize = (3, 3)
    else:
        figsize = (12, 12)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks([]); ax.set_yticks([])
    denorm_image = denormalize(image.unsqueeze(0).cpu(), *stats)
    ax.imshow(denorm_image.squeeze().permute(1, 2, 0).clamp(0,1))
    
    if save_path is None:
        plt.show()
    
    # save image if save_path is provided
    if save_path is not None:
        # make path if it does not exist
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
        
from PIL import Image

def gaussian_noise_image_rescaled(width, height, mean=0, std=1, mode='L'):
    """
    Generate an image with Gaussian noise and rescale to 0-255 range.

    Parameters:
    - width, height: Dimensions of the image.
    - mean, std: Mean and standard deviation of the Gaussian noise.
    - mode: Color mode. 'L' for grayscale, 'RGB' for color.

    Returns:
    - An image object from PIL.
    """

    # Generate Gaussian noise
    if mode == 'L':
        noise_array = np.random.normal(mean, std, (height, width))
    elif mode == 'RGB':
        noise_array = np.random.normal(mean, std, (height, width, 3))
    else:
        raise ValueError("Unsupported mode. Choose either 'L' or 'RGB'.")

    # Rescale noise from 0-1 to 0-255
    noise_array = ((noise_array - noise_array.min()) / 
                   (noise_array.max() - noise_array.min()) * 255).astype(np.uint8)

    # Convert array to PIL Image
    noise_image = Image.fromarray(noise_array, mode)

    return noise_image


def reshape_for_multiple_choice(labels, image_embeddings, text_embeddings, goal_method_id):
    n = labels.size(0)
    
    # Adjust size if n is not divisible by 4
    if n % 4 != 0:
        padding_size = 4 - (n % 4)
        random_indices = torch.randint(0, n, (padding_size,))
        labels = torch.cat([labels, labels[random_indices]])
        image_embeddings = torch.cat([image_embeddings, image_embeddings[random_indices]])
        text_embeddings = torch.cat([text_embeddings, text_embeddings[random_indices]])
        
        gmi_to_add = goal_method_id[random_indices]
        if len(random_indices) == 1:
            gmi_to_add = np.array([gmi_to_add])
        
        goal_method_id = np.concatenate([goal_method_id, gmi_to_add])
        # Update n
        n += padding_size

    # Initialize the new tensors
    labels_new = torch.zeros((n // 4, 4), dtype=torch.long)
    image_embeddings_new = torch.zeros((n // 4, 4, image_embeddings.shape[-1]))
    text_embeddings_new = torch.zeros((n // 4, 4, text_embeddings.shape[-1]))
    
    # Vectorized mask for all different labels
    diff_labels_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
    
    for i in range(0, n, 4):
        # Get indices of rows that have different labels
        diff_label_indices = torch.nonzero(diff_labels_mask[i]).squeeze()
        
        # Randomly select 3 of them
        selected_indices = diff_label_indices[torch.randperm(diff_label_indices.size(0))[:3]]
        
        # Form the group
        group_indices = torch.cat([torch.tensor([i]).to(selected_indices.device), selected_indices])
        
        # Assign to new tensors
        labels_new[i // 4] = labels[group_indices]
        image_embeddings_new[i // 4] = image_embeddings[group_indices]
        text_embeddings_new[i // 4] = text_embeddings[group_indices]
    
    return labels_new.to(image_embeddings.device), image_embeddings_new.to(image_embeddings.device), text_embeddings_new.to(image_embeddings.device), goal_method_id

def reshape_for_multiple_choice_step(labels, image_embeddings, text_embeddings, goal_method_id):
    n = labels.size(0)
    
    # Adjust size if n is not divisible by 4
    if n % 4 != 0:
        padding_size = 4 - (n % 4)
        random_indices = torch.randint(0, n, (padding_size,))
        labels = torch.cat([labels, labels[random_indices]])
        image_embeddings = torch.cat([image_embeddings, image_embeddings[random_indices]])
        text_embeddings = torch.cat([text_embeddings, text_embeddings[random_indices]])
        
        gmi_to_add = goal_method_id[random_indices]
        if len(random_indices) == 1:
            gmi_to_add = np.array([gmi_to_add])
        
        goal_method_id = np.concatenate([goal_method_id, gmi_to_add])
        # Update n
        n += padding_size

    # Initialize the new tensors
    labels_new = torch.zeros((n // 4, 4), dtype=torch.long)
    image_embeddings_new = torch.zeros((n // 4, 4, image_embeddings.shape[-1]))
    text_embeddings_new = torch.zeros((n // 4, 4, text_embeddings.shape[-1]))
    
    # Vectorized mask for all different labels
    diff_labels_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
    
    for i in range(0, n, 4):
        # Get indices of rows that have different labels
        diff_label_indices = torch.nonzero(diff_labels_mask[i]).squeeze()
        
        # Randomly select 3 of them
        selected_indices = diff_label_indices[torch.randperm(diff_label_indices.size(0))[:3]]
        
        # Form the group
        group_indices = torch.cat([torch.tensor([i]).to(selected_indices.device), selected_indices])
        
        # Assign to new tensors
        labels_new[i // 4] = labels[group_indices]
        image_embeddings_new[i // 4] = image_embeddings[group_indices]
        text_embeddings_new[i // 4] = text_embeddings[group_indices]
    
    return labels_new.to(image_embeddings.device), image_embeddings_new.to(image_embeddings.device), text_embeddings_new.to(image_embeddings.device), goal_method_id

def generate_filename(sample_num, goal_method_id, i, cfg):
    """Generate filename based on provided parameters."""
    output_image_fname = f"{sample_num}-{goal_method_id}_{i}.png"
    output_image_fname = output_image_fname.replace('/', '_')
    output_image_fname = os.path.join(cfg.diagram_dir, output_image_fname)
    return output_image_fname

def batch_has_existing_files(batch, cfg):
    """Check if batch has entries with filenames that already exist."""
    curr_batch_size = batch['input_ids'].shape[0]
    goal_method_ids = batch['goal_method_id']
    
    for sample_num in range(cfg.num_validation_images):
        for index in range(curr_batch_size):
            for i in range(cfg.model.sequence_length):
                curr_goal_method_id = goal_method_ids[index]
                output_image_fname = generate_filename(sample_num, curr_goal_method_id, i, cfg)
                if os.path.exists(output_image_fname):
                    return True
    return False