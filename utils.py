import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from PIL import Image


def is_interactive() -> bool:
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
        else:
            return False
    except NameError:
        return False  # Probably standard Python interpreter


def denormalize(images, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225)):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


def show_single_image(image, denormalize_stats=None, bgr_image=False, save_path=None, size='small'):
    if size == 'size_img':
        figsize = (image.shape[2]/100, image.shape[1]/100)  # The default dpi of plt.savefig is 100
    elif size == 'small':
        figsize = (4, 4)
    else:
        figsize = (12, 12)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if denormalize_stats is not None:
            image = denormalize(image.unsqueeze(0), *denormalize_stats)
        if image.dtype == torch.float32:
            image = image.clamp(0, 1)
        ax.imshow(image.squeeze(0).permute(1, 2, 0))
    else:
        if bgr_image:
            image = image[..., ::-1]
        ax.imshow(image)

    if save_path is None:
        plt.show()
    # save image if save_path is provided
    if save_path is not None:
        # make path if it does not exist
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
        
  
import re
def extract_argparse_info(argparse_string):
    arguments = re.findall(r'parser\.add_argument\((.*?)\).', argparse_string, re.DOTALL)
    extracted_info = []

    for argument in arguments:        
        arg = re.search(r"--([\w_]+)", argument)
        arg_type = re.search(r"type=(\w+)", argument)
        required = re.search(r"required=(.+?),", argument)
        default = re.search(r"default=(.+?),", argument)
        help_texts = re.findall(r'help=\((.*)', argument, re.DOTALL) + re.findall(r"help=\"(.*?)\"", argument, re.DOTALL)
        final_help_text = ' '.join([t.strip().replace('\n', ' ') for help_text in help_texts for t in help_text.split('"')])
        # if 'validation_epochs' in arg.group(1):
        #     break
        
        extracted_info.append({
            "argument": arg.group(1) if arg else None,
            "type": arg_type.group(1) if arg_type else None,
            "required": (required.group(1).strip() == "True") if required else False,
            "default": default.group(1).strip() if default else None,
            "help": final_help_text if final_help_text else None
        })
    return extracted_info

def extracted_info_to_hydra_config(argdictlist):
    result = ""
    for argdict in argdictlist:
        param_name = argdict['argument']
        param_type = argdict['type']
        param_required = argdict['required']
        param_default = argdict['default']
        param_help = argdict['help']
        
        param_hydra_value = ""
        
        if param_required and param_default is None:
            param_hydra_value = "???"
        elif param_default is not None:
            param_hydra_value = param_default
        else:
            param_hydra_value = "null"
        
        result += f"# {param_help}\n"
        result += f"{param_name}: {param_default}\n\n"
    return result


# def sv(x, csc=False, index=0):
#     if csc: x = x/csc
#     self.image_processor.postprocess(self.vae.decode(x/self.vae.config.scaling_factor)[0])[index].save('test.png')
# svc = lambda x: self.image_processor.postprocess(self.vae.decode(x/self.vae.config.scaling_factor)[0])[0].save('test.png')