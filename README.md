# Generating Illustrated Instructions

This repository is a reproduction of the method from the paper ``Generating Illustrated Instructions."

[**Generating Illustrated Instructions**](https://arxiv.org/abs/2312.04552)

Sachit Menon, Ishan Misra, Rohit Girdhar

[website](https://facebookresearch.github.io/IllustratedInstructions/) | [arxiv](https://arxiv.org/abs/2312.04552) | [bibtex](#citation)

TL;DR: A picture is worth a thousand words. LLMs are great at answering "how to" questions, but adding images can improve their answers significantly. We combine LLMs with diffusion models and present way to generate custom visual "how-to" answers for a user’s question. Our method beats prior SOTA significantly in human evals.

https://github.com/sachit-menon/generating-illustrated-instructions/assets/21482705/762ddb5d-e229-4432-9135-897dfdfda7a6



## QUICKSTART
To enable getting started with inference as quickly as possible, we provide a HuggingFace Diffusers pipeline for convenience.

```
pip install diffusers
```
then
```
pipeline = DiffusionPipeline.from_pretrained("sachit-menon/illustrated_instructions", custom_pipeline="snt_pipeline", trust_remote_code=True)
```
This will download a pretrained model and the required code to perform inference, ready to use as usual.
A full example of inference with this setup can be found in `quick_inference.py`.

If you would like to expand on the method in more detail, proceed as below. 

## Installation

Example conda environment setup:
```
conda create --name illust python=3.9 -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 mpi4py -c pytorch -c nvidia
pip install -r requirements.txt
```

## Data
We reuse the VGSI dataset from https://github.com/YueYANG1996/wikiHow-VGSI. Please see [DATA.md](docs/DATA.md) for further instructions. 

## Training

To train your own models, see `trainer/scripts/train.py`. Example usage:
```
accelerate launch --config_file ./trainer/ds8.yaml --use_deepspeed ./trainer/scripts/train.py +experiment=sd accelerator.project_name=test
```

Further details can be found in [TRAIN.md](docs/TRAIN.md).

## Metrics

See [METRICS.md](docs/METRICS.md) for how to compute the metrics the paper introduces (goal faithfulness, step faithfulness, and cross-image consistency), as well as for implementation differences between this reimplementation and the original work.

## Intended Usage Note

This model is intended for research purposes only. Its primary use case is to generate full instructional articles of recipes. Further, it has not been trained on images of humans (past pretraining). As this model is based on Stable Diffusion, it likely inherits some of its biases and safety issues.


## Citation

```bibtex
@article{menon2023generating,
  title={Generating Illustrated Instructions},
  author={Sachit Menon and Ishan Misra and Rohit Girdhar},
  journal={arXiv preprint arXiv:2312.04552},
  year={2023}
}
```
