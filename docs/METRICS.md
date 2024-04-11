## Metrics

Goal/Step Faithfulness:
These are measured with the CLIP model, with the script `clip_test_cmd.py`. Example usage:
`python clip_test_cmd.py '+gpu_number=0' '+metric=goal' --config-path --config-path /default/`

Cross-Image Consistency:
These are measured with the DINOv2 model, with the script `clip_image_test.py`. Example usage (after modifying appropriate variables in the script):
`python clip_image_test.py`


### Implementation differences vs original paper

Since the model in the paper builds upon a proprietary T2I model, we reimplement the same techniques here using the publicly available SD2.1 model.


The metrics for this open-source reimplementation are as follows:

| Metric              |   This repo   |   Original   |
|---------------------|------|------|
| Goal Faithfulness   | 83.5 | 74.3 |
| Step Faithfulness   | 70.1 | 61.5 |
| Cross-image Consistency   | 49.7 | 50.7 |

This result is with the pretrained model that creates up to 6 steps, based on Stable Diffusion v1.5 as a pretrained model, trained with the StackedDiffusion method detailed in the paper for 70000 steps. All other hyperparameters, etc are as described in the paper. 