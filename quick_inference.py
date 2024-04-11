
# %%
import torch
from diffusers import DiffusionPipeline
import os

# only needed if using GPT-4 for LLM
import openai 
import re

# %%
cfg = {}
cfg['manual_steps'] = True
cfg['gpu_number'] = '0'
cfg['device'] = f'cuda:{cfg["gpu_number"]}'
cfg['save_dir'] = './quick_inference'

pipeline = DiffusionPipeline.from_pretrained("sachit-menon/illustrated_instructions", custom_pipeline="snt_pipeline", trust_remote_code=True)
tokenizer = pipeline.model.tokenizer
generator = torch.Generator(device=cfg['device']).manual_seed(0)

# %%

pipeline = pipeline.to(cfg['device'])



# %%
goal = "How to Make a Delicious Apple Crisp"
prompt = f"Goal: {goal}."

if cfg['manual_steps']:
    step_texts = [
        "Preheat oven to 350°F (175°C).",
        "Peel, core, and slice apples; place in a baking dish.",
        "Mix sugar, cinnamon, and a pinch of salt; sprinkle over apples.",
        "Combine oats, flour, brown sugar, and butter; crumble over apples.",
        "Bake for 45 minutes or until golden brown.",
        "Serve warm with vanilla ice cream.",
    ]
else:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    llm_prompt = f"Provide instructions for the goal: {goal}. Write your response in the form of a goal 'Goal: GOAL, METHOD' followed by concise numbered headline steps, each one line, without any other text. Use at most 6 steps."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": llm_prompt},
        ]
    )
    out = response['choices'][0]['message']['content']
    step_texts = re.findall(r'\d+\.\s(.*?)\s*(?=\d+\.|$)', out)
    

for i, step in enumerate(step_texts):
    step_texts[i] = f'{i}. {step}'

input_texts = [
    prompt,
]

input_texts.extend(step_texts)

prompts = tokenizer(
                input_texts,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids

prompts = prompts.unsqueeze(0).to(cfg['device'])



for sample_num in range(10):
    with torch.autocast(device_type=str(cfg['device']).replace(f":{cfg.gpu_number}", ""), 
                        dtype=torch.bfloat16, 
                        enabled= True):
        output_images = pipeline(
            prompts,
            image=torch.zeros((1,3,256,256)).to(cfg['device']) + 0.0039, # vestigial, ignore
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=7,
            generator=generator,
        ).images
        

    for i in range(cfg.model.sequence_length):
        curr_goal_method_id = input_texts[0].replace(' ', '_').replace('/', '_')
        # pattern is samplenum-originalfname.png
        output_image_fname = f"{sample_num}-{curr_goal_method_id}_{input_texts[i+1]}_{i}.png"
        output_image_fname = output_image_fname.replace('/', '_')
        save_dir = cfg['save_dir']
        # make the save_dir
        os.makedirs(save_dir, exist_ok=True)
        output_image_fname = os.path.join(save_dir, output_image_fname)
        output_images[i].save(output_image_fname)