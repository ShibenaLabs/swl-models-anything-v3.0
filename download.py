# In this file, we define download_model
# It runs during container build time to get model weights built into the container
import os
import time
import torch
from diffusers import StableDiffusionPipeline

def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time
    t1 = time.time()
    model_id = "swl-models/anything-v3.0"
    model = StableDiffusionPipeline.from_pretrained(model_id, revision=branch_name, torch_dtype=torch.float16)
    
    t2 = time.time()
    print("Download took - ",t2-t1,"seconds")

if __name__ == "__main__":
    download_model()

    from diffusers import StableDiffusionPipeline


branch_name= "diffusers"
pipe = pipe.to("cuda")

