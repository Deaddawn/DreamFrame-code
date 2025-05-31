import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:50"
os.environ['FORCE_MEM_EFFICIENT_ATTN'] = "1"
os.environ['CUDA_VISIBLE_DEVICES']="0"
from diffusers import DiffusionPipeline,DDIMScheduler,DDPMScheduler
import torch
from PIL import Image,ImageEnhance
import torchvision.transforms as T
from tqdm import tqdm
import random
import numpy as np
import re
import json
import argparse

def set_XLembedding(base,emb,token="my"):
    with torch.no_grad():            
        # Embeddings[tokenNo] to learn
        tokens=base.components["tokenizer"].encode(token)
        assert len(tokens)==3, "token is not a single token in 'tokenizer'"
        tokenNo=tokens[1]
        tokens=base.components["tokenizer_2"].encode(token)
        assert len(tokens)==3, "token is not a single token in 'tokenizer_2'"
        tokenNo2=tokens[1]
        embs=base.components["text_encoder"].text_model.embeddings.token_embedding.weight
        embs2=base.components["text_encoder_2"].text_model.embeddings.token_embedding.weight
        assert embs[tokenNo].shape==emb["emb"].shape, "different 'text_encoder'"
        assert embs2[tokenNo2].shape==emb["emb2"].shape, "different 'text_encoder_2'"
        embs[tokenNo]=emb["emb"].to(embs.dtype).to(embs.device)
        embs2[tokenNo2]=emb["emb2"].to(embs2.dtype).to(embs2.device)

def load_XLembedding(base,token="my",path="path"):
    emb=torch.load(path)
    set_XLembedding(base,emb,token)




if __name__=='__main__':

    parser = argparse.ArgumentParser(description="style embedding generate")
    parser.add_argument("--js_path", default='./story_info_0.json', type=str, help="js_path")
    parser.add_argument("--embed_path", default='nothing', type=str, help="embed_path")
    parser.add_argument("--keyword", default='nothing', type=str, help="keyword")
    parser.add_argument("--save_path", default='nothing', type=str, help="save_path")
    opt = parser.parse_args()


    base_path="./model/stable-diffusion-xl-base-1.0"
    refiner_path="./model/stable-diffusion-xl-refiner-1.0"
    device=torch.device('cuda')


    base = DiffusionPipeline.from_pretrained(
    base_path, 
    torch_dtype=torch.float16, #torch.bfloat16
    variant="fp32", 
    use_safetensors=True,
    add_watermarker=False,
    ).to(device)
    base.enable_xformers_memory_efficient_attention()
    torch.set_grad_enabled(False)


    refiner = DiffusionPipeline.from_pretrained(
    refiner_path,
    text_encoder_2=base.text_encoder_2,  
    vae=base.vae,
    torch_dtype=torch.float16,
    variant="fp32",
    use_safetensors=True,
    add_watermarker=False,
    ).to(device)
    refiner.enable_xformers_memory_efficient_attention()

    negative_prompt="sketches, (worst quality:2), (low quality:2), (normal quality:2), ((non-symmetrical eyes)), chromatic aberration, lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, bad anatomy, DeepNegative, facing away, tilted head, lowres, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed, mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot, totem, totem body, double body"
    n_steps=40
    high_noise_frac=.75


    
    load_XLembedding(base,token=opt.keyword,path=opt.embed_path)

    f=open(opt.js_path,'r')
    js=json.load(f)
    f.close()

    img=[im[1] for im in js['img_diag'] if im[0]=='frame']

    img_path=opt.save_path
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    else:
        pass

    cc=1
    for seed,sample_prompt in zip([x for x in range(20,20+len(img))],img): 
        prompt=sample_prompt
        with torch.no_grad():    
            torch.manual_seed(seed)
            image = base(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                output_type="latent",
            ).images
            image = refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=n_steps,
                denoising_start=high_noise_frac,
                image=image        
            ).images
            image[0].save(os.path.join(img_path,str(cc)+'.png'))
            
            cc+=1