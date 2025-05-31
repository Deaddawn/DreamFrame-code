import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:50"
os.environ['FORCE_MEM_EFFICIENT_ATTN'] = "0"
from diffusers import DiffusionPipeline,DDPMScheduler
import torch
from PIL import Image,ImageEnhance
import torchvision.transforms as T
from tqdm import auto
import random
import argparse

def force_training_grad(model,bT=True,bG=True):
    model.training=bT
    model.requires_grad_=bG
    for module in model.children():
        force_training_grad(module,bT,bG)
        
def load_imgs(path,wh=(1024,1024),flip=True,preview=(64,64)):
    files=list()
    imgs=list()
    PILimgs=list()
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if (f.endswith(".jpg") or f.endswith(".JPG") or f.endswith(".png") or f.endswith(".JPEG") or f.endswith(".jpeg"))]:
            fname = os.path.join(dirpath, filename)
            files.append(fname)
    for f in files:
        img = Image.open(f).convert("RGB")
        img = T.RandomAutocontrast(p=1.0)(img)
        img = T.Resize(wh, interpolation=T.InterpolationMode.LANCZOS)(img)
        #img = ImageEnhance.Contrast(T.RandomAutocontrast(p=1.0)(img)).enhance(5.0)
        PILimgs.append(T.Resize(preview, interpolation=T.InterpolationMode.LANCZOS)(img))
        img0 = T.ToTensor()(img)
        img0 = img0 *2.- 1.0
        imgs.append(img0[None].clip(-1.,1.))
        # plus horizontally mirrowed
        if flip:
            img0 = T.RandomHorizontalFlip(p=1.0)(img0)  
            imgs.append(img0[None].clip(-1.,1.)) 
            img = T.RandomHorizontalFlip(p=1.0)(img)
            PILimgs.append(T.Resize(preview, interpolation=T.InterpolationMode.LANCZOS)(img))
    return imgs,PILimgs

def make_grid(imgs):
    n=len(imgs)
    cols=1
    while cols*cols<n:
        cols+=1
    rows=n//cols+int(n%cols>0)
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))  
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def save_XLembedding(emb,embedding_file="myToken.pt",path="./Embeddings/"):
    torch.save(emb,path+embedding_file)

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

def load_XLembedding(base,token="my",embedding_file="myToken.pt",path="./Embeddings/"):
    emb=torch.load(path+embedding_file)
    set_XLembedding(base,emb,token)



def XL_textual_inversion(base,imgs,prompts,prompt_variations=None,token="my",start_token=None,negative_prompt=None,learning_rates=[(5,1e-3),(10,9e-4),(20,8e-4),(35,7e-4),(55,6e-4),(80,5e-4),(110,4e-4),(145,3e-4)],intermediate_steps=9):
    
    XLt1=base.components["text_encoder"]
    XLt2=base.components["text_encoder_2"]
    XLtok1=base.components["tokenizer"]
    XLtok2=base.components["tokenizer_2"]
    XLunet=base.components["unet"]
    XLvae=base.components['vae']
    XLsch=base.components['scheduler']
    base.upcast_vae() # vae does not work correctly in 16 bit mode -> force fp32
    
    # Check Scheduler
    schedulerType=XLsch.config.prediction_type
    assert schedulerType in ["epsilon","sample"], "{} scheduler not supported".format(schedulerType)

    # Embeddings to Finetune
    embs=XLt1.text_model.embeddings.token_embedding.weight
    embs2=XLt2.text_model.embeddings.token_embedding.weight

    with torch.no_grad():       
        # Embeddings[tokenNo] to learn
        tokens=XLtok1.encode(token)
        assert len(tokens)==3, "token is not a single token in 'tokenizer'"
        tokenNo=tokens[1]
        tokens=XLtok2.encode(token)
        assert len(tokens)==3, "token is not a single token in 'tokenizer_2'"
        tokenNo2=tokens[1]            

        # init Embedding[tokenNo] with noise or with a copy of an existing embedding
        if start_token=="randn_like" or start_token==None:
            # Original value range: [-0.5059,0.6538] # regular [-0.05,+0.05]
            embs[tokenNo]=(torch.randn_like(embs[tokenNo])*.01).clone() # start with [-0.04,+0.04]
            # Original value range 2: [-0.6885,0.1948] # regular [-0.05,+0.05]
            embs2[tokenNo2]=(torch.randn_like(embs2[tokenNo2])*.01).clone() # start [-0.04,+0.04]
            startNo="~"
            startNo2="~"
        else:  
            tokens=XLtok1.encode(start_token)
            assert len(tokens)==3, "start_token is not a single token in 'tokenizer'"
            startNo=tokens[1]
            tokens=XLtok2.encode(start_token)
            assert len(tokens)==3, "start_token is not a single token in 'tokenizer_2'"
            startNo2=tokens[1]
            embs[tokenNo]=embs[startNo].clone()
            embs2[tokenNo2]=embs2[startNo2].clone()

        # Make a copy of all embeddings to keep all but the embedding[tokenNo] constant 
        index_no_updates = torch.arange(len(embs)) != tokenNo
        orig=embs.clone()
        index_no_updates2 = torch.arange(len(embs2)) != tokenNo2
        orig2=embs2.clone()
 
        print("Begin with '{}'=({}/{}) for '{}'=({}/{})".format(start_token,startNo,startNo2,token,tokenNo,tokenNo2))

        # Create all combinations [prompts] X [promt_variations]
        if prompt_variations:
            token=token+" "
        else:
            prompt_variations=[""]            

        txt_prompts=list()
        for p in prompts:
            for c in prompt_variations:
                txt_prompts.append(p.format(token+c))
        noPrompts=len(txt_prompts)
        
        # convert imgs to latents
        samples=list()
        for img in imgs:
            samples.append(((XLvae.encode(img.to(XLvae.device)).latent_dist.sample(None))*XLvae.config.scaling_factor).to(XLunet.dtype)) # *XLvae.config.scaling_factor=0.13025:  0.18215    
        noSamples=len(samples)
           
        # Training Parameters
        batch_size=1
        acc_size=2
        total_steps=sum(i for i, _ in learning_rates)
        # record_every_nth step is recorded in the progression list
        record_every_nth=(total_steps//(intermediate_steps+1)+1)*acc_size
        total_steps*=acc_size

        # Prompt Parametrs
        lora_scale = [0.6]  
        time_ids = torch.tensor(list(imgs[0].shape[2:4])+[0,0]+[1024,1024]).to(XLunet.dtype).to(XLunet.device)

    
    with torch.enable_grad():
        # Switch Models into training mode
        force_training_grad(XLunet,True,True)
        force_training_grad(XLt1,True,True)
        force_training_grad(XLt2,True,True)
        XLt1.text_model.train()
        XLt2.text_model.train()
        XLunet.train()
        XLunet.enable_gradient_checkpointing()
       
        # Optimizer Parameters        
        learning_rates=iter(learning_rates+[(0,0.0)]) #dummy for last update
        sp,lr=next(learning_rates)
        optimizer = torch.optim.AdamW([embs,embs2], lr=lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)   # 1e-7
        optimizer.zero_grad()
                
        # Progrssion List collects intermediate and final embedding
        progression=list()
        emb=embs[tokenNo].clone()
        emb2=embs2[tokenNo2].clone()
        progression.append({"emb":emb,"emb2":emb2})
        
        # Display [min (mean) max] of embeddings & current learning rate during training
        desc="[{0:2.3f} ({1:2.3f}) +{2:2.3f}] [{3:2.3f} ({4:2.3f}) +{5:2.3f}] lr={6:1.6f}".format(
                        torch.min(emb.to(float)).detach().cpu().numpy(),
                        torch.mean(emb.to(float)).detach().cpu().numpy(),
                        torch.max(emb.to(float)).detach().cpu().numpy(),
                        torch.min(emb2.to(float)).detach().cpu().numpy(),
                        torch.mean(emb2.to(float)).detach().cpu().numpy(),
                        torch.max(emb2.to(float)).detach().cpu().numpy(),
                        lr)

        # Training Loop
        t=auto.trange(total_steps, desc=desc,leave=True)
        # print("total_steps:",total_steps)
        for i in t:
            # use random prompt, random time stepNo, random input image sample
            prompt=txt_prompts[random.randrange(noPrompts)]
            stepNo=torch.tensor(random.randrange(XLsch.config.num_train_timesteps)).unsqueeze(0).long().to(XLunet.device)
            sample=samples[random.randrange(noSamples)].to(XLunet.device)

            ### Target
            noise = torch.randn_like(sample).to(XLunet.device)
            target = noise
            noised_sample=XLsch.add_noise(sample,noise,stepNo)

            # Prediction
            (prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds) = base.encode_prompt(
                prompt=prompt,prompt_2=prompt,
                negative_prompt=negative_prompt,negative_prompt_2=negative_prompt,
                do_classifier_free_guidance=True,lora_scale=lora_scale)
            cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}
            pred = XLunet.forward(noised_sample,stepNo,prompt_embeds,added_cond_kwargs=cond_kwargs)['sample']
                        
            # Loss
            loss = torch.nn.functional.mse_loss((pred).float(), (target).float(), reduction="mean")                  
            loss/=float(acc_size)
            loss.backward() 
            
            # One Optimization Step for acc_size gradient accumulation steps
            if ((i+1)%acc_size)==0:
                # keep Embeddings in normal value range
                torch.nn.utils.clip_grad_norm_(XLt1.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(XLt2.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()
                
                with torch.no_grad():                    
                    # keep Embeddings for all other tokens stable      
                    embs[index_no_updates]= orig[index_no_updates]
                    embs2[index_no_updates2]= orig2[index_no_updates2]      
                        
                    # Current Embedding
                    emb=embs[tokenNo].clone()        
                    emb2=embs2[tokenNo2].clone()        
                            
                    if ((i+1)%(record_every_nth))==0:
                        progression.append({"emb":emb,"emb2":emb2})
                        
                    # adjust learning rate?
                    sp-=1
                    if sp<1:
                        sp,lr=next(learning_rates)
                        for g in optimizer.param_groups:
                            g['lr'] = lr
                            
                    # update display
                    t.set_description("[{0:2.3f} ({1:2.3f}) +{2:2.3f}] [{3:2.3f} ({4:2.3f}) +{5:2.3f}] lr={6:1.6f}".format(
                        torch.min(emb.to(float)).detach().cpu().numpy(),
                        torch.mean(emb.to(float)).detach().cpu().numpy(),
                        torch.max(emb.to(float)).detach().cpu().numpy(),
                        torch.min(emb2.to(float)).detach().cpu().numpy(),
                        torch.mean(emb2.to(float)).detach().cpu().numpy(),
                        torch.max(emb2.to(float)).detach().cpu().numpy(),
                        lr))

        # append final Embedding
        progression.append({"emb":emb,"emb2":emb2})
        
        return progression

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="style embedding train")
    parser.add_argument("--image_path", default='./Images/Figure/', type=str, help="image_style_path")
    parser.add_argument("--style_keyword", default='nothing', type=str, help="learnable style keyword")
    opt = parser.parse_args()


    
    base_model_path="./model/stable-diffusion-xl-base-1.0/"
    base = DiffusionPipeline.from_pretrained(
    base_model_path, 
    torch_dtype=torch.bfloat16,
    variant="fp32", 
    use_safetensors=False,
    add_watermarker=False,
    # use DDPM DDPMScheduler instead of default EulerDiscreteScheduler 
    scheduler = DDPMScheduler(num_train_timesteps=1000,prediction_type="epsilon",beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False)
    )
    base.disable_xformers_memory_efficient_attention()
    torch.set_grad_enabled(True)
    _=base.to("cuda")

    # A single token to be used during the learning process; should NOT be used in "prompts" below
    learn_token=opt.style_keyword
    # start learning with an embedding of single token or "randn_like" 
    start_token="randn_like"
    # list of learning rates [(#steps,learning_rate)] ; 4 gradient accumulation steps per step
    learning_rates=[(4,1e-3),(8,9e-4),(13,8e-4),(20,7e-4),(35,6e-4),(60,5e-4),(100,4e-4),(160,3e-4)]

    # Templates for training: {} defines the token to be learned (learn_token)
    template_prompts_for_styles=["generate an image in {} style","an image in {} style","a picture of people walking in {} style","depicted in a {} style","in style of {}"]

    # Define prompts for training
    prompts=template_prompts_for_styles
    negative_prompt="sketches, (worst quality:2), (low quality:2), (normal quality:2), ((non-symmetrical eyes)), chromatic aberration, lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, bad anatomy, DeepNegative, facing away, tilted head, lowres, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed, mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot, totem, totem body, double body"

    # prompt_variations (randomly added to {} in prompts) 
    prompt_variations=["painting","acryl","art","picture"]

    # INPUT images
    imgs_path=opt.image_path
    #imgs_wh=(1024,1024) # 25 min for 500 steps (3090TI) -> noisy when used with lower INPUT image resolution
    imgs_wh=(768,768) # 15 min for 500 steps (3090TI) -> good results
    #imgs_wh=(512,512) # 10 min for 500 steps (3090TI) -> fastest
    imgs_flip=True # additionally use horizontally mirrored INPUT images

    # OUTPUT embedding
    embs_path="./Embeddings/"
    emb_file=f"{opt.style_keyword}.pt"

    # Visualize intermediate optimization steps
    test_prompt="generate an image in style {}"
    intermediate_steps=9
    imgs,PILimgs=load_imgs(imgs_path,wh=imgs_wh,flip=imgs_flip)
    torch.manual_seed(46)
    progression=XL_textual_inversion(base,imgs=imgs,prompts=prompts,prompt_variations=prompt_variations,token=learn_token,start_token=start_token,negative_prompt=negative_prompt,learning_rates=learning_rates,intermediate_steps=intermediate_steps)

    # save final embedding
    save_XLembedding(progression[-1],embedding_file=emb_file,path=embs_path)
    # save intermediate embeddings
    save_XLembedding(progression,embedding_file="all"+emb_file,path=embs_path)
    print('saving embedding...')
