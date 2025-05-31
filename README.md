

<p align="center">

  <h2 align="center">DreamFrame: <br> Enhancing Video Understanding via
Automatically Generated QA and Style-Consistent Keyframes</h2>
  <p align="center">
    <a href="https://github.com/Deaddawn"><strong>Zhende Song</strong></a>
    ·
    <a href="https://github.com/doctorlightt"><strong>Chenchen Wang</strong></a>
    ·  
    <a href="https://github.com/sjmFDU"><strong>Jiamu Sheng</strong></a>
    ·
    <a href="https://icoz69.github.io/"><strong>Chi Zhang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=K7drMDgAAAAJ&hl=en"><strong>Shengji Tang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=gsLd2ccAAAAJ"><strong>Jiayuan Fan✦</strong></a>
    ·
    <a href="https://eetchen.github.io/"><strong>Tao Chen</strong></a>
    <!-- <br>
    (✦ Corresponding Author )
    <br>
    From Fudan University and Tencent PCG -->
    <br>
    </br>
        <!-- <a href="https://arxiv.org/abs/2403.01422">
        <img src='https://img.shields.io/badge/arxiv-MovieLLM-b31b1b.svg' alt='Paper PDF'></a> -->
        <a href="https://deaddawn.github.io/DreamFrame/">
        <img src='https://img.shields.io/badge/Project-Website-green' alt='Project Page'></a>
  </p>
</p>




<image src="docs/fig1.png" />
We propose DreamFrame, a novel framework designed to create synthetic, high-quality data for video understanding.


<!-- ## Changelog
- __[2024.03.03]__: Release inference code, evaluation code and model weights.
- __[2024.03.13]__: Release raw data, check it out [here](https://huggingface.co/datasets/sfsdfsafsddsfsdafsa/MovieLLM-raw-data/tree/main)
- __[2024.07.02]__: All generation code will be released after the work is accepted. -->


## Contents
- [Install](#install)
- [Data Generation](#​generation)
- [Model](#model)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Evaluation](#evaluation)
- [Results](#results)
- [Acknowledgement](#acknowledgement)


## Install
Please follow the instructions below to install the required packages. Our training process is mainly based on LLaMA-VID. And our short video evaluation process is mainly based on quantitative_evaluation from Video-ChatGPT.

Clone this repository
```bash
git clone https://github.com/Deaddawn/DreamFrame-code.git
```

Install Package (Tested on A100 and RTX3090, CUDA 11.8. We recommend sticking to the package versions we provided, as changes in the versions of __diffusers__ and __transformers__ may lead to certain issues.)
```bash
conda create -n DreamFrame python=3.10 -y
conda activate DreamFrame
cd DreamFrame
pip install -r requirements.txt
```

## ​Generation
The data generation process of DreamFrame mainly consists of three stage: (1) __Move Plot Generation__ (2) __Style Immobilization Process__ (3) __Video Instruction Data Generation__

### Move Plot Generation
We basically adopt a story expanding strategy which incrementally generates frame descriptions through three levels. We provide three-level example [prompts](https://github.com/Deaddawn/DreamFrame-code/tree/main/prompt). Use any LLM(We use GPT-4) to generate frame descriptions and organize them into a JSON file like this [story_js](https://github.com/Deaddawn/DreamFrame-code/blob/main/json/story_info_0.json)


### Style Immobilization Process
Style Immobilization is to learn a style embedding which can be used to generate style consistent key frames. To learn the style embedding, we will need a style-related __keyword__ and a set of style-related __images__. Keyword can be obtained from stage one. For style-related images, we simply use sdxl-1.0-base to generate these based on the detail style description (you can find an example in the prompt we provide).

Here, we provide an example to show how you can train a style embedding. We use keyword "Dramatic".

```bash
cd StyleImmobilization
python style_embedding.py --style_keyword Dramatic --image_path ./style
```
The learned style embedding will be saved at folder "Embeddings". This should only take 5~10 minutes (tested on A100). 

## Video Instruction Data Generation
After train a style embedding, you can start to generate consistent keyframes based on the aformentioned json file like this:
```bash
cd StyleImmobilization
python generate.py --js_path ./json/story_info_0.json --embed_path ./Embeddings/story_0_Dramatic.pt --keyword Dramatic --save_path ./save_path
```


## Model
We provide our baseline model and model trained on our generated dataset. For more detailed information, refer to [LLaMA-VID-model](https://github.com/dvlab-research/LLaMA-VID#model). And please follow [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID) to prepare the necessary settings and feel free to use our provided checkpiont.

| Type | Max Token | Base LLM | Finetuning Data | Finetuning schedule | Download |
|----------|----------|----------|---------------|--------------------|------------------|
|Base Model|64K | Vicuna-7B-v1.5 | LLaVA1.5-VideoChatGPT-Instruct | full_ft-1e | [ckpt]() |
|DreamFrame-7B|64K | Vicuna-7B-v1.5 | LLaVA1.5-VideoChatGPT-Instruct + DreamFrameQA | full_ft-1e | [ckpt]() |




## Dataset
We provide raw dataset generated from our pipeline and also related training data based on LLaMA-VID.

### Our Raw Data
Data generated from our pipeline consists of key frame images, corresponding QAs and dialogues. You can download it from here [DreamFrame-Data]()
<image src="docs/tuning_data_distribution.png" />







## Pipeline
<image src="docs/PIPELINE.png" />






## Evaluation
We follow [MVBench](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2), [Video-Bench](https://github.com/PKU-YuanGroup/Video-Bench) and [TempCompass](https://github.com/llyx97/TempCompass) to conduct evaluations.
### Evaluation Results
<image src="docs/res.png" />



## Results
### Generation Results
<image src="docs/res1.png" />

### Comparison Results
<image src="docs/res2.png" />





## Acknowledgement
We would like to thank the following repos for their great work:

- Our model is trained based on [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID/).
- We build our pipeline based on [textual-inversion](https://github.com/oss-roettger/XL-Textual-Inversion)

