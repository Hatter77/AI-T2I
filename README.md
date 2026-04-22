AI-T2I: Aggregating-and-Isolating Cross-Attention to Diffusion Models for Text-to-Image Synthesis


## Description  
Implementation of our AI-T2I paper.

## Setup

### Environment
Our code builds on the requirement of the official [Stable Diffusion repository](https://github.com/CompVis/stable-diffusion). To set up their environment, please run:

```
conda env create -f environment/environment.yaml
conda activate ldm
```

On top of these requirements, we add several requirements which can be found in `environment/requirements.txt`. These requirements will be installed in the above command.

### Hugging Face Diffusers Library
Our code relies also on Hugging Face's [diffusers](https://github.com/huggingface/diffusers) library for downloading the Stable Diffusion v1.4 model. 


To generate an image, for example:
```
python explain.py
```
Notes:

- To apply AI-T2I on Stable Diffusion 2.1, specify: `--sd_2_1 True`
- You may run multiple seeds by passing a list of seeds. For example, `--seeds [0,1,2,3]`.
- If you do not provide a list of which token indices to alter using `--token_indices`, we will split the text according to the Stable Diffusion's tokenizer and display the index of each token. You will then be able to input which indices you wish to alter.
- If you wish to run the standard Stable Diffusion model without AI-T2I, you can do so by passing `--run_standard_sd True`.
- All parameters are defined in `config.py` and are set to their defaults according to the official paper.

All generated images will be saved to the path `"{config.output_path}/{prompt}"`. We will also save a grid of all images (in the case of multiple seeds) under `config.output_path`.


## Metrics 
In `metrics/` we provide code needed to reproduce the quantitative experiments presented in the paper: 
1. In `compute_clip_similarity.py`, we provide the code needed for computing the image-based CLIP similarities. Here, we compute the CLIP-space similarities between the generated images and the guiding text prompt. 
2. In `blip_captioning_and_clip_similarity.py`, we provide the code needed for computing the text-based CLIP similarities. Here, we generate captions for each generated image using BLIP and compute the CLIP-space similarities between the generated captions and the guiding text prompt. 
    - Note: to run this script you need to install the `lavis` library. This can be done using `pip install lavis`.

To run the scripts, you simply need to pass the output directory containing the generated images. The direcory structure should be as follows:
```
outputs/
|-- prompt_1/
|   |-- 0.png 
|   |-- 1.png
|   |-- ...
|   |-- 64.png
|-- prompt_2/
|   |-- 0.png 
|   |-- 1.png
|   |-- ...
|   |-- 64.png
...
```
The scripts will iterate through all the prompt outputs provided in the root output directory and aggregate results across all images. 

The metrics will be saved to a `json` file under the path specified by `--metrics_save_path`. 


## Acknowledgements 
This code is builds on the code from the [diffusers](https://github.com/huggingface/diffusers) library, the [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt/) as well as the [Attend-and-Excite-main](https://yuval-alaluf.github.io/Attend-and-Excite/) codebase.

