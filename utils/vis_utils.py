import math
from typing import List
from PIL import Image
import cv2
import numpy as np
import torch
#########
import time
from PIL import Image
from torchvision import transforms as T
import os
from sklearn.decomposition import PCA
from math import sqrt
########

from utils import ptp_utils
from utils.ptp_utils import AttentionStore, aggregate_attention
from torchvision import transforms as T

def show_cross_attention(prompt: str,
                         attention_store: AttentionStore,
                         tokenizer,
                         indices_to_alter: List[int],
                         res: int,
                         from_where: List[str],
                         select: int = 0,
                         orig_image=None):
    tokens = tokenizer.encode(prompt)
    # print("tokens", tokens) # tokens输出的值为 [49406, 550, 14004, 593, 320, 6902, 49407]
    # print("tokens类型为", type(tokens)) # <class 'list'>

    #######
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select).detach().cpu()
    # print('attention_maps维度', attention_maps.shape)   # torch.Size([16, 16, 77])
    images = []
    masks = []
    token_norm_crosses = []
    # show spatial attention for indices of tokens to strengthen

    # for i in range(len(tokens)):
    # image = attention_maps[:, :, i]
    # if i in indices_to_alter:
    #     image = show_image_relevance(image, orig_image)
    #     image = image.astype(np.uint8)
    #     image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
    #     image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
    #     images.append(image)
    
    # 为了可视化所有token的图像，修改为如下形式：
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        # if i in indices_to_alter:
        image = show_image_relevance(image, orig_image)
        image = image.astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        # print('image维度为', image.shape)   # 维度为(307,256,3)
        images.append(image)

    ptp_utils.view_images(np.stack(images, axis=0))

    # 基于cross生成mask
    thres = 0.5
    for i in range(len(tokens)):
        token_norm_cross = attention_maps[:, :, i]
        timestamp = int(time.time())
        # cross_attn = token_norm_cross
        # cross_attn_img = Image.fromarray((cross_attn.cpu().numpy() * 255).astype(np.uint8))
        # cross_attn_img = T.Resize(256, interpolation=T.InterpolationMode.NEAREST)(cross_attn_img)
        # cross_attn_img.save(f"/home/u2023030056/network/Attend-and-Excite-main/cross_{i}_{decoder(int(tokens[i]))}_{timestamp}.png")

        mask = torch.ones_like(token_norm_cross)
        # print('最大值为', token_norm_cross.max())
        token_norm_cross = (token_norm_cross - token_norm_cross.min()) / (token_norm_cross.max() - token_norm_cross.min())

        # # ## 处理mask 
        # mask[token_norm_cross >= thres] = 1
        # mask[token_norm_cross < thres] = 0
        # mask = mask.unsqueeze(dim=-1).repeat(1,1,3).cpu().numpy()
        # # print('mask维度', mask.shape)
        # mask = Image.fromarray((mask * 255).astype(np.uint8))
        # mask = T.Resize(256, interpolation=T.InterpolationMode.NEAREST)(mask)
        # mask = ptp_utils.text_under_image(np.array(mask), decoder(int(tokens[i])))
        # masks.append(mask)

        ## 处理获得的norm cross
        token_norm_cross = token_norm_cross.unsqueeze(dim=-1).repeat(1,1,3)
        token_norm_cross = token_norm_cross.cpu().numpy()

        # token_norm_cross = np.array(Image.fromarray(token_norm_cross * 255).resize((res ** 2, res ** 2)))
        token_norm_cross = Image.fromarray((token_norm_cross * 255).astype(np.uint8))
        token_norm_cross = T.Resize(256, interpolation=T.InterpolationMode.NEAREST)(token_norm_cross)
        token_norm_cross = ptp_utils.text_under_image(np.array(token_norm_cross), decoder(int(tokens[i])))
        ## print('token_norm_cross维度为', token_norm_cross.shape) # 维度为(307,256,3)
        token_norm_crosses.append(token_norm_cross)

    # ptp_utils.view_images(np.stack(masks, axis=0))
    ptp_utils.view_images(np.stack(token_norm_crosses, axis=0))

def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    # print('image_relevance维度', image_relevance.shape) # torch.Size([16, 16])
    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    # print('reshape之后image_relevance维度', image_relevance.shape)  # torch.Size([1, 1, 16, 16])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image

def save_mask_numpy(image_relevance, save_dir, name, t, attn_layer):
    # 输入的mask是numpy形式
    image_relevance = Image.fromarray((image_relevance * 255).astype(np.uint8))
    image_relevance = T.Resize(256, interpolation=T.InterpolationMode.NEAREST)(image_relevance)
    image_relevance.save(os.path.join(save_dir, f"{name}_time_{t}_layers_{attn_layer}.png"))


def save_mask(mask, save_dir, name, t, attn_layer):
    # 输入的mask是cuda()上的tensor
    image_relevance = mask.detach().cpu().numpy() # send it back to cpu
    # image_relevance = mask # send it back to cpu
    image_relevance = Image.fromarray((image_relevance * 255).astype(np.uint8))
    image_relevance = T.Resize(256, interpolation=T.InterpolationMode.NEAREST)(image_relevance)
    image_relevance.save(os.path.join(save_dir, f"{name}_time_{t}_layers_{attn_layer}.png"))
    

def cross_show(image_relevance, save_dir, name, t, attn_layer):

    image_relevance = image_relevance.detach().cpu().numpy() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = Image.fromarray((image_relevance * 255).astype(np.uint8))
    image_relevance = T.Resize(256, interpolation=T.InterpolationMode.NEAREST)(image_relevance)
    image_relevance.save(os.path.join(save_dir, f"{name}_time_{t}_layers_{attn_layer}.png"))



## 存储self-attention pca 可视化
def self_pca(attn, copmponent, save_dir, prompt, t, attn_layer):

    attn = attn.detach().cpu().numpy()
    # 对head做均值
    self_head_mean = attn.sum(0) / attn.shape[0]
    # # 不对head做均值
    # self_pca_new = rearrange(attn, 'h n m -> n (h m)')

    pca = PCA(n_components=copmponent)
    self_head_mean_fit = self_head_mean
    pca.fit(self_head_mean_fit)
    pca_img = pca.transform(self_head_mean)  # N X 3, 注: N=H*W

    h = w = int(sqrt(pca_img.shape[0]))
    if copmponent == 1:
        pca_img = pca_img.reshape(h, w)
    else:
        pca_img = pca_img.reshape(h, w, copmponent)
    pca_img_min = pca_img.min(axis=(0, 1))
    pca_img_max = pca_img.max(axis=(0, 1))
    pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
    pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
    pca_img = T.Resize(256, interpolation=T.InterpolationMode.NEAREST)(pca_img)
    pca_img.save(os.path.join(save_dir, f"{prompt}_pca_time_{t}_layer_{attn_layer}.png"))


