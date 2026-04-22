
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, is_accelerate_available, logging, randn_tensor, replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

from utils.gaussian_smoothing import GaussianSmoothing
from utils.ptp_utils import AttentionStore, aggregate_attention, aggregate_self_attn

import nltk
import math
# from kmeans_pytorch import kmeans
from sklearn.decomposition import PCA


from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from utils.vis_utils import save_mask, cross_show, self_pca#, cross_show_no_norm
from torchvision import transforms as T

from PIL import Image
import os
import torch.distributions as dist

logger = logging.get_logger(__name__)

class AttendAndExcitePipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds

    def compute_multi_token_loss(
        self,
        ave_cross: torch.Tensor,
        token_indices: List[int],
        timestep: int,
        prompt: str
    ) -> torch.Tensor:
        """
        Compute comprehensive loss for multiple tokens.
        Refactored version of _compute_multi_loss.
        
        Args:
            ave_cross: Cross attention map of shape [h, w, n]
            token_indices: List of token indices to process
            timestep: Current timestep (inverted, where 0 is the start)
            p: Unused parameter (kept for compatibility)
            loss_weights: Dictionary of loss weights
            
        Returns:
            Total weighted loss
        """

        loss_weights = {
            'centroid_distance': 2.0,
            'radius_aggregate': 1.25,
            'max_loss': 0.25,
            'adjective_align': 0.75
        }
        
        # Extract attention maps for each token
        attn_list = [ave_cross[:, :, idx] for idx in token_indices]
        
        # 1. Centroid distance loss (minimize distance between token centroids)
        centroid_loss = self.compute_centroid_distance_loss(attn_list[0], attn_list[1])
        
        # 2. Radius aggregate loss (encourage compact regions)
        agg_loss_0, max_center_0 = self.compute_radius_centroid_loss(
            attn_list[0], timestep, num_regions=3, radius_start=5, radius_end=5
        )
        agg_loss_1, max_center_1 = self.compute_radius_centroid_loss(
            attn_list[1], timestep, num_regions=3, radius_start=5, radius_end=5
        )
        agg_loss = agg_loss_0 + agg_loss_1
        
        # 3. Maximum value loss (from process_cross - smoothing and max)
        images_list = self.process_cross(ave_cross, token_indices, prompt)
        max_loss = self._compute_loss_2token_max(images_list)
        
        # 4. Adjective alignment loss (align adjectives to noun centers)
        adj_attn_list = [ave_cross[:, :, idx - 1] for idx in token_indices]
        adj_loss_0 = self.compute_adjective_alignment_loss(
            adj_attn_list[0], max_center_0, timestep, num_regions=3, radius_start=6, radius_end=6
        )
        adj_loss_1 = self.compute_adjective_alignment_loss(
            adj_attn_list[1], max_center_1, timestep, num_regions=3, radius_start=6, radius_end=6
        )
        adj_loss = adj_loss_0 + adj_loss_1
        
        # Combine losses
        total_loss = (
            loss_weights['centroid_distance'] * (1-centroid_loss) +
            loss_weights['radius_aggregate'] * agg_loss +
            loss_weights['max_loss'] * max_loss  +
            loss_weights['adjective_align'] * adj_loss
        )
        
        # print(f'Aggregate loss: {radius_loss.item():.4f}')
      
        return total_loss


    def normalize_attention(self, attn, eps=1e-8):
        """
        对attention map进行归一化，避免出现NaN值
        
        参数:
            attn: 输入的attention张量
            eps: 用于数值稳定性的小值
            
        返回:
            归一化后的attention张量
    """
        # 检查并替换可能存在的NaN值
        if torch.isnan(attn).any():
            attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        
        # 检查并替换可能存在的无穷大值
        if torch.isinf(attn).any():
            attn = torch.where(torch.isinf(attn), torch.zeros_like(attn), attn)
        
        # 计算总和，确保为正值
        attn_sum = attn.sum()
        
        # 处理总和为零或负值的情况
        if attn_sum <= 0:
            # 如果总和为零或负值，返回均匀分布
            return torch.ones_like(attn) / attn.numel()
        
        # 进行归一化
        # attn_norm = attn / (attn_sum + eps)
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + eps)
        
        # 最后检查是否还有NaN值，确保输出安全
        if torch.isnan(attn_norm).any():
            attn_norm = torch.where(torch.isnan(attn_norm), torch.zeros_like(attn_norm), attn_norm)
     
        return attn_norm 


    # 余弦距离
    def cos_dist(self, attention_map1, attention_map2):
        # Convert map into a single distribution: 16x16 -> 256
        if len(attention_map1.shape) > 1:
            attention_map1 = attention_map1.reshape(-1)
        if len(attention_map2.shape) > 1:
            attention_map2 = attention_map2.reshape(-1)

        p = dist.Categorical(probs=attention_map1)
        q = dist.Categorical(probs=attention_map2)

        cos_dist = 1 - (p.probs * q.probs).sum() / (p.probs.norm() * q.probs.norm())
        return cos_dist


    def get_spatial_dims(self, attn: torch.Tensor) -> Tuple[int, int]:
        """
        Dynamically extract spatial dimensions from attention map.
        
        Args:
            attn: Attention tensor of shape [h, w] or [h, w, n]
            
        Returns:
            Tuple of (height, width)
        """
        if len(attn.shape) == 2:
            h, w = attn.shape
        elif len(attn.shape) == 3:
            h, w, _ = attn.shape
        else:
            raise ValueError(f"Unexpected attention shape: {attn.shape}")
        return h, w
    
    def normalize_attention(self, attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Normalize attention map to avoid NaN values.
        
        Args:
            attn: Input attention tensor
            eps: Small value for numerical stability
            
        Returns:
            Normalized attention tensor
        """
        # Replace NaN and Inf values
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        attn = torch.where(torch.isinf(attn), torch.zeros_like(attn), attn)
        
        # Check if sum is valid
        attn_sum = attn.sum()
        if attn_sum <= 0:
            return torch.ones_like(attn) / attn.numel()
        
        # Min-max normalization
        attn_min = attn.min()
        attn_max = attn.max()
        
        if attn_max - attn_min < eps:
            return torch.ones_like(attn) / attn.numel()
            
        attn_norm = (attn - attn_min) / (attn_max - attn_min + eps)
        
        # Final NaN check
        attn_norm = torch.where(torch.isnan(attn_norm), torch.zeros_like(attn_norm), attn_norm)
        
        return attn_norm
    

    def compute_centroid(self, attn: torch.Tensor) -> torch.Tensor:
        """
        Compute the weighted centroid of an attention map.
        
        Args:
            attn: Attention map of shape [h, w]
            
        Returns:
            Centroid coordinates [x, y]
        """
        h = w = int(attn.shape[0])
        ws = torch.arange(w).view(1, -1).to(attn.device)
        hs = torch.arange(h).view(-1, 1).to(attn.device)   
        weighted_w = torch.sum(ws * attn, dim=[0,1])
        weighted_h = torch.sum(hs * attn, dim=[0,1])
        bray = torch.stack([weighted_w, weighted_h]) / attn.sum((0,1)) # 除以质量和

        return bray


    def index_to_coord(self, index: int, width: int) -> torch.Tensor:
        """
        Convert a flattened index to 2D coordinates.
        
        Args:
            index: Flattened index
            width: Width of the 2D grid
            
        Returns:
            Coordinates [x, y] 
        """
        yh = index // width
        xw = index % width
        return torch.tensor([xw, yh], device=self.device, dtype=torch.float32)  # ⭐ 改为 [x, y]


    def create_ball_mask(self, attn: torch.Tensor, center_idx: int, radius: float) -> torch.Tensor:
        """
        Create a circular mask around a center point.
        
        Args:
            attn: Attention map of shape [h, w]
            center_idx: Flattened index of center
            radius: Radius of the ball
            
        Returns:
            Binary mask of shape [h, w]
        """
        h, w = self.get_spatial_dims(attn)
        
        # Convert center index to coordinates
        center_y = center_idx // w
        center_x = center_idx % w
        
        # Create coordinate meshgrid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=attn.device),
            torch.arange(w, device=attn.device),
            indexing='ij'
        )
        
        # Compute distance from center
        dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
        mask = (dist_sq < radius ** 2).float()
        
        return mask
    
    def compute_cosine_distance(self, attn1: torch.Tensor, attn2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine distance between two attention maps.
        
        Args:
            attn1: First attention map
            attn2: Second attention map
            
        Returns:
            Cosine distance (0 = identical, 2 = opposite)
        """
        # Flatten and normalize
        p = attn1.reshape(-1)
        q = attn2.reshape(-1)
        
        # Add small epsilon for numerical stability
        eps = 1e-8
        p_norm = p / (p.norm() + eps)
        q_norm = q / (q.norm() + eps)
        
        # Cosine distance = 1 - cosine similarity
        cos_dist = 1 - (p_norm * q_norm).sum()
        
        return cos_dist
    
    def find_top_regions(
        self, 
        attn: torch.Tensor, 
        radius: float, 
        num_regions: int = 2
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:  
        """
        Find top-k regions in attention map using non-maximum suppression.
        
        Args:
            attn: Attention map of shape [h, w]
            radius: Radius for ball query
            num_regions: Number of regions to find
            
        Returns:
            Tuple of (centroids, centers)
        """
        h, w = self.get_spatial_dims(attn)
        attn_remaining = attn.clone()
        
        centroids = []
        centers = []
        
        for i in range(num_regions):
            # Find maximum in remaining attention
            attn_flat = attn_remaining.reshape(-1)
            max_idx = attn_flat.argmax().item()
            
            # Create ball mask around maximum
            mask = self.create_ball_mask(attn, max_idx, radius)
            
            # Compute centroid within masked region
            masked_attn = mask * attn 
            centroid = self.compute_centroid(masked_attn)
            
            # Store results
            centroids.append(centroid)
            centers.append(self.index_to_coord(max_idx, w))
            
            # Remove this region from remaining attention
            attn_remaining = attn_remaining * (1 - mask)
        
        return centroids, centers 

    def compute_radius_centroid_loss(
        self,
        attn: torch.Tensor,
        timestep: int,
        num_regions: int = 3,  # 固定为3个区域
        radius_start: float = 2.0,
        radius_end: float = 8.0,
        total_steps: int = 25
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        让3个ball的重心都趋于最大值点
        """
        h, w = self.get_spatial_dims(attn)
        # print('h为', h)
        # # Compute dynamic radius
        # t_normalized = timestep / total_steps
        # radius = radius_start + (radius_end - radius_start) * (1 - t_normalized)
        # radius = max(radius_end, min(radius_start, round(radius)))

        # 使用固定半径
        radius = radius_start
        
        # 找到前3个区域
        centroids, centers = self.find_top_regions(attn, radius, num_regions)
        
        # 获取第一个最大值点的坐标
        max_center = centers[0]
        
        # 计算三个重心到最大值点的距离（类似代码二的bray_dist_0,1,2）
        total_dist = torch.tensor(0.0, device=attn.device)
        
        # 1. 三个重心到最大值点的距离
        for i in range(num_regions):
            dist = torch.norm(max_center.detach() - centroids[i])
            total_dist += dist
        
        # 2. 第二个和第三个重心之间的距离（类似代码二的bray_dist_3）
        if num_regions >= 3:
            dist_2_3 = torch.norm(centroids[1] - centroids[2])
            total_dist += dist_2_3
        
        # Compute pairwise Euclidean distances normalized by max distance
        # dist_max = torch.sqrt(torch.tensor(h**2 + w**2, device=attn.device))
        dist_max = torch.sqrt(torch.tensor((h-1)**2 + (w-1)**2, device=attn.device))

        loss = total_dist / dist_max
        
        return loss, max_center


    def compute_adjective_alignment_loss(
        self,
        adj_attn: torch.Tensor,
        n_bray: torch.Tensor,
        timestep: int,
        num_regions: int = 3,
        radius_start: float = 4.0,
        radius_end: float = 8.0,
        total_steps: int = 25
    ) -> torch.Tensor:
        
        h, w = self.get_spatial_dims(adj_attn)
        
        # # Compute dynamic radius
        # t_normalized = timestep / total_steps
        # radius = radius_start + (radius_end - radius_start) * (1 - t_normalized)
        # radius = max(radius_end, min(radius_start, round(radius)))

        radius = radius_start
        
        # 找到形容词的前3个区域
        centroids, _ = self.find_top_regions(adj_attn, radius, num_regions)
        
        # 计算距离（只计算到名词重心的距离 + 第2、3个重心之间距离）
        total_dist = torch.tensor(0.0, device=adj_attn.device)
        
        # 1. 三个重心到名词重心的距离
        for centroid in centroids:
            dist = torch.norm(n_bray.detach() - centroid)
            total_dist += dist
        
        # 2. 仅第2和第3个重心之间的距离
        if len(centroids) >= 3:
            dist_2_3 = torch.norm(centroids[1] - centroids[2])
            total_dist += dist_2_3
        
        # Compute distances to noun center
        # dist_max = torch.sqrt(torch.tensor(h**2 + w**2, device=adj_attn.device))
        dist_max = torch.sqrt(torch.tensor((h-1)**2 + (w-1)**2, device=adj_attn.device))        
        
        loss = total_dist / dist_max
        
        return loss


    def compute_centroid_distance_loss(
        self,
        attn1: torch.Tensor,
        attn2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normalized Euclidean distance between two attention centroids.
        
        Args:
            attn1: First attention map of shape [h, w]
            attn2: Second attention map of shape [h, w]
            
        Returns:
            Normalized distance loss
        """
        h, w = self.get_spatial_dims(attn1)
        # print('距离loss中h为', h)
        
        # Compute centroids
        centroid1 = self.compute_centroid(attn1)
        centroid2 = self.compute_centroid(attn2)
        
        # Compute Euclidean distance
        distance = torch.norm(centroid1 - centroid2)
        
        # Normalize by maximum possible distance (diagonal)
        max_dist = torch.sqrt(torch.tensor(h**2 + w**2, device=attn1.device))
        # max_dist = torch.sqrt(torch.tensor((h-1)**2 + (w-1)**2, device=attn1.device))
        
        return distance / max_dist

    def _smoothing_ave_cross(self, attention_for_text, token_index):  
        smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=0.5, dim=2).cuda()
        image = attention_for_text[:, :, (token_index - 1)]
        image = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
        image = smoothing(image).squeeze(0).squeeze(0)
        return image

    def process_cross(self, ave_cross, token_index, prompt):
        
        images_list = []

        attention_for_text = ave_cross[:, :, 1:-1].clone()
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        for idx in token_index:
            images_list.append(self._smoothing_ave_cross(attention_for_text, idx))

        return images_list

    # 对两个token, 增大其中较小的token的最大值
    def _compute_loss_2token_max(self, images_list):        

        losses = []
        # 获取每个token的最大值
        for i in range (len(images_list)):
            losses.append(images_list[i].max())

        max_loss = max(0, 1. - min(losses))

        return max_loss

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents

 

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            attention_store: AttentionStore,
            indices_to_alter: List[int],
            attention_res: int = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            min_iter_to_alter: Optional[int] = 2,
            max_iter_to_alter: Optional[int] = 25,
            run_standard_sd: bool = False,
            # thresholds: Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8},
            thresholds: Optional[dict] = {0: 0.99, 5: 0.5, 20: 0.8},
            scale_factor: int = 20, # 默认为20
            scale_range: Tuple[float, float] = (1., 0.5),
            smooth_attentions: bool = True,
            sigma: float = 0.5,
            kernel_size: int = 3,
            sd_2_1: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        Examples:
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
            :type attention_store: object
        """

        self.bear_interest_mask = None
        self.new_interest_mask = None
        self.get_in = min_iter_to_alter
        self.cross_attn_label = None
        self.token_max_list = []
        self.bear_max_list = []
        self.mask_lists = None
        # self.last_time_cross = None
        # self.w2 = 0
        # self.w3 = 0
        self.pro_token = None
        self.tokens = indices_to_alter
        self.jiaodu_last_time = None
        self.init_time_bray = None
        self.last_time_bray = None
        # self.seed = generator

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_inputs, prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                with torch.enable_grad():

                    latents = latents.clone().detach().requires_grad_(True)

                    # Forward pass of denoising with text conditioning
                    noise_pred_text = self.unet(latents, t,
                                                encoder_hidden_states=prompt_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                    self.unet.zero_grad()

                    if not run_standard_sd:
 
                        if i < 25:

                            ave_cross_attn = aggregate_attention(attention_store=attention_store, res=16, from_where=("up", "down"), is_cross=True, select=0)
                            loss = self.compute_multi_token_loss(ave_cross_attn, indices_to_alter, i, prompt)

                            if loss != 0:
                                latents = self._update_latent(latents=latents, loss=loss,
                                                                step_size=scale_factor * np.sqrt(scale_range[i]))
                            print(f'Iteration {i} | Loss: {loss:0.4f}')
                
                torch.cuda.empty_cache()

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
