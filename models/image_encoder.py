import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from einops import rearrange
from typing import List, Tuple, Optional

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
except ImportError:
    print("---IMPORT ERROR ---")
    print("Could not import MultiScaleDeformableAttention from mmcv.ops")
    print("Please ensure mmcv is installed correctly for your PyTorch/CUDA version.")
    print("Example: pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.3/index.html")
    MultiScaleDeformableAttention = None 


# Add this right after imports
if MultiScaleDeformableAttention is not None:
    import inspect
    print("="*80)
    print("MultiScaleDeformableAttention.forward signature:")
    print(inspect.signature(MultiScaleDeformableAttention.forward))
    print("="*80)
# --- Positional Encoding ---
# We vendor this in to remove the mmdet/mmengine dependency
class SinePositionalEncoding(nn.Module):
    """
    Sine-Cosine Positional Encoding for DETR-like models.
    """
    def __init__(self,
                 num_feats: int,
                 temperature: int = 10000,
                 normalize: bool = True,
                 scale: float = 2 * 3.141592653589793,
                 eps: float = 1e-6,
                 offset: float = 0.):
        super().__init__()
        if not normalize:
            raise ValueError("normalize=False is not supported")
            
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask (torch.Tensor): A boolean mask of shape [B, H, W]
                                 (False for valid pixels, True for padding).
        Returns:
            torch.Tensor: Positional encoding of shape [B, C, H, W]
        """
        assert mask.dim() == 3, f"Mask must be 3D [B, H, W], but got {mask.dim()}D"
        
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        
        # Reshape dim_t for broadcasting
        dim_t = dim_t.view(1, -1, 1, 1)

        pos_x = x_embed.unsqueeze(1) / dim_t
        pos_y = y_embed.unsqueeze(1) / dim_t
        
        pos_x = torch.stack((pos_x[:, 0::2, :, :].sin(), pos_x[:, 1::2, :, :].cos()), dim=2).flatten(1, 2)
        pos_y = torch.stack((pos_y[:, 0::2, :, :].sin(), pos_y[:, 1::2, :, :].cos()), dim=2).flatten(1, 2)
        
        pos = torch.cat((pos_y, pos_x), dim=1).permute(0, 2, 3, 1) # [B, H, W, C]
        return pos.permute(0, 3, 1, 2) # [B, C, H, W]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_feats={self.num_feats}, ' \
               f'temperature={self.temperature}, normalize={self.normalize}, ' \
               f'scale={self.scale}, offset={self.offset})'

class DeformableDetrTransformerEncoderLayer(nn.Module):
    """
    This is the "vendored" implementation of the Deformable DETR Encoder Layer.
    It inherits directly from nn.Module to avoid mmdet/mmengine dependencies.
    """
    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_points: int = 4,
                 feedforward_channels: int = 1024,
                 dropout: float = 0.1):
        super().__init__()

        self.embed_dims = embed_dims

        # Self-Attention (Deformable)
        self.self_attn = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=True,
        )
        
        self.dropout1 = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(embed_dims)

        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_channels, embed_dims),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(embed_dims)

    def forward(self,
            query: torch.Tensor,
            query_pos: torch.Tensor,
            key_padding_mask: torch.Tensor,
            spatial_shapes: torch.Tensor,
            level_start_index: torch.Tensor,
            valid_ratios: torch.Tensor) -> torch.Tensor:
    
        # Generate reference points
        reference_points = _generate_reference_points(
            spatial_shapes, valid_ratios, query.device
        )
        
        print(f"[DEBUG] query shape: {query.shape}")
        print(f"[DEBUG] query_pos shape: {query_pos.shape}")
        print(f"[DEBUG] reference_points shape: {reference_points.shape}")
        print(f"[DEBUG] spatial_shapes: {spatial_shapes}")
        print(f"[DEBUG] level_start_index: {level_start_index}")
        print(f"[DEBUG] valid_ratios shape: {valid_ratios.shape}")
        
        # Check if reference points are correct
        expected_total = spatial_shapes.prod(dim=1).sum().item()
        print(f"[DEBUG] Expected total sequence length: {expected_total}")
        print(f"[DEBUG] Actual query sequence length: {query.shape[1]}")
        
        identity = query
        query_with_pos = query + query_pos
        
        # Try calling with minimal parameters first
        try:
            attn_output = self.self_attn(
                query=query_with_pos,
                value=query,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=None#key_padding_mask
            )
            print("[DEBUG] Attention call succeeded!")
        except Exception as e:
            print(f"[DEBUG] Attention call failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        query = identity + self.dropout1(attn_output)
        query = self.attn_norm(query)

        # FFN
        identity = query
        ffn_output = self.ffn(query)
        query = identity + self.dropout2(ffn_output)
        query = self.ffn_norm(query)

        return query    
class DeformableDetrTransformerEncoder(nn.Module):
    """
    This is the "vendored" implementation of the Deformable DETR Encoder.
    It is a stack of Encoder Layers.
    """
    def __init__(self,
                 num_layers: int,
                 layer_cfg: dict):
        super().__init__()
        self.layers = nn.ModuleList([
            DeformableDetrTransformerEncoderLayer(**layer_cfg)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    # This forward pass is correct. It passes `query_pos` to each layer.
    def forward(self,
                query: torch.Tensor,
                query_pos: torch.Tensor,
                key_padding_mask: torch.Tensor,
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor,
                valid_ratios: torch.Tensor) -> torch.Tensor:
        
        output = query
        
        for layer in self.layers:
            output = layer(
                query=output, # This is the output from the previous layer
                query_pos=query_pos, # This is the ORIGINAL pos encoding
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios
            )

        return output
    # --- END FIX ---

# --- Helper function to generate reference points ---
def _generate_reference_points(spatial_shapes: torch.Tensor, 
                             valid_ratios: torch.Tensor, 
                             device: torch.device) -> torch.Tensor:
    """
    Generates normalized (0, 1) reference points for all pixels in a feature map.
    
    Args:
        spatial_shapes (torch.Tensor): Shape [L, 2] of feature maps, e.g., [[56, 56], [28, 28], ...].
        valid_ratios (torch.Tensor): Shape [B, L, 2] valid ratios (1.0 for unpadded).
        device: The device to create the tensors on.
        
    Returns:
        torch.Tensor: Reference points, shape [B, S_total, L, 2].
    """
    reference_points_list = []
    for lvl, (H, W) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            indexing='ij'  # Fixes the UserWarning
        )
        
        # Get ratios for the current level [B, 2]
        vr_lvl = valid_ratios[:, lvl]
        
        # Normalize and reshape
        # [H, W] -> [1, H, W] / [B, 1, 1] -> [B, H, W]
        ref_y = ref_y.unsqueeze(0) / (vr_lvl[:, 1].unsqueeze(1).unsqueeze(2) * H)
        ref_x = ref_x.unsqueeze(0) / (vr_lvl[:, 0].unsqueeze(1).unsqueeze(2) * W)
        
        # Stack (x, y) -> [B, H, W, 2] and flatten -> [B, H*W, 2]
        # (x comes first, so stack ref_x, ref_y)
        ref = torch.stack((ref_x, ref_y), -1).flatten(1, 2)
        
        reference_points_list.append(ref)
        
    # Concat all levels -> [B, S_total, 2]
    reference_points = torch.cat(reference_points_list, 1)
    
    # Expand to [B, S_total, L, 2] for multi-level attention
    reference_points = reference_points.unsqueeze(2).repeat(1, 1, len(spatial_shapes), 1)
    return reference_points


# Helper function for device
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class SwinBackbone(nn.Module):
    """
    Loads a Swin Transformer from timm and configures it to return 
    multi-scale feature maps, as required by DETR-like models.
    """
    def __init__(self, model_name='swin_tiny_patch4_window7_224', 
                 output_indices=(0, 1, 2, 3), 
                 pretrained=True):
        """
        Args:
            model_name (str): The name of the Swin model in timm.
            output_indices (tuple): Which stages to return features from.
                                    (0, 1, 2, 3) corresponds to C2, C3, C4, C5 
                                    (strides 4, 8, 16, 32). This is the 4 levels
                                    expected by Deformable DETR.
            pretrained (bool): Whether to load pre-trained ImageNet weights.
        """
        super().__init__()
        
        # Load the Swin model as a feature extractor
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=output_indices
        )
        
        # Get the channel dimensions for the selected stages
        self.feature_info = self.model.feature_info.get_dicts(
            keys=['num_chs', 'reduction']
        )
        
        # Store the output channel dimensions
        # Example for 'tiny' at (0, 1, 2, 3): [96, 192, 384, 768]
        self.output_channels = [info['num_chs'] for info in self.feature_info]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
        Returns:
            list[torch.Tensor]: A list of feature maps from the selected stages.
                                The Swin model in timm returns channels-last [B, H, W, C]
                                by default, but we'll permute them.
        """
        # Swin backbone in timm returns a list of tensors [B, H, W, C]
        features = self.model(x)
        
        # Permute to [B, C, H, W] (channels-first) for convolutional layers
        return [f.permute(0, 3, 1, 2) for f in features]


class FullImageEncoder(nn.Module):
    """
    The complete T-Rex2 Image Encoder.
    This combines the Swin backbone with the Deformable Transformer Encoder.
    """
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 backbone_model_name='swin_tiny_patch4_window7_224',
                 backbone_output_indices=(0, 1, 2, 3)): # Use C2, C3, C4, C5
        
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_levels = len(backbone_output_indices) # This is 4

        # --- 1. Backbone ---
        self.backbone = SwinBackbone(
            model_name=backbone_model_name,
            output_indices=backbone_output_indices
        )
        
        # --- 2. 1x1 Convs for Channel Projection ---
        # We need to project the backbone's feature channels (e.g., [96, 192, 384, 768])
        # to our transformer's hidden dimension (d_model = 256).
        self.input_proj = nn.ModuleList()
        for in_channels in self.backbone.output_channels:
            self.input_proj.append(
                nn.Conv2d(in_channels, d_model, kernel_size=1)
            )

        # --- 3. Positional Encoding ---
        # We use our vendored SinePositionalEncoding
        self.position_encoder = SinePositionalEncoding(
            num_feats=d_model // 2,
            normalize=True
        )

        # --- 4. Deformable Transformer Encoder ---
        # This is the 6-layer encoder stack.
        # We use our self-contained, vendored implementation.
        self.encoder = DeformableDetrTransformerEncoder(
            num_layers=num_encoder_layers,
            layer_cfg=dict(
                embed_dims=d_model,
                num_heads=nhead,
                num_levels=self.num_levels, # Pass 4 levels to the layer
                num_points=4, # Pass num_points
                feedforward_channels=dim_feedforward,
                dropout=dropout
            )
        )
        
        # --- 5. Level Embeddings ---
        # Learnable embeddings to distinguish between feature levels
        self.level_embed = nn.Parameter(
            torch.randn(self.num_levels, d_model)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Runs the full image encoding pipeline.
        
        Args:
            x (torch.Tensor): Input image tensor, shape [B, 3, H, W]
        
        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]:
            - encoded_memory (torch.Tensor): The final output of the transformer,
                                             shape [B, S, C]
            - projected_features (list[torch.Tensor]): The [B, C, H, W] projected features
        """
        
        # Get batch size
        b = x.shape[0]
        
        # 1. Get multi-scale features from the backbone
        # List of [B, C_in, H_i, W_i]
        backbone_features = self.backbone(x)
        
        projected_features = []
        pos_embeddings = []
        
        # 2. Project channels and create positional embeddings
        for i, feat in enumerate(backbone_features):
            # Project [B, C_in, H_i, W_i] -> [B, d_model, H_i, W_i]
            projected_feat = self.input_proj[i](feat)
            
            # Create a mask (all False, since we don't use padding)
            mask = torch.zeros(
                (feat.shape[0], feat.shape[2], feat.shape[3]), 
                device=feat.device, 
                dtype=torch.bool
            )
            
            # Calculate positional embedding [B, d_model, H_i, W_i]
            pos_embed = self.position_encoder(mask)
            
            projected_features.append(projected_feat)
            pos_embeddings.append(pos_embed)

        # 3. Prepare features for the Transformer Encoder
        # Our encoder expects (with batch_first=True):
        # - src: [B, S, C]
        # - pos: [B, S, C]
        # - key_padding_mask: [B, S]
        # - spatial_shapes: [L, 2]
        # - level_start_index: [L]
        
        srcs = []
        masks = []
        poses = []
        spatial_shapes = []
        
        for i, (feat, pos) in enumerate(zip(projected_features, pos_embeddings)):
            # feat shape [B, C, H, W]
            spatial_shapes.append(feat.shape[2:])
            
            # Flatten [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C] (B, S, C)
            feat_flat = rearrange(feat, 'b c h w -> b (h w) c')
            
            # Flatten [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
            pos_flat = rearrange(pos, 'b c h w -> b (h w) c')
            
            # Create mask [B, H, W] -> [B, H*W]
            mask_flat = rearrange(mask, 'b h w -> b (h w)')
            
            # Add level embedding [B, S, C] + [1, 1, C]
            feat_flat = feat_flat + self.level_embed[i].view(1, 1, -1)
            
            srcs.append(feat_flat)
            masks.append(mask_flat)
            poses.append(pos_flat)
            
        if not srcs:
            raise ValueError("No features produced by backbone. Check output_indices.")

        # Concat all levels
        src_flat = torch.cat(srcs, dim=1)             # [B, S_total, C]
        mask_flat = torch.cat(masks, dim=1)           # [B, S_total]
        pos_flat = torch.cat(poses, dim=1)            # [B, S_total, C]
        
        # Generate all required arguments
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=src_flat.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)), 
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ))
        
        # Create valid_ratios for unpadded inputs, shape [B, L, 2]
        valid_ratios = torch.ones((b, self.num_levels, 2), device=x.device)

        # 4. Pass through the Deformable Transformer Encoder
        encoded_memory = self.encoder(
            query=src_flat,
            query_pos=pos_flat,
            key_padding_mask=mask_flat,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios
        )
        
        return encoded_memory, projected_features

if __name__ == "__main__":
    # Test the full image encoder
    device = get_device()
    
    try:
        image_encoder = FullImageEncoder(backbone_model_name='swin_tiny_patch4_window7_224').to(device)
        image_encoder.eval()
        
        print(f"Loaded full ImageEncoder.")
        print(f"Backbone: {image_encoder.backbone.model.default_cfg['architecture']}")
        print(f"Transformer: {image_encoder.encoder.num_layers} layers, {image_encoder.nhead} heads, {image_encoder.d_model} d_model")
        
        # Create a dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        # Perform a forward pass
        with torch.no_grad():
            encoded_memory, projected_features = image_encoder(dummy_input)
        
        print("\n--- Output Shapes ---")
        print(f"Input shape: {dummy_input.shape}")
        
        print("\nProjected Features (for Visual Prompt Encoder):")
        # Note: Swin-T w/ indices (0, 1, 2, 3) gives strides 4, 8, 16, 32
        # C2 @ s4: [1, 256, 56, 56]
        # C3 @ s8: [1, 256, 28, 28]
        # C4 @ s16: [1, 256, 14, 14]
        # C5 @ s32: [1, 256, 7, 7]
        for i, feat in enumerate(projected_features):
            print(f"  Level {i} (from C{i+2}): {feat.shape}")
            
        print("\nEncoded Memory (for Box Decoder):")
        # Encoded Memory: [1, 56*56 + 28*28 + 14*14 + 7*7, 256] = [1, 3136 + 784 + 196 + 49, 256] = [1, 4165, 256]
        print(f"  Shape: {encoded_memory.shape}")
        
    except ImportError as e:
        print(f"\n--- IMPORT ERROR ---")
        print(f"{e}")
        print("Please ensure 'mmcv' is installed correctly.")
        print("See installation guide in the `if __name__ == '__main__':` block.")
    except Exception as e:
        print(f"\n--- An error occurred ---")
        print(e)