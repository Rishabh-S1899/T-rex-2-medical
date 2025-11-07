import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Tuple, Optional

# --- Copy Core Deformable Attention Operator ---
# We copy this from image_encoder.py to keep the file self-contained
try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
except ImportError:
    print("---IMPORT ERROR ---")
    print("Could not import MultiScaleDeformableAttention from mmcv.ops")
    print("Please ensure mmcv is installed correctly for your PyTorch/CUDA version.")
    print("Example: pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.3/index.html")
    MultiScaleDeformableAttention = None 

# --- Copy Helper function to generate reference points ---
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

# --- Simple MLP Helper Class ---
class MLP(nn.Module):
    """ Simple MLP (Feed-Forward Network) """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class VisualPromptEncoder(nn.Module):
    """
    Implements the Visual Prompt Encoder from T-Rex2 (Fig 3).
    
    This module takes user-provided points or boxes, combines them with
    image features from the FullImageEncoder, and produces a single
    visual prompt embedding (V).
    """
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_levels=4, # Must match the number of feature levels from image_encoder
                 num_points=4, # Number of deformable attention sampling points
                 dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_levels = num_levels
        self.num_heads = nhead
        self.num_points = num_points
        
        # --- 1. Prompt Embedders ---
        # MLPs to create position embeddings from coordinates
        self.point_pos_embed = MLP(2, d_model, d_model, 3) # (x, y) -> d_model
        self.box_pos_embed = MLP(4, d_model, d_model, 3)   # (x, y, w, h) -> d_model
        
        # Learnable content embeddings (C in Eq. 3)
        self.point_content_embed = nn.Embedding(1, d_model)
        self.box_content_embed = nn.Embedding(1, d_model)
        
        # Learnable global/aggregator token embeddings (C' and B' in Eq. 3)
        self.global_content_embed = nn.Embedding(1, d_model)
        self.global_pos_embed = nn.Embedding(1, d_model)
        
        # --- 2. Deformable Cross-Attention ---
        # This layer's QUERY will be the prompt embeddings.
        # Its KEY and VALUE will be the image feature maps.
        self.cross_attn = MultiScaleDeformableAttention(
            embed_dims=d_model,
            num_heads=nhead,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # --- 3. Aggregator (Eq 5) ---
        # A simple self-attention layer and FFN to "aggregate" the
        # cross-attention outputs into a single embedding.
        
        # We use a standard nn.MultiheadAttention here, NOT deformable.
        self.aggregator_self_attn = nn.MultiheadAttention(
            embed_dim=d_model, # <-- This is your fix (embed_dim, not embed_dims)
            num_heads=nhead,
            dropout=dropout,
            batch_first=True # Use batch_first for consistency
        )
        self.aggregator_self_attn_norm = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        self.aggregator_ffn = MLP(d_model, dim_feedforward, d_model, 2)
        self.aggregator_ffn_norm = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)


    def forward(self,
                # Inputs from FullImageEncoder
                projected_features: List[torch.Tensor],
                pos_embeddings: List[torch.Tensor],
                # User-provided prompts
                boxes: Optional[torch.Tensor] = None,
                points: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            projected_features (List[torch.Tensor]): List of 4 feature maps 
                from the Image Encoder, shape [B, C, H_i, W_i].
            pos_embeddings (List[torch.Tensor]): List of 4 positional embeddings
                for the feature maps, shape [B, C, H_i, W_i].
            boxes (torch.Tensor, optional): Box prompts, shape [B, K, 4].
            points (torch.Tensor, optional): Point prompts, shape [B, K, 2].

        Returns:
            torch.Tensor: The final visual prompt embedding (V), shape [B, d_model].
        """
        
        if boxes is None and points is None:
            raise ValueError("Must provide either boxes or points as visual prompts.")
        
        b = projected_features[0].shape[0]
        
        # --- 1. Prepare Image Features (Key/Value for Cross-Attention) ---
        
        srcs = []
        poses = []
        spatial_shapes = []
        
        for i, (feat, pos) in enumerate(zip(projected_features, pos_embeddings)):
            # feat shape [B, C, H, W]
            spatial_shapes.append(feat.shape[2:])
            
            # Flatten [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C] (B, S, C)
            feat_flat = rearrange(feat, 'b c h w -> b (h w) c')
            pos_flat = rearrange(pos, 'b c h w -> b (h w) c')
            
            srcs.append(feat_flat)
            poses.append(pos_flat)
            
        # Concat all levels
        value = torch.cat(srcs, dim=1)           # [B, S_total, C]
        value_pos = torch.cat(poses, dim=1)      # [B, S_total, C]
        
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=value.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)), 
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ))
        valid_ratios = torch.ones((b, self.num_levels, 2), device=value.device)

        # --- 2. Prepare Prompt Embeddings (Query for Cross-Attention) ---
        
        if boxes is not None:
            # `boxes` shape is [B, K, 4]
            K = boxes.shape[1]
            # Position embedding from (x,y,w,h) coordinates
            pos_embed = self.box_pos_embed(boxes) # [B, K, C]
            # Content embedding (same for all boxes)
            content_embed = self.box_content_embed.weight.unsqueeze(0).repeat(b, K, 1) # [B, K, C]
            # Reference points are the box centers (x, y)
            reference_points_prompt = boxes[..., :2] # [B, K, 2]
        
        elif points is not None:
            # `points` shape is [B, K, 2]
            K = points.shape[1]
            # Position embedding from (x,y) coordinates
            pos_embed = self.point_pos_embed(points) # [B, K, C]
            # Content embedding (same for all points)
            content_embed = self.point_content_embed.weight.unsqueeze(0).repeat(b, K, 1) # [B, K, C]
            # Reference points are the points themselves
            reference_points_prompt = points # [B, K, 2]

        # Get global "aggregator" token embeddings
        global_content = self.global_content_embed.weight.unsqueeze(0).repeat(b, 1, 1) # [B, 1, C]
        global_pos = self.global_pos_embed.weight.unsqueeze(0).repeat(b, 1, 1)       # [B, 1, C]
        
        # Create a global reference point at the center [0.5, 0.5]
        global_ref_point = torch.full((b, 1, 2), 0.5, device=value.device, dtype=value.dtype)

        # Concatenate prompts and global token
        query = torch.cat([content_embed, global_content], dim=1)        # [B, K+1, C]
        query_pos = torch.cat([pos_embed, global_pos], dim=1)            # [B, K+1, C]
        reference_points = torch.cat([reference_points_prompt, global_ref_point], dim=1) # [B, K+1, 2]
        
        # Expand reference points for multi-level attention: [B, K+1, L, 2]
        reference_points = reference_points.unsqueeze(2).repeat(1, 1, self.num_levels, 1)

        # --- 3. Run Deformable Cross-Attention (Eq 4) ---
        
        # Query: Prompt embeddings (query + pos)
        # Value: Image features (value + pos)
        # Reference Points: Prompt coordinates
        
        attn_output = self.cross_attn(
            query=query + query_pos,
            value=value + value_pos, # Add pos encoding to value
            key=None, # Key is same as Value
            query_pos=None,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios
        )
        
        # Residual connection and norm
        identity = query
        query = identity + self.dropout1(attn_output)
        query = self.cross_attn_norm(query)
        
        # --- 4. Run Aggregator (Eq 5) ---
        
        # 4.a. Self-Attention
        identity = query
        attn_output, _ = self.aggregator_self_attn(
            query=query,
            key=query,
            value=query
        )
        
        # Residual and norm
        query = identity + self.dropout2(attn_output)
        query = self.aggregator_self_attn_norm(query)
        
        # 4.b. FFN
        identity = query
        ffn_output = self.aggregator_ffn(query)
        
        # Residual and norm
        query = identity + self.dropout3(ffn_output)
        query = self.aggregator_ffn_norm(query)
        
        # --- 5. Get Final Embedding ---
        # The final visual embedding 'V' is the last token in the sequence,
        # which is our global aggregator token.
        visual_prompt_embedding = query[:, -1, :] # [B, C]
        
        return visual_prompt_embedding


if __name__ == "__main__":
    # Test the VisualPromptEncoder
    # We need to import the FullImageEncoder to generate its inputs
    
    # Temporarily add image_encoder to sys.path to import it
    import sys
    import os
    # Add the directory containing 'models' to the path
    # This is a bit of a hack for testing, but demonstrates the principle
    try:
        from image_encoder import FullImageEncoder, SinePositionalEncoding
        print("Successfully imported FullImageEncoder.")
    except ImportError:
        print("Could not import FullImageEncoder. Make sure image_encoder.py is in the same directory.")
        sys.exit(1)

    device = get_device()
    
    # 1. Init FullImageEncoder to get inputs
    image_encoder = FullImageEncoder().to(device)
    image_encoder.eval()

    # 2. Init VisualPromptEncoder
    visual_encoder = VisualPromptEncoder().to(device)
    visual_encoder.eval()
    
    # 3. Create dummy inputs
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    
    # Dummy prompts: 2 boxes, 4D (x_c, y_c, w, h) normalized
    dummy_boxes = torch.tensor([
        [0.25, 0.25, 0.1, 0.1],
        [0.75, 0.75, 0.2, 0.2]
    ]).unsqueeze(0).to(device) # Shape [1, 2, 4]

    # Dummy prompts: 3 points, 2D (x, y) normalized
    dummy_points = torch.tensor([
        [0.1, 0.1],
        [0.5, 0.5],
        [0.9, 0.9]
    ]).unsqueeze(0).to(device) # Shape [1, 3, 2]

    # 4. Perform forward pass
    with torch.no_grad():
        # Get outputs from the image encoder
        encoded_memory, projected_features = image_encoder(dummy_image)
        
        # We need the projected_features and their pos_embeddings
        # Let's generate the pos_embeddings just like the image_encoder does
        pos_embeddings = []
        for feat in projected_features:
            mask = torch.zeros(
                (feat.shape[0], feat.shape[2], feat.shape[3]), 
                device=feat.device, 
                dtype=torch.bool
            )
            pos_embeddings.append(image_encoder.position_encoder(mask))

        print("\n--- Testing with BOX prompts ---")
        v_embedding_box = visual_encoder(
            projected_features=projected_features,
            pos_embeddings=pos_embeddings,
            boxes=dummy_boxes
        )
        print(f"Box-prompted embedding shape: {v_embedding_box.shape}") # Expected: [1, 256]

        print("\n--- Testing with POINT prompts ---")
        v_embedding_point = visual_encoder(
            projected_features=projected_features,
            pos_embeddings=pos_embeddings,
            points=dummy_points
        )
        print(f"Point-prompted embedding shape: {v_embedding_point.shape}") # Expected: [1, 256]

    print("\nVisualPromptEncoder test successful!")