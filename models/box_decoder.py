import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Tuple, Optional

# --- Core Deformable Attention Operator ---
# We copy this from image_encoder.py
try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
except ImportError:
    print("---IMPORT ERROR ---")
    print("Could not import MultiScaleDeformableAttention from mmcv.ops")
    print("Please ensure mmcv is installed correctly for your PyTorch/CUDA version.")
    MultiScaleDeformableAttention = None 

# --- Copy Helper function to generate reference points ---
# We need this again for the decoder's cross-attention
def _generate_reference_points(spatial_shapes: torch.Tensor, 
                             valid_ratios: torch.Tensor, 
                             device: torch.device) -> torch.Tensor:
    """
    Generates normalized (0, 1) reference points for all pixels in a feature map.
    
    Args:
        spatial_shapes (torch.Tensor): Shape [L, 2] of feature maps.
        valid_ratios (torch.Tensor): Shape [B, L, 2] valid ratios.
        device: The device to create the tensors on.
        
    Returns:
        torch.Tensor: Reference points, shape [B, S_total, L, 2].
    """
    reference_points_list = []
    for lvl, (H, W) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        vr_lvl = valid_ratios[:, lvl]
        ref_y = ref_y.unsqueeze(0) / (vr_lvl[:, 1].unsqueeze(1).unsqueeze(2) * H)
        ref_x = ref_x.unsqueeze(0) / (vr_lvl[:, 0].unsqueeze(1).unsqueeze(2) * W)
        ref = torch.stack((ref_x, ref_y), -1).flatten(1, 2)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points.unsqueeze(2).repeat(1, 1, len(spatial_shapes), 1)
    return reference_points

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def inverse_sigmoid(x, eps=1e-6):
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


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

# --- Deformable Transformer DECODER Layer ---
# This is DIFFERENT from the Encoder Layer
class DeformableDetrTransformerDecoderLayer(nn.Module):
    """
    Vendored implementation of the Deformable DETR Decoder Layer.
    """
    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_points: int = 4,
                 feedforward_channels: int = 1024,
                 dropout: float = 0.1):
        super().__init__()

        # 1. Standard Self-Attention (Query-to-Query)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Our convention is Batch First
        )
        self.attn_norm = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)

        # 2. Deformable Cross-Attention (Query-to-Image)
        self.cross_attn = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dims)
        self.dropout2 = nn.Dropout(dropout)

        # 3. Feed-Forward Network (FFN)
        self.ffn = MLP(embed_dims, feedforward_channels, embed_dims, 2)
        self.ffn_norm = nn.LayerNorm(embed_dims)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                query_pos: torch.Tensor,
                key: torch.Tensor, # This is the encoded_memory
                key_pos: torch.Tensor, # This is the pos encodings for the memory
                key_padding_mask: torch.Tensor,
                reference_points: torch.Tensor, # The (x,y) coords for the query
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor,
                valid_ratios: torch.Tensor) -> torch.Tensor:
        
        # 1. Standard Self-Attention (query -> query)
        q = k = query + query_pos
        attn_output, _ = self.self_attn(q, k, value=query)
        
        identity = query
        query = identity + self.dropout1(attn_output)
        query = self.attn_norm(query)
        
        # 2. Deformable Cross-Attention (query -> key/value)
        attn_query = query + query_pos
        
        attn_output = self.cross_attn(
            query=attn_query,
            value=key + key_pos, # Add pos encoding to image memory
            key=None,
            query_pos=None,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            key_padding_mask=key_padding_mask
        )
        
        identity = query
        query = identity + self.dropout2(attn_output)
        query = self.cross_attn_norm(query)

        # 3. FFN
        identity = query
        ffn_output = self.ffn(query)
        query = identity + self.dropout3(ffn_output)
        query = self.ffn_norm(query)

        return query

class DeformableDetrTransformerDecoder(nn.Module):
    """
    A stack of DeformableDetrTransformerDecoderLayer.
    This also handles the iterative refinement of reference points (boxes).
    """
    def __init__(self,
                 num_layers: int,
                 layer_cfg: dict,
                 d_model: int):
        super().__init__()
        self.layers = nn.ModuleList([
            DeformableDetrTransformerDecoderLayer(**layer_cfg)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.num_levels = layer_cfg["num_levels"]
        # --- Box Prediction Head ---
        # This is shared across all decoder layers
        self.box_prediction_head = MLP(d_model, d_model, 4, 3) # 4-dim (x, y, w, h)
        
        # Initialize the bias of the final box prediction layer to 0
        nn.init.constant_(self.box_prediction_head.layers[-1].bias.data, 0)

    def forward(self,
                query: torch.Tensor,
                query_pos: torch.Tensor,
                key_memory: torch.Tensor,
                key_pos: torch.Tensor,
                key_padding_mask: torch.Tensor,
                reference_points: torch.Tensor, # Initial reference points
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor,
                valid_ratios: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        output = query
        intermediate_outputs = []
        intermediate_reference_points = []
        
        for i, layer in enumerate(self.layers):
            # The reference points for deformable attention are the *predicted boxes*
            # from the previous layer.
            # For the first layer, it uses the initial reference_points.
            
            # Expand ref_points from [B, Nq, 2] -> [B, Nq, L, 2]
            # Ensure reference_points always have shape [B, Nq, num_levels, 2]
            if reference_points.shape[-1] == 4:
                # Keep only center (x, y) for attention reference points
                reference_points_input = reference_points[..., :2].unsqueeze(2).repeat(1, 1, self.num_levels, 1)
            else:
                reference_points_input = reference_points.unsqueeze(2).repeat(1, 1, self.num_levels, 1)

            
            output = layer(
                query=output,
                query_pos=query_pos,
                key=key_memory,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios
            )
            
            # --- Iterative Box Refinement ---
            # Predict box *deltas* (offsets) from the current layer's output
            delta_boxes = self.box_prediction_head(output)
            
            # Add deltas to the previous reference points to get new reference points
            # 1. inverse_sigmoid to convert normalized (0,1) boxes to logits
            # 2. add the predicted deltas
            # 3. sigmoid to convert back to normalized (0,1) boxes
            if reference_points.shape[-1] == 4: # (x,y,w,h)
                # After the first layer, ref_points are 4D
                new_reference_points = (delta_boxes + inverse_sigmoid(reference_points)).sigmoid()
            else: # (x,y)
                # For the first layer, ref_points are 2D
                # Add 0s for (w,h) deltas
                delta_boxes[..., :2] += inverse_sigmoid(reference_points)
                new_reference_points = delta_boxes.sigmoid()

            # The new reference points are detached, as gradients don't flow
            # back through the reference point creation.
            reference_points = new_reference_points.detach()
            
            intermediate_outputs.append(output)
            intermediate_reference_points.append(new_reference_points)

        # Stack the intermediate outputs from all decoder layers
        # [L, B, Nq, C]
        all_layer_outputs = torch.stack(intermediate_outputs)
        # [L, B, Nq, 4]
        all_layer_boxes = torch.stack(intermediate_reference_points)

        return all_layer_outputs, all_layer_boxes


class BoxDecoder(nn.Module):
    """
    The main Box Decoder module.
    This wraps the Deformable Transformer Decoder.
    """
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_decoder_layers=6,
                 num_levels=4,
                 num_points=4,
                 dim_feedforward=1024,
                 dropout=0.1,
                 num_queries=900):
        super().__init__()

        self.d_model = d_model
        self.num_queries = num_queries
        self.num_levels = num_levels

        # --- 1. Learnable Queries ---
        # The 900 "object queries"
        self.query_embedding = nn.Embedding(num_queries, d_model)
        # The 900 positional encodings for the queries
        self.query_pos_embedding = nn.Embedding(num_queries, d_model)
        
        # --- 2. The Decoder Stack ---
        self.decoder = DeformableDetrTransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            layer_cfg=dict(
                embed_dims=d_model,
                num_heads=nhead,
                num_levels=num_levels,
                num_points=num_points,
                feedforward_channels=dim_feedforward,
                dropout=dropout
            )
        )
        
        # --- 3. Reference Point Embedder ---
        # A simple MLP to create the *initial* reference points (x,y)
        # from the query's positional embedding.
        self.reference_point_embed = nn.Linear(d_model, 2)

    def forward(self, 
                encoded_memory: torch.Tensor,
                encoded_pos: torch.Tensor,
                key_padding_mask: torch.Tensor,
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor,
                valid_ratios: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoded_memory (torch.Tensor): [B, S_total, C] from FullImageEncoder
            encoded_pos (torch.Tensor): [B, S_total, C] pos encodings for memory
            key_padding_mask (torch.Tensor): [B, S_total] mask for memory
            spatial_shapes (torch.Tensor): [L, 2]
            level_start_index (torch.Tensor): [L]
            valid_ratios (torch.Tensor): [B, L, 2]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - all_layer_outputs: [L, B, Nq, C] (L=num_decoder_layers, Nq=num_queries)
            - all_layer_boxes: [L, B, Nq, 4]
        """
        b = encoded_memory.shape[0] # Batch size
        
        # --- 1. Get query embeddings ---
        # [Nq, C] -> [1, Nq, C] -> [B, Nq, C]
        query_embed = self.query_embedding.weight.unsqueeze(0).repeat(b, 1, 1)
        query_pos = self.query_pos_embedding.weight.unsqueeze(0).repeat(b, 1, 1)

        # --- 2. Generate initial reference points (boxes) ---
        # [B, Nq, C] -> [B, Nq, 2] -> sigmoid -> [B, Nq, 2]
        # This creates the initial (x,y) coordinates for each query
        initial_reference_points = self.reference_point_embed(query_pos).sigmoid()

        # --- 3. Run the Decoder ---
        all_layer_outputs, all_layer_boxes = self.decoder(
            query=query_embed,
            query_pos=query_pos,
            key_memory=encoded_memory,
            key_pos=encoded_pos,
            key_padding_mask=key_padding_mask,
            reference_points=initial_reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios
        )

        return all_layer_outputs, all_layer_boxes


if __name__ == "__main__":
    # Test the BoxDecoder
    # We need to import the FullImageEncoder to generate its inputs
    
    import sys
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

    # 2. Init BoxDecoder
    d_model = 256
    num_queries = 900
    num_decoder_layers = 6
    box_decoder = BoxDecoder(
        d_model=d_model,
        num_decoder_layers=num_decoder_layers,
        num_queries=num_queries
    ).to(device)
    box_decoder.eval()
    
    # 3. Create dummy inputs
    dummy_image = torch.randn(1, 3, 224, 224).to(device) # Batch size 1
    
    # 4. Perform forward pass
    with torch.no_grad():
        # --- Get Image Encoder outputs ---
        encoded_memory, projected_features = image_encoder(dummy_image)
        
        # --- Prepare inputs for the Decoder ---
        # 1. Create pos embeddings for the memory
        pos_embeddings = []
        masks = []
        spatial_shapes = []
        for feat in projected_features:
            spatial_shapes.append(feat.shape[2:])
            mask = torch.zeros(
                (feat.shape[0], feat.shape[2], feat.shape[3]), 
                device=feat.device, 
                dtype=torch.bool
            )
            pos_embeddings.append(image_encoder.position_encoder(mask))
            masks.append(rearrange(mask, 'b h w -> b (h w)'))

        encoded_pos = torch.cat(
            [rearrange(p, 'b c h w -> b (h w) c') for p in pos_embeddings], 
            dim=1
        )
        key_padding_mask = torch.cat(masks, dim=1)
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)), 
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ))
        valid_ratios = torch.ones((1, len(projected_features), 2), device=device)
        
        print("\n--- Decoder Input Shapes ---")
        print(f"encoded_memory: {encoded_memory.shape}")
        print(f"encoded_pos: {encoded_pos.shape}")
        print(f"key_padding_mask: {key_padding_mask.shape}")
        print(f"spatial_shapes: {spatial_shapes}")
        
        # --- Get Decoder outputs ---
        all_layer_outputs, all_layer_boxes = box_decoder(
            encoded_memory=encoded_memory,
            encoded_pos=encoded_pos,
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios
        )

        print("\n--- Decoder Output Shapes ---")
        print(f"all_layer_outputs shape: {all_layer_outputs.shape}")
        print(f"all_layer_boxes shape: {all_layer_boxes.shape}")
        
        # --- Check shapes ---
        assert all_layer_outputs.shape == (num_decoder_layers, 1, num_queries, d_model)
        assert all_layer_boxes.shape == (num_decoder_layers, 1, num_queries, 4)
        
    print("\nBoxDecoder test successful!")