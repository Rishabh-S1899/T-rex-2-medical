import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Tuple, Optional, Dict

# Import all the modules we built
from image_encoder import FullImageEncoder
from text_encoder import CLIPTextEncoder
from visual_prompt_encoder import VisualPromptEncoder
from box_decoder import BoxDecoder, inverse_sigmoid

# Helper function for device
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class T_Rex2(nn.Module):
    """
    T-Rex2: The complete, assembled model.
    This class combines all the components we built:
    1. FullImageEncoder: Processes the image.
    2. CLIPTextEncoder: Encodes text prompts.
    3. VisualPromptEncoder: Encodes box/point prompts.
    4. BoxDecoder: Detects objects based on prompts and image.
    """
    def __init__(self,
                 # Model component configs
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 num_levels=4, # Number of feature levels from backbone
                 num_points=4, # Number of deformable points
                 num_queries=900,
                 # Backbone model names
                 swin_model_name='swin_tiny_patch4_window7_224',
                 clip_model_name='openai/clip-vit-base-patch32'):
        
        super().__init__()
        
        self.d_model = d_model
        self.num_levels = num_levels

        # --- 1. Image Encoder ---
        self.image_encoder = FullImageEncoder(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            backbone_model_name=swin_model_name,
            backbone_output_indices=tuple(range(num_levels))
        )

        # --- 2. Text Encoder ---
        self.text_encoder = CLIPTextEncoder(
            model_name=clip_model_name
        )
        
        # --- 3. Visual Prompt Encoder ---
        self.visual_prompt_encoder = VisualPromptEncoder(
            d_model=d_model,
            nhead=nhead,
            num_levels=num_levels,
            num_points=num_points,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # --- 4. Box Decoder ---
        self.box_decoder = BoxDecoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            num_levels=num_levels,
            num_points=num_points,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_queries=num_queries
        )

    def forward(self, 
                images: torch.Tensor,
                text_prompts: Optional[List[str]] = None,
                visual_prompts_boxes: Optional[torch.Tensor] = None,
                visual_prompts_points: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Main forward pass for the T-Rex2 model.
        
        Args:
            images (torch.Tensor): Batch of images, shape [B, 3, H, W].
            text_prompts (List[str], optional): List of text prompts.
            visual_prompts_boxes (torch.Tensor, optional): Box prompts, shape [B, K, 4].
            visual_prompts_points (torch.Tensor, optional): Point prompts, shape [B, K, 2].
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - "pred_logits": [L, B, Nq, 1] (L=num_decoder_layers)
                - "pred_boxes": [L, B, Nq, 4]
        """
        
        b = images.shape[0]
        
        # --- 1. Run Image Encoder ---
        # This is our "Phase 2.1" module
        encoded_memory, projected_features = self.image_encoder(images)
        
        # --- 2. Get Prompt Embedding ---
        # This is "Phase 1.2" or "Phase 2.2"
        if text_prompts is not None:
            # `text_prompts` is a list of strings, e.g., ["a dog", "a cat"]
            # We assume one prompt *per batch item*
            assert len(text_prompts) == b, "Number of text prompts must match batch size"
            prompt_embed = self.text_encoder(text_prompts) # [B, C]
        
        elif visual_prompts_boxes is not None or visual_prompts_points is not None:
            # We need the pos_embeddings for the projected_features
            pos_embeddings = []
            for feat in projected_features:
                mask = torch.zeros(
                    (feat.shape[0], feat.shape[2], feat.shape[3]), 
                    device=feat.device, 
                    dtype=torch.bool
                )
                pos_embeddings.append(self.image_encoder.position_encoder(mask))
            
            prompt_embed = self.visual_prompt_encoder(
                projected_features=projected_features,
                pos_embeddings=pos_embeddings,
                boxes=visual_prompts_boxes,
                points=visual_prompts_points
            ) # [B, C]
        
        else:
            raise ValueError("Must provide either text_prompts or visual_prompts")

        # --- 3. Prepare Inputs for Box Decoder ---
        # This is the "glue logic" from the test script in box_decoder.py
        
        pos_embeddings_list = []
        masks_list = []
        spatial_shapes = []
        for feat in projected_features:
            spatial_shapes.append(feat.shape[2:])
            mask = torch.zeros(
                (feat.shape[0], feat.shape[2], feat.shape[3]), 
                device=feat.device, 
                dtype=torch.bool
            )
            pos_embeddings_list.append(self.image_encoder.position_encoder(mask))
            masks_list.append(rearrange(mask, 'b h w -> b (h w)'))

        encoded_pos = torch.cat(
            [rearrange(p, 'b c h w -> b (h w) c') for p in pos_embeddings_list], 
            dim=1
        )
        key_padding_mask = torch.cat(masks_list, dim=1)
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=images.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)), 
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ))
        valid_ratios = torch.ones((b, self.num_levels, 2), device=images.device)
        
        # --- 4. Run Box Decoder ---
        # This is our "Phase 2.3" module
        all_layer_outputs, all_layer_boxes = self.box_decoder(
            encoded_memory=encoded_memory,
            encoded_pos=encoded_pos,
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios
        )
        # all_layer_outputs: [L, B, Nq, C]
        # all_layer_boxes: [L, B, Nq, 4]
        
        # --- 5. Calculate Classification Logits (Eq. 8) ---
        # This is the dot product between decoder queries and the prompt embedding
        
        # Reshape prompt_embed for broadcasting:
        # [B, C] -> [1, B, 1, C]
        prompt_embed_bcast = prompt_embed.unsqueeze(0).unsqueeze(2)
        
        # (all_layer_outputs * prompt_embed_bcast): [L, B, Nq, C]
        # .sum(-1): [L, B, Nq]
        # .unsqueeze(-1): [L, B, Nq, 1] (for compatibility with sigmoid loss)
        pred_logits = (all_layer_outputs * prompt_embed_bcast).sum(-1).unsqueeze(-1)
        
        return {
            "pred_logits": pred_logits,
            "pred_boxes": all_layer_boxes
        }


if __name__ == "__main__":
    # Test the full T_Rex2 model
    
    device = get_device()
    
    # 1. Init model
    d_model = 256
    num_queries = 900
    num_decoder_layers = 6
    
    model = T_Rex2(
        d_model=d_model,
        num_decoder_layers=num_decoder_layers,
        num_queries=num_queries
    ).to(device)
    model.eval()
    
    # 2. Create dummy inputs
    dummy_image = torch.randn(2, 3, 224, 224).to(device) # Batch size 2
    
    # 3. Test Text Prompt Workflow
    dummy_text = ["a photo of a dog", "a picture of a cat"] # Batch size 2
    
    with torch.no_grad():
        print("--- Testing Text Prompt Workflow ---")
        outputs_text = model(images=dummy_image, text_prompts=dummy_text)
        print(f"pred_logits shape: {outputs_text['pred_logits'].shape}")
        print(f"pred_boxes shape: {outputs_text['pred_boxes'].shape}")
        
        # Check shapes
        assert outputs_text['pred_logits'].shape == (num_decoder_layers, 2, num_queries, 1)
        assert outputs_text['pred_boxes'].shape == (num_decoder_layers, 2, num_queries, 4)
        print("Text prompt test successful!")

    # 4. Test Visual Prompt (Box) Workflow
    dummy_boxes = torch.tensor([
        # Batch item 1
        [[0.25, 0.25, 0.1, 0.1], [0.75, 0.75, 0.2, 0.2]],
        # Batch item 2
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.1, 0.1]]
    ]).to(device) # Shape [2, 2, 4]
    
    with torch.no_grad():
        print("\n--- Testing Visual (Box) Prompt Workflow ---")
        outputs_box = model(images=dummy_image, visual_prompts_boxes=dummy_boxes)
        print(f"pred_logits shape: {outputs_box['pred_logits'].shape}")
        print(f"pred_boxes shape: {outputs_box['pred_boxes'].shape}")
        
        # Check shapes
        assert outputs_box['pred_logits'].shape == (num_decoder_layers, 2, num_queries, 1)
        assert outputs_box['pred_boxes'].shape == (num_decoder_layers, 2, num_queries, 4)
        print("Visual prompt (box) test successful!")
        
    print("\nFull T-Rex2 model test successful!")