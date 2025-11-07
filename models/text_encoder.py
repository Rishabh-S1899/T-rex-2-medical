import torch
import torch.nn as nn
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

class CLIPTextEncoder(nn.Module):
    """
    Loads the pre-trained CLIP text encoder (CLIP-B/32).
    This module will be fine-tuned during training, as specified in the T-Rex2 paper.
    It takes a list of text strings and returns their [CLS] token embeddings.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initializes the CLIP text encoder and tokenizer.
        
        Args:
            model_name (str): The name of the Hugging Face model to load.
        """
        super().__init__()
        
        # Load the pre-trained model and tokenizer.
        # We use CLIPTextModelWithProjection to get the final text embeddings
        # that are projected into the shared (image-text) latent space.
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name, use_safetensors=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        
        # Get the dimension of the output embeddings (e.g., 512 for CLIP-B)
        self.embedding_dim = self.model.config.projection_dim

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Tokenizes and encodes a batch of text strings.
        
        Args:
            texts (list[str]): A list of text prompts (e.g., ["a person", "a dog"]).
        
        Returns:
            torch.Tensor: The projected [CLS] token embeddings, shape [B, embedding_dim].
        """
        # Tokenize the text.
        # - padding="max_length": Pad all sequences to the model's max length (77 for CLIP).
        # - truncation=True: Truncate any sequences longer than the max length.
        # - return_tensors="pt": Return PyTorch tensors.
        inputs = self.tokenizer(
            texts, 
            padding="max_length", 
            truncation=True, 
            max_length=77, # CLIP's standard max length
            return_tensors="pt"
        )
        
        # Move inputs to the same device as the model
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
        
        # Get the model outputs
        outputs = self.model(**inputs)
        
        # The 'pooler_output' is the projected [CLS] token embedding
        return outputs.text_embeds

if __name__ == "__main__":
    # This block allows us to run this file directly to test the module
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize our text encoder
    text_encoder = CLIPTextEncoder().to(device)
    text_encoder.eval() # Set to evaluation mode
    
    print(f"Loaded CLIPTextEncoder with model: 'openai/clip-vit-base-patch32'")
    print(f"Output embedding dimension: {text_encoder.embedding_dim}")
    
    # Create dummy text prompts
    dummy_prompts = ["a photo of a cat", "a drawing of a dog"]
    
    # Perform a forward pass
    with torch.no_grad(): # We don't need gradients for this test
        text_embeddings = text_encoder(dummy_prompts)
    
    print("\n--- Output Embedding Shape ---")
    print(f"Input prompts: {dummy_prompts}")
    print(f"Output shape: {text_embeddings.shape}")

    # Expected output shape for CLIP-B: [2, 512]
    # (Batch size 2, Embedding dim 512)