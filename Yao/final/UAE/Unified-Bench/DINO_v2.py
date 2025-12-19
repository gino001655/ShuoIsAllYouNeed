import torch
from torch.nn import functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

class Score_Cal():
    def __init__(self, model_name="facebook/dinov2-large"):
        """
        Initialize DINOv2 model and processor
        Args:
            model_name: DINOv2 model name, default uses facebook/dinov2-large
                       Options: facebook/dinov2-base, facebook/dinov2-large, facebook/dinov2-giant
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._load_model_and_transform()

    def _load_model_and_transform(self):
        """
        Load DINOv2 model and processor
        """
        try:
            print(f"Loading DINOv2 model: {self.model_name}")
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def encode_image(self, image):
        """
        Encode image and return feature vector
        Args:
            image: PIL Image, image path string, or torch.Tensor [b, 3, h, w]
        Returns:
            torch.Tensor: Image feature vector (using CLS token)
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                image_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
                image_features = F.normalize(image_features, p=2, dim=1)
        elif isinstance(image, Image.Image):
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                image_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
                image_features = F.normalize(image_features, p=2, dim=1)
        elif isinstance(image, torch.Tensor):
            # Handle tensor data in [b, 3, h, w] format
            if len(image.shape) != 4 or image.shape[1] != 3:
                raise ValueError("Tensor input must have shape [b, 3, h, w]")
            
            image = image.to(self.device)
            with torch.no_grad():
                # Use tensor directly as model input
                outputs = self.model(pixel_values=image)
                image_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
                image_features = F.normalize(image_features, p=2, dim=1)
        else:
            raise ValueError("Image must be PIL Image, file path string, or torch.Tensor [b, 3, h, w]")
        
        return image_features

    def calculate_similarity(self, image1, image2):
        """
        Calculate similarity between two images
        Args:
            image1: First image (PIL Image, path, or torch.Tensor [b, 3, h, w])
            image2: Second image (PIL Image, path, or torch.Tensor [b, 3, h, w])
        Returns:
            torch.Tensor: Similarity scores, returns batch results if input is batch data
        """
        features1 = self.encode_image(image1)
        features2 = self.encode_image(image2)
        
        # Calculate cosine similarity
        similarity = torch.cosine_similarity(features1, features2, dim=1)
        
        # Return scalar if single image, return tensor if batch
        if similarity.shape[0] == 1:
            return similarity.item()
        else:
            return similarity


# Example usage functions
def example_usage():
    """
    Usage example
    """
    # Initialize DINOv2 score calculator
    score_calculator = Score_Cal()
    
    # Example 1: Calculate similarity between two images (file paths)
    # similarity = score_calculator.calculate_similarity("image1.jpg", "image2.jpg")
    # print(f"Image similarity: {similarity:.4f}")
    
    # Example 2: Calculate batch image similarity (tensor format)
    # batch_size = 4
    # image1_batch = torch.randn(batch_size, 3, 224, 224)  # Random batch images 1
    # image2_batch = torch.randn(batch_size, 3, 224, 224)  # Random batch images 2
    # similarities = score_calculator.calculate_similarity(image1_batch, image2_batch)
    # print(f"Batch similarities shape: {similarities.shape}")
    # print(f"Batch similarities: {similarities}")
    
    print("DINOv2 Score calculator initialized successfully!")
    print("Supports input formats:")
    print("- PIL Image")
    print("- Image file path (string)")
    print("- torch.Tensor [b, 3, h, w] for batch processing")


if __name__ == "__main__":
    example_usage()