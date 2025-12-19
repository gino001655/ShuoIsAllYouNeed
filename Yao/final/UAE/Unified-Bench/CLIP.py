import torch
from torch.nn import functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class Score_Cal():
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        """
        Initialize CLIP model and processor
        Args:
            model_name: CLIP model name, default uses openai/clip-vit-large-patch14
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._load_model_and_transform()

    def _load_model_and_transform(self):
        """
        Load CLIP model and processor
        """
        try:
            print(f"Loading CLIP model: {self.model_name}")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def encode_image(self, image):
        """
        Encode a single image and return feature vector
        Args:
            image: PIL Image or image path string
        Returns:
            torch.Tensor: Image feature vector
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image or file path string")
        
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            # Normalize feature vectors
            image_features = F.normalize(image_features, p=2, dim=1)
        
        return image_features

    def encode_text(self, text):
        """
        Encode text and return feature vector
        Args:
            text: Text string or list of text strings
        Returns:
            torch.Tensor: Text feature vector
        """
        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            # Normalize feature vectors
            text_features = F.normalize(text_features, p=2, dim=1)
        
        return text_features

    def calculate_similarity(self, image1, image2):
        """
        Calculate similarity between two images
        Args:
            image1: First image (PIL Image or path)
            image2: Second image (PIL Image or path)
        Returns:
            float: Similarity score (between 0-1)
        """
        features1 = self.encode_image(image1)
        features2 = self.encode_image(image2)
        
        # Calculate cosine similarity
        similarity = torch.cosine_similarity(features1, features2, dim=1)
        return similarity.item()

    def calculate_text_image_similarity(self, text, image):
        """
        Calculate similarity between text and image
        Args:
            text: Text description
            image: Image (PIL Image or path)
        Returns:
            float: Similarity score (between 0-1)
        """
        text_features = self.encode_text(text)
        image_features = self.encode_image(image)
        
        # Calculate cosine similarity
        similarity = torch.cosine_similarity(text_features, image_features, dim=1)
        return similarity.item()
