import torch
from torch.nn import functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

class DinoV3Score():
    def __init__(self, model_name="/home/tione/notebook/linkaiqing/code/Unigen_08_08/dinov3/dinov3-vitl16-pretrain-lvd1689m",
                 device=None):
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._load_model_and_transform()

    def _load_model_and_transform(self):
        """
        加载DINOv3模型和处理器
        """
        try:
            print(f"Loading DINOv3 model: {self.model_name}")
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def encode_image(self, image):
        """
        编码图像，返回特征向量
        Args:
            image: PIL Image, 图像路径字符串, 或 torch.Tensor [b, 3, h, w]
        Returns:
            torch.Tensor: 图像特征向量 (使用CLS token)
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
            # 处理 [b, 3, h, w] 格式的tensor数据
            if len(image.shape) != 4 or image.shape[1] != 3:
                raise ValueError("Tensor input must have shape [b, 3, h, w]")
            
            image = image.to(self.device)
            with torch.no_grad():
                # 直接使用tensor作为模型输入
                outputs = self.model(pixel_values=image)
                image_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
                image_features = F.normalize(image_features, p=2, dim=1)
        else:
            raise ValueError("Image must be PIL Image, file path string, or torch.Tensor [b, 3, h, w]")
        
        return image_features

    def calculate_similarity(self, image1, image2):
        """
        计算两张图像之间的相似度
        Args:
            image1: 第一张图像 (PIL Image, 路径, 或 torch.Tensor [b, 3, h, w])
            image2: 第二张图像 (PIL Image, 路径, 或 torch.Tensor [b, 3, h, w])
        Returns:
            torch.Tensor: 相似度分数，如果输入是批量数据则返回批量结果
        """
        features1 = self.encode_image(image1)
        features2 = self.encode_image(image2)
        
        # 计算余弦相似度
        similarity = torch.cosine_similarity(features1, features2, dim=1)
        
        # 如果是单张图像，返回标量；如果是批量，返回tensor
        if similarity.shape[0] == 1:
            return similarity.item()
        else:
            return similarity


# 使用示例函数
def example_usage():
    """
    使用示例
    """
    # 初始化DINOv2评分计算器
    score_calculator = DinoV3Score()
    
    # 示例1: 计算两张图像的相似度（文件路径）
    # similarity = score_calculator.calculate_similarity("image1.jpg", "image2.jpg")
    # print(f"Image similarity: {similarity:.4f}")
    
    # 示例2: 计算批量图像的相似度（tensor格式）
    # batch_size = 4
    # image1_batch = torch.randn(batch_size, 3, 224, 224)  # 随机批量图像1
    # image2_batch = torch.randn(batch_size, 3, 224, 224)  # 随机批量图像2
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