import sys
import warnings
import torch
from torch.nn import functional as F
sys.path.append("/home/tione/notebook/linkaiqing/code/OmiAE/flow_grpo/flow_grpo")
from long_clip.model import longclip
from PIL import Image


class Score_Cal():
    def __init__(self, model_path="/home/tione/notebook/linkaiqing/code/OmiAE/flow_grpo/flow_grpo/long_clip/checkpoints/longclip-L.pt"):
        """
        初始化LongCLIP模型和处理器
        Args:
            model_path: LongCLIP模型路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.tform = None
        self._load_model_and_transform()

    def _load_model_and_transform(self):
        """
        加载LongCLIP模型和处理器
        """
        try:
            print(f"Loading LongCLIP model from: {self.model_path}")
            self.processor = longclip.tokenize
            self.model, self.tform = longclip.load(self.model_path, device=self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def _process_image(self, image):
        """
        处理单张图像，转换为模型输入格式
        Args:
            image: PIL Image 或 torch.Tensor
        Returns:
            torch.Tensor: 处理后的图像张量
        """
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            image = image.mul(255).byte().cpu().numpy().astype(np.uint8)
            image = image.transpose(1, 2, 0)  # C,H,W -> H,W,C
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        tensor_image = self.tform(pil_image)
        return tensor_image

    def _process_images(self, images):
        """
        批量处理图像
        Args:
            images: 图像列表
        Returns:
            torch.Tensor: 批量图像张量
        """
        dtype = self.model.dtype
        pixel_list = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            tensor_image = self._process_image(image)
            pixel_list.append(tensor_image)
        pixels = torch.stack(pixel_list, dim=0)
        pixels = pixels.to(dtype=dtype, device=self.device)
        return pixels

    def encode_image(self, image):
        """
        编码单张图像，返回特征向量
        Args:
            image: PIL Image 或 图像路径字符串
        Returns:
            torch.Tensor: 图像特征向量
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image or file path string")
        
        with torch.no_grad():
            pixels = self._process_images([image])
            image_features = self.model.encode_image(pixels)
            # 归一化特征向量
            image_features = F.normalize(image_features, p=2, dim=1)
        
        return image_features

    def encode_text(self, text):
        """
        编码文本，返回特征向量
        Args:
            text: 文本字符串或文本列表
        Returns:
            torch.Tensor: 文本特征向量
        """
        with torch.no_grad():
            if isinstance(text, str):
                text = [text]
            texts = self.processor(text, truncate=True).to(self.device)
            text_features = self.model.encode_text(texts)
            # 归一化特征向量
            text_features = F.normalize(text_features, p=2, dim=1)
        
        return text_features

    def calculate_similarity(self, image1, image2):
        """
        计算两张图像之间的相似度
        Args:
            image1: 第一张图像 (PIL Image 或 路径)
            image2: 第二张图像 (PIL Image 或 路径)
        Returns:
            float: 相似度分数 (0-1之间)
        """
        features1 = self.encode_image(image1)
        features2 = self.encode_image(image2)
        
        # 计算余弦相似度
        similarity = torch.cosine_similarity(features1, features2, dim=1)
        return similarity.item()

    def calculate_text_image_similarity(self, text, image):
        """
        计算文本和图像之间的相似度
        Args:
            text: 文本描述
            image: 图像 (PIL Image 或 路径)
        Returns:
            float: 相似度分数 (0-1之间)
        """
        text_features = self.encode_text(text)
        image_features = self.encode_image(image)
        
        # 计算余弦相似度
        similarity = torch.cosine_similarity(text_features, image_features, dim=1)
        return similarity.item()

    def batch_text_image_similarity(self, texts, images):
        """
        批量计算文本和图像之间的相似度 (对应clip_scorer.py中的__call__方法)
        Args:
            texts: 文本列表
            images: 图像列表 (PIL Images 或 路径列表)
        Returns:
            torch.Tensor: 相似度分数张量
        """
        with torch.no_grad():
            # 处理文本
            if isinstance(texts, str):
                texts = [texts]
            text_tokens = self.processor(texts, truncate=True).to(self.device)
            
            # 处理图像
            if isinstance(images, (str, Image.Image)):
                images = [images]
            
            processed_images = []
            for img in images:
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                processed_images.append(img)
            
            pixels = self._process_images(processed_images)
            
            # 获取特征
            image_features = self.model.encode_image(pixels)
            text_features = self.model.encode_text(text_tokens)
            
            # 计算相似度矩阵
            logits_per_image = image_features @ text_features.T
            
            # 返回对角线元素 (对应的文本-图像对的相似度)，除以30进行缩放
            return logits_per_image.diagonal() / 30

    def image_similarity(self, images, ref_images):
        """
        计算图像与参考图像之间的相似度 (对应clip_scorer.py中的image_similarity方法)
        Args:
            images: 图像列表
            ref_images: 参考图像列表
        Returns:
            torch.Tensor: 相似度分数张量
        """
        with torch.no_grad():
            # 处理图像
            if isinstance(images, (str, Image.Image)):
                images = [images]
            if isinstance(ref_images, (str, Image.Image)):
                ref_images = [ref_images]
            
            processed_images = []
            for img in images:
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                processed_images.append(img)
            
            processed_ref_images = []
            for img in ref_images:
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                processed_ref_images.append(img)
            
            pixels = self._process_images(processed_images)
            ref_pixels = self._process_images(processed_ref_images)
            
            # 获取图像特征
            pixel_embeds = self.model.encode_image(pixels)
            ref_embeds = self.model.encode_image(ref_pixels)
            
            # 归一化
            pixel_embeds = pixel_embeds / pixel_embeds.norm(p=2, dim=-1, keepdim=True)
            ref_embeds = ref_embeds / ref_embeds.norm(p=2, dim=-1, keepdim=True)
            
            # 计算相似度矩阵
            sim = pixel_embeds @ ref_embeds.T
            sim = torch.diagonal(sim, 0)
            
            return sim


def main():
    """
    测试函数
    """
    # try:
    scorer = Score_Cal()
        
    #     # 测试文本-图像相似度
    #     test_text = "A beautiful landscape with mountains"
    #     test_image_path = "/path/to/test/image.jpg"  # 请替换为实际的图像路径
        
    #     print("LongCLIP Score Calculator initialized successfully!")
    #     print("You can use the following methods:")
    #     print("- calculate_text_image_similarity(text, image)")
    #     print("- calculate_similarity(image1, image2)")
    #     print("- batch_text_image_similarity(texts, images)")
    #     print("- image_similarity(images, ref_images)")
        
    # except Exception as e:
    #     print(f"Error in main: {e}")
    #     print("Please check if the LongCLIP model path is correct and the model files exist.")


if __name__ == "__main__":
    main()
