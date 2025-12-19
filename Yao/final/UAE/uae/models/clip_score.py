import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel, AutoProcessor
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, center_crop


class ClipScoreTransform_in_Dataset:
    def __init__(self, tar_size, resize_mode, normalize_mean, normalize_std):
        self.tar_size = tar_size
        self.resize_mode = resize_mode 
        self.normalize_mean = normalize_mean 
        self.normalize_std = normalize_std

    def __call__(self, images):
        # ① Normalize to [0,1]
        images = images * 0.00392156862745098

        batch = []
        B = images.size(0)
        for i in range(B):
            img = images[i]

            # ② Resize: shortest side to 224, maintain aspect ratio
            _, h, w = img.shape
            scale = 224 / min(h, w)
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            # img = resize(img, [new_h, new_w], interpolation=F.InterpolationMode.BICUBIC)
            img = F.interpolate(
                img.unsqueeze(0), size=(self.tar_size, self.tar_size), mode=self.resize_mode, align_corners=False
            )
            # ③ Center-crop to 224×224
            # img = center_crop(img, [224, 224])

            batch.append(img.squeeze(dim=0))

        images = torch.stack(batch, dim=0)  # (B, 3, 224, 224)

        # ④ Normalize by CLIP's mean/std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=images.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=images.device).view(1, 3, 1, 1)

        images = (images - mean) / std

        return images

class ClipScore:
    def __init__(self, clip_name='clip-vit-large-patch14'):
        self.clip_name = clip_name
        self.clip_model, self.clip_processor, self.tar_size, \
            self.resize_mode, self.normalize_mean, self.normalize_std = \
                self.load_model(clip_name)

    def load_model(self, clip_name='clip-vit-large-patch14'):
        """
        Load the CLIP model.
        """
        if clip_name is None:
            clip_name = self.clip_name

        if clip_name == 'clip-vit-large-patch14':
            from transformers import CLIPProcessor, CLIPModel
            model = CLIPModel.from_pretrained("/root/paddlejob/workspace/env_run/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41")
            processor = CLIPProcessor.from_pretrained("/root/paddlejob/workspace/env_run/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41")
            tar_size, resize_mode, normalize_mean, normalize_std = self.clip_process_params()

        return model, processor, tar_size, resize_mode, normalize_mean, normalize_std
    
    @staticmethod
    def clip_process_params():
        resize_size = 224
        resize_mode = 'bicubic'
        normalize_mean = (0.48145466, 0.4578275, 0.40821073)
        normalize_std = (0.26862954, 0.26130258, 0.27577711)
        return resize_size, resize_mode, normalize_mean, normalize_std

    def get_clip_transform(self):
        return ClipScoreTransform_in_Dataset(self.tar_size, self.resize_mode, self.normalize_mean, self.normalize_std)

    def preprocess_clip_tensor_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, H, W)  ── pixel values in range [0, 255], dtype=float32/float64/uint8 all supported
        returns: (B, 3, 224, 224) ── normalized CLIP input
        """
        
        # ① Normalize to [0,1]
        images = images * 0.00392156862745098

        batch = []
        B = images.size(0)
        for i in range(B):
            img = images[i]

            # ② Resize: shortest side to 224, maintain aspect ratio
            _, h, w = img.shape
            scale = 224 / min(h, w)
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            # img = resize(img, [new_h, new_w], interpolation=F.InterpolationMode.BICUBIC)
            img = F.interpolate(
                img.unsqueeze(0), size=(self.tar_size, self.tar_size), mode=self.resize_mode, align_corners=False
            )
            # ③ Center-crop to 224×224
            # img = center_crop(img, [224, 224])

            batch.append(img.squeeze(dim=0))

        images = torch.stack(batch, dim=0)  # (B, 3, 224, 224)

        # ④ Normalize by CLIP's mean/std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=images.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=images.device).view(1, 3, 1, 1)

        images = (images - mean) / std
        return images

    @staticmethod    
    def normalize_tensor(image_tensor, mean, std):
        # Convert mean and std to tensors, ensure their shapes match the input image_tensor
        mean = torch.tensor(mean, device=image_tensor.device, dtype=image_tensor.dtype)
        std = torch.tensor(std, device=image_tensor.device, dtype=image_tensor.dtype)
        
        # Normalize the image_tensor
        # Since image_tensor shape is (1, 3, 224, 224), mean and std should be expanded to (3, 1, 1)
        return (image_tensor - mean[:, None, None]) / std[:, None, None]

    def __call__(self, images: torch.Tensor, gt_images):
        """
        Forward pass for the preprocess and calculate the score.

        """
        self.clip_model.eval()
        
        images = self.preprocess_clip_tensor_batch(images)
        image_features = self.clip_model.get_image_features(images)
        with torch.no_grad():        
            gt_image_features = self.clip_model.get_image_features(gt_images)
        
        # Calculate the cosine similarity between the image features and gt image features
        image_features = F.normalize(image_features, dim=-1)        # dim: [B, D]
        gt_image_features = F.normalize(gt_image_features, dim=-1)      # dim: [B, D]
        
        sim = (image_features * gt_image_features).sum(dim=-1)       # shape [B]        
        return (1-sim).mean()


def preprocess_clip_tensor_batch(images: torch.Tensor) -> torch.Tensor:
    """
    images: (B, 3, H, W)  ── pixel values in range [0, 255], dtype=float32/float64/uint8 all supported
    returns: (B, 3, 224, 224) ── normalized CLIP input
    """
    
    # ① Normalize to [0,1]
    # images = images * 0.00392156862745098

    batch = []
    B = images.size(0)
    for i in range(B):
        img = images[i]

        # ② Resize: shortest side to 224, maintain aspect ratio
        _, h, w = img.shape

        # img = F.interpolate(
        #     img.unsqueeze(0), size=(224, 224), mode='bicubic', align_corners=False
        # )
        # ③ Center-crop to 224×224
        # img = center_crop(img, [224, 224])

        batch.append(img.squeeze(dim=0))

    images = torch.stack(batch, dim=0)  # (B, 3, 224, 224)

    # ④ Normalize by CLIP's mean/std
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=images.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                        device=images.device).view(1, 3, 1, 1)

    images = (images - mean) / std
    return images

def load_image_safe(img_path):
    try:
        img = Image.open(img_path)
        if img.mode == 'P': 
            return None
        else:
            img = img.convert("RGB")
        return img
    except ValueError as e:
        print(f"Error loading image {img_path}: {e}")
        return Image.open("/root/paddlejob/workspace/env_run/100.png").convert("RGB")
    
    

if __name__ == '__main__':
    from transformers import AutoModel, AutoProcessor
    from transformers.image_utils import load_image
    from transformers import CLIPProcessor, CLIPModel
    import numpy as np
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    from PIL import Image
    
    image_np = load_image_safe('/home/tione/notebook/linkaiqing/code/Unigen/tmp_debug/2.png')
    processor.image_processor.do_center_crop = False
    processor.image_processor.do_resize = True
    processor.image_processor.do_rescale = False
    processor.image_processor.do_normalize = False

    img = processor.image_processor(image_np, return_tensors='pt')['pixel_values'][0]

    image_np = load_image_safe('/home/tione/notebook/linkaiqing/code/Unigen/tmp_debug/2.png')
    image_pt = torch.FloatTensor(np.array(image_np)).to(torch.float32)
    image_pt = image_pt.permute(2, 0, 1)
    image_pt = image_pt.unsqueeze(0)
    image_pt = preprocess_clip_tensor_batch(image_pt)
    image_pt = image_pt.to(torch.float32).squeeze()

    print(1)


