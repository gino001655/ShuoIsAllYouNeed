"""
DLCVLayoutDataset with Index-based Caption Matching
直接用 index 從 caption.json 查找 caption，支持 TAData 的 Image 對象
"""

import json
import re
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset


class DLCVLayoutDatasetIndexed(Dataset):
    """
    支持直接讀取 TAData (Image 對象) + 用 index 匹配 caption.json
    """
    
    def __init__(
        self,
        data_dir,
        caption_json_path=None,
        enable_debug=False,
        max_layers=20,
    ):
        """
        Args:
            data_dir: TAData 目錄路徑
            caption_json_path: caption.json 的路徑
            enable_debug: 是否顯示 debug 資訊
            max_layers: 最大 layer 數量
        """
        self.data_dir = Path(data_dir)
        self.enable_debug = enable_debug
        self.max_layers = max_layers
        
        # 載入 dataset（使用 pyarrow 避免 metadata 問題）
        print(f"載入 dataset from {data_dir}...")
        try:
            self.dataset = load_dataset(
                'parquet',
                data_dir=str(self.data_dir),
                split='train'
            )
            print(f"✓ 載入 {len(self.dataset)} 個樣本")
        except (TypeError, KeyError) as e:
            # Fallback: 直接用 pyarrow 讀取，避免 metadata 解析問題
            print(f"  ⚠️  load_dataset 失敗，使用 pyarrow 直接讀取...")
            import pyarrow.parquet as pq
            from pathlib import Path
            
            # 找到所有 parquet 文件
            parquet_files = sorted(self.data_dir.glob("*.parquet"))
            if not parquet_files:
                # 如果 data_dir 是 data/ 目錄，則在裡面找
                data_subdir = self.data_dir / "data"
                if data_subdir.exists():
                    parquet_files = sorted(data_subdir.glob("*.parquet"))
            
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in {self.data_dir}")
            
            print(f"  找到 {len(parquet_files)} 個 parquet 文件")
            
            # 讀取所有文件
            all_rows = []
            for pf in parquet_files:
                table = pq.read_table(str(pf))
                for i in range(len(table)):
                    row = {col: table[col][i].as_py() for col in table.column_names}
                    all_rows.append(row)
            
            self.dataset = all_rows
            print(f"  ✓ 載入 {len(self.dataset)} 個樣本")
        
        # 載入 caption mapping (index-based)
        self.caption_mapping = self._load_caption_mapping(caption_json_path)
        
    def _load_caption_mapping(self, caption_json_path):
        """
        載入 caption.json 並轉換為 index-based mapping
        
        從路徑 "/workspace/.../00000123.png" 提取數字 123
        建立 mapping: {123: "caption text", ...}
        """
        if not caption_json_path:
            print("未提供 caption_json_path，將使用 dataset 內的 title")
            return {}
        
        caption_json_path = Path(caption_json_path)
        if not caption_json_path.exists():
            print(f"⚠️  Caption JSON 不存在: {caption_json_path}")
            return {}
        
        print(f"載入 caption mapping from {caption_json_path}...")
        with open(caption_json_path, 'r', encoding='utf-8') as f:
            path_based_captions = json.load(f)
        
        # 轉換為 index-based
        index_based_captions = {}
        pattern = re.compile(r'(\d{8})\.png$')  # 匹配 00000123.png
        
        for path, caption in path_based_captions.items():
            match = pattern.search(path)
            if match:
                idx = int(match.group(1))  # 提取數字，去掉前導 0
                index_based_captions[idx] = caption
        
        print(f"✓ 載入 {len(index_based_captions)} 個 captions")
        return index_based_captions
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        返回一個樣本
        
        Returns:
            dict with keys:
                - whole_img: PIL Image (preview)
                - caption: str
                - layout: list of dicts, each with:
                    - layer_img: PIL Image (RGBA)
                    - left, top, width, height: int
                    - type: str
                - height: int (canvas height)
                - width: int (canvas width)
        """
        show_debug = self.enable_debug and idx < 3
        
        if show_debug:
            print(f"\n{'='*60}")
            print(f"[LOAD] Sample {idx}")
            print(f"{'='*60}")
        
        # 支援 Dataset 對象和 list
        if isinstance(self.dataset, list):
            item = self.dataset[idx]
        else:
            item = self.dataset[idx]
        
        # === 1. 取得 preview 圖片 ===
        preview = item['preview']
        
        # 處理不同格式的 preview
        if isinstance(preview, Image.Image):
            whole_img = preview
            if show_debug:
                print(f"[IMG] Preview: PIL Image {whole_img.size}")
        elif isinstance(preview, dict) and 'bytes' in preview:
            # HuggingFace Image feature format
            from io import BytesIO
            whole_img = Image.open(BytesIO(preview['bytes']))
            if show_debug:
                print(f"[IMG] Preview: bytes → PIL Image {whole_img.size}")
        elif isinstance(preview, bytes):
            from io import BytesIO
            whole_img = Image.open(BytesIO(preview))
            if show_debug:
                print(f"[IMG] Preview: bytes → PIL Image {whole_img.size}")
        else:
            # 路徑
            whole_img = Image.open(preview)
            if show_debug:
                print(f"[IMG] Preview: loaded from {preview}")
        
        # 轉換為 RGBA
        if whole_img.mode != 'RGBA':
            whole_img = whole_img.convert('RGBA')
        
        # === 2. 取得 caption (用 index 查找) ===
        if self.caption_mapping:
            caption = self.caption_mapping.get(idx, item.get('title', ''))
            if show_debug:
                caption_preview = caption[:100] + '...' if len(caption) > 100 else caption
                print(f"[CAPTION] From index {idx}: {caption_preview}")
        else:
            caption = item.get('title', '')
            if show_debug:
                print(f"[CAPTION] From dataset title: {caption[:100]}...")
        
        # === 3. 取得 canvas 尺寸 ===
        canvas_width = item.get('canvas_width', whole_img.width)
        canvas_height = item.get('canvas_height', whole_img.height)
        
        if show_debug:
            print(f"[CANVAS] {canvas_width} x {canvas_height}")
        
        # === 4. 處理 layers ===
        left_list = item.get('left', [])
        top_list = item.get('top', [])
        width_list = item.get('width', [])
        height_list = item.get('height', [])
        type_list = item.get('type', [])
        layer_images = item.get('image', [])
        
        num_layers = item.get('length', len(left_list))
        num_layers = min(num_layers, self.max_layers)
        
        if show_debug:
            print(f"[LAYERS] Total: {num_layers}")
        
        layout = []
        
        for i in range(num_layers):
            # 取得 bbox
            x = left_list[i] if i < len(left_list) else 0
            y = top_list[i] if i < len(top_list) else 0
            w = width_list[i] if i < len(width_list) else 0
            h = height_list[i] if i < len(height_list) else 0
            layer_type = type_list[i] if i < len(type_list) else 'unknown'
            
            # 取得 layer image
            layer_img = layer_images[i] if i < len(layer_images) else None
            
            # 處理不同格式的 layer_img
            if layer_img is None:
                # 從 preview crop
                if show_debug:
                    print(f"  [CROP] Layer {i}: No image, cropping from preview")
                
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(whole_img.width, x + w)
                y2 = min(whole_img.height, y + h)
                
                if x2 > x1 and y2 > y1:
                    layer_img = whole_img.crop((x1, y1, x2, y2))
                else:
                    if show_debug:
                        print(f"  [SKIP] Layer {i}: Invalid bbox")
                    continue
            
            elif isinstance(layer_img, Image.Image):
                # 已經是 PIL Image
                if show_debug:
                    print(f"  [IMG] Layer {i}: PIL Image {layer_img.size}")
            
            elif isinstance(layer_img, dict) and 'bytes' in layer_img:
                # HuggingFace format
                from io import BytesIO
                layer_img = Image.open(BytesIO(layer_img['bytes']))
                if show_debug:
                    print(f"  [IMG] Layer {i}: bytes → PIL Image {layer_img.size}")
            
            elif isinstance(layer_img, bytes):
                from io import BytesIO
                layer_img = Image.open(BytesIO(layer_img))
                if show_debug:
                    print(f"  [IMG] Layer {i}: bytes → PIL Image {layer_img.size}")
            
            else:
                # 路徑
                try:
                    layer_img = Image.open(layer_img)
                    if show_debug:
                        print(f"  [IMG] Layer {i}: loaded from path")
                except Exception as e:
                    if show_debug:
                        print(f"  [SKIP] Layer {i}: Cannot load: {e}")
                    continue
            
            # 轉換為 RGBA
            if layer_img.mode != 'RGBA':
                layer_img = layer_img.convert('RGBA')
            
            layout.append({
                'layer_img': layer_img,
                'left': x,
                'top': y,
                'width': w,
                'height': h,
                'type': layer_type,
            })
        
        if show_debug:
            print(f"[RESULT] Loaded {len(layout)} layers")
        
        return {
            'whole_img': whole_img,
            'caption': caption,
            'layout': layout,
            'height': canvas_height,
            'width': canvas_width,
        }


def collate_fn(batch):
    """
    Collate function for DataLoader
    因為 batch_size=1，直接返回第一個元素
    """
    if len(batch) == 1:
        return batch[0]
    return batch

