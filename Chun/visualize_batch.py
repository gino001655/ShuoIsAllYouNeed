import argparse
import json
import cv2
import os
import sys
from glob import glob
from tqdm import tqdm  # 如果沒安裝 tqdm 可以拿掉，只是為了顯示進度條

def sanitize_coords(box, img_w, img_h):
    """座標防呆處理"""
    x1, y1, x2, y2 = box[:4]
    x1 = max(0, min(int(x1), img_w - 1))
    y1 = max(0, min(int(y1), img_h - 1))
    x2 = max(0, min(int(x2), img_w - 1))
    y2 = max(0, min(int(y2), img_h - 1))
    return [x1, y1, x2, y2]

def process_single_image(json_path, image_path, output_path):
    """處理單張圖片的繪圖邏輯"""
    img = cv2.imread(image_path)
    if img is None:
        return False, "無法讀取圖片"
    
    h, w = img.shape[:2]

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            boxes = data.get('boxes', [])
    except Exception as e:
        return False, f"JSON 損毀: {e}"

    # 顏色定義
    COLOR_GREEN = (0, 255, 0)    # > 0.75
    COLOR_YELLOW = (0, 255, 255) # 0.5 - 0.75
    COLOR_RED = (0, 0, 255)      # < 0.5

    for box in boxes:
        x1, y1, x2, y2 = sanitize_coords(box, w, h)
        conf = box[4] if len(box) > 4 else 0.0

        # 依照分數決定顏色
        if conf > 0.75:
            color = COLOR_GREEN
            status = ""
        elif conf >= 0.5:
            color = COLOR_YELLOW
            status = "WARN"
        else:
            color = COLOR_RED
            status = "LOW"

        # 畫框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 標籤文字
        label = f"{conf:.2f}"
        if status:
            label = f"[{status}] {label}"
        
        # 文字底色
        font_scale = 0.6
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        text_y = y1 - 5
        if text_y < text_h: 
            text_y = y1 + text_h + 5

        cv2.rectangle(img, (x1, text_y - text_h - 5), (x1 + text_w, text_y + 5), color, -1)
        cv2.putText(img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # 儲存
    cv2.imwrite(output_path, img)
    return True, "成功"

def main():
    parser = argparse.ArgumentParser(description="批次將整個資料夾的 JSON Bbox 畫在圖片上")
    parser.add_argument("--json_dir", required=True, help="JSON 檔案所在的資料夾")
    parser.add_argument("--img_dir", required=True, help="原始圖片所在的資料夾")
    parser.add_argument("--output_dir", required=True, help="結果輸出的資料夾")
    
    args = parser.parse_args()

    # 1. 建立輸出目錄
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"📁 已建立輸出目錄: {args.output_dir}")

    # 2. 搜尋所有 JSON 檔案
    json_files = glob(os.path.join(args.json_dir, "*.json"))
    json_files.sort()
    
    print(f"🔍 在 {args.json_dir} 找到了 {len(json_files)} 個 JSON 檔")
    print("-" * 40)

    count = 0
    # 3. 開始批次處理
    # 如果有裝 tqdm 就用進度條，沒有就用普通迴圈
    iterator = tqdm(json_files) if 'tqdm' in sys.modules else json_files

    for json_path in iterator:
        # 取得檔名 (不含副檔名)，例如 "00017532"
        filename = os.path.splitext(os.path.basename(json_path))[0]
        
        # 嘗試尋找對應的圖片 (優先找 png, 再找 jpg)
        img_path_png = os.path.join(args.img_dir, f"{filename}.png")
        img_path_jpg = os.path.join(args.img_dir, f"{filename}.jpg")
        
        target_img_path = None
        if os.path.exists(img_path_png):
            target_img_path = img_path_png
        elif os.path.exists(img_path_jpg):
            target_img_path = img_path_jpg
        
        if target_img_path:
            output_path = os.path.join(args.output_dir, f"{filename}_vis.png")
            success, msg = process_single_image(json_path, target_img_path, output_path)
            if success:
                count += 1
            else:
                print(f"❌ {filename} 失敗: {msg}")
        else:
            # 找不到圖片就跳過，不報錯，避免洗版
            pass

    print("-" * 40)
    print(f"🎉 批次處理完成！共產生 {count} 張圖片")
    print(f"📂 結果已存至: {args.output_dir}")

if __name__ == "__main__":
    main()
