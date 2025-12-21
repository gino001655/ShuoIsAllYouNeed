import argparse
import json
import cv2
import numpy as np
import os

def sanitize_coords(box, img_w, img_h):
    """座標防呆：確保不超出圖片範圍"""
    x1, y1, x2, y2 = box[:4]
    x1 = max(0, min(int(x1), img_w - 1))
    y1 = max(0, min(int(y1), img_h - 1))
    x2 = max(0, min(int(x2), img_w - 1))
    y2 = max(0, min(int(y2), img_h - 1))
    return [x1, y1, x2, y2]

def calculate_iou(boxA, boxB):
    """計算 IoU：交集 / 聯集"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    
    if unionArea == 0: return 0
    return interArea / unionArea

def nms_filter(boxes, iou_threshold=0.7):
    """NMS 過濾器：刪除重疊度過高(IoU > 0.7)的重複框"""
    n = len(boxes)
    if n == 0: return set()
    
    # 格式化並加入 index 方便追蹤
    candidates = []
    for i, box in enumerate(boxes):
        score = box[4] if len(box) > 4 else 0.0
        candidates.append({'id': i, 'box': box[:4], 'score': score})

    # 依照分數高低排序 (分數高的優先保留)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    drop_indices = set()
    
    for i in range(len(candidates)):
        if candidates[i]['id'] in drop_indices: continue
            
        for j in range(i + 1, len(candidates)):
            if candidates[j]['id'] in drop_indices: continue

            # 計算這兩個框有多像 (IoU)
            iou = calculate_iou(candidates[i]['box'], candidates[j]['box'])

            # 如果太像了 (超過 0.7)，就刪掉分數比較低的那個 (j)
            if iou > iou_threshold:
                drop_indices.add(candidates[j]['id'])

    return drop_indices

def draw_bboxes(image_path, json_path, output_path=None):
    if not os.path.exists(image_path) or not os.path.exists(json_path):
        print("❌ 找不到檔案")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("❌ 無法讀取圖片")
        return
    
    h, w = img.shape[:2]

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            boxes = data.get('boxes', [])
    except Exception as e:
        print(f"❌ JSON 錯誤: {e}")
        return

    # 1. 先把重複的框找出來 (NMS)
    # 這裡的座標轉換只是為了算 IoU，不影響畫圖
    nms_boxes = [sanitize_coords(b, w, h) + [b[4] if len(b)>4 else 0] for b in boxes]
    drop_indices = nms_filter(nms_boxes, iou_threshold=0.7)
    
    print(f"📦 原始數量: {len(boxes)}, 移除重複: {len(drop_indices)}, 剩餘: {len(boxes) - len(drop_indices)}")

    # 定義顏色 (BGR 格式)
    COLOR_GREEN = (0, 255, 0)    # > 0.75
    COLOR_YELLOW = (0, 255, 255) # 0.5 - 0.75
    COLOR_RED = (0, 0, 255)      # < 0.5
    COLOR_GRAY = (200, 200, 200) # 被 NMS 刪掉的框 (畫淡一點)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = sanitize_coords(box, w, h)
        conf = box[4] if len(box) > 4 else 0.0

        # 如果是被 NMS 刪掉的重複框，我們跳過不畫 (或者你可以選擇畫灰色)
        if i in drop_indices:
            continue 

        # 2. 依照老闆的標準決定顏色
        if conf > 0.75:
            color = COLOR_GREEN
            status = "GOOD"
        elif conf >= 0.5:
            color = COLOR_YELLOW
            status = "WARN"
        else:
            color = COLOR_RED
            status = "POOR"

        # 畫框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 標籤文字
        label = f"{conf:.2f}"
        
        # 文字底色
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        
        # 畫字 (黑色字體對比比較高)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_color{ext}"

    cv2.imwrite(output_path, img)
    print(f"✅ 完成！請查看: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=str)
    parser.add_argument("image_path", type=str)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    draw_bboxes(args.image_path, args.json_path, args.output)
