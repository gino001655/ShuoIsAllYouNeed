import argparse
import json
import cv2
import os
import sys
import math
from glob import glob

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False


# ============================================================
# Basic helpers
# ============================================================
def sanitize_coords(box, img_w, img_h):
    """座標防呆處理 + 確保 x1<x2, y1<y2"""
    x1, y1, x2, y2 = box[:4]
    x1 = max(0, min(int(round(x1)), img_w - 1))
    y1 = max(0, min(int(round(y1)), img_h - 1))
    x2 = max(0, min(int(round(x2)), img_w - 1))
    y2 = max(0, min(int(round(y2)), img_h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def box_w(b): return max(0, b[2] - b[0])
def box_h(b): return max(0, b[3] - b[1])
def box_area(b): return box_w(b) * box_h(b)


def y_overlap_ratio(a, b):
    """垂直方向重疊比例 (intersection / min(height))"""
    ay1, ay2 = a[1], a[3]
    by1, by2 = b[1], b[3]
    inter = max(0, min(ay2, by2) - max(ay1, by1))
    denom = max(1, min(box_h(a), box_h(b)))
    return inter / denom


def size_similar(a, b, ratio_thresh=2.0):
    """高度相近：max(h)/min(h) <= ratio_thresh"""
    ha, hb = box_h(a), box_h(b)
    if ha <= 0 or hb <= 0:
        return False
    r = max(ha, hb) / max(1e-6, min(ha, hb))
    return r <= ratio_thresh


def x_gap(a, b):
    """a 在左 b 在右時的水平間距；若重疊則為 0 或負"""
    return b[0] - a[2]


def merge_boxes(boxes):
    """把一組 boxes 合成最小包覆框；conf 用 max"""
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    confs = [b[4] for b in boxes if len(b) > 4]
    conf = max(confs) if confs else 0.0
    return [x1, y1, x2, y2, conf]


# ============================================================
# Semantic grouping / merging
# ============================================================
def group_into_lines(sorted_boxes, y_overlap_thresh=0.6, size_ratio_thresh=2.0):
    """
    先用「垂直重疊 + 尺寸相近」把 boxes 粗分成一行一行
    sorted_boxes: 建議先依 y_center 排序
    """
    lines = []
    for b in sorted_boxes:
        placed = False
        for line in lines:
            rep = line["rep"]
            if y_overlap_ratio(rep, b) >= y_overlap_thresh and size_similar(rep, b, size_ratio_thresh):
                line["boxes"].append(b)
                line["rep"] = merge_boxes(line["boxes"])  # update representative
                placed = True
                break
        if not placed:
            lines.append({"boxes": [b], "rep": b[:]})
    return [ln["boxes"] for ln in lines]


def merge_in_line(line_boxes, merge_mode="word",
                  gap_ratio=0.35, y_overlap_thresh=0.6, size_ratio_thresh=2.0):
    """
    同一行內合併：
    - word: 合併相鄰且距離小的框成 word-level
    - line: 整行合成一個 line-level
    """
    if not line_boxes:
        return []

    if merge_mode == "line":
        return [merge_boxes(line_boxes)]

    line_boxes = sorted(line_boxes, key=lambda b: (b[0], b[1]))
    groups = []
    cur = [line_boxes[0]]

    for b in line_boxes[1:]:
        prev = cur[-1]
        h_ref = max(1, int(round((box_h(prev) + box_h(b)) / 2)))
        gap = x_gap(prev, b)

        ok_y = (y_overlap_ratio(prev, b) >= y_overlap_thresh)
        ok_size = size_similar(prev, b, size_ratio_thresh)
        ok_gap = (gap <= gap_ratio * h_ref)

        if ok_y and ok_size and ok_gap:
            cur.append(b)
        else:
            groups.append(cur)
            cur = [b]
    groups.append(cur)

    return [merge_boxes(g) for g in groups]


def semantic_merge(boxes, img_w, img_h,
                   merge_mode="word",
                   min_conf=0.5,
                   y_overlap_thresh=0.6,
                   gap_ratio=0.35,
                   size_ratio_thresh=2.0,
                   small_area_ratio=0.08):
    """
    主流程：
    1) 過濾：只對較可信且相對小的框做合併（避免把大背景框亂合）
    2) 依 y_center 排序 → 分行 → 行內合併成 word/line
    """
    if merge_mode == "none":
        return []

    img_area = img_w * img_h

    merged_candidates = []
    for b in boxes:
        if len(b) < 5:
            continue
        conf = b[4]
        if conf < min_conf:
            continue
        if box_area(b) / max(1, img_area) > small_area_ratio:
            continue
        merged_candidates.append(b)

    if not merged_candidates:
        return []

    merged_candidates.sort(key=lambda b: ((b[1] + b[3]) * 0.5, b[0]))

    lines = group_into_lines(
        merged_candidates,
        y_overlap_thresh=y_overlap_thresh,
        size_ratio_thresh=size_ratio_thresh
    )

    merged = []
    for line in lines:
        merged.extend(
            merge_in_line(
                line,
                merge_mode=merge_mode,
                gap_ratio=gap_ratio,
                y_overlap_thresh=y_overlap_thresh,
                size_ratio_thresh=size_ratio_thresh
            )
        )

    out = []
    for mb in merged:
        x1, y1, x2, y2 = sanitize_coords(mb, img_w, img_h)
        out.append([x1, y1, x2, y2, float(mb[4])])
    return out


# ============================================================
# Image-level final score (IQS)
# ============================================================
def compute_image_score(clean_boxes, merged_boxes, img_w, img_h,
                        conf_thr=0.5, cov_lo=0.02, cov_hi=0.35):
    """
    回傳：
      score_0_100, label(str), details(dict)
    """
    if img_w <= 0 or img_h <= 0:
        return 0.0, "BAD", {"reason": "invalid_image_size"}

    N_all = len(clean_boxes)
    if N_all == 0:
        details = {
            "H_high_conf_strength": 0.0,
            "L_low_conf_penalty": 0.0,
            "E_merge_effectiveness": 0.0,
            "C_coverage_sanity": 0.0,
            "coverage_A": 0.0,
            "trusted_raw_N": 0,
            "all_raw_N": 0,
            "merged_K": 0,
            "ratio_r_K_over_N": 0.0,
            "score": 0.0,
            "label": "BAD",
            "reason": "no_boxes",
        }
        return 0.0, "BAD", details


    img_area = img_w * img_h

    hi = [b for b in clean_boxes if b[4] >= conf_thr]
    N = len(hi)                    # trusted raw boxes
    K = len(merged_boxes)          # merged boxes count

    # (A) H: high-confidence strength
    H = sum(max(0.0, (b[4] - conf_thr) / (1.0 - conf_thr)) for b in clean_boxes) / max(1, N_all)
    H = max(0.0, min(1.0, H))

    # (B) L: low-confidence penalty
    L = sum(max(0.0, (conf_thr - b[4]) / conf_thr) for b in clean_boxes) / max(1, N_all)
    L = max(0.0, min(1.0, L))

    # (C) E: merge effectiveness
    r = K / max(1, N)
    r_clip = min(1.0, max(0.0, r))
    E = 1.0 - math.sqrt(r_clip)
    E = max(0.0, min(1.0, E))

    # (D) C: coverage sanity (use max-box coverage, not sum)
    if len(hi) == 0:
        A = 0.0
    else:
        A = max(box_area(b) for b in hi) / max(1, img_area)   # A = max coverage

    # For max-coverage, a large background box is normal.
    # We only penalize if it's *too small* (almost nothing detected) or *impossibly large* (> ~98%).
    cov_lo2 = 0.02
    cov_hi2 = 0.98

    dist = max(0.0, A - cov_hi2) + max(0.0, cov_lo2 - A)
    C = math.exp(- (dist / 0.25) ** 2)   # wider tolerance than before
    C = max(0.0, min(1.0, C))


    pos_sum = 0.45 + 0.25 + 0.15
    raw = (0.45 * H + 0.25 * E + 0.15 * C - 0.15 * L) / pos_sum
    raw = max(0.0, min(1.0, raw))
    score = 100.0 * raw


    if score >= 80:
        label = "GOOD"
    elif score >= 60:
        label = "OK"
    else:
        label = "BAD"

    details = {
        "H_high_conf_strength": H,
        "L_low_conf_penalty": L,
        "E_merge_effectiveness": E,
        "C_coverage_sanity": C,
        "coverage_A": A,
        "trusted_raw_N": N,
        "all_raw_N": N_all,
        "merged_K": K,
        "ratio_r_K_over_N": r,
        "score": score,
        "label": label,
    }
    return score, label, details

def draw_score_panel(img, score, label, details=None):
    """左上角畫 IQS 分數面板（含容錯）"""
    if label == "GOOD":
        panel_color = (0, 255, 0)
    elif label == "OK":
        panel_color = (0, 255, 255)
    else:
        panel_color = (0, 0, 255)

    text = f"IQS {score:.1f}  [{label}]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = 10, 10 + th
    cv2.rectangle(img, (x - 8, y - th - 10), (x + tw + 8, y + 10), panel_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness)

    if details is None:
        return

    # --- 容錯：如果是 early-return 的 details 只會有 reason ---
    if "reason" in details:
        reason = str(details.get("reason"))
        small = f"reason: {reason}"
        fs2, thk2 = 0.55, 1
        (tw2, th2), _ = cv2.getTextSize(small, font, fs2, thk2)
        y2 = y + th2 + 14
        cv2.rectangle(img, (x - 8, y2 - th2 - 10), (x + tw2 + 8, y2 + 10), (230, 230, 230), -1)
        cv2.putText(img, small, (x, y2), font, fs2, (0, 0, 0), thk2)
        return

    # --- 正常情況：安全取值 ---
    H = details.get("H_high_conf_strength", None)
    E = details.get("E_merge_effectiveness", None)
    C = details.get("C_coverage_sanity", None)
    L = details.get("L_low_conf_penalty", None)

    if any(v is None for v in [H, E, C, L]):
        # 缺 key 就不畫細節，避免再炸
        return

    small = f"H={H:.2f} E={E:.2f} C={C:.2f} L={L:.2f}"
    fs2, thk2 = 0.55, 1
    (tw2, th2), _ = cv2.getTextSize(small, font, fs2, thk2)
    y2 = y + th2 + 14
    cv2.rectangle(img, (x - 8, y2 - th2 - 10), (x + tw2 + 8, y2 + 10), (230, 230, 230), -1)
    cv2.putText(img, small, (x, y2), font, fs2, (0, 0, 0), thk2)

# ============================================================
# Visualization main logic
# ============================================================
def process_single_image(json_path, image_path, output_path, args):
    img = cv2.imread(image_path)
    if img is None:
        return False, "無法讀取圖片", None

    h, w = img.shape[:2]

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            boxes = data.get("boxes", [])
    except Exception as e:
        return False, f"JSON 損毀: {e}", None

    # sanitize all boxes
    clean_boxes = []
    for box in boxes:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = sanitize_coords(box, w, h)
        conf = float(box[4]) if len(box) > 4 else 0.0
        clean_boxes.append([x1, y1, x2, y2, conf])

    # 原始框顏色
    COLOR_GREEN = (0, 255, 0)     # > 0.75
    COLOR_YELLOW = (0, 255, 255)  # 0.5 - 0.75
    COLOR_RED = (0, 0, 255)       # < 0.5

    # 先畫原始框（細線）
    for b in clean_boxes:
        x1, y1, x2, y2, conf = b
        if conf > 0.75:
            color, status = COLOR_GREEN, ""
        elif conf >= 0.5:
            color, status = COLOR_YELLOW, "WARN"
        else:
            color, status = COLOR_RED, "LOW"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        if args.draw_scores:
            label = f"{conf:.2f}"
            if status:
                label = f"[{status}] {label}"

            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            ty = y1 - 4
            if ty < th:
                ty = y1 + th + 4
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw, ty + 4), color, -1)
            cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # 語義合併 + 畫合併框（粗線）
    merged = semantic_merge(
        clean_boxes, w, h,
        merge_mode=args.merge_mode,
        min_conf=args.min_conf,
        y_overlap_thresh=args.y_overlap,
        gap_ratio=args.gap_ratio,
        size_ratio_thresh=args.size_ratio,
        small_area_ratio=args.small_area_ratio
    )

    if merged:
        COLOR_MERGED = (255, 0, 255)  # 紫色
        for mb in merged:
            x1, y1, x2, y2, conf = mb
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_MERGED, 2)
            label = f"MERGED {conf:.2f}"
            font_scale = 0.55
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            ty = y1 - 4
            if ty < th:
                ty = y1 + th + 4
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw, ty + 4), COLOR_MERGED, -1)
            cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # --- 新增：整張圖最終分數 ---
    score, label, details = compute_image_score(
        clean_boxes=clean_boxes,
        merged_boxes=merged,
        img_w=w,
        img_h=h,
        conf_thr=args.score_conf_thr,
        cov_lo=args.cov_lo,
        cov_hi=args.cov_hi
    )
    draw_score_panel(img, score, label, details if args.draw_score_details else None)

    # 存檔
    cv2.imwrite(output_path, img)
    return True, "成功", details


def main():
    parser = argparse.ArgumentParser(description="批次將 JSON Bbox 畫在圖片上 + 語義分組合併 + 每張圖 IQS 分數")
    parser.add_argument("--json_dir", required=True, help="JSON 檔案所在的資料夾")
    parser.add_argument("--img_dir", required=True, help="原始圖片所在的資料夾")
    parser.add_argument("--output_dir", required=True, help="結果輸出的資料夾")

    # --- 合併參數 ---
    parser.add_argument("--merge_mode", default="word", choices=["none", "word", "line"],
                        help="語義合併模式：none=不合併, word=字/單詞級, line=行級")
    parser.add_argument("--min_conf", type=float, default=0.5, help="參與合併的最小信心分數")
    parser.add_argument("--gap_ratio", type=float, default=0.35,
                        help="word 合併的水平間距門檻（gap <= gap_ratio * 字高）")
    parser.add_argument("--y_overlap", type=float, default=0.6,
                        help="判定同一行/可合併的垂直重疊比例門檻")
    parser.add_argument("--size_ratio", type=float, default=2.0,
                        help="高度相似門檻：max(h)/min(h) <= size_ratio")
    parser.add_argument("--small_area_ratio", type=float, default=0.08,
                        help="參與合併的小框面積上限（相對於整張圖面積）")

    # --- 視覺化開關 ---
    parser.add_argument("--draw_scores", action="store_true", help="是否在原框上畫 conf 分數")
    parser.add_argument("--draw_score_details", action="store_true", help="是否在左上角 IQS 下方再畫 H/E/C/L 細節")

    # --- IQS 分數參數 ---
    parser.add_argument("--score_conf_thr", type=float, default=0.5, help="計分時，可信框 conf 門檻")
    parser.add_argument("--cov_lo", type=float, default=0.02, help="coverage 合理下界")
    parser.add_argument("--cov_hi", type=float, default=0.35, help="coverage 合理上界")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    json_files = glob(os.path.join(args.json_dir, "*.json"))
    json_files.sort()
    print(f"🔍 在 {args.json_dir} 找到了 {len(json_files)} 個 JSON 檔")
    print(f"🧩 merge_mode={args.merge_mode} | IQS_conf_thr={args.score_conf_thr}")
    print("-" * 60)

    iterator = tqdm(json_files) if (HAS_TQDM and sys.stdout.isatty()) else json_files

    count = 0
    miss = 0
    summaries = []

    for json_path in iterator:
        filename = os.path.splitext(os.path.basename(json_path))[0]
        img_path_png = os.path.join(args.img_dir, f"{filename}.png")
        img_path_jpg = os.path.join(args.img_dir, f"{filename}.jpg")

        if os.path.exists(img_path_png):
            target_img_path = img_path_png
        elif os.path.exists(img_path_jpg):
            target_img_path = img_path_jpg
        else:
            miss += 1
            continue

        output_path = os.path.join(args.output_dir, f"{filename}_vis.png")
        success, msg, details = process_single_image(json_path, target_img_path, output_path, args)

        if success:
            count += 1
            if details is not None:
                details["filename"] = filename
                details["json_path"] = json_path
                details["image_path"] = target_img_path
                details["output_path"] = output_path
                summaries.append(details)
        else:
            print(f"❌ {filename} 失敗: {msg}")

    # 輸出 summary.json
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("-" * 60)
    print(f"🎉 完成！產生 {count} 張圖片（找不到圖片 {miss} 筆）")
    print(f"📂 輸出位置: {args.output_dir}")
    print(f"🧾 Summary 已輸出: {summary_path}")


if __name__ == "__main__":
    main()
