#只做單張

# 基本用法 (自動產生檔名)
python draw_bbox.py configs/exp001/outputs/pipeline_outputs/rtdetr/00017532.json configs/exp001/inputs/00017532.png

# 或者指定輸出的檔名
python draw_bbox.py configs/exp001/outputs/pipeline_outputs/rtdetr/00017532.json configs/exp001/inputs/00017532.png --output evaluation_result_00017532.png


#做整個資料夾


# 安裝 tqdm 讓進度條跑起來看起來比較專業 (選用)
pip install tqdm

# 執行批次轉換
python visualize_batch.py \
  --json_dir configs/exp001/outputs/pipeline_outputs/rtdetr \
  --img_dir configs/exp001/inputs \
  --output_dir visual_results
