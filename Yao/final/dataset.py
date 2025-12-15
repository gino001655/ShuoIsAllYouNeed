from huggingface_hub import hf_hub_download

# 資料集在 Hugging Face 上的名稱
repo_id = "WalkerHsu/DLCV2025_final_project_piccollage"

# 要下載的檔案名稱
# 這是 Hugging Face 轉換後的 Parquet 檔案路徑
filename = "default/train/0000.parquet"  # 根據頁面結構，實際名稱可能略有不同，請核對「Files and versions」頁面

# 執行下載
downloaded_file_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
    subfolder="data"  # 通常 Parquet 檔案會在 data/default/train/ 下，建議先確認頁面結構
)

print(f"檔案已下載到: {downloaded_file_path}")