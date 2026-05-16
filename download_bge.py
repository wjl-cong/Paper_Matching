# -*- coding: utf-8 -*-
"""下载 BGE-M3 模型"""
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

from huggingface_hub import snapshot_download

local_dir = r"C:\Users\w2004\Desktop\简历\面_论文\journal_matcher\models\bge-m3"

print("开始下载 BGE-M3 模型...")
print(f"保存目录: {local_dir}")

try:
    snapshot_download(
        "BAAI/bge-m3",
        local_dir=local_dir,
        resume_download=True
    )
    print("下载完成!")
except Exception as e:
    print(f"下载失败: {e}")
    print("\n尝试手动下载:")
    print("1. 访问 https://huggingface.co/BAAI/bge-m3")
    print("2. 下载 pytorch_model.bin 文件")
    print("3. 放到上述目录中")
