import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download

print("开始下载 BAAI/bge-m3 模型...")
snapshot_download(
    repo_id='BAAI/bge-m3', 
    local_dir='C:/Users/w2004/Desktop/简历/面_论文/journal_matcher/models/bge-m3'
)
print("下载完成！")
