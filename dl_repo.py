import os
from huggingface_hub import snapshot_download
import sys
repo_id = sys.argv[1]
out_dir = sys.argv[2]
os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT','https://hf-mirror.com')
os.environ['HUGGINGFACE_HUB_BASE_URL'] = os.environ.get('HUGGINGFACE_HUB_BASE_URL',os.environ['HF_ENDPOINT'])
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
snapshot_download(repo_id=repo_id, repo_type='model', local_dir=out_dir, local_dir_use_symlinks=False, allow_patterns=['*'])
print('DONE', repo_id)
