#!/usr/bin/env bash
set -euo pipefail

# 配置
ENV_NAME="${ENV_NAME:-hybrid_sd}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${ROOT_DIR}/pretrained_models"
mkdir -p "${TARGET_DIR}"

# 使用镜像（可通过环境变量覆盖）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_BASE_URL="${HUGGINGFACE_HUB_BASE_URL:-${HF_ENDPOINT}}"

echo "[Info] 使用镜像: ${HF_ENDPOINT}"
echo "[Info] 目标目录: ${TARGET_DIR}"

# 选择 huggingface-cli 命令（若系统无 cli，则尝试在 conda 环境中安装/调用）
HF_CMD="huggingface-cli"
if ! command -v "${HF_CMD}" >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    echo "[Info] 未检测到 huggingface-cli，尝试在 conda 环境 ${ENV_NAME} 中安装 huggingface_hub ..."
    if ! conda run -n "${ENV_NAME}" python - <<'PY' >/dev/null 2>&1; then
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("huggingface_hub") else 1)
PY
    then
      conda run -n "${ENV_NAME}" pip install -q huggingface_hub
    fi
    HF_CMD="conda run -n ${ENV_NAME} huggingface-cli"
  else
    echo "[Error] 未找到 huggingface-cli 且未检测到 conda，请先安装 huggingface_hub 或 conda。"
    exit 1
  fi
fi

download_repo() {
  local repo="${1}"
  local out_dir="${2}"
  mkdir -p "${out_dir}"
  echo "[Info] 开始下载 ${repo} -> ${out_dir}"
  # --local-dir-use-symlinks False 确保存到真实文件，不使用软链接
  ${HF_CMD} download \
    --repo-type model "${repo}" \
    --local-dir "${out_dir}" \
    --local-dir-use-symlinks False \
    --include "*"
  echo "[Info] 完成下载 ${repo}"
}

# 下载两个模型
download_repo "zai-org/CogVideoX-2b" "${TARGET_DIR}/CogVideoX-2b"
download_repo "zai-org/CogVideoX-5b" "${TARGET_DIR}/CogVideoX-5b"

echo "[Done] 所有模型已下载到: ${TARGET_DIR}"


