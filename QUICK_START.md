# HybridSD 快速开始指南

## 模型准备

已准备好的模型：
- **大模型（云端）**: `pretrained_models/CompVis--stable-diffusion-v1-4`
- **小模型（边缘）**: `pretrained_models/bk-sdm-tiny`

## 协同推理原理

HybridSD 通过以下方式实现协同推理：
1. **大模型运行前N步**：在云端服务器使用大模型进行初始去噪，获得高质量特征
2. **小模型运行后M步**：在边缘设备使用小模型完成剩余去噪步骤
3. **总步数**：N + M = 总推理步数（通常25步）

## 运行方式

### 方式一：使用脚本运行（推荐）

```bash
# 激活环境
conda activate hybrid_sd

# 进入项目目录
cd /home/dataset-assist-0/Hybrid-SD-main_for_v2i

# 运行脚本
bash scripts/hybrid_sd/run_bk_sdm_tiny.sh
```

### 方式二：直接运行Python命令

```bash
# 激活环境
conda activate hybrid_sd

# 进入项目目录
cd /home/dataset-assist-0/Hybrid-SD-main_for_v2i

# 设置环境变量
export PYTHONPATH='.'

# 运行协同推理（示例：大模型10步 + 小模型15步）
CUDA_VISIBLE_DEVICES=0 python3 examples/hybrid_sd/hybrid.py \
    --model_id pretrained_models/CompVis--stable-diffusion-v1-4 pretrained_models/bk-sdm-tiny \
    --steps 10,15 \
    --prompts_file examples/hybrid_sd/prompts.txt \
    --seed 1674753452 \
    --img_sz 512 \
    --output_dir results/test_output \
    --num_images_per_prompt 1 \
    --num_images 1 \
    --enable_xformers_memory_efficient_attention \
    --save_middle \
    --use_dpm_solver \
    --guidance_scale 7
```

## 参数说明

### 核心参数

- `--model_id`: 模型路径，**第一个是大模型，第二个是小模型**
- `--steps`: 步数配置，格式为 `"大模型步数,小模型步数"`
  - 例如：`"10,15"` 表示大模型运行10步，小模型运行15步
  - 例如：`"0,25"` 表示只用小模型运行25步
  - 例如：`"25,0"` 表示只用大模型运行25步
- `--prompts_file`: 提示词文件路径
- `--img_sz`: 生成图像尺寸（默认512）
- `--guidance_scale`: 引导尺度（默认7）
- `--output_dir`: 输出目录

### 步数配置策略

脚本中预设了几种步数配置：
- `"0,25"`: 纯小模型推理（基准对比）
- `"10,15"`: 大模型10步 + 小模型15步
- `"15,10"`: 大模型15步 + 小模型10步
- `"25,0"`: 纯大模型推理（基准对比）

**更多大模型步数 → 更高质量，但速度更慢**
**更多小模型步数 → 更快速度，但质量可能略降**

## 提示词文件格式

`prompts.txt` 文件格式：
- 奇数行（第1, 3, 5...行）：正提示词（prompt）
- 偶数行（第2, 4, 6...行）：负提示词（negative prompt）

示例：
```
a beautiful landscape
low quality, blurry
a cat sitting on a chair
deformed, ugly
```

## 输出结果

结果保存在 `results/HybridSD_bk_sdm_tiny/` 目录下：
```
results/
└── HybridSD_bk_sdm_tiny/
    └── CompVis--stable-diffusion-v1-4-bk-sdm-tiny-10,15/
        ├── prompt_0/
        │   ├── 0.png          # 最终生成的图像
        │   └── image_0/       # 中间步骤（如果启用--save_middle）
        ├── prompt_1/
        └── log.txt            # 运行日志
```

## 故障排查

1. **CUDA内存不足**：减小 `--img_sz` 或使用更小的步数
2. **模型加载失败**：检查模型路径是否正确
3. **提示词格式错误**：确保提示词文件正负提示词数量相等

## 性能优化建议

1. 使用 `--enable_xformers_memory_efficient_attention` 可以节省内存
2. 使用 `--use_dpm_solver` 可以加速推理
3. 根据实际需求调整步数比例，找到质量和速度的平衡点

