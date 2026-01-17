# T2V 协同推理实现框架

## 一、核心架构

### 1.1 文件结构
```
compression/hybrid_sd/
├── diffusers/
│   └── pipeline_cogvideox.py          # 核心Pipeline（参考pipeline_stable_diffusion.py）
├── inference_pipeline.py               # 添加HybridVideoInferencePipeline类
└── ...

examples/hybrid_sd/
└── hybrid_video.py                     # 入口脚本（参考hybrid.py）

scripts/hybrid_sd/
└── run_cogvideox.sh                    # 运行脚本（参考run_bk_sdm_tiny.sh）
```

### 1.2 核心组件
- **HybridCogVideoXPipeline**: 实现transformer切换逻辑
- **HybridVideoInferencePipeline**: 封装Pipeline，管理模型加载
- **hybrid_video.py**: 入口脚本，解析参数并执行推理

## 二、关键实现点

### 2.1 模型加载（inference_pipeline.py）
- 加载Text Encoder（共享，T5）
- 加载VAE（共享，AutoencoderKLCogVideoX）
- 加载多个Transformer（5B + 2B）
- 创建HybridCogVideoXPipeline

### 2.2 模型切换逻辑（pipeline_cogvideox.py）
- 在denoising loop中根据`step_config`选择transformer
- **关键**: 根据模型类型设置`image_rotary_emb`
  - 5B: 使用`_prepare_rotary_positional_embeddings`生成
  - 2B: 传递`None`

### 2.3 步数配置（复用T2I逻辑）
- `get_step_config()`: 将`steps=[10,15]`转换为step_config字典
- 格式: `{0:0, ..., 9:0, 10:1, ..., 24:1}`

## 三、与T2I的主要差异

### 3.1 模型类型
| 组件 | T2I | T2V |
|------|-----|-----|
| UNet | UNet2DConditionModel | CogVideoXTransformer3DModel |
| VAE | AutoencoderKL | AutoencoderKLCogVideoX |
| Text Encoder | CLIPTextModel | T5EncoderModel |

### 3.2 Latent形状
- T2I: `(B, C, H, W)`
- T2V: `(B, T, C, H, W)` - 增加时间维度T=49

### 3.3 输出格式
- T2I: 图像（PIL.Image）
- T2V: 视频（frames列表或视频文件）

### 3.4 特殊处理
- **Rotary Positional Embeddings**: 5B需要，2B不需要
- **VAE scale factor**: 需要获取`vae_scale_factor_spatial`

## 四、实现步骤

### 步骤1: 创建HybridCogVideoXPipeline
- 参考`pipeline_stable_diffusion.py`的`HybridStableDiffusionPipeline`
- 修改点:
  - UNet → Transformer
  - 2D latent → 3D latent (B,T,C,H,W)
  - 添加`image_rotary_emb`处理逻辑

### 步骤2: 扩展HybridVideoInferencePipeline
- 在`inference_pipeline.py`中添加新类
- 修改点:
  - 加载CogVideoX模型（Transformer, VAE, Text Encoder）
  - 复用`get_step_config()`逻辑
  - 输出视频而非图像

### 步骤3: 创建hybrid_video.py
- 参考`hybrid.py`
- 修改点:
  - 参数: 添加`num_frames`等视频相关参数
  - 输出: 保存视频文件而非图像

### 步骤4: 测试脚本
- 创建`run_cogvideox.sh`
- 测试不同step配置: `("0,25" "10,15" "15,10" "25,0")`

## 五、注意事项

### 5.1 Rotary Positional Embeddings
- **必须**: 在切换transformer时检查`use_rotary_positional_embeddings`
- 5B模型: 生成并传递`image_rotary_emb`
- 2B模型: 传递`None`

### 5.2 Latent形状一致性
- 两个模型的latent形状完全相同: `(B, 49, 16, 60, 90)`
- 切换时无需转换，直接传递

### 5.3 共享组件
- VAE和Text Encoder可以共享
- 减少内存占用

### 5.4 模型路径
- 5B: `pretrained_models/THUDM--CogVideoX-5B`
- 2B: `pretrained_models/THUDM--CogVideoX-2B`

## 六、可复用部分

✅ **完全复用**:
- `get_step_config()`逻辑
- 模型切换机制（denoising loop中的选择逻辑）
- 整体架构设计

⚠️ **需要修改**:
- 模型加载（UNet → Transformer）
- Latent形状处理（2D → 3D）
- Rotary embeddings处理
- 输出格式（图像 → 视频）

## 七、参考文件

- T2I核心: `compression/hybrid_sd/diffusers/pipeline_stable_diffusion.py:705`
- T2I封装: `compression/hybrid_sd/inference_pipeline.py:243`
- T2I入口: `examples/hybrid_sd/hybrid.py`
- CogVideoX兼容性: `COGVIDEOX_TRANSFORMER_COMPATIBILITY.md`
- CogVideo源码: `CogVideo-main/inference/`

---

**核心思想**: 复用T2I的模型切换机制，适配CogVideoX的3D transformer和rotary embeddings处理。

