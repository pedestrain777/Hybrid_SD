# T2V 协同推理技术验证报告

## 📅 验证时间
2025-11-09

## 🎯 验证目标
验证CogVideoX-5B和CogVideoX-2B协同推理的技术可行性，确定实现方案。

## ✅ 已验证的关键信息

### 1. 模型架构 ✅

#### 1.1 核心组件
**验证结果**: ✅ 已确认

- **Transformer**: `CogVideoXTransformer3DModel` 
  - 位置: `diffusers.models.transformers.cogvideox_transformer_3d`
  - 作用: 类似UNet，但使用Transformer架构（DiT架构）
  - 输入输出: 处理3D latent (B, T, C, H, W)

- **VAE**: `AutoencoderKLCogVideoX`
  - 位置: `diffusers.models.autoencoders`
  - 作用: 3D VAE，编码解码视频
  - 输入: 视频帧 (B, C, T, H, W)
  - 输出: Latent (B, T, C, H, W) 或 (B, C, T, H, W)

- **Text Encoder**: `T5EncoderModel`
  - 位置: `transformers.T5EncoderModel`
  - 作用: 文本编码（使用T5，不是CLIP）
  - 是否可以共享: ✅ **可以共享**（两个模型使用相同的T5编码器）

- **Scheduler**: `CogVideoXDPMScheduler` 或 `CogVideoXDDIMScheduler`
  - 位置: `diffusers.schedulers`
  - 作用: 扩散调度器
  - 推荐: CogVideoX-2B使用DDIM，CogVideoX-5B使用DPM

#### 1.2 模型加载方式
**验证结果**: ✅ 已确认

```python
# 方式1: 完整Pipeline加载
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=dtype)

# 方式2: 组件单独加载（可用于模型切换）
text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer")
vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")
```

**关键发现**:
- ✅ 两个模型（5B和2B）使用相同的加载方式
- ✅ 可以从HuggingFace或本地路径加载
- ✅ 可以单独加载组件，然后组合成Pipeline

### 2. Latent空间 ✅

#### 2.1 Latent形状
**验证结果**: ✅ 已确认

从代码分析 (`ddim_inversion.py:375-377`):
```python
height=latents.size(3) * pipeline.vae_scale_factor_spatial,
width=latents.size(4) * pipeline.vae_scale_factor_spatial,
num_frames=latents.size(1),
```

**Latent形状**: `(B, T, C, H, W)`
- B: Batch size
- T: 帧数（时序维度）
- C: Latent通道数
- H: Latent高度
- W: Latent宽度

**编码过程** (`ddim_inversion.py:308`):
```python
latent_dist = vae.encode(x=video_frames).latent_dist.sample().transpose(1, 2)
# video_frames: (B, C, F, H, W) -> latent: (B, T, C, H, W)
```

#### 2.2 Latent兼容性
**验证结果**: ✅ **理论上兼容**

**代码证据** (`cli_vae_demo.py:50, 80`):
```python
# 两个模型使用相同的VAE加载方式
model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype)
```

**理论分析**:
- ✅ CogVideoX-5B和CogVideoX-2B使用相同的VAE架构 `AutoencoderKLCogVideoX`
- ✅ Latent形状相同: `(B, T, C, H, W)`
- ✅ 编码解码过程相同
- ⚠️ **关键问题**: 两个模型的transformer输入输出形状是否一致？

**需要验证**:
- [ ] 两个模型的transformer输入输出形状（最重要）
- [ ] Latent在不同模型间是否可以无缝传递（理论上可以）
- [ ] 模型切换时是否需要latent转换（理论上不需要）

### 3. 推理流程 ✅

#### 3.1 Denoising Loop
**验证结果**: ✅ 已确认

从代码分析 (`ddim_inversion.py:389-406`):
```python
for i, t in enumerate(timesteps):
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    # ...
    noise_pred = pipeline.transformer(
        latent_model_input,
        timestep=t,
        encoder_hidden_states=prompt_embeds,
        # ...
    )
    # ...
    latents = scheduler.step(noise_pred, t, latents, ...).prev_sample
```

**关键发现**:
- ✅ Denoising loop结构与Stable Diffusion类似
- ✅ 使用transformer进行噪声预测
- ✅ 支持classifier-free guidance
- ✅ 可以使用相同的scheduler进行步进

#### 3.2 模型切换可行性
**验证结果**: ✅ **理论上可行**

**分析**:
1. Denoising loop中，每一步都调用 `pipeline.transformer` 进行预测
2. 可以动态替换 `pipeline.transformer` 为不同的模型
3. 只要两个模型的输入输出形状兼容，就可以切换

**实现方式**:
```python
# 类似T2I的实现
transformers = [transformer_5b, transformer_2b]
step_config = {"step": {0: 0, ..., 9: 0, 10: 1, ..., 24: 1}}

for i, t in enumerate(timesteps):
    model_index = step_config["step"][i]
    noise_pred = transformers[model_index](latent_model_input, ...)
```

### 4. 组件共享 ✅

#### 4.1 Text Encoder共享
**验证结果**: ✅ **可以共享**

**证据**:
- 两个模型都使用 `T5EncoderModel`
- Text Encoder不依赖于transformer模型
- 可以加载一次，多个模型共享使用

#### 4.2 VAE共享
**验证结果**: ✅ **可以共享**

**证据**:
- 两个模型使用相同的VAE架构 `AutoencoderKLCogVideoX`
- VAE不依赖于transformer模型
- 可以加载一次，多个模型共享使用

#### 4.3 Transformer不共享
**验证结果**: ✅ **不共享，这是需要切换的组件**

**证据**:
- CogVideoX-5B和CogVideoX-2B的transformer参数不同
- Transformer是主要的模型差异
- 需要在denoising过程中动态切换

### 5. Pipeline结构 ✅

#### 5.1 Pipeline类
**验证结果**: ✅ 已确认

- **类名**: `CogVideoXPipeline`
- **位置**: `diffusers.pipelines.cogvideo.pipeline_cogvideox`
- **功能**: 完整的T2V推理Pipeline

#### 5.2 Pipeline组件
**验证结果**: ✅ 已确认

Pipeline包含以下组件:
- `transformer`: CogVideoXTransformer3DModel
- `vae`: AutoencoderKLCogVideoX
- `text_encoder`: T5EncoderModel
- `scheduler`: CogVideoXDPMScheduler 或 CogVideoXDDIMScheduler
- `tokenizer`: T5Tokenizer

### 6. 模型兼容性 ⚠️

#### 6.1 Transformer兼容性
**验证结果**: ✅ **高度可能兼容**

**代码证据**:
1. **相同的加载方式** (`cli_demo_quantization.py:80-82`):
   ```python
   transformer = CogVideoXTransformer3DModel.from_pretrained(
       model_path, subfolder="transformer", torch_dtype=dtype
   )
   ```
   - 两个模型使用相同的类 `CogVideoXTransformer3DModel`
   - 使用相同的加载接口

2. **相同的调用方式** (`ddim_inversion.py:406-413`):
   ```python
   noise_pred = pipeline.transformer(
       hidden_states=latent_model_input,
       encoder_hidden_states=prompt_embeds,
       timestep=timestep,
       ...
   )
   ```
   - 输入输出接口相同
   - 参数格式相同

**理论分析**:
- ✅ 两个模型使用相同的transformer类
- ✅ 使用相同的latent空间（相同的VAE）
- ✅ Transformer的输入是latent: `(B, T, C, H, W)`
- ✅ 输出是noise prediction: `(B, T, C, H, W)`
- ✅ **如果形状一致，理论上可以无缝切换**

**需要确认**:
- [ ] 两个模型的transformer config参数（特别是latent通道数）
- [ ] 两个模型的 `use_rotary_positional_embeddings` 配置
- [ ] 实际测试模型切换

#### 6.2 配置兼容性
**验证结果**: ⚠️ 需要进一步验证

**需要确认**:
- [ ] 两个模型的config参数
- [ ] Scheduler配置是否兼容
- [ ] 其他配置参数

### 7. 时序处理 ✅

#### 7.1 时序一致性
**验证结果**: ✅ 已确认

**证据**:
- CogVideoX使用3D Transformer处理时序信息
- Latent包含时序维度: `(B, T, C, H, W)`
- Transformer内部处理时序关系

#### 7.2 模型切换对时序的影响
**验证结果**: ⚠️ 需要实验验证

**理论分析**:
- 模型切换时，latent应该保持时序连续性
- 只要两个模型的latent空间兼容，切换应该不会破坏时序
- **需要实验验证**: 切换后视频的时序一致性

### 8. 实现方案 ✅

#### 8.1 可以复用的部分
**验证结果**: ✅ 已确认

1. **架构设计**: ✅ 可以复用T2I的架构设计
   - Pipeline封装模式
   - 步数配置机制
   - 模型切换逻辑框架

2. **代码结构**: ✅ 可以复用
   - 文件组织方式
   - 类的设计模式
   - 接口定义方式

3. **核心逻辑**: ✅ 可以复用
   - step_config生成逻辑
   - 模型选择逻辑
   - 推理循环框架

#### 8.2 需要修改的部分
**验证结果**: ✅ 已确认

1. **Pipeline实现**: ⚠️ 需要修改
   - 使用 `CogVideoXPipeline` 而不是 `HybridStableDiffusionPipeline`
   - 修改为处理3D latent
   - 修改transformer切换逻辑（而不是UNet）

2. **模型加载**: ✅ 需要修改
   - 加载 `CogVideoXTransformer3DModel` 而不是 `UNet2DConditionModel`
   - 加载 `AutoencoderKLCogVideoX` 而不是 `AutoencoderKL`
   - 加载 `T5EncoderModel` 而不是 `CLIPTextModel`

3. **Latent处理**: ✅ 需要修改
   - Latent形状: `(B, T, C, H, W)` 而不是 `(B, C, H, W)`
   - 处理时序维度

4. **输出处理**: ✅ 需要修改
   - 输出视频而不是图像
   - 视频保存格式

## ⚠️ 待验证的关键问题

### 1. Transformer兼容性（高优先级）
**问题**: CogVideoX-5B和CogVideoX-2B的transformer是否可以无缝切换？

**代码证据** (`cli_demo_quantization.py:80-82`):
```python
transformer = CogVideoXTransformer3DModel.from_pretrained(
    model_path, subfolder="transformer", torch_dtype=dtype
)
```
- 两个模型使用相同的加载方式
- 都使用 `subfolder="transformer"`
- 理论上应该兼容

**需要验证**:
1. 加载两个模型的transformer
2. 检查输入输出形状
3. 检查config参数（特别是 `use_rotary_positional_embeddings`）
4. 测试模型切换

**代码位置**: 
- `CogVideo-main/inference/cli_demo_quantization.py`
- `CogVideo-main/inference/ddim_inversion.py:477` - 检查config使用

### 2. Latent兼容性（高优先级）
**问题**: 两个模型的latent空间是否完全兼容？

**验证方法**:
1. 使用相同的VAE编码视频
2. 用两个transformer分别处理
3. 检查输出形状和数值范围

### 3. 时序一致性（中优先级）
**问题**: 模型切换后，视频的时序一致性如何？

**验证方法**:
1. 实现基础协同推理
2. 测试不同步数配置
3. 评估视频质量 and 时序连续性

### 4. 性能影响（中优先级）
**问题**: 模型切换对性能的影响？

**验证方法**:
1. 测试单模型推理速度
2. 测试协同推理速度
3. 对比内存使用

## 📊 技术可行性评估

### 可行性: ✅ **高度可行**

**理由**:
1. ✅ CogVideoX使用diffusers库，结构与Stable Diffusion类似
2. ✅ 两个模型使用相同的VAE和Text Encoder
3. ✅ 两个模型使用相同的transformer类 `CogVideoXTransformer3DModel`
4. ✅ 两个模型使用相同的加载和调用接口
5. ✅ Denoising loop结构类似，可以复用切换逻辑
6. ✅ Latent形状明确: `(B, T, C, H, W)`
7. ✅ 模型切换逻辑与T2I完全相同
8. ⚠️ 需要实际验证transformer兼容性（但代码证据显示高度兼容）

### 风险点

1. **Transformer兼容性**: ⚠️ 中等风险
   - 需要验证两个模型的transformer输入输出是否完全一致
   - 如果形状不一致，需要额外的转换层

2. **时序一致性**: ⚠️ 低风险
   - 理论上应该保持，但需要实验验证
   - 可能需要额外的时序对齐机制

3. **性能优化**: ⚠️ 低风险
   - 视频生成比图像生成更耗内存
   - 可能需要优化内存使用

## 🎯 实现建议

### 阶段1: 基础验证（1-2天）
1. 验证transformer兼容性
2. 验证latent兼容性
3. 实现单模型推理测试

### 阶段2: 协同推理实现（2-3天）
1. 创建HybridCogVideoXPipeline
2. 实现模型切换逻辑
3. 实现步数配置
4. 测试基础功能

### 阶段3: 测试和优化（2-3天）
1. 测试不同步数配置
2. 优化内存使用
3. 优化推理速度
4. 验证视频质量

## 📝 关键代码位置

### CogVideo代码库
- **推理代码**: `inference/cli_demo.py`
- **VAE演示**: `inference/cli_vae_demo.py`
- **量化演示**: `inference/cli_demo_quantization.py`
- **DDIM反转**: `inference/ddim_inversion.py`

### Diffusers库（需要查看）
- **Pipeline**: `diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline`
- **Transformer**: `diffusers.models.transformers.cogvideox_transformer_3d.CogVideoXTransformer3DModel`
- **VAE**: `diffusers.models.autoencoders.AutoencoderKLCogVideoX`
- **Scheduler**: `diffusers.schedulers.CogVideoXDPMScheduler`

## ✅ 验证总结

### 已确认 ✅
1. ✅ 模型架构和组件
2. ✅ 模型加载方式
3. ✅ Latent形状和编码方式
4. ✅ Denoising loop结构
5. ✅ 组件共享可行性
6. ✅ Pipeline结构
7. ✅ 可以复用的部分
8. ✅ 需要修改的部分

### 待验证 ⚠️
1. ⚠️ Transformer兼容性（关键）
2. ⚠️ Latent兼容性（关键）
3. ⚠️ 时序一致性（重要）
4. ⚠️ 性能影响（次要）

### 结论
**技术可行性**: ✅ **高度可行**

**下一步**: 
1. 验证transformer兼容性（最重要）
2. 实现基础协同推理
3. 测试和优化

## 🔗 参考资源

1. **CogVideo GitHub**: https://github.com/zai-org/CogVideo
2. **推理代码**: `CogVideo-main/inference/cli_demo.py`
3. **VAE演示**: `CogVideo-main/inference/cli_vae_demo.py`
4. **DDIM反转**: `CogVideo-main/inference/ddim_inversion.py`

