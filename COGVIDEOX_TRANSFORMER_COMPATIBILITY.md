# CogVideoX-5B 和 CogVideoX-2B Transformer 兼容性验证报告

## 执行日期
2025年（基于实际测试结果）

## 测试方法
通过分析模型配置文件和实际加载pipeline进行验证。

## 一、关键参数对比

### 1.1 Transformer 关键参数

| 参数 | CogVideoX-5B | CogVideoX-2B | 兼容性 |
|------|--------------|--------------|--------|
| `in_channels` | 16 | 16 | ✅ 相同 |
| `out_channels` | 16 | 16 | ✅ 相同 |
| `sample_frames` | 49 | 49 | ✅ 相同 |
| `sample_height` | 60 | 60 | ✅ 相同 |
| `sample_width` | 90 | 90 | ✅ 相同 |
| `text_embed_dim` | 4096 | 4096 | ✅ 相同 |
| `time_embed_dim` | 512 | 512 | ✅ 相同 |
| `temporal_compression_ratio` | 4 | 4 | ✅ 相同 |
| `patch_size` | 2 | 2 | ✅ 相同 |
| `max_text_seq_length` | 226 | 226 | ✅ 相同 |

### 1.2 VAE 关键参数

| 参数 | CogVideoX-5B | CogVideoX-2B | 兼容性 |
|------|--------------|--------------|--------|
| `latent_channels` | 16 | 16 | ✅ 相同 |
| `sample_height` | 480 | 480 | ✅ 相同 |
| `sample_width` | 720 | 720 | ✅ 相同 |
| `temporal_compression_ratio` | 4 | 4 | ✅ 相同 |

### 1.3 模型架构差异（不影响latent兼容性）

| 参数 | CogVideoX-5B | CogVideoX-2B | 说明 |
|------|--------------|--------------|------|
| `num_attention_heads` | 48 | 30 | 注意力头数不同，不影响输入输出形状 |
| `num_layers` | 42 | 30 | 层数不同，不影响输入输出形状 |
| `use_rotary_positional_embeddings` | True | False | **重要：位置编码方式不同** |
| `_diffusers_version` | 0.31.0.dev0 | 0.30.0.dev0 | 版本不同，但API兼容 |

## 二、Latent 形状分析

### 2.1 Latent 输入输出形状

两个模型的latent形状完全一致：

```
Latent形状: (B, T, C, H, W)
- B: Batch size
- T: 49 (帧数)
- C: 16 (通道数)
- H: 60 (高度)
- W: 90 (宽度)
```

### 2.2 Patch 后的序列长度

```
Patch大小: 2x2
空间Patch数量: (60/2) * (90/2) = 30 * 45 = 1350
总图像序列长度: 49 * 1350 = 66,150
文本序列长度: 最多226
```

### 2.3 输入输出兼容性

✅ **完全兼容**
- Transformer输入形状: `(B, 49, 16, 60, 90)`
- Transformer输出形状: `(B, 49, 16, 60, 90)`
- 两个模型的输入输出形状完全相同

## 三、兼容性结论

### 3.1 ✅ 可以实现的特性

1. **Latent空间完全兼容**
   - 两个模型使用相同的latent空间维度
   - 可以在denoising loop中无缝切换transformer模型
   - 不需要latent转换层

2. **共享组件**
   - 可以共享VAE（编码器和解码器）
   - 可以共享Text Encoder
   - 可以共享Scheduler

3. **动态模型切换**
   - 可以在不同的denoising step使用不同的transformer
   - 例如：前10步用5B模型，后15步用2B模型

### 3.2 ⚠️ 需要注意的问题

1. **Rotary Positional Embeddings**
   - 5B模型使用rotary positional embeddings
   - 2B模型不使用rotary positional embeddings
   - **解决方案**: 在切换transformer时，根据模型类型设置`image_rotary_emb`参数
     - 5B模型: 传递`image_rotary_emb`（通过`pipeline._prepare_rotary_positional_embeddings`生成）
     - 2B模型: 传递`image_rotary_emb=None`

2. **模型架构差异**
   - `num_attention_heads`和`num_layers`不同，但不影响latent兼容性
   - 这些差异只影响模型内部计算，不影响输入输出形状

3. **实际推理测试**
   - 需要在实际推理中测试跨模型切换的稳定性
   - 可能需要调整切换点的位置以获得最佳效果

## 四、实现建议

### 4.1 协同推理实现方案

```python
# 伪代码示例
def hybrid_cogvideox_inference(
    pipeline_5b, pipeline_2b, 
    latents, prompt_embeds, 
    timesteps, step_config
):
    """
    step_config: {0: '5b', 1: '5b', ..., 9: '5b', 10: '2b', ..., 24: '2b'}
    """
    # 准备rotary embeddings（仅5B需要）
    image_rotary_emb_5b = pipeline_5b._prepare_rotary_positional_embeddings(
        height=latents.size(3) * pipeline_5b.vae_scale_factor_spatial,
        width=latents.size(4) * pipeline_5b.vae_scale_factor_spatial,
        num_frames=latents.size(1),
        device=latents.device,
    )
    image_rotary_emb_2b = None  # 2B不使用
    
    for i, t in enumerate(timesteps):
        # 根据step_config选择模型
        model_name = step_config[i]
        
        if model_name == '5b':
            transformer = pipeline_5b.transformer
            image_rotary_emb = image_rotary_emb_5b
        else:
            transformer = pipeline_2b.transformer
            image_rotary_emb = image_rotary_emb_2b
        
        # 调用transformer
        noise_pred = transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            image_rotary_emb=image_rotary_emb,  # 关键：根据模型类型设置
            return_dict=False,
        )[0]
        
        # 更新latents...
```

### 4.2 关键实现点

1. **模型切换逻辑**
   - 在denoising loop中根据step配置选择transformer
   - 保持latent形状不变

2. **Rotary Embeddings处理**
   - 5B模型: 使用`pipeline._prepare_rotary_positional_embeddings`生成
   - 2B模型: 传递`None`

3. **共享组件**
   - VAE: 两个模型使用相同的VAE
   - Text Encoder: 两个模型使用相同的Text Encoder
   - Scheduler: 可以使用相同的scheduler配置

## 五、验证结果总结

### 5.1 配置验证
- ✅ 所有关键参数完全兼容
- ✅ Latent空间维度完全相同
- ✅ 输入输出形状完全相同

### 5.2 Pipeline验证
- ✅ 5B pipeline可以成功加载
- ✅ 2B pipeline可以成功加载
- ✅ 两个pipeline的latent形状要求相同

### 5.3 兼容性结论
- ✅✅✅ **完全兼容，可以实现协同推理**
- ⚠️ 需要注意rotary positional embeddings的处理
- ⚠️ 需要在实际推理中测试跨模型切换的稳定性

## 六、下一步工作

1. **实现Hybrid CogVideoX Pipeline**
   - 参考HybridSD的T2I实现
   - 实现动态transformer切换
   - 正确处理rotary positional embeddings

2. **测试协同推理**
   - 测试不同的step配置（如10,15, 15,10等）
   - 验证输出质量
   - 优化切换点位置

3. **性能优化**
   - 减少模型加载时间
   - 优化内存使用
   - 实现模型卸载机制

## 七、参考文件

- 测试脚本: `scripts/test_cogvideox_shapes.py`
- 验证脚本: `scripts/verify_cogvideox_compatibility.py`
- 模型路径: 
  - 5B: `/data/models/hybridsd_checkpoint/THUDM--CogVideoX-5B`
  - 2B: `/data/models/hybridsd_checkpoint/THUDM--CogVideoX-2B`

---

**结论**: CogVideoX-5B和CogVideoX-2B的transformer输入输出形状完全兼容，可以实现协同推理。主要需要注意rotary positional embeddings的处理。

