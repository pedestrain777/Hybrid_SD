# T2V 视频生成问题调试总结

本文档总结了在实现 Hybrid-SD T2V（Text-to-Video）功能过程中遇到的主要问题及其解决方案。

## 问题1: 视频全黑 - NaN/Inf 值问题

### 问题发现
- **现象**: 生成的视频文件存在但内容全黑
- **日志线索**: 
  - `Latents before decode - min: 0.0000, max: 0.0000, mean: 0.0000`
  - `Before scheduler.step (last step) - latents min: nan, max: nan, mean: nan`
  - `Latents contain NaN or Inf values before VAE decoding!`

### 根本原因
`CogVideoXDPMScheduler.step()` 调用方式不正确：
- `old_pred_original_sample` 被错误初始化为 `torch.zeros_like(latents)`，应该为 `None`
- `timestep_back` 在第一步应该为 `None`，而不是下一个 timestep 的值
- 参数类型不正确（应为 `int` 而不是 `torch.Tensor`）

### 解决方案
参考父类 `CogVideoXPipeline` 的实现，修复了 `CogVideoXDPMScheduler` 的调用：
```python
# 修复前
old_pred_original_sample = torch.zeros_like(latents)
timestep_back = timesteps[i + 1].item() if i + 1 < len(timesteps) else 0

# 修复后
old_pred_original_sample = None  # 初始化为 None
timestep_back = timesteps[i - 1] if i > 0 else None  # 第一步为 None
t_int = t.item() if isinstance(t, torch.Tensor) else t  # 确保为 int 类型
```

**文件**: `compression/hybrid_sd/diffusers/pipeline_cogvideox.py`

---

## 问题2: VAE 解码 dtype 不匹配

### 问题发现
- **错误信息**: `RuntimeError: Input type (float) and bias type (c10::Half) should be the same`
- **发生位置**: VAE 解码 latents 时

### 根本原因
- latents 在去噪过程中保持 `float32` 或 `float16`
- VAE 权重为 `float16`
- 调用 `decode_latents` 时 dtype 不匹配

### 解决方案
1. 在 `scheduler.step` 后将 latents 转换为 `prompt_embeds.dtype`（与父类一致）
2. 添加 try-except 处理 VAE 解码时的 dtype 转换：
```python
latents = latents.to(prompt_embeds.dtype)  # 转换 dtype

try:
    video = self.decode_latents(latents)
except RuntimeError as e:
    if "Input type" in str(e) and "bias type" in str(e):
        vae_dtype = next(self.vae.parameters()).dtype
        latents_converted = latents.to(dtype=vae_dtype)
        video = self.decode_latents(latents_converted)
```

**文件**: `compression/hybrid_sd/diffusers/pipeline_cogvideox.py`

---

## 问题3: Prompts 解析错误导致视频与 prompt 对应关系错乱

### 问题发现
- **现象**: `prompt_3` 的视频内容是第三个 prompt，但应该是 `prompt_2`
- **日志线索**: `Loaded 7 prompts from examples/hybrid_sd/prompts.txt`（实际只有 5 个 prompts）
- **用户反馈**: 发现视频内容与 prompt 索引不匹配

### 根本原因
原代码使用 `lines[0::2]` 和 `lines[1::2]` 提取 prompts，但：
- 没有过滤空行
- 将每行都当作一个 prompt，导致 prompt 和 negative_prompt 配对错乱
- 例如：空行被当作 prompt，导致后续所有 prompts 索引错位

### 解决方案
修复 prompts.txt 解析逻辑：
```python
# 修复前
val_prompts = [line.strip() for line in lines[0::2]]
neg_val_prompts = [line.strip() for line in lines[1::2]]

# 修复后
lines = [line.strip() for line in file.readlines()]
lines = [line for line in lines if line]  # 过滤空行
val_prompts = []
neg_val_prompts = []
for i in range(0, len(lines), 2):  # 每两行为一组
    if i + 1 < len(lines):
        val_prompts.append(lines[i])
        neg_val_prompts.append(lines[i + 1])
```

**文件**: `examples/hybrid_sd/hybrid_video.py`

---

## 调试技巧总结

1. **添加详细的日志记录**:
   - 在关键步骤记录 latents 的统计信息（min, max, mean, std, dtype, shape）
   - 检查 NaN/Inf 值

2. **对比父类实现**:
   - 当遇到调度器调用问题时，参考父类 `CogVideoXPipeline` 的实现
   - 确保参数类型和调用方式一致

3. **逐步验证**:
   - 先验证 prompts 解析是否正确
   - 再验证去噪过程是否正常
   - 最后验证 VAE 解码和视频保存

4. **使用测试脚本验证**:
   - 创建简单的测试脚本验证修复是否生效
   - 检查生成的视频文件是否正常

---

## 验证结果

修复后测试结果：
- ✅ 正确加载 5 个 prompts
- ✅ 成功生成 5 个视频（prompt_0 到 prompt_4）
- ✅ 视频与 prompt 对应关系正确
- ✅ 视频内容正常（非全黑）
- ✅ 平均延迟：75.51 秒/视频

所有视频保存在 `results/test_fixed/` 目录下。

