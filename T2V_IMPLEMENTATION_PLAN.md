# T2V 协同推理实现方案

## 📋 基于验证结果的实现计划

### ✅ 已验证的关键信息

1. **模型架构**: CogVideoX使用Transformer架构（不是UNet），但原理类似
2. **组件**: Transformer, VAE, Text Encoder (T5), Scheduler
3. **Latent形状**: (B, T, C, H, W) - 包含时序维度
4. **模型加载**: 可以使用diffusers库加载
5. **组件共享**: Text Encoder和VAE可以共享，只有Transformer需要切换

### ⚠️ 待验证（但理论上可行）

1. **Transformer兼容性**: 需要验证两个模型的transformer输入输出形状
2. **Latent兼容性**: 需要验证latent在不同模型间的兼容性

## 🎯 实现方案

### 方案1: 基于diffusers Pipeline（推荐）

#### 优势
- ✅ 直接使用diffusers库的CogVideoXPipeline
- ✅ 代码结构清晰
- ✅ 易于维护

#### 实现步骤

1. **创建HybridCogVideoXPipeline类**
   - 基于 `CogVideoXPipeline`
   - 添加多transformer支持
   - 添加step_config支持
   - 实现模型切换逻辑

2. **创建HybridVideoInferencePipeline类**
   - 封装HybridCogVideoXPipeline
   - 实现模型加载
   - 实现步数配置
   - 实现推理方法

3. **创建入口脚本**
   - 基于 `hybrid.py`
   - 修改为视频生成
   - 添加视频保存功能

### 方案2: 直接修改Pipeline（备选）

如果方案1不可行，可以考虑直接修改CogVideoXPipeline的源代码。

## 📝 具体实现细节

### 1. HybridCogVideoXPipeline

#### 关键修改点

1. **添加多transformer支持**
```python
class HybridCogVideoXPipeline(CogVideoXPipeline):
    def __init__(self, ...):
        super().__init__(...)
        self.transformers = None  # 多个transformer列表
        self.step_config = None   # 步数配置
    
    def set_transformers(self, transformers):
        self.transformers = transformers
    
    def set_step_config(self, step_config):
        self.step_config = step_config
```

2. **修改denoising loop**
```python
# 在denoising loop中
for i, t in enumerate(timesteps):
    model_index = self.step_config["step"][i]
    transformer = self.transformers[model_index]
    
    noise_pred = transformer(
        hidden_states=latent_model_input,
        encoder_hidden_states=prompt_embeds,
        timestep=timestep,
        ...
    )
```

### 2. HybridVideoInferencePipeline

#### 关键实现

1. **模型加载**
```python
def set_pipe_and_generator(self):
    # 加载共享组件
    text_encoder = T5EncoderModel.from_pretrained(...)
    vae = AutoencoderKLCogVideoX.from_pretrained(...)
    
    # 加载多个transformer
    transformers = []
    for path in self.weight_folders:
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            path, subfolder="transformer"
        )
        transformers.append(transformer)
    
    # 创建Pipeline
    self.pipe = HybridCogVideoXPipeline.from_pretrained(
        self.weight_folders[0],
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformers[0]
    )
    
    # 设置多transformer
    self.pipe.set_transformers(transformers)
    self.pipe.set_step_config(step_config)
```

2. **步数配置**
```python
def get_step_config(self, args):
    # 与T2I相同的逻辑
    step_config = {"step": {}, "name": {}}
    total_step = 0
    for index, model_step in enumerate(args.steps):
        for i in range(model_step):
            step_config["step"][total_step] = index
            total_step += 1
    return total_step, step_config
```

### 3. 入口脚本

#### 关键修改

1. **使用HybridVideoInferencePipeline**
```python
from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline

pipeline = HybridVideoInferencePipeline(
    weight_folders=["path/to/cogvideox_5b", "path/to/cogvideox_2b"],
    seed=1234,
    device="cuda:0",
    args=args
)
```

2. **生成视频**
```python
video = pipeline.generate(
    prompt=prompt,
    num_frames=81,
    height=720,
    width=480,
    guidance_scale=6.0,
    steps="10,15"  # 大模型10步，小模型15步
)
```

## 🔍 关键技术点

### 1. Transformer切换

**关键代码位置**: denoising loop中的transformer调用

**实现方式**:
- 存储多个transformer在列表中
- 根据step_config选择transformer
- 调用选定的transformer进行预测

### 2. Latent处理

**Latent形状**: (B, T, C, H, W)
- B: Batch size
- T: 帧数（时序维度）
- C: Latent通道数
- H, W: 空间维度

**处理方式**:
- 保持latent形状不变
- 在denoising过程中传递latent
- 模型切换时保持latent形状

### 3. 时序一致性

**保证方式**:
- Latent包含完整的时序信息
- Transformer处理时序关系
- 模型切换时保持latent连续性

## 📊 与T2I实现的对比

### 相同点 ✅
1. ✅ 架构设计模式
2. ✅ 步数配置机制
3. ✅ 模型切换逻辑框架
4. ✅ 推理循环结构

### 不同点 ⚠️
1. ⚠️ 使用Transformer而不是UNet
2. ⚠️ Latent包含时序维度
3. ⚠️ 使用T5而不是CLIP
4. ⚠️ 输出视频而不是图像

### 需要修改的部分
1. Pipeline类: CogVideoXPipeline
2. 模型类: CogVideoXTransformer3DModel
3. VAE类: AutoencoderKLCogVideoX
4. Text Encoder: T5EncoderModel
5. Latent形状: 添加时序维度
6. 输出处理: 视频保存

## 🎯 实现优先级

### 阶段1: 基础验证（高优先级）
1. 验证transformer兼容性
2. 验证latent兼容性
3. 实现单模型推理测试

### 阶段2: 协同推理实现（高优先级）
1. 创建HybridCogVideoXPipeline
2. 实现模型切换逻辑
3. 实现步数配置
4. 测试基础功能

### 阶段3: 测试和优化（中优先级）
1. 测试不同步数配置
2. 优化内存使用
3. 优化推理速度
4. 验证视频质量

## ⚠️ 潜在问题和解决方案

### 问题1: Transformer兼容性
**问题**: 两个模型的transformer输入输出形状可能不一致

**解决方案**:
1. 验证两个模型的config
2. 如果形状不一致，添加转换层
3. 或者使用latent对齐机制

### 问题2: 内存占用
**问题**: 视频生成内存占用大

**解决方案**:
1. 使用CPU offload
2. 使用gradient checkpointing
3. 优化batch size
4. 使用量化模型

### 问题3: 时序一致性
**问题**: 模型切换可能影响时序一致性

**解决方案**:
1. 测试不同步数配置
2. 优化切换时机
3. 可能需要额外的时序对齐机制

## 📚 参考代码

### T2I实现
- `compression/hybrid_sd/diffusers/pipeline_stable_diffusion.py`
- `compression/hybrid_sd/inference_pipeline.py`
- `examples/hybrid_sd/hybrid.py`

### CogVideoX实现
- `CogVideo-main/inference/cli_demo.py`
- `CogVideo-main/inference/ddim_inversion.py`
- `CogVideo-main/inference/cli_demo_quantization.py`

## 🚀 快速开始

### 步骤1: 验证环境
```bash
# 检查diffusers版本
pip list | grep diffusers

# 检查CogVideoX是否可用
python -c "from diffusers import CogVideoXPipeline; print('OK')"
```

### 步骤2: 实现基础Pipeline
1. 创建 `HybridCogVideoXPipeline`
2. 实现模型切换逻辑
3. 测试单模型推理

### 步骤3: 实现协同推理
1. 创建 `HybridVideoInferencePipeline`
2. 实现多模型加载
3. 实现步数配置
4. 测试协同推理

### 步骤4: 测试和优化
1. 测试不同步数配置
2. 优化性能
3. 验证视频质量

