# T2I 协同推理核心代码文件清单

## 已完成的基础协同推理测试使用的代码文件

### 1. 核心Pipeline实现（最重要）

#### `compression/hybrid_sd/diffusers/pipeline_stable_diffusion.py`
**作用**: 实现HybridStableDiffusionPipeline，核心的协同推理逻辑

**关键部分**:
- **第81行**: `class HybridStableDiffusionPipeline` - 主Pipeline类
- **第201-204行**: `self.unets = None` 和 `self.step_config = None` - 多模型和步数配置
- **第547行**: `def __call__` - 主推理方法
- **第705-715行**: **核心模型切换逻辑**
  ```python
  model_index = self.step_config["step"][i]  # 获取当前步使用的模型索引
  noise_pred = self.unets[model_index](...)  # 使用选定的模型进行推理
  ```
- **第732行**: `latents = self.scheduler.step(...)` - 更新latent

**关键机制**:
- 在denoising loop的每一步，根据 `step_config` 动态选择使用哪个UNet模型
- 大模型处理前N步，小模型处理后M步

---

### 2. Pipeline封装类

#### `compression/hybrid_sd/inference_pipeline.py`
**作用**: 封装HybridStableDiffusionPipeline，提供高级接口

**关键类**: `HybridInferencePipeline` (第124行开始)

**关键方法**:
1. **`__init__`** (第125行): 初始化，保存模型路径列表
2. **`set_pipe_and_generator`** (第140行): 
   - 加载Text Encoder（共享）
   - 加载VAE（共享）
   - 加载多个UNet模型（第160-171行）
   - 创建HybridStableDiffusionPipeline（第186行）
   - 设置step_config（第195行）
3. **`get_step_config`** (第243行): **核心步数配置逻辑**
   ```python
   # 将 steps=[10, 15] 转换为 step_config
   # step_config["step"] = {0: 0, 1: 0, ..., 9: 0, 10: 1, ..., 24: 1}
   ```
4. **`generate`** (第224行): 调用Pipeline生成图像

---

### 3. 入口脚本

#### `examples/hybrid_sd/hybrid.py`
**作用**: 主入口脚本，解析参数并执行推理

**关键部分**:
- **第38-109行**: `parse_args()` - 解析命令行参数
- **第141-147行**: 创建HybridInferencePipeline
- **第190-213行**: 推理循环
  ```python
  img = pipeline.generate(
      prompt=prompt,
      negative_prompt=neg_prompt,
      img_sz=args.img_sz,
      guidance_scale=args.guidance_scale,
      ...
  )
  ```

---

### 4. Pipeline工具类

#### `compression/hybrid_sd/diffusers/pipeline_utils.py`
**作用**: Pipeline基础类和工具函数

**关键类**: `DiffusionPipeline` - Pipeline基类

---

### 5. 运行脚本

#### `scripts/hybrid_sd/run_bk_sdm_tiny.sh`
**作用**: 批量测试不同步数配置

**关键内容**:
- 定义模型路径
- 定义步数配置列表: `step_list=("0,25" "10,15" "15,10" "25,0")`
- 循环调用 `hybrid.py`

---

## 核心工作流程

### 1. 初始化阶段
```
hybrid.py
  → HybridInferencePipeline.__init__()
  → HybridInferencePipeline.set_pipe_and_generator()
    → 加载Text Encoder (共享)
    → 加载VAE (共享)
    → 加载多个UNet (大模型+小模型)
    → 创建HybridStableDiffusionPipeline
    → 设置step_config
```

### 2. 推理阶段
```
hybrid.py
  → pipeline.generate()
    → HybridStableDiffusionPipeline.__call__()
      → Denoising Loop:
        → 每一步:
          1. 根据step_config选择模型 (第705行)
          2. 使用选定的UNet进行预测 (第709行)
          3. 更新latent (第732行)
      → VAE解码
      → 返回图像
```

### 3. 步数配置机制
```
steps = [10, 15]  # 大模型10步，小模型15步

→ get_step_config() 生成:
step_config = {
    "step": {
        0: 0, 1: 0, ..., 9: 0,    # 前10步用模型0 (大模型)
        10: 1, 11: 1, ..., 24: 1   # 后15步用模型1 (小模型)
    },
    "name": {
        0: "CompVis--stable-diffusion-v1-4",
        1: "bk-sdm-tiny"
    }
}
```

---

## 关键代码片段

### 1. 模型切换核心代码 (pipeline_stable_diffusion.py:705-715)
```python
# 在denoising loop的每一步
model_index = self.step_config["step"][i]  # 获取模型索引
model_name = self.step_config["name"][model_index]

# 使用选定的模型
noise_pred = self.unets[model_index](
    latent_model_input,
    t,
    encoder_hidden_states=prompt_embeds,
    ...
)
```

### 2. 步数配置生成 (inference_pipeline.py:243-260)
```python
def get_step_config(self, args):
    step_config = {"step": {}, "name": {}}
    total_step = 0
    for index, model_step in enumerate(args.steps):
        for i in range(model_step):
            step_config["step"][total_step] = index
            total_step += 1
    for index, model_name in enumerate(self.weight_folders):
        step_config["name"][index] = model_name.split("/")[-1]
    return total_step, step_config
```

### 3. 多模型加载 (inference_pipeline.py:160-171)
```python
unets = []
for path in self.weight_folders:  # [大模型路径, 小模型路径]
    unets.append(
        MODEL_OBJ.from_pretrained(
            path, subfolder="unet"
        ).to(self.device, dtype=torch.float16)
    )
self.pipe.unets = unets  # 设置多个UNet
```

---

## 文件依赖关系

```
hybrid.py (入口)
    ↓
HybridInferencePipeline (封装)
    ↓
HybridStableDiffusionPipeline (核心)
    ↓
DiffusionPipeline (基类)
```

---

## 用于T2V迁移的关键文件

### 必须修改的文件:
1. ✅ `pipeline_stable_diffusion.py` → `pipeline_cogvideox.py`
   - 修改UNet为3D版本
   - 修改latent形状处理
   - **保持模型切换逻辑不变**

2. ✅ `inference_pipeline.py` → 添加 `HybridVideoInferencePipeline`
   - 修改模型加载（CogVideoX）
   - **保持步数配置逻辑不变**
   - 修改输出为视频

3. ✅ `hybrid.py` → `hybrid_video.py`
   - 修改参数（视频相关）
   - 修改输出保存（视频格式）
   - **保持推理流程不变**

### 可以复用的部分:
- ✅ 步数配置逻辑 (`get_step_config`)
- ✅ 模型切换机制 (denoising loop中的模型选择)
- ✅ 整体架构设计

### 需要适配的部分:
- ❌ UNet模型 (2D → 3D)
- ❌ VAE模型 (2D → 3D)
- ❌ Latent形状 (B,C,H,W → B,C,T,H,W)
- ❌ 输出格式 (Image → Video)

---

## 快速参考

### 关键行号:
- **模型切换**: `pipeline_stable_diffusion.py:705`
- **步数配置**: `inference_pipeline.py:243`
- **模型加载**: `inference_pipeline.py:160`
- **Pipeline创建**: `inference_pipeline.py:186`
- **推理入口**: `hybrid.py:202`

### 关键变量:
- `self.unets`: 多个UNet模型列表
- `self.step_config`: 步数配置字典
- `model_index`: 当前步使用的模型索引
- `latent_model_input`: 输入latent
- `noise_pred`: 噪声预测

---

## 总结

**核心思想**: 在denoising loop中，根据预定义的步数配置，动态切换不同大小的UNet模型。

**关键设计**:
1. 多个UNet模型存储在列表中
2. step_config指定每步使用哪个模型
3. 在denoising loop中根据step_config选择模型
4. 共享Text Encoder和VAE

**迁移要点**:
- 保持模型切换机制
- 修改UNet为3D版本
- 修改latent形状处理
- 保持步数配置逻辑

