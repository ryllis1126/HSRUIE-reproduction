# 当前源码的模型结构

更新日期：2026-11-02；源码 HEAD：21ff69916b5ce95a1f4a1aea89b23d15c0a15d4b。本文描述实际发布代码，不将它等同于论文完整实现。论文协议见 [paper_protocol](paper_protocol.md)。

## 当前活动入口

**源码事实**：[test.py](../test.py):48-58 解析参数、创建数据和模型并 setup；[模型工厂](../models/__init__.py):25-65 按 cycle_gan 选择 CycleGANModel。[BaseModel.test](../models/base_model.py):98-106 用 no_grad 包裹 forward。实际使用的不是 DenseNet.Mynet。

~~~mermaid
flowchart TD
    X[RGB 输入 B×3×H×W] --> E[VGG19 前三组特征]
    E --> C[NC / FE 三尺度内容变换]
    C --> C1[fc1: B×64×H×W]
    C --> C2[fc2: B×128×H/2×W/2]
    C --> C3[fc3: B×256×H/4×W/4]
    C1 --> PASS[当前 LOW-off: fe1 = fc1]
    C1 -. 未调用 .-> LOW[LOW / DetailEnhancement]
    X -. 未调用 .-> LOW
    C2 --> M[HIGH2 / MIDDLE: DEM + SEM]
    X --> M
    C3 --> S[HIGH3 / FEM]
    PASS --> G[SAD 三尺度解码]
    M --> G
    S --> G
    G --> Y[Tanh: I_e, B×3×H×W]
~~~

来源：[CycleGANModel.forward](../models/cycle_gan_model.py):109-121。LOW 虽被构造和加载，其输出没有进入 SAD。图中虚线是未执行分支，不是自动配置开关。

## 输入、shape 与输出

**源码事实与静态推导**：RGB 由 SingleDataset 转换，get_transform 实际只执行 ToTensor 和 0.5 均值/标准差归一化，输入范围 [-1,1]。[single_dataset.py](../data/single_dataset.py):19-36；[base_dataset.py](../data/base_dataset.py):81-112。

令 B 为 batch size，H/W 为 16 倍数的非极小尺寸：

| 模块 | 输入 | 输出 / Tensor Shape | 当前推理 | 当前训练 |
|---|---|---|---|---|
| VGG19 / E | B×3×H×W | relu1_2: B×64×H×W；relu2_2: B×128×H/2×W/2；relu3_2: B×256×H/4×W/4 | 执行，参数冻结 | 训练初始化有断点；按源码参数冻结 |
| NC / FE / 候选 C | 三尺度原始特征列表 | 三尺度保持 shape，每尺度 6 个残差块 | 执行 | 默认 requires_grad 未关闭，但不在当前 optimizer 中；没有可执行训练闭环 |
| LOW / 候选浅层 DEM | X 与 fc1 | 若调用为 B×64×H×W；当前 fe1=fc1 | 旁路，仍加载 | 列入 optimizer，但 Forward 旁路；训练闭环不可运行 |
| HIGH2 / MIDDLE | X 与 fc2 | B×128×H/2×W/2；0.5×DetailEnhancement2 + 0.5×FEM | 执行 | 列入 optimizer，受训练断点阻塞 |
| HIGH3 / FEM | fc3 | B×256×H/4×W/4 | 执行 | 列入 optimizer，受训练断点阻塞 |
| SAD / 候选 G | [fe1,fe2,fe3] | B×3×H×W，Tanh 范围 [-1,1] | 执行 | 默认 requires_grad 未关闭，不在 optimizer 中；无训练闭环 |
| CLIP RN50 | 若使用应适配 RGB 范围/尺寸 | 本地 visual 返回多层列表 | 构造并加载，但不调用编码器 | 未接入 Lclip；不能声称已参与优化 |
| U-Net / T | 若接入为任务协议下的 I_e | 预期 B×K×H×W logits | 最终图像推理不需要 | 当前未创建 netUNET、unet_out、label |
| 失真提取器 D | 论文要求三尺度 Fu | 预期与内容特征同 shape | 论文最终推理不需要 | 未发现活动实例 |

来源：[vggloss.py](../models/vggloss.py):13-47；[FE](../models/DenseNet.py):190-248；[FEM.py](../models/FEM.py):5-181；[cycle_gan_model.py](../models/cycle_gan_model.py):50-137；[CLIP 返回值](../CLIP/clip/model.py):138-154。

以上“当前训练”是静态配置解释，不是实际完成训练。训练构造会访问未定义 netUNET，不能按表中的 optimizer 成员推断参数已更新。

## SAD 内部空间流

**源码事实**：[SAD.forward](../models/DenseNet.py):330-376 将 64/128/256 通道分别投影到 64，保留三种空间尺度。最深尺度 H/4 再下采样至 H/8、H/16，再逐级上采样，与 H/8、H/4、H/2、H 的 skip feature 拼接融合，最终卷积和 Tanh 生成 RGB；不是给原图加 RGB 残差。

H/W 不是 16 倍数时，向下取整的 VGG pooling、向上取整的 stride-2 卷积和固定 2 倍上采样可能导致拼接失败。极小输入还需满足 InstanceNorm 的空间元素数限制。UVTD 当前全部尺寸已满足空间约束，但没有认证大图显存可行性。

VGG slice4/5 被构造但不调用；SAD trans_down_1/2 被注册但不在当前 SAD.forward 使用。不要由构造出的模块或注释直接推断活动图。

## checkpoint 对应与路径

| 模块 | checkpoint | 当前加载方式 |
|---|---|---|
| VGG | model_data/vgg19-dcbb9e9d.pth | vggloss.py:17 硬编码 HSRUIE/model_data/...，相对 cwd |
| NC | checkpoints/maps_cyclegan/latest_net_NC.pth | cycle_gan_model.py:73-76 硬编码 HSRUIE/checkpoints/... |
| SAD | checkpoints/maps_cyclegan/latest_net_SAD.pth | cycle_gan_model.py:70-71 同类硬编码 |
| LOW | checkpoints/maps_cyclegan/latest_net_LOW.pth | BaseModel.setup/load_networks 按 checkpoints_dir/name/epoch |
| HIGH2 | checkpoints/maps_cyclegan/latest_net_HIGH2.pth | 同上 |
| HIGH3 | checkpoints/maps_cyclegan/latest_net_HIGH3.pth | 同上 |
| CLIP | RN50 缓存，由 clip.load 查找 | cycle_gan_model.py:90；虽不参与生成，构造仍加载 |
| U-Net | 正式任务权重未认证 | 当前没有活动加载调用 |

第二轮已验证五组自定义权重的 key/shape 匹配且为有限值，历史哈希见 [reproduction_status](reproduction_status.md)。--epoch 或 --checkpoints_dir 不会同步切换硬编码的 NC/SAD/VGG。

## Mynet 与当前入口差别

**源码事实**：[Mynet](../models/DenseNet.py):11-31 调用 LOW 并返回 I_e；其构造函数没有自动加载五组自定义权重，也没有提供 BaseModel 的 setup/test 语义。直接改成 Mynet() 不能保证同权重、同设备、同推理行为。

LOW-on 不改变后续接口 shape，也不改变独立的 HIGH2/HIGH3 输入；它改变 SAD 第一尺度的数值，并增加计算和显存。LOW-off 当前仅是发布代码基线，作者意图仍待确认。论文 Fig.2 的浅层 DEM 及其他结构差异见 [known_issues](known_issues.md)。

## 保存与训练边界

- I_e 经 [tensor2im](../util/util.py):9-27 从 [-1,1] 映射到 uint8；[save_images](../util/visualizer.py):22-54 使用输入 stem 保存 PNG。文件名没有 visual 标签，多视觉输出或重复 stem 时存在覆盖风险。
- [BaseModel.eval](../models/base_model.py):91-96 仅遍历 model_names；推理名单只有 LOW/HIGH2/HIGH3，NC/SAD 不在列表。不能把 --eval 理解为所有组件都被统一切换。
- 训练名单含 UNET，optimizer 却没有对应实例；NC/SAD 也未显式冻结或加入 optimizer。当前没有可验证参数更新。
- SAD 与论文 G 的高层接口相符，但逐层结构、阶段来源及冻结配置未完全认证；这不等于“MS-UDAG 完全不存在”。
