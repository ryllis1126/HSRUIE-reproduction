# 论文原始实验协议

更新日期：2026-11-02。源码对照版本：21ff69916b5ce95a1f4a1aea89b23d15c0a15d4b。

## 来源与证据边界

- **论文事实**：Task-Driven Underwater Image Enhancement via Hierarchical Semantic Refinement，IEEE TIP 35:42-56，2026，DOI 10.1109/TIP.2025.3647323。
- 本地 PDF：C:/Users/a1825/Desktop/新/Task-Driven_Underwater_Image_Enhancement_via_Hierarchical_Semantic_Refinement.pdf，共 15 页。下文均用印刷页码；印刷第 45/46/47 页分别为 PDF 第 4/5/6 页。
- 图 2-4 已在第二轮审计中进行内存渲染核查。本轮重新读取方法和实验文字，不创建 PDF 副本或渲染文件。
- **源码事实**不是论文事实；实现差异见 [已知问题](known_issues.md)。**计划**见 [实验计划](experiment_plan.md)，均未执行。

## 符号与总体流程

**论文事实，p.45 Fig.2、式(1)，p.46-47 III.C-D**：E 为固定的预训练 VGG 特征提取器，G 为 MS-UDAG，C/D 分别为内容/失真提取器，Ec 为 CLIP 图像编码器，T 为下游任务网络。l 属于 {1,2,3} 表示三层语义特征。

训练依次建立失真感知生成先验、进行无干净参考的特征解耦、利用任务监督增强内容。最终推理只需要提取内容、分层增强及 G 生成图像；失真分支、CLIP 和任务网络用于训练约束，不必进入最终图像推理链。

## Stage 1：MS-UDAG 预训练

**论文事实，p.45 III.B、式(1)-(3)，p.46 式(4)-(6)**：使用不配对的水下图像 Xu 和空气图像 Xa。

$$
F_a^l=E(X_a),\qquad F_u^l=E(X_u),\quad l\in\{1,2,3\}.
$$

空气域执行高保真重建；水下域使用较弱重建约束，学习对失真特征产生异常响应的生成先验：

$$
\hat X_a=G(F_a),\quad A=G(F_u),\quad
L_{rec}^{a}=\|X_a-\hat X_a\|_1,\quad
L_{rec}^{u}=\|X_u-A\|_1.
$$

相似度排序项为：

$$
L_{sr}=\max\{0,-[d(X_a,\hat X_a)-d(X_u,A)]+m\},\qquad m=1.
$$

式(5)给出余弦相似度示例 d(x,y)=(x·y)/(||x||2 ||y||2)。它是相似度，不能将排序项按“距离越小越好”改写反号。

$$
L_{MS\text{-}UDAG}=L_{sr}+\lambda_{rec}^{a}L_{rec}^{a}+\lambda_{rec}^{u}L_{rec}^{u},
\qquad \lambda_{rec}^{a}=1,\quad\lambda_{rec}^{u}=0.1.
$$

论文这一目标没有列出 GAN loss 或 CycleGAN cycle loss。不能因入口名为 cycle_gan_model 就把这些损失添加为论文要求。

**源码事实**：[VGG](../models/vggloss.py):13-47 和 [SAD](../models/DenseNet.py):250-376 提供相符的三尺度输入/图像输出组件；当前 [CycleGANModel](../models/cycle_gan_model.py):109-137 未实现上述两路重建及排序目标。SAD 与 G 的逐层对应、预训练来源尚未认证。

## Stage 2：自监督特征解耦

**论文事实，p.46 III.C、式(7)-(14)**：C、D 均由六个残差块构成，分别提取内容和失真特征。

$$
F_d^l=D(F_u^l),\qquad F_c^l=C(F_u^l),\qquad X_c=G(F_c).
$$

$$
L_{sep}=\sum_{i=1}^{3}\|(F_d^i+F_c^i)-F_u^i\|_1.
$$

CLIP 使用预训练 ResNet 图像编码器的第 2-4 层特征，约束原图 Xu 与生成图 Xc 的内容一致：

$$
L_{clip}=\sum_{i=2}^{4}\|E_c^i(X_u)-E_c^i(X_c)\|_1.
$$

身份分支直接将 Xc 重新经过 E 和 G，不在这条分支中自行再增加一次 C：

$$
F_{idt}=E(X_c),\qquad X_{idt}=G(F_{idt}),
$$

$$
L_{idt1}=\sum_{i=1}^{3}\|F_{idt}^i-F_c^i\|_1,\qquad
L_{idt2}=\|X_{idt}-X_c\|_1.
$$

$$
L_{dis}=L_{sep}+L_{clip}+2L_{idt1}+L_{idt2}.
$$

**论文图示事实，p.45 Fig.2**：E、后续使用的 G、CLIP 带冻结标记。冻结参数不意味着可以切断其对输入的梯度；这是实现注意事项，不是额外论文损失。

**源码事实**：[FE](../models/DenseNet.py):207-248 每个尺度有六个残差块；当前仅构造 NC 一个实例，没有活动的独立 D。修改版 [CLIP](../CLIP/clip/model.py):138-154 返回 [relu3, layer1, layer2, layer3, out]，已有多层接口，但论文层级与列表索引尚需明确映射。[当前损失](../models/cycle_gan_model.py):123-128 没有实现四项解耦目标。

## Stage 3：分层内容增强与任务监督

**论文事实，p.45 Fig.2，p.46-47 III.D、Fig.3-4、式(15)**：浅层使用 DEM；中层同时使用 DEM、SEM；深层使用 SEM；增强后的多层内容交给 G 生成 Xenh。

- DEM：空间平均池化差分提取局部变化；FFT、高通滤波、IFFT 提取高频；融合并使用注意力增强细节。Fig.3 的 HDE 末端标注 3×3 Conv。
- SEM：局部卷积、中尺度空洞卷积、全局池化/门控三分支，使用 softmax 权重自适应融合。
- 任务网络使用 U-Net，以语义分割提供密集监督。

$$
L_{task}=CE(T(X_{enh}),mask).
$$

**源码事实**：[FEM.py](../models/FEM.py):5-16 对中层两分支执行 0.5/0.5 融合；[当前 Forward](../models/cycle_gan_model.py):116-121 旁路浅层 LOW；[EdgeEnhancer](../models/FEM.py):192-196 实际末端是 1×1；U-Net 和标签未接入。不能把这些实现细节当成论文明确规定。

## 训练超参数与数据用途

**论文事实，p.47 IV.A**：

| 阶段 | 数据 | Adam 初始学习率 | 固定学习率阶段 | 衰减至零阶段 | 总 Epoch | Batch Size |
|---|---|---:|---:|---:|---:|---:|
| Stage 1 | UVTD + Flickr2K，不配对 | 1e-4 | 50 | 50 | 100 | 2 |
| Stage 2 | UVTD，无需干净参考 | 1e-4 | 100 | 100 | 200 | 2 |
| Stage 3 | UVTD + 语义分割标签 | 1e-4 | 50 | 50 | 100 | 2 |

原文记录 256×256 随机训练裁剪；该句直接承接 Stage 1 的两数据集描述。Stage 2/3 的具体裁剪实施及图像/标签同步方式仍需落实，不将源码中缺失的增强步骤当作已实现。论文实验平台为 Linux、NVIDIA RTX 2070、PyTorch；这不是本地环境配置。

**源码事实**：[train_options.py](../options/train_options.py):30-36 默认为 40+40 epoch、lr=1e-4、beta1=0.5；[get_transform](../data/base_dataset.py):81-112 未执行随机裁剪。beta1=0.5 是源码默认值，不能据此声称论文指定该值。

## 数据集与评价指标

**论文事实，p.47 IV.B、p.48-49 IV.C**：

| 数据/任务 | 论文用途与数量 | 模型或指标 |
|---|---|---|
| UVTD | 5380 张；三阶段训练及下游任务实验 | 分割、检测、显著性 |
| Flickr2K | 高质量空气图像，Stage 1 | 不配对空气域；文中此处未列本地资产数量 |
| SUIM | 文中记为 1634 张，额外分割评价 | U-Net、DeepLabv3+；mIoU、mPA |
| UVTD 检测 | 检测泛化评价 | SSD、YOLOv8；mAP50、mAP@0.5:0.95，并报告类别结果 |
| USOD10K | 文中记为 10255 张，额外显著性评价 | U2-Net；F-measure、E-measure、S-measure、MAE |

论文 p.47 说明以分割指导增强训练，检测与显著性任务直接评价增强图像，不对这些任务进行该处所述的额外微调。任务模型的具体变体和权重仍须核实。

**文件事实**：本地 SUIM 为 1525 train_val + 110 TEST = 1635 张，比论文记载多 1 张；不能自动删除、调整划分或改写论文数量，见 [数据清单](dataset_inventory.md)。本地 TEST 110 张是测试资产，不能据论文总量描述其为训练样本。

## 原文尚未明确、不得自行补成原始协议的细节

- SAD/G 的完整逐层对应、所用公开 checkpoint 的训练阶段及数据版本。
- Stage 3 的 C、T 是否同步更新，以及任务模型初始化/冻结协议。
- Adam 其余超参数、学习率衰减曲线的精确实现、损失的 sum/mean 和 batch 归约方式。
- CLIP 层索引、插值和输入适配的精确协议；VGG 归一化的原文细节。
- 数据版本、去重、划分文件、类别集合、ignore 规则、评价脚本和测试尺寸协议。
- LOW 旁路、HDE 卷积、DEM 池化、SEM 残差差异是否为发布遗漏、图示简化或其他设置。

以上应保持“待确认”，不能用经验默认值替代论文证据。
