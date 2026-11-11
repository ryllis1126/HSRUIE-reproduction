# Stage 2 原文与源码审计（2026-11-11）

**当前状态（用户确认后更新）**：已按用户后续决定实现独立 Stage 2 代码、Dataset、四项损失、训练入口和测试。原先的结构/CLIP/PDF/短图决策均已解除待决状态，详见本页末尾“用户确认后的审计结论”。以下初次审计正文保留为历史证据，其中“尚未确认/未实现”描述的是第一次交付时的状态，不能用来覆盖后续明确授权。最终实现与测试报告见 [stage2_implementation.md](stage2_implementation.md)。

## 审计范围与版本

- **本地源码事实**：分支 `repro/stage1`，HEAD `47147e7cf67df22be271ed425e80b87d2df7f84a`；开始时工作区干净。本次不改动现有源码、测试或 Stage 1 记录。
- **作者源码事实**：2026-11-11 读取公开仓库，其 main 指向 `397da5e492b309f5c1af4f64df73c5a647e038cd`。以下引用固定至该 commit，不把本地适配误认为作者代码。
- **任务依据**：用户本次附件《HSRUIE Stage 2：自监督内容与失真解耦模块重建》，以及当前 `AGENTS.md`；已阅读 2026-11-10 完整交接、现有论文协议、网络结构、复现状态、实验计划、决策日志、Stage 1 实现与性能记录。交接中的早期测试数量和短图待决状态属于历史信息，当前 D-11 和源码优先。
- **论文版本边界**：在仓库、Codex 附件以及本机下载、桌面、文档目录的本次检查中，未找到指定文件名末尾带 `(1)` 的 PDF。实际读取 `C:/Users/a1825/Desktop/新/Task-Driven_Underwater_Image_Enhancement_via_Hierarchical_Semantic_Refinement.pdf`，15 页，SHA-256 `7824c43bc1149c6f8b537c213816b42777b79d32b3696be3f6ad191eac209803`。已逐段核对印刷页 45–47，并渲染查看 p.45 Fig.2；未证明它与指定 `(1)` 文件逐字节相同。
- **实验边界**：只运行 CPU 合成回归、原始 FE 的小尺寸结构探针，以及已校验哈希的本地 RN50 归档元数据读取。没有对 CLIP 或真实 VGG/SAD 运行 Stage 2 前后向，没有访问服务器、下载权重、操作现有训练或执行正式训练。

本次独立记录目录为 `runs/stage2-audit-20261111-01/`（被 Git 忽略）。包含开始快照、论文摘录、Fig.2 页渲染、FE 探针、CLIP 归档探针和完整回归日志。它们是本次本地审计证据，不是服务器训练记录。

## 论文、作者代码与拟实现的对应关系

表中“工程假设”尚未落地；“作者源码确认”只说明能在公开代码中找到对应事实，不证明公开工程具备完整的 Stage 2 训练过程。

| 项目 | 论文依据 | 作者源码依据 | 当前实现方案 | 属性 |
|---|---|---|---|---|
| C 结构 | p.46 III.C，式 (7)，内容提取器作用于三个尺度 | `models/DenseNet.py:207–248` 的 FE；`models/cycle_gan_model.py:73–76,113` 以 netNC 使用 FE 并加载 NC 权重 | 候选为一个原始 FE，保持三个分支，不接入 DEM/SEM | 作者源码确认；与论文 C 对应是有依据的候选，训练实现仍待确认 |
| D 结构 | p.46 III.C，式 (7)，与 C 独立的失真提取器 | FE 可保持三个尺度 shape；当前 CycleGANModel 没有独立的 Stage 2 D 训练路径 | 候选为第二个独立 FE，不共享参数；不能声称作者已公开该训练接法 | 论文明确；具体源码映射尚未确认 |
| Residual Blocks | p.46 III.C 称两个提取器均含 six residual blocks，没有明说三个独立分支的总块数 | `DenseNet.py:190–248`：三个分支各 6 个，单 FE 合计 18 个独立块 | 建议原样复用 FE，测试“每分支 6、总计 18”；需先确认与任务“各 6 块”的验收解释 | 作者源码确认；论文与验收文字存在解释差异 |
| CLIP 版本 | p.46 式 (10)：预训练 CLIP ResNet，未注明 RN50/RN101 | `cycle_gan_model.py:90` 明确 `clip.load("RN50")` | 使用作者仓库的 RN50 结构和显式本地权重路径，禁止自动下载和 ViT 替代 | 作者源码确认；论文没有明确 RN50 |
| CLIP 特征层 | p.46 式 (10)–(11)，scale 2–4 | `CLIP/clip/model.py:138–154` 返回 `[relu3, layer1, layer2, layer3, out]`；未返回 layer4 空间张量 | 尚不选择 list[1:4] 或 ResNet layer2/3/4，见下文 | 尚未确认，阻塞训练图实现 |
| VGG 特征层 | p.45 III.B/Fig.2 三尺度 E；p.46 式 (7)、(12) 复用 E | 当前 `models/vggloss.py:14–55` 的 Vgg19_out 返回 relu1_2、relu2_2、relu3_2；作者原版具有相同切片和 forward | 复用 Stage 1 E，显式权重路径，严格加载；与 Stage 1 VGG 哈希核对 | 作者源码确认；路径/device 是已有工程适配 |
| G* 解码 | p.45 Fig.2 与 p.46 式 (9)、(12)，复用冻结的 MS-UDAG | `DenseNet.py:250–376` SAD；当前 Stage 1 使用原始 SAD | 复用当前 Stage 1 的 SAD，加载 checkpoint['G']，strict=True；不改结构 | 当前工程事实；真实最终 G* 尚未验收，不能宣称已完整认证论文对应 |
| 四项 Loss | p.46 式 (8)、(11)、(13)、(14)，权重 1、1、2、1 | 已检查的公开 CycleGANModel 没有实现这一完整 Stage 2 损失图 | 仅实现这四项；各尺度 L1 元素均值后求和作为显式工程假设 | 公式及权重论文明确；reduction 工程假设 |
| 参数冻结 | p.45 Fig.2 的 E、G、CLIP 冻结标识 | 公开旧训练入口不足以作为 Stage 2 冻结/optimizer 依据 | 仅 C/D 更新；E/G/CLIP requires_grad=False 且 eval，保留生成图像路径 autograd | 论文明确；实现与梯度验收尚未执行 |
| 数据处理 | p.47 IV.A，Stage 2 仅 UVTD；256 crop 的原文直接描述第一阶段 | Stage 1 Dataset 有 RGB、随机 crop256、[-1,1] 以及 D-11 精确反射策略 | 按本次任务使用 Stage 2 crop256 与固定清单；Stage 1 实际 run 清单/config 尚未取得，D-11 是否沿用需明确 | UVTD 论文明确；Stage 2 crop 为任务指定，短图策略尚未确认 |
| LR Scheduler | p.47 IV.A，100 epoch 固定 1e-4，后 100 逐渐衰减至 0，Adam，batch 2 | `stage1.py:75–88` 提供 Adam 和 epoch 边界线性调度；旧入口 Adam 使用 beta1=0.5、beta2=0.999 | 候选复用调度函数，参数 100/100；每轮 optimizer 更新后才 scheduler.step | 总日程论文明确；精确离散公式与 beta 的 Stage 2 使用为工程假设 |

作者代码固定链接：[FE/ResBlock 与 SAD](https://github.com/gemyumeng/HSRUIE/blob/397da5e492b309f5c1af4f64df73c5a647e038cd/models/DenseNet.py#L190)、[旧入口的 NC/CLIP/Adam](https://github.com/gemyumeng/HSRUIE/blob/397da5e492b309f5c1af4f64df73c5a647e038cd/models/cycle_gan_model.py#L63)、[CLIP forward](https://github.com/gemyumeng/HSRUIE/blob/397da5e492b309f5c1af4f64df73c5a647e038cd/CLIP/clip/model.py#L138)、[CLIP 加载和预处理](https://github.com/gemyumeng/HSRUIE/blob/397da5e492b309f5c1af4f64df73c5a647e038cd/CLIP/clip/clip.py#L79)、[作者原始 VGG](https://github.com/gemyumeng/HSRUIE/blob/397da5e492b309f5c1af4f64df73c5a647e038cd/models/vggloss.py#L13)。

## 需要先确定的结构和输入协议

### C/D 的六块如何计数

**源码事实**：FE 的 conv11/conv12/conv13 分别处理 64/128/256 通道，各串联 6 个 ResBlock。每块是两次 3×3、bias=False 卷积和 GroupNorm(32)，第一层后及残差相加后使用 ReLU（`DenseNet.py:190–248`）。没有通道投影、跨尺度融合或权重共享。

**本次 CPU 结构探针事实**：单 FE 有 18 个 ResBlock、9,300,480 个参数、108 个 state_dict 项。两个 FE 实例参数不共享，共 18,600,960 个参数。探针输入 `[2,64,32,32]`、`[2,128,16,16]`、`[2,256,8,8]`，输出 shape 相同且均有限；这是缩小空间尺寸的结构检查，不是 256 crop 的完整 Stage 2 验收。原始 256 crop 对应三个尺度应为 `[B,64,256,256]`、`[B,128,128,128]`、`[B,256,64,64]`。

**待确认方案**：C=FE()、D=FE()，按“每尺度 6 块”解释论文文字；不能为了通过“每个提取器总计 6 块”的断言而删除作者分支。若用户要求总计 6 块，需另有权威结构依据，目前无法从公开代码确定该结构。

**初始化边界**：旧入口直接构造 FE 后加载作者 NC 权重，没有公开从零训练 C/D 的完整初始化流程（`cycle_gan_model.py:73–76`）。候选为固定种子后两次独立调用原始 FE 构造器，沿用 PyTorch Conv2d/GroupNorm 默认初始化；不调用无依据的 networks.init_weights，不拿作者 NC 当作新训练 C/D 的起点。该从零训练选择必须记录为工程假设，不能称为作者已核实的 Stage 2 初始化。

### CLIP 的层与预处理不能从未调用的对象推定

**源码事实**：作者 ModifiedResNet.forward 的返回顺序如下（`CLIP/clip/model.py:138–154`）。下表空间尺寸是 RN50、224×224 输入的结构推导，未在本次运行真实 CLIP forward 验证。

| 返回索引（0 起） | 模块 | 预期 shape |
|---|---|---|
| 0 | relu3，avgpool 前的 stem | `[B,64,112,112]` |
| 1 | layer1 | `[B,256,56,56]` |
| 2 | layer2 | `[B,512,28,28]` |
| 3 | layer3 | `[B,1024,14,14]` |
| 4 | layer4 经 attention pooling 的 out | `[B,1024]` |

“第 2–4 个返回特征”对应 list[1:4]，即 layer1/2/3；“ResNet 的 layer2/3/4”则需要取得未返回的 layer4 空间输出。两者不同。list[2:5] 会混入最终向量，不能满足三层空间特征的要求。公开旧入口未给出式 (11) 的实际索引使用，论文也没有给出模块名；因此不能仅根据列表次序认定映射。

**源码事实**：`CLIP/clip/clip.py:79–87` 的加载器预处理是 PIL bicubic Resize、CenterCrop、RGB、ToTensor、CLIP mean/std。`cycle_gan_model.py:92–96` 另定义 Tensor Resize((224,224)) 和相同 mean/std，但该对象及加载器返回的 preprocess 没有在已检查的 forward 中调用。这不能作为作者 Stage 2 的有效输入处理证据，且其定义没有明确把 [-1,1] 映射回 [0,1]。

**待确认方案**：显式、可微的 Tensor 预处理，以相同流程处理 Xu 与 Xc；候选是 `[-1,1] -> [0,1] -> resize224 -> CLIP mean/std`。仍需确定是否采用该候选，以及 resize 的插值/antialias 与层映射。CLIP mean/std 为 `(0.48145466,0.4578275,0.40821073)` / `(0.26862954,0.26130258,0.27577711)`。不得在 Xc 分支转 PIL/NumPy、detach 或 no_grad。也不能把 CLIP 的规范化加到 VGG/SAD 输入上。

**精度差异已确认**：`CLIP/clip/model.py:375–436` 的 convert_weights/build_model 会转 half，`clip.py:165–166` 仅 CPU 分支恢复 float。按原始 GPU loader 直接加载会与本次禁止 FP16 的要求冲突。本地 RN50 归档元数据也确认 visual.conv1 为 float16。未来训练应明确以 FP32 执行，避免不必要的 FP32→FP16→FP32 再量化；这是本次用户约束下的工程适配。原始权重归档自身的存储精度不等于允许启用 FP16 训练。

## 损失与梯度合同（待实现）

**论文事实**，p.46 式 (7)–(14)：

```text
Fu = E(Xu)
Fc = C(Fu), Fd = D(Fu)
Xc = G*(Fc)
Fidt = E(Xc), Xidt = G*(Fidt)

Lsep  = Σ(l=1..3) ||Fd_l + Fc_l - Fu_l||₁
Lclip = Σ(i=2..4) ||CLIP_i(Xu) - CLIP_i(Xc)||₁
Lidt1 = Σ(l=1..3) ||Fidt_l - Fc_l||₁
Lidt2 = ||Xidt - Xc||₁
Ldis  = Lsep + Lclip + 2 Lidt1 + Lidt2
```

不在式 (12) 中额外执行 C，不将 Lidt2 改成 Xu 与 Xc 的重建。每项相减之前检查完整 shape、尺度数量、浮点类型和有限性，拒绝广播。候选 reduction 为每个对应 Tensor 的绝对误差元素均值，再跨尺度求和；论文未明确元素平均与求和，该选择必须在有效配置和文档中标记。

| 路径 | 参数状态 | autograd 合同 |
|---|---|---|
| E(Xu) | E 冻结/eval | Xu 不求梯度，可 no_grad；Fu 在同一 batch 内复用 |
| C(Fu)、D(Fu) | 两者更新 | 建图；optimizer 仅含两者，且无重复参数 |
| G*(Fc) | G* 冻结/eval | 必须建图，梯度回到 Fc/C |
| CLIP(Xu) | CLIP 冻结/eval | 可 no_grad；该 batch 目标特征只计算一次 |
| CLIP(Xc) | CLIP 冻结/eval | 必须建图，经可微预处理、Xc、G* 回到 C |
| E(Xc) | E 冻结/eval | 必须建图，保留 Fidt 和 Fc 两侧应有的导数 |
| G*(Fidt) | 同一个 G* 冻结/eval | 必须建图，不能 detach Xidt 或 Xc |

只冻结参数不能代替 eval；后者防止 CLIP BatchNorm 运行状态变化。未来单独隔离四项损失测试各自梯度，检查冻结模块的参数、buffer 和 grad，确认 C、D 的梯度有限且有有效更新。D 来自 Lsep 的监督足够，不人为连接其他损失。本次没有执行这些 Stage 2 梯度测试。

## 权重接入与 Stage 1 边界

**本地文件事实**：本次重新计算了以下完整文件 SHA-256，并读取 RN50 TorchScript 归档的结构元数据。没有下载权重。

| 文件 | 大小（bytes） | SHA-256 | 本次验证范围 |
|---|---:|---|---|
| `D:/Research/HSRUIE/model_data/vgg19-dcbb9e9d.pth` | 574673361 | `dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44` | 文件哈希；未在本次加载 VGG 前向 |
| `C:/Users/a1825/.cache/clip/RN50.pt` | 255827503 | `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762` | 与仓库 RN50 下载 URL 中的预期哈希一致；CPU 读取归档元数据，无 forward |

RN50 元数据：输入分辨率 224、ResNet blocks `[3,4,6,3]`、attention position shape `[50,2048]`、输出 embedding 1024、state_dict 492 项。归档匹配不证明层选择、可微预处理或梯度已正确实现，也不证明服务器文件与本地相同。

**当前源码事实**：`stage1.py:120–138` 的训练 checkpoint 包含 `G, optimizer, scheduler, epoch, global_step, rng, config`。Stage 2 应仅将 `checkpoint['G']` 交给原始 SAD 的 strict=True 加载，另行记录整个来源 checkpoint 的哈希；不得把整个训练包当作 SAD state_dict。Stage 1 格式不改动。

**验收计划**：正式 G* 要核查 completed epoch=100、50/50 日程、mode=train、结构/shape/有限值、有效配置、VGG 哈希和训练结束验收记录；checkpoint 数值与文件名仅是证据的一部分。`train_stage1.py:253–254` 当前命名为 `epoch_0100.pth`，不据此声称某目录已有此文件。诊断权重必须显式标记且只能用于 smoke；正式 train 拒绝诊断/未完成来源。

用户指定的候选来源 `/data/czy/experiments/HSRUIE/stage1/stage1-optimized-20261111T131422-2491194` 本次未访问，不推断运行状态。本地已检查的运行记录中未取得该 run 的真实 manifest、effective_config 或最终 G；这只说明本次本地证据缺失，不证明服务器没有。最终 G 暂缺本身不阻止将来的代码和合成测试，阻止真实权重验收；当前暂停实现的首要原因是结构/CLIP 协议未确定。

## 数据、优化器与日程

**当前工程事实**：`data/stage1_dataset.py` 的 `UVTD_REFLECT_POLICY` 是 `uvtd-reflect16-352x224-224x352-v1`。只对 UVTD 的 352×224 图上下各 reflect16，对 224×352 图左右各 reflect16，其余尺寸合法图不填充；不支持的短图仍报错。随后沿用随机 crop256、RGB、float32 和 [-1,1]。D-11 记录的 5380/504/4876 是已有用户数据盘点证据，本次没有重扫服务器复核这些数量。

**Stage 2 待决**：本次任务要求先核对 Stage 1 实际清单/config，且不擅自 padding。旧 D-11 的明确授权对象是 Stage 1；不能直接把它当作本次 Stage 2 已确认的数据协议。建议经确认后原样复用该精确策略及实际清单，同时对全量数据做存在/解码/尺寸检查；不能过滤或自动生成小图豁免清单。若不沿用，短图如何满足 256 crop 仍未解决。固定 manifest 需保存数量、路径和内容指纹；不得使用 Reference 或任务标签。

本次任务指定 Stage 2 随机 crop256、batch2、单 GPU；论文 p.47 明确所有阶段 batch2、Adam，Stage 2 100+100 和初始 LR 1e-4。原始作者 Adam 调用只显式传 `lr` 与 `betas=(opt.beta1,0.999)`（`cycle_gan_model.py:85–86`），`options/train_options.py:32` 的 beta1 默认 0.5。该调用训练的是旧入口其他模块，不是作者已核实的 C/D optimizer。候选 Stage 2 沿用 `(0.5,0.999)`，eps、weight_decay 等使用固定 PyTorch 版本默认值并记录，不另加优化器超参数；仍属于工程选择。

候选 LR 为复用 `stage1.py:80–88` 的离散公式 `factor(c)=max(0,1-max(0,c-100)/100)`，c 表示已完成 epoch；每轮更新完参数再调度。由此 epoch1–100 用 1e-4，epoch101 仍用 1e-4，epoch102 用 9.9e-5，epoch200 用 1e-6，完成200后设为0。最后一轮有参数更新。论文只有“逐渐”衰减，未规定此边界，不能将该离散公式写成论文事实；实施时需专门测试边界和恢复。

## 确认后的最小实施与验证计划

以下文件尚未创建，不提供指向不存在入口的可执行服务器 smoke 命令：

| 文件 | 拟定职责 |
|---|---|
| `data/stage2_dataset.py` | 单域固定清单、RGB/random crop/range 契约；复用已确认的 Stage 1 公共读取逻辑，不改 Stage 1 |
| `stage2.py` | 两个独立 FE、冻结 E/G/CLIP、四项损失、训练一步、Stage 2 checkpoint；不接入 FEM/NEDB/DEM/SEM/任务网络 |
| `train_stage2.py` | 必填 mode={data-check,smoke,train}，显式路径及配置、独立 run、不覆盖、epoch 边界恢复、失败诊断 |
| `tests/test_stage2.py` | 任务 A–E 的结构/数值/shape/finite/逐损失梯度/冻结 buffer/Adam 更新/checkpoint/RNG/CLI 安全测试 |
| `docs/stage2_implementation.md` | 最终有效协议、假设、测试与实际可用的服务器命令 |

Stage 2 checkpoint 独立保存 C/D、optimizer、scheduler、completed_epoch、global_step、Python/NumPy/Torch/CUDA RNG（如初始化）、effective_config、Git commit/源码差异、G*/VGG/CLIP 哈希、manifest 指纹和代码版本。加阶段/模式标识，禁止混用 Stage 1、诊断 smoke 和正式 Stage 2 包；epoch 边界恢复须比较数据、权重及配置，不实现任意 batch 中断恢复。所有输出与参数日志都用新 run 目录，独占创建，不改作者入口和 Stage 1 格式。

**性能审计候选，未实施、未测速**：同一 batch 的 E(Xu) 和 CLIP(Xu) 各算一次，只有这两条固定输入路径可 no_grad；汇总日志标量后一次传 CPU，保留全部损失与有限性检查。不跨随机 crop 缓存特征，不改数据顺序/worker RNG/持久 worker、不自动启用 pin_memory/后端或数值格式。先用合成输入逐项对照 loss、C/D 梯度、更新参数、冻结 buffers 和 RNG，再在另行获准的 GPU benchmark 中测量。不能把减少同步次数直接宣称为实际训练提速。本次没有修改任何性能代码。

## 本次测试报告

**新实验事实**：CPU、Python 3.8.20、PyTorch 2.0.1+cu118；测试子进程 `CUDA_VISIBLE_DEVICES=''`。命令 cwd 为 `D:/Research/HSRUIE`：

```powershell
D:/Miniconda3/envs/hsruie/python.exe -B -m unittest discover -s tests -p 'test_stage1*.py' -v
```

开始 UTC `2026-11-11T13:53:49.788779+00:00`，结束 `2026-11-11T13:54:14.051086+00:00`；unittest 报告 **43 tests，21.668s，OK，0 失败、0 跳过**，进程退出码 0。原有测试、断言没有修改。完整输出见 `runs/stage2-audit-20261111-01/stage1_regression.txt`，命令/设备/时间见 `regression_record.json`。这是 Stage 1 回归结果，不是 Stage 2 A–E 测试结果。

FE 探针记录 `fe_probe.json`：CPU，seed=20261111，threads=2，torch.no_grad 下小尺寸结构前向通过。CLIP 归档结构探针记录 `clip_archive_probe.json`：只加载已校验的本地归档到 CPU 读取元数据，没有图像输入、前向或梯度验证。没有正式实验所需的 Stage 2 全套权重加载和样本验收，因此不宣称已具备真实 Stage 2 smoke 条件。

## 初次审计时需要用户确认（历史，现已回复）

1. **C/D**：是否接受复用原始 FE，并将“6 块”落实为每尺度 6 块、每提取器总计 18 块？这保留作者结构，需相应明确结构测试计数口径。
2. **CLIP**：scale2–4 采用返回列表第2–4项（layer1/2/3），还是 ResNet layer2/3/4？可微输入是否采用先回到 [0,1]、再 resize224 和 CLIP mean/std；插值与 antialias 使用何种已确认协议？目前没有证据允许把其中一种称为作者实际训练流程。
3. **Stage 2 短图**：是否批准原样沿用 D-11 的两个精确尺寸反射策略？需要取得指定 Stage 1 run 的实际 manifest/config 进行一致性核对，当前不能声明核对完成。

指定 `(1)` PDF 的版本一致性仍未确认，可通过其本地路径或 SHA-256 解除这一资料边界。Stage 1 最终 G 的验收则可在代码/合成测试完成后、真实 smoke 前补齐。除上述关键设计外，块初始化、L1 reduction、Adam beta 和离散 LR 均按本文显式列为必要工程假设，不冒充论文明确设置。

本次新增受版本控制的交付文件仅为本审计文档；没有修改任何既有文件，没有 Git commit/push/reset、切换分支、服务器操作、GPU 测试或正式训练。

## 用户确认后的审计结论（当前有效）

以下决定来自用户对审计问题的明确回复，不是从作者代码补推出来的论文事实。用户随后单独批准 Stage 2 沿用 D-11 的精确反射策略；无需再次为同一操作请求许可。

| 项目 | 论文依据 | 作者源码依据 | 当前实现方案 | 属性 |
|---|---|---|---|---|
| C 结构 | p.46 III.C、式 (7) | `DenseNet.py:207–248` FE；`cycle_gan_model.py:73–76,113` netNC | 原样 FE，三独立分支 64/128/256 | 作者源码确认；用户确认映射 |
| D 结构 | p.46 III.C、式 (7) | 没有找到作者完整 D 训练实现；FE 给出对称候选 | 第二个独立 FE，无 C/D 参数共享 | 用户批准的有源码依据的工程重建假设 |
| Residual Blocks | p.46 的 six residual blocks 表述未明确总数口径 | 每个分支 6 块，单 FE 总计 18 块 | C/D 均每分支6/总计18，单个参数量9,300,480；保留 GN32、3×3、ReLU | 作者源码确认；论文文字解释经用户确认 |
| CLIP 版本 | p.46 只明确预训练 CLIP ResNet | `cycle_gan_model.py:90` RN50 | 原始 ModifiedResNet RN50，严格加载已知哈希的本地归档 visual 权重，FP32 | RN50 作者源码确认；FP32 执行是本次约束 |
| CLIP 特征层 | p.46 式 (10)–(11)，scale2–4 | `model.py:138–154` 有 layer2/3/4；`CLIP/test_vis.py:79–104` 演示这些模块的 hook | hook visual.layer2/3/4，最后一层是 AttentionPool2d 前的空间特征 | 用户确认的层映射；hook 位置源码确认，Stage2 loss 索引未由作者训练代码确认 |
| VGG 特征层 | p.45 Fig.2、p.46 式 (7)/(12) 的 E | `vggloss.py:34–55` 三个切片 | 复用 relu1_2/2_2/3_2，原始 [-1,1] 输入，无额外 ImageNet 标准化 | 作者源码事实及 Stage1 工程接口 |
| G* 解码 | p.46 式 (9)/(12) | SAD、现有 Stage1 的 checkpoint['G'] | 原始 SAD，strict=True；E/G/CLIP eval 冻结 | 源码事实；真实最终 G 尚未接入验收 |
| 四项 Loss | p.46 式 (8)/(11)/(13)/(14) | 公开旧入口未给出完整实现 | `stage2.py:90` 四项均实现，权重1/1/2/1，先元素均值再跨尺度相加 | 公式论文明确；mean reduction 是工程假设 |
| 参数冻结 | p.45 Fig.2 | 原始模块定义可复用；旧入口不能代替本阶段训练图 | `stage2.py:118,154`：只更新 C/D；原图目标路径 no_grad，生成路径保留图 | 论文明确；独立实现并以合成梯度测试验证 |
| 数据处理 | p.47 IV.A：Stage2 仅 UVTD | 原 Stage1 RGB/crop256/[-1,1] 和 D-11 实现 | `data/stage2_dataset.py:7` 复用精确 reflect16→原随机 crop；无其他增强 | 用户批准的工程适配，不是论文明确操作 |
| LR Scheduler | p.47 IV.A 的100固定+100逐渐衰减、Adam、batch2 | 原 Adam 参数、当前 Stage1 make_scheduler | Adam(0.5,0.999)、1e-4，复用 epoch 边界线性公式；最终轮非零更新 | 总日程论文明确；beta、精确边界是工程假设 |

**新增源码证据**：本次读取作者固定 commit 的 [CLIP/test_vis.py:79–104](https://github.com/gemyumeng/HSRUIE/blob/397da5e492b309f5c1af4f64df73c5a647e038cd/CLIP/test_vis.py#L79)，确有 layer1–layer4 的空间 hook。该脚本为可视化使用 `detach().cpu()` 和 no_grad；本次训练适配器没有复制这两项操作到 Xc 分支。它证明模块位置可获取，不证明作者 Stage 2 实际采用了本次层映射/预处理。

**CLIP 预处理工程假设**：用户指定先从 [-1,1] 回到 [0,1]，再 Tensor 双三次 resize224、官方 mean/std。实现显式固定 `align_corners=False, antialias=True`，不 clamp、不量化、不经过 PIL/NumPy。原 loader 的 PIL Resize/CenterCrop/ToTensor/Normalize 在方形 crop256 上具有相同目标尺寸，但 PIL uint8 插值有截断和量化，与 Tensor 流程不逐像素等价。高对比合成图的本地探针得到 resize 后范围 `[-0.0569285,1.0569285]`，与 PIL 标准化结果的最大绝对差0.2064295、平均绝对差0.0230886；不能据此宣称完整预处理与作者训练一致。该过冲来自双三次负瓣，没有悄悄增加 clamp 来改变协议。

**本地新实验事实**：2026-11-11 已验证本地官方哈希 RN50，在 CPU/FP32 上对一张合成图做前后向：layer2/3/4 分别 `[1,512,28,28]`、`[1,1024,14,14]`、`[1,2048,7,7]`，与手工顺序调用原始模块逐位相等；输入梯度有限且非零，参数冻结、BN buffers 未变。证据 `runs/stage2-build-20261111-01/clip_cpu_probe.json`。这不是完整 VGG/SAD/C/D 的真实模型 Smoke。

**资料决策**：用户允许直接使用不带 `(1)` 的 PDF，本页已记录的实际 PDF 哈希继续作为论文版本依据，不再将文件名后缀列为阻塞。**数据边界**：Stage2 精确反射已获批准；真实 Stage1 run manifest/config 和最终 G 仍未在本地取得。正式 train 入口会核对来源 G 的 config 中实际 underwater records 与 Stage2 全域清单一致，并要求 hash-bound 的 epoch100 验收 JSON；不能只凭候选目录名认定完成。

**当前执行边界**：新增独立 Stage2 文件，不改任何已跟踪的 Stage1/作者源码或原有测试。没有服务器操作、Git commit/push/reset、分支切换、GPU 操作或正式训练。全部最终测试、审查与剩余问题以实现文档为准。
