# Stage 1 MS-UDAG 预训练代码审计

审计日期：2026-11-09（Asia/Shanghai）。工作区：`D:/Research/HSRUIE`；源码 HEAD：`21ff69916b5ce95a1f4a1aea89b23d15c0a15d4b`。本报告为静态审计与重建方案，**不是训练完成记录或修改授权**。

## 结论与检查边界

**源码事实**：已有候选组件 E=`models.vggloss.Vgg19_out`、G=`models.DenseNet.SAD`，但当前 `train.py → CycleGANModel` 没有 Stage 1 双域前向、重建目标或排序目标；G 不在当前优化器和保存名单。单独更改 epoch、补 B_size 或设置 `--netG SAD` 都不足以恢复 Stage 1。当前训练链还存在未构造 `netUNET` 等明确断点。

**关键发现**：仓库存在两个同名 `Vgg19_out`，接口和权重来源不同。活动模型导入的是 `models/vggloss.py:13–47` 的三尺度、加载本地权重版本；`networks.define_G('vgg')` 实际构造 `models/networks.py:323–357` 的随机初始化、单尺度版本，不能作为等价替换。

本次只读取源码、Git 历史、已有文档和论文；不访问本地/服务器数据集、不加载 checkpoint、不导入项目模型、不执行 GPU/训练/推理、不安装依赖、不运行现有参数解析入口。唯一新增文件为本文，创建前该路径不存在。已用 `git diff --exit-code -- <逐个源码文件>` 确认所审计的 17 个 Python 文件与 HEAD 一致；工作区原有 README、`.pyc` 改动及未跟踪 AGENTS/docs 均保留。Linux `/home/czy/projects/HSRUIE` 的实际源码与本地是否一致未核验；此前单图成功不认证训练路径。

**证据标签**：“论文要求/论文事实”来自下述原文；“源码事实”来自本次静态读取；“工程适配建议/计划”均未实施；“未确认项”不得用默认值冒充原始协议。数据结论只引用 [已有数据审计](dataset_inventory.md:101)，不视为本次重新盘点。

## 1. 论文目标与逐式对照

**论文事实（本次重新读取原文）**：本地 `C:/Users/a1825/Desktop/新/Task-Driven_Underwater_Image_Enhancement_via_Hierarchical_Semantic_Refinement.pdf`，15 页；印刷 p.45/46/47 对应 PDF 第 4/5/6 页。以已安装的 Miniconda Python 3.13.9、PyMuPDF 1.28.2 在内存提取这三页文本，未保存中间文件。默认 Python 3.12 缺少 pypdf，改用现成环境，无依赖安装。论文 Section III.B、式(1)–(6) 与 IV.A 已直接核对；其他历史结论见 [论文协议](paper_protocol.md)。

记 Xu 为 UVTD 水下输入，Xa 为 Flickr2K 空气域输入，二者不配对；E 为固定的预训练 VGG，G 为共享 MS-UDAG。

| 原文位置 | 论文要求 | 当前实现与判定 |
|---|---|---|
| p.45 III.B，式(1) | `Fa^l=E(Xa)`、`Fu^l=E(Xu)`，l=1,2,3 | 有三尺度 E 组件；`CycleGANModel.set_input/forward` 只读 A 并计算一次 E，没有 Xa/Fa（`cycle_gan_model.py:97–121`）。**组件已有，双域分支缺失**。 |
| p.45，式(2) | `Xa_hat=G(Fa)`；`L_rec_a=||Xa−Xa_hat||1`；正文权重 1 | 有 SAD 图像生成组件，没有空气重建分支/损失（同文件 `109–128`）。 |
| p.45，式(3) | `A=G(Fu)`；`L_rec_u=||Xu−A||1`；正文权重 0.1 | 当前为 E→NC→HIGH2/HIGH3→SAD，第一尺度旁路 LOW；不等于 E→G 的 Stage 1 水下分支（同文件 `112–121`）。无水下重建损失。 |
| p.46，式(4) | `L_sr=max(0, −(d(Xa,Xa_hat)−d(Xu,A))+m)`；m=1 | 没有排序目标、margin 或相似度调用（同文件 `123–128`）；**缺失，无法评为方向已正确**。 |
| p.46，式(5) | 以 `dot(Xa,Xa_hat)/(||Xa||2 ||Xa_hat||2)` 举例定义相似度 | 这是 similarity，越大越相似；不是距离。原文写“for example”，具体实现仍须明确采用该示例。 |
| p.46，式(6) | `L=L_sr+1·L_rec_a+0.1·L_rec_u` | 当前仅 `loss_G=CE_Loss(unet_out,label)`，且属性未赋值（同文件 `125–128`）。**目标不一致且训练断裂**。 |

**源码事实**：`CycleGANModel.__init__:83` 只构造了 `criterionCycle=L1Loss()`，在该模型中未调用；不能把“存在 L1 类实例”写成已实现两个重建项。`modify_commandline_options:34–38` 的 lambda_A/B/SA/SB/identity 未进入上述目标，不等于论文的 lambda_rec。`models/networks.py:171–212,218–318` 有通用判别器/GAN 工具，但当前链未使用；论文 Stage 1 式(6)不含 GAN、cycle consistency、感知损失、CLIP 或 CE，重建方案不加入它们。

**方向核验依据（解析推导，非模型实验）**：应写为 `relu(m−s_a+s_u)`。铰链激活时，增大 s_a 使损失下降，增大 s_u 使损失上升；例 `(s_a,s_u)=(1,0)` 时为 0，交换后为 2。不能用 `relu(m+s_a−s_u)`，也不能直接把 `1−cosine` 代入仍保留原符号。

**未确认项**：L1 的像素/通道/batch 归约；cosine 的展平维度、零范数 epsilon、逐样本 hinge 后平均还是先平均相似度；计算相似度时的图像值域。建议候选是每张图展平 C/H/W、逐样本计算式(4)再 batch mean，但这不是原文明确给出的实现细节。对归一化结果加偏移会改变 cosine，不能将 `[0,1]` 与 `[-1,1]` 视作等价。L1 sum/mean 会改变其相对排序项的尺度，权重 1/0.1 不能脱离归约方式认证。

## 2. 组件定义、调用与 shape

**源码事实**：工厂 `models/__init__.py:25–65` 由 `--model cycle_gan` 选择 `CycleGANModel`；`train.py:28–35` 创建数据、取长度、创建模型、setup。默认 netG 虽是 `resnet_9blocks`（`options/base_options.py:34`），活动构造却硬编码 LOW/HIGH2/HIGH3，SAD 直接实例化（`cycle_gan_model.py:63–76`）；单改 netG 标志没有 Stage 1 切换作用。

| 对象/函数 | 具体源码 | 实际行为 |
|---|---|---|
| `models.vggloss.Vgg19_out.__init__/forward`，活动 E | `models/vggloss.py:13–47`；导入/调用 `cycle_gan_model.py:8,63,112` | `vgg19(pretrained=False)` 后显式加载 `HSRUIE/model_data/vgg19-dcbb9e9d.pth`；前三组 feature 索引 0:4、4:9、9:14，输出三张量列表。参数默认冻结；slice4/5 被注册但未调用。 |
| `models.networks.Vgg19_out`，同名非活动版本 | `models/networks.py:323–357`；`define_G:156–168` | 未加载预训练权重，只返回 h_relu3 张量。经 `init_net:104–119` 还会初始化 Conv 权重；冻结 requires_grad 不阻止初始化。不能通过 `define_G('vgg')` 获得论文 E。 |
| `SAD.__init__/forward`，候选 G | `models/DenseNet.py:250–376` | 三尺度 64/128/256 通道各投影到 64；密集块、下采样、上采样和 skip 融合，输出 3 通道 Tanh；不是原图 RGB 残差相加。 |
| G 的下/上采样 | `DenseNet.py:166–187,342–369` | stride=2 卷积下采样、双线性固定 2 倍上采样，含 InstanceNorm；融合卷积用 GroupNorm。256 方块能在空间尺度上闭合，此为静态 shape 推导。 |
| `Dense_Block_IN` | `models/NEDB_IN.py:6–47` | 三组 Conv/ReLU 密集拼接、1×1 融合和残差；同一 group 同时通过命名属性和 ModuleList 注册，后续导出应保留现有 state_dict 键结构。 |
| `networks.define_G('SAD')` | `models/networks.py:152–153,168` | 提供 SAD 工厂并走初始化/DataParallel；活动 CycleGANModel 没走它，而是 `SAD().cuda()` 后加载现有权重。两者初始化/封装行为不同。 |
| `Mynet.forward` | `models/DenseNet.py:11–31` | E→NC→LOW/HIGH2/HIGH3→SAD 的完整增强候选，不是 E→G 预训练，也不是 train.py 实际实例。不能为 Stage 1 直接替换。 |

**源码事实与静态推导**：`[B,3,256,256]` 经活动 E 产生 relu1_2 `[B,64,256,256]`、relu2_2 `[B,128,128,128]`、relu3_2 `[B,256,64,64]`；直接输入 SAD 后输出 `[B,3,256,256]`，范围由 Tanh 限定为 `[-1,1]`。SAD 最低空间尺度为 16×16。此接口适合候选 E→G 重建，不构成逐层结构与论文完全一致或数值运行已通过的证明。

**重要的无效注册模块**：`SAD.__init__:283–286` 注册 trans_down_1/2/3/4，但 `forward:330–376` 只调用 trans_down_3/4；trans_down_1/2 的参数预计无梯度。重建时不要擅自删除或接回，避免改变结构/权重键；参数更新验收必须区分活动与未调用模块，不能要求每个 SAD 注册参数都有非零梯度。

## 3. 参数、冻结、Optimizer 与梯度

**论文要求**：III.B p.45 指 E 是 pretrained、fixed，Stage 1 预训练 G；不需要 C/NC、D、DEM/SEM、U-Net 或 CLIP。应训练 SAD 活动路径的 Conv、GroupNorm 等可学习参数，冻结整个 E；Stage 2/3 对 G 的冻结协议不应倒灌成 Stage 1 冻结 G。

| 参数/状态 | 当前源码事实 | 对 Stage 1 的判定 |
|---|---|---|
| E | `vggloss.py:36–38` 设所有参数 requires_grad=False，无输入标准化代码（`40–47`） | 冻结符合目标；是否需要额外 ImageNet mean/std 未由论文明确，不能擅自补入。 |
| SAD/G | `cycle_gan_model.py:70–71` 直接构造、加载；无显式冻结，不在 optimizer | 默认需梯度与会被优化是两件事。当前 G 没有可验证更新，更不是 Stage 1 优化器。 |
| NC/FE | `cycle_gan_model.py:73–76,113` 构造/加载/调用，未冻结且不在 optimizer | Stage 1 不应经过 NC。若未来仅修复下游损失，NC/SAD 可能产生未被 optimizer 清零/更新的梯度；这是条件推导，不是当前训练事实。 |
| LOW/HIGH2/HIGH3/UNET | `cycle_gan_model.py:85–87` 一个 Adam 参数组包含这四者，lr=opt.lr、betas=(opt.beta1,0.999) | netUNET 从未赋值，构造会失败；LOW 在 forward:116–117 被旁路，即使补齐其他断点也没有活动损失路径。 |
| 实际 backward/logging | `cycle_gan_model.py:123–137,50`；`base_model.py:136–142` | unet_out、label、loss_clip 均未赋值；CE 无有效输入，loss_names 还会读取不存在的 loss_clip。不能声称 optimizer.step 已有效执行。 |

**错误断梯度检查**：活动 `CycleGANModel.forward`、`SAD.forward` 中未发现 `detach()` 或 `no_grad()`；问题是目标/优化对象缺失，不是已有 Stage 1 被某个 detach 截断。`vggloss.py:61–63,83` 的 detach 属于其他感知损失类，当前 Stage 1/活动 CycleGANModel 没有调用它们。`BaseModel.test:98–106` 则明确包裹 no_grad，不能借用其 test() 来训练。

**工程适配建议**：独立 Stage 1 显式 `E.eval()`/冻结、`G.train()`；优化器只包含去重后的 G 参数，核对参数 ID 集合。原始输入不需反向，E(Xa/Xu) 可在 no_grad 内提特征而不妨碍随后对 G 求导；**G 的前向、重建图和 loss 不可 detach/no_grad**。若未来上游输入本身需学习，E 参数冻结也不能据此截断对输入的梯度。当前 BaseModel 是 ABC，不是 nn.Module（`base_model.py:8`），不能假定调用外层 `.train()` 能递归管理所有组件；其 eval() 也只遍历 model_names（`91–96`）。

## 4. Dataset/DataLoader 与两个不配对域

**源码事实**：`UnalignedDataset` 的说明称支持 A/B，但实现不支持。

- `__init__:26–31` 仅拼 `<dataroot>/<phase>A`，只创建 A_paths/A_size/transform_A；没有 dir_B/B_paths/B_size/transform_B。
- `__getitem__:45–51` 只读取 RGB A 并返回 A、A_paths；没有任何 B 采样。`__len__:59` 引用不存在的 B_size。`train.py:30` 取长度就会触发该缺陷（前提是 trainA 路径已存在），不必等到 Forward。
- `CycleGANModel.set_input:105–107` 即使收到 B 也不使用，AtoB 局部变量不控制实际读入方向。Stage 1 两域不能仅靠修 Dataset 一处完成。
- `data/base_dataset.py:get_transform:81–112` 的 resize/crop/flip 都被注释，只执行可选灰度、ToTensor、Normalize(0.5)。`get_params:63–78` 能产生 crop_pos，但 UnalignedDataset 不调用，且当前 get_transform 不执行它。`crop_size=256`、`preprocess=resize_and_crop` 不等于生效。
- `data/__init__.py:75–79` 的 DataLoader 用 batch_size、shuffle、num_workers，未指定 collate_fn、drop_last、worker_init_fn、generator（使用库默认）；不同原始宽高无法保证默认 stack 成 batch=2。`__len__:84–86` 返回样本数，迭代限制 `90–92` 按批边界截断；max_dataset_size 非整批时需注意实际数量。
- `data/image_folder.py:12–32` 支持 BMP/JPG/PNG 并递归枚举。指向 UVTD 整根会混入 Reference/两类 mask，应显式指向 Raw；这来自已有数据报告，不是本次数据访问。`SingleDataset:19–36` 只有单域；`AlignedDataset:21–22,40–56` 假设左右拼接 AB，也不等价于不配对双域。

**论文要求**：UVTD Raw 与 Flickr2K 独立抽样，各自以本域原图为重建目标；不需要同名、不需要图像对齐，不将 Flickr2K 当成某张水下图的干净配对目标。Reference 不作为 Stage 1 默认输入/监督。

**工程适配建议/待确认**：新 Dataset 用显式 `underwater_root`、`inair_root` 与清单，返回 Xu/Xa 及各自路径；两个域的 crop 坐标独立。域长度不同，`max(Nu,Na)` 搭配短域循环/随机采样、以 UVTD 为 epoch 基准或固定步数都可行，但属于不同采样协议，需选择并记录每域曝光次数；不能把修好的 max(A_size,B_size) 当作论文指定。batch=2 拟按每域 2 张理解，论文未明确“双域合计还是每域”，应人工确认。训练集两域均须非空，固定文件清单、排序、worker/采样 RNG。

## 5. 短边不足 256 的候选处理策略

**历史证据**：[数据报告](dataset_inventory.md:130) 记录本地样本 Raw/39.bmp=352×224；2026-11-02 全量尺寸表记录 352×224 共 483 张、224×352 共 21 张，合计 504 张。此处只引用文档，不接触数据，不重新确认数量。满足 16 倍数不保证能直接裁出 256 方块。

**论文要求**：p.47 IV.A 写随机裁成 256×256，未说明短边不足时如何处理。下表均为候选工程方案，**不选择最终策略**；Flickr2K 也应进行相同边界条件检查。

| 候选策略 | 优点 | 对论文裁剪/分布的影响 |
|---|---|---|
| 仅不足时等比例放大短边至 256，再随机 crop | 保持比例、保留样本；352×224 会约变为 402/403×256 | 增加未明确的插值，改变纹理/频谱；短轴无随机位置。需确认尺寸取整、插值和抗锯齿。 |
| 不足边反射/复制填充后 crop | 不缩放内部内容，保留样本 | 边界含人工重复；改变生成先验与相似度。填充量、位置、模式须确定，不能称论文原始设计。 |
| 全部 resize 为 256×256 再 crop | 简单且固定 shape | 非方图拉伸，crop 坐标固定没有空间随机性；与“随机裁剪”明显不同，不宜作为无说明默认。 |
| 全部等比例 resize 到短边≥256（可设更大尺寸）再 crop | 两域变换统一，可提供随机位置 | 对大图也重新采样；load_size 及裁剪尺度成为额外协议，不能直接恢复旧模板即称忠实。 |
| 排除/重采样短边不足图 | 不插值、不填充 | 改变训练样本分布与有效样本数；不能自行丢弃这 504 张或按目录改写数据。 |
| 原尺寸越界 PIL crop、零填充；或可变小 patch | 易实现 | 越界 crop 产生黑边，可变 patch 不满足 256 设置并妨碍默认 batch；均不是无影响修复。 |

重建时需将最终策略作为显式配置，并记录原尺寸→变换尺寸→crop 坐标。恢复 `get_transform` 的所有注释会同时启用 resize/flip 等多项计算变化，不是本任务授权；也会影响现有测试基线。

## 6. batch、epoch、调度和保存恢复

| 项目 | 论文/目标 | 当前源码事实与风险 |
|---|---|---|
| Batch | p.47 IV.A：2 | `options/base_options.py:45` 默认 2，与标量一致；双域含义/尾批仍待确认，当前根本无 B 域。 |
| Epoch | p.47 IV.A：50 固定 + 50 衰减，共 100 | `train_options.py:27,30–31` 默认 epoch_count=1、40+40；`train.py:38` 包含末端，默认 80 轮。改为 50/50 还须处理 scheduler 边界。 |
| Adam | p.47：初始 1e-4、Adam | `train_options.py:32–33`、`cycle_gan_model.py:85–86` 为 lr=1e-4、betas=(0.5,0.999)，未传入项按所用版本默认；beta/epsilon/weight_decay 不是原文确认配置。 |
| LR | 后 50 轮逐渐降至 0；原文未写精确曲线 | `networks.get_scheduler:54–58` 为线性 lambda，分母 decay+1；`train.py:43` 每轮开始先 step，`base_model.py:116–126` 注释却称末尾。 |
| 保存频率 | 原文未规定 | `train_options.py:23–25` latest_freq=5380、epoch_freq=5；`train.py:49–50,66–75` 的 total_iters 按配置 batch_size 加，实际是名义样本数，不是 optimizer step 数；尾批也按完整 batch 计。 |
| 自动重载 | 区分新训、权重初始化、精确恢复 | `BaseModel.setup:84–89` continue_train 只 load_networks；epoch_count 要手动给；SAD/NC 在构造时无条件硬编码加载，和 continue_train 无关。 |

**源码事实与版本化静态推导**：PyTorch 2.0.1 scheduler 初始化先执行一次 step；结合当前 lambda 和 epoch 开始 step，epoch_count=1、计划改成 50/50 时，训练 epoch k 使用倍率 `1−max(0,k+1−50)/51`。于是 1–49 轮为 1，第 50 轮为 50/51，第 51 轮为 49/51，第 99 轮为 1/51，第 100 轮为 0。默认 40/40 则 1–39 轮固定，第 40 轮开始衰减，第 80 轮为 0。**这是基于源码公式和官方 2.0.1 行为的推导，未运行 torch；现有训练早先就被其他缺陷阻塞。**零 LR 的最后一轮仍会更新 Adam 状态。应明确“50 个完整训练 epoch 固定 LR、何时到零”的边界，再画出完整日程验收；不直接照抄 scheduler 或宣称线性就是原文唯一曲线。依据：[PyTorch v2.0.1 scheduler 源码](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/optim/lr_scheduler.py#L24-L149)。

**checkpoint 事实与兼容性**：

1. `CycleGANModel.model_names:58–61` 训练为 LOW/HIGH2/HIGH3/UNET、测试为 LOW/HIGH2/HIGH3；`BaseModel.save_networks:144–160` 只保存名单内的模型 state_dict。**SAD/G 不会被这条保存链保存**，E/NC 也不在其中；而 UNET 属性不存在。
2. 保存没有 optimizer/scheduler/epoch/step/RNG/sampler 状态。`train.py:36` 还将 total_iters 重置为 0。`continue_train` 至多是部分权重重载，不是精确续训；SAD/NC 无条件加载的固定文件也不受 `--epoch`、`--checkpoints_dir`、`--name` 控制。
3. `save_networks:156–160` 在 CUDA 分支假设 `net.module`，但活动 SAD 是裸模块；将 SAD 加入名单而不处理封装会引入新错误。建议 Stage 1 专用保存器只在 isinstance(DataParallel) 时解包，并保存 CPU state_dict，避免修改共享保存器影响现有基线。
4. `load_networks:182–199` 先解包 DataParallel、加载普通 state_dict，默认严格匹配；加载一个包含 optimizer 等字段的“训练包”不能直接交给现有 SAD.load_state_dict。后续应区分完整训练包与 **裸 SAD state_dict 导出**，保留 DenseNet.SAD 的层名、通道、GroupNorm/InstanceNorm 和 Tanh。已有未调用模块和重复注册键也不要随意整理。
5. 当前测试 SAD/NC 加载在 `cycle_gan_model.py:70–76`，VGG 在 `vggloss.py:16–17`，都是相对 cwd 的 `HSRUIE/...`。在服务器 cwd `/home/czy/projects/HSRUIE` 下会解析到额外嵌套的 `/home/czy/projects/HSRUIE/HSRUIE/...`；部署可能已有外部适配，本次未核验。独立 Stage 1 须用显式绝对路径/项目根解析，不覆盖作者文件。
6. `test.py:48–58` 解析后把 batch/线程等强制改写，`BaseOptions.print_options:109–114` 已先以写模式保存 `<checkpoints_dir>/<name>/<phase>_opt.txt`。训练同样会覆盖 train_opt.txt；`Visualizer:93–102` 写 web 并追加 loss_log.txt。必须在解析前确认新 run 目录，测试 results_dir 不保护 test_opt.txt，记录配置应在所有覆盖后进行。
7. **结构兼容不等于实验兼容**：新 Stage 1 裸 G 权重可在相同 SAD 类中做 strict key/shape 验收；不代表将它单独替换到既有 NC/HIGH 组合可保持结果。下游组件与 G 的联合来源/阶段匹配仍须认证。当前“可加载/单图成功”的作者 checkpoint 不能直接认定为 Stage 1 产物。本次未读取权重内容或计算哈希；历史匹配结论仅见既有文档。

## 7. Linux PyTorch 2.0.1+cu118 的兼容性边界

**工程适配建议（静态风险，未在服务器测试）**：

| 风险/证据 | 最少处理或核验 |
|---|---|
| `environment.yml:17,48,51,71–73` 固定 Python3.8、NumPy1.24.4、Pillow9.4、torch1.8+cu111/vision0.9 | 与服务器已部署 2.0.1+cu118 是不同环境，不能重跑 YAML 覆盖成功环境。官方对应 torchvision=0.15.2，需读取服务器实际版本确认；本次不安装。依据：[官方历史版本表](https://pytorch.org/get-started/previous-versions/#v201)。 |
| `.cuda()` 强制 GPU：`vggloss.py:16`、`cycle_gan_model.py:70,73`；默认 gpu_ids='4'（base_options.py:25,135–136） | CPU smoke 不可直接运行；单卡机器默认卡号也可能越界。未来显式 device、从 CPU map_location 加载后统一搬移，记清可见设备与逻辑编号。 |
| map_location 仅映射 cuda:1→cuda:0（vggloss.py:17、cycle_gan_model.py:71,75–76） | 不覆盖其他保存设备；用显式加载设备，不修改权重内容；核对所有输入/模型/optimizer 状态设备一致。 |
| torchvision `pretrained=False`（vggloss.py:16） | 0.15.2 保留旧接口但弃用警告，不能误称必然报错；未来可按版本改 weights=None 后仍加载同一权重，避免更换预训练版本。依据：[vision v0.15.2 兼容处理](https://github.com/pytorch/vision/blob/v0.15.2/torchvision/models/_utils.py#L142-L228)。 |
| Resize/PIL 插值行为（base_dataset.py:81,85–104） | 当前未执行 resize 不代表重建后无版本差异；固定 PIL/Tensor 变换顺序、插值和 antialias。vision0.15.2 与后续版本默认不同，需显式配置。依据：[v0.15.2 Resize](https://github.com/pytorch/vision/blob/v0.15.2/torchvision/transforms/transforms.py#L283-L332)。 |
| scheduler 先于 optimizer（train.py:43） | 2.0.1 有调用顺序警告与首步偏移；按上一节完整日程核验，不靠消除警告认定符合论文。 |
| 无关 CLIP 构造（cycle_gan_model.py:90；CLIP/clip/clip.py:119–139） | 即便 Stage 1 不调用也会加载/可能下载 RN50、引入依赖；独立 Stage 1 不构造 CLIP/任务网。Stage 1 无需运行 DEM FFT 分支。 |
| `BaseModel.__init__:37–38` 启用 cudnn.benchmark；所审计训练链无显式种子策略 | 需确定随机种子、worker seed、采样 generator 和确定性设置，作为复现工程配置记录，不能称论文原设。 |

没有发现需要因 torch2.0.1 而修改 SAD 卷积、归一化或 Forward 的证据；设备/路径/保存适配与网络结构变更应严格区分。运行时仍需服务器小规模独立验收，单图前向成功不覆盖 backward/optimizer/续训。

## 8. 最小重建方案与拟改文件（全部未实施）

**方案原则（工程适配建议）**：新增专用 Stage 1 路径，复用活动三尺度 E 和原样 SAD，避开修改当前 CycleGANModel/UnalignedDataset/共享 get_transform 对已验证推理链的影响。“最小”指变更行为范围，不以强行复用错误分支来减少文件数。以下名称为拟议接口，不是已存在代码；正式实施需用户另行授权。

| 拟新增/修改文件 | 计划内容 | 论文依据、影响及兼容性 |
|---|---|---|
| 新增 `models/stage1_model.py`，`Stage1Model` | 双域 `set_input`；共享 E/G 两次前向；仅三个目标；Adam 仅 G；日志含各 loss/similarity/margin；专用完整保存恢复与裸 G 导出 | III.B 式(1)–(6)；改变的仅是新增训练路径。复用 SAD，不接 NC/LOW/HIGH/UNET/CLIP，不加载作者 SAD 作为“新训”默认；初始化来源显式记录。 |
| 新增 `data/stage1_unpaired_dataset.py` | 显式 Raw/Flickr2K 根和清单、非空检查、独立抽样与 crop、返回双域路径/变换元数据；短边策略配置化 | III.B 不配对、IV.A 两数据集/256 crop；采样与短边处理为待确认工程协议。保留现有 unaligned 的问题作为报告，不在本阶段顺手修复发布基线。 |
| 新增 `options/stage1_options.py` | 阶段专用参数：域路径/清单、VGG路径、G初始化或resume、2/50+50/1e-4、1/0.1/m1、归约和值域、seed与run目录；解析不抢先覆盖既有日志 | IV.A 和式(2)–(6)规定部分默认；beta、epsilon、LR曲线等另标为工程配置，未确认值不得伪装原文。 |
| 新增 `train_stage1.py` | 串起 Dataset/Model；确认输出不存在或明确续训；optimizer之后调度、准确计数、100轮日程、完整恢复/配置与审计记录；无默认全量启动 | IV.A 的训练目标日程；路径/日志/恢复/确定性属工程补全。可复用工厂与 DataLoader机制，但不照搬 train.py 的保存/调度缺陷。 |
| 最小修改 `models/vggloss.py:Vgg19_out.__init__` | 可选 weight_path/device 参数，安全加载到显式设备；保持旧调用默认语义、相同 slices/输出、冻结和数值计算 | 式(1)固定预训练 E；仅设备/路径适配。不得同时加入 ImageNet normalization 或切换到 networks.Vgg19_out。修改后需旧/新调用的同权重特征一致性检查。 |
| 新增 `tests/test_stage1_contract.py`（或同等独立验收脚本） | 下节的数值方向、batch形状、梯度归属、checkpoint往返与日程边界检查 | 验收手段为工程配置，不是论文额外损失；只在后续授权后编写/执行。 |

**明确不拟改**：`models/DenseNet.py`、`NEDB_IN.py`、`FEM.py` 的结构/Forward/归一化；当前 `cycle_gan_model.py`、`train.py`、`test.py`、`data/unaligned_dataset.py`、`data/base_dataset.py` 的发布行为；作者权重和数据本体。若后来选择直接在这些共享文件上重建，须重新说明影响，不能将此报告当作批准。

**权重与指标边界**：新的随机 G 初始化（采用直接 SAD 的 PyTorch 默认初始化还是现有 init_weights normal(0,0.02)）需确认；二者不同，论文未规定。只有显式 resume 才重载新 run 的训练状态；作者 SAD 作为初始化需单独标注为 warm-start，不能混称从头复现。保存训练包和导出权重到全新 run，不自动替换 `latest_net_SAD.pth`。任何预处理/归约/初始化差异都会影响训练先验和后续指标可比性，应记录为工程协议或待验证设置。

## 9. 单 Batch 验收计划（本次全部未执行）

验收前先完成下节人工确认及实现授权。服务器 cwd `/home/czy/projects/HSRUIE`；所有输出使用唯一新 run ID。CPU 优先验证接口/公式，CUDA 只在之后明确获准的小规模检查中使用；本报告不提供当前可直接启动训练的命令。

1. **无模型数据契约**：`num_workers=0`、固定 seed，只取一批；Xu/Xa 各 2 张、float32 `[2,3,256,256]`、有限值、值域按批准配置。验证路径来自各自清单、无 Reference/任务标签；显式纳入历史小尺寸样本 39 验证已批准策略，图像本体不写回。该检查不证明全库可读。
2. **前向接口**：加载指定 SHA-256 的 VGG；E 两域各返回上节三尺度，G 两域输出与各自输入同 shape/值域；使用同一个 G 参数对象，未构造任务网/CLIP/内容提取器。首次只做一批，不遍历全库。
3. **目标数值与方向**：单独用可控标量验证 hinge 的 `(1,0)→0`、`(0,1)→2`；铰链激活区增加 s_a 下降、增加 s_u 上升。验证逐样本独立计算和归约、`L=L_sr+L_rec_a+0.1L_rec_u`，输出两个 similarity 和全部分项。若采用 cosine，覆盖零范数、相同/反向向量与 batch维隔离；不要求真实初始 batch 的 L_sr 必须等于 0。
4. **一次 backward 与一次 step**：E全部参数 requires_grad=False、grad=None、数值前后不变；G的活动分支有有限梯度，两个生成图到 loss 的梯度链都存在，至少一个活动 G 参数发生有限变化；optimizer参数ID只含 G且去重。单独核查已知 trans_down_1/2 无调用/无梯度，不据此接回模块，也不要求每个活动 ReLU后参数的梯度都非零。无其他模块更新，无 detach掉生成图。
5. **checkpoint 契约**：在独立临时 run 路径保存一次完整训练包及裸 G；新实例 strict 加载裸 G，报告 missing/unexpected keys=0、同一输入同模式输出在规定容差内一致。训练包恢复 optimizer/scheduler/epoch/step/RNG/采样状态；若验证恢复后下一更新的一致性，另用一批固定合成输入做对照并明确它是小型恢复检查，非额外正式训练。不得覆盖原 checkpoint。
6. **日程与日志**：无需100轮模型计算，用纯配置/调度检查验证第1/50/51/100轮及终止后的LR、恢复位置和更新顺序；记录实际每域batch/尾批/step数。核实 train_opt、loss_log、导出目录不会触及旧 run，落盘的是最终生效配置。精确到零的时点及最后一轮是否有非零LR须与已批准日程一致。

所有检查结果后续应标注处理/失败/跳过数量、结束状态；一次 batch通过不等于完成训练、数据质量审计或论文指标复现。正式训练仍需单独授权。

## 10. 实施前必须人工确认与留档

- **数据划分与泄漏边界**：Stage 1 使用哪些 UVTD ID、如何与后续评估隔离；检测 train_3380/test_2000 的官方用途尚未认证，不能直接移作 Stage 1 split；不能一边训练其全部图像一边声称2000张为未见测试。Flickr2K 是否另设验证/早停数据，验证协议与模型选择原则须记录；SUIM TEST 110 张不得介入训练/选型。
- **Flickr2K 来源**：官方来源/版本、具体高质量图像子目录（如HR而非降采样派生目录）、真实服务器路径、样本清单/数量和许可情况。已有数据报告未确认它已就绪，本次不访问数据目录。
- **预处理**：上节短边策略、插值/antialias/取整、是否对大图resize、随机flip或其他增强、两域变换是否一致；256×256裁剪和各域batch的解释。不能默认恢复模板的全部增强。
- **值域与目标细节**：输入与重建L1/cosine在哪个范围计算；VGG输入有无专用ImageNet normalization；L1 sum/mean；cosine展平/epsilon；hinge与batch归约。拟用式(5)示例须明确确认。
- **优化日程**：G初始化来源/seed，Adam beta1/beta2/eps/weight_decay，50+50的曲线及边界、AMP/梯度累积/裁剪是否使用（原文未说明，均不自动加入）；每域采样、epoch定义、尾批和续训精度要求。
- **组件与来源**：活动三尺度VGG层与论文E的对应、SAD与论文G逐层对应及原始训练代码是否另有发布；已有checkpoint的阶段/训练来源、VGG与后续阶段权重组合的认证。不能从文件名latest或可加载推断阶段。
- **最小运行记录**：唯一run ID、开始/结束时间、完整命令/cwd、服务器Python及软件版本/设备、seed/确定性、commit及未提交源码diff、所有实际组件路径与SHA-256、最终生效配置/Forward、数据清单和split、独立输出目录/计数/状态。Stage 1未用的NC/LOW/HIGH/任务网等明确记“未加载/不适用”，不伪造权重记录。

**审计结束状态**：论文相关三页已重新读取，所列本地源码/选定Git提交已静态检查；未发现可直接执行的 Stage 1 训练实现。报告给出的独立重建方案与验收均为计划，未修改源码、数据或权重，未执行任何模型实验。下一步是确认上述协议并另行授权实现；本任务到此停止。
