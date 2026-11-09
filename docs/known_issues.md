# 已知问题与待确认差异

更新日期：2026-11-02；源码基线：21ff69916b5ce95a1f4a1aea89b23d15c0a15d4b。本轮仅文档化，没有修复源码。分类沿用审计要求：通用复现知识、实验规范风险、项目特有异常。论文原始协议见 [paper_protocol](paper_protocol.md)。

“图示不同”不自动等于“源码应立即改正”。涉及权重兼容或指标可比性的差异，须先报告并获得结构变更授权。

## KI-01：LOW 旁路

- **分类/发现位置**：项目特有异常；[cycle_gan_model.py](../models/cycle_gan_model.py):64-65、116-117；[Mynet](../models/DenseNet.py):21-30。
- **实际表现**：LOW 被创建并加载，活动 Forward 却使用 fe1=fc1；Mynet 会调用 LOW。
- **证据**：源码事实；第二轮权重 key/shape 匹配。作者历史 8a9a5c3 已含旁路；论文 p.45 Fig.2、p.46 III.D 指定浅层 DEM，但没有解释发布代码旁路。
- **影响**：接回 LOW 不改变 shape，却改变 SAD 输入数值、结果、计算和显存；影响论文指标可比性。
- **解决状态**：保留 LOW-off 发布代码基线；作者意图待确认。不是已经认定的笔误，也不是完整 DEM-off 消融，因为 HIGH2 的 DEM 仍执行。

## KI-02：resize_and_crop 不执行缩放或裁剪

- **分类/发现位置**：项目特有异常；[base_dataset.py](../data/base_dataset.py):81-112。
- **实际表现**：resize、crop、flip、尺寸对齐均被注释；RGB 只执行 ToTensor 和 Normalize(0.5,0.5)。
- **证据**：源码事实；第二轮实际构造三种 preprocess 均为同一转换链；论文 p.47 记录 256×256 随机训练裁剪。
- **影响**：日志中的参数名不代表实际预处理；变尺寸 batch_size=2 的训练还可能无法直接堆叠。训练裁剪不能简单恢复为推理默认行为。
- **解决状态**：未修复。当前推理保持原尺寸；未来训练专门落实同步裁剪与记录。

## KI-03：SAD 与 MS-UDAG 未完成逐层对应

- **分类/发现位置**：实验规范风险；[SAD](../models/DenseNet.py):250-376；[加载/调用](../models/cycle_gan_model.py):70-76、121。
- **实际表现**：SAD 接收三尺度特征并生成 RGB，与论文 G 的高层接口相符；但没有认证逐层结构与 Stage 1 训练来源。
- **证据**：源码事实；论文 p.45 Fig.2、式(1)-(6)；第二轮 checkpoint 结构兼容检查。
- **影响**：不能宣称完整 MS-UDAG 已复现，也不能宣称该模块完全不存在。结构兼容不证明训练履历。
- **解决状态**：接口对应暂可采用，逐层与阶段来源待确认；不擅自替换 G。

## KI-04：HDE 末端 1×1 与论文 3×3 不同

- **分类/发现位置**：项目特有异常；[EdgeEnhancer.out_conv](../models/FEM.py):192-196。
- **实际表现**：源码使用 1×1，论文 p.46 Fig.3 末端标注 3×3。
- **证据**：论文图示事实与源码事实；第二轮 LOW 权重对应 [32,32,1,1]，HIGH2 为 [64,64,1,1]。
- **影响**：直接改成 3×3 会破坏 checkpoint shape 兼容，不能作为现有推理的无害修复。
- **解决状态**：保留 1×1，不擅改 HDE；作者说明、对应权重或单独结构重建待后续。

## KI-05：DEM 池化与下采样表述的差异

- **分类/发现位置**：论文与源码待对应；实验规范风险；[MEEM](../models/FEM.py):255、279-283；[DetailEnhancement2](../models/FEM.py):109-113。
- **实际表现**：MEEM 的 AvgPool2d 为 kernel=3、stride=1、padding=1，不缩小空间尺寸；中层图像入口另有 stride=2 卷积。
- **证据**：源码事实；论文 p.46-47 III.D 提到两次 downsampling，Fig.3 画 AP 但未列其 stride。
- **影响**：不能把所有 AP 当成 stride=2；这样会影响分支拼接和数值。图示省略与实现差异尚未分清。
- **解决状态**：待作者/更完整配置确认，不自动改变 stride。

## KI-06：SEM 的输入残差未在图中画出

- **分类/发现位置**：可能的实现差异；实验规范风险；[FEM.forward](../models/FEM.py):65-86。
- **实际表现**：最后执行 out=out*self.scale+x，且构造中 self.scale=1；论文 p.47 Fig.4 没有画出该输入残差。
- **证据**：图示与源码事实；尚无作者说明证明它是错误。
- **影响**：删掉残差不改变 shape，但改变数值和 checkpoint 的行为；不能无记录修正。
- **解决状态**：待确认，保持当前发布计算。

## KI-07：三阶段训练计算图和交接缺失

- **分类/发现位置**：项目特有异常；[train.py](../train.py):27-52；[cycle_gan_model.py](../models/cycle_gan_model.py):82-137。
- **实际表现**：没有 Stage 1 双域重建/排序目标，没有 Stage 2 独立 D 和四项损失，也没有完整 Stage 3；只保留不完整 CE 调用。默认日程为 40+40。
- **证据**：源码事实；论文 p.45-47 式(1)-(15)和 IV.A，要求 Stage 1/3 为 50+50、Stage 2 为 100+100。
- **影响**：不能直接开始论文三阶段训练或把普通 train.py 运行称为三阶段复现。
- **解决状态**：未完成，按 P2-P4 分阶段规划；不额外添加论文没有要求的 GAN/cycle loss。

## KI-08：Stage 3 的 U-Net、标签和 CE 未接通

- **分类/发现位置**：项目特有异常；[cycle_gan_model.py](../models/cycle_gan_model.py):85-86、97-107、123-128；[UNET 工厂](../models/networks.py):164-165。
- **实际表现**：optimizer 访问未创建的 netUNET；没有 unet_out 和 label 赋值，也没有 I_e→U-Net 调用。
- **证据**：源码事实；论文 p.47 式(15)。仅在工厂定义 Unet 不表示已经构造。
- **影响**：训练初始化/反向路径无法完成；当前 UVTD 语义标注也未定位。
- **解决状态**：未修复，P4 被标注及计算图共同阻塞。T/C 是否同步优化仍待确定，不能默认任务网络必冻或必训。

## KI-09：test_opt.txt 会被覆盖且不等于最终配置

- **分类/发现位置**：实验规范风险；[BaseOptions](../options/base_options.py):91-126；[test.py](../test.py):48-54。
- **实际表现**：print_options 用 wt 写 checkpoints/<name>/<phase>_opt.txt；测试脚本随后覆盖 batch_size、线程数、排序、翻转、display_id。
- **证据**：源码事实。参数文件在模型创建/成功推理之前即写入。
- **影响**：新 results_dir 也不能避免覆盖旧参数记录；已有 opt 不能单独证明成功运行或最终配置。
- **解决状态**：待处理，列入 P0 的启动前条件。当前不得直接重跑入口；需先设计独立运行记录和旧记录保护。

## KI-10：checkpoint 使用两套加载规则

- **分类/发现位置**：项目特有异常；[vggloss.py](../models/vggloss.py):16-17；[cycle_gan_model.py](../models/cycle_gan_model.py):70-76；[BaseModel](../models/base_model.py):176-199。
- **实际表现**：VGG/NC/SAD 固定使用相对 cwd 的 HSRUIE/... 路径，LOW/HIGH2/HIGH3 使用 checkpoints_dir/name/epoch。
- **证据**：源码事实；第二轮确认项目根目录内不存在再嵌套的 HSRUIE 目录。
- **影响**：cwd 敏感；改 epoch/目录不能切换全模型，可能形成混合阶段权重；部分 .cuda()/map_location 也限制设备适配。
- **解决状态**：未修复；P0 先明确六份权重路径、cwd 与哈希，再考虑不改计算的路径适配。

## KI-11：冻结、优化、保存、eval 名单不统一

- **分类/发现位置**：项目特有异常；[模型配置](../models/cycle_gan_model.py):58-90；[BaseModel](../models/base_model.py):91-96、144-160。
- **实际表现**：NC/SAD 未显式冻结，却不在 optimizer/model_names 中；LOW 被优化器列入但被旁路；eval 仅遍历 model_names。
- **证据**：源码事实。论文 Fig.2 对 E/G/CLIP 给出冻结标记；并不充分说明 Stage 3 中 C/T 的所有优化细节。
- **影响**：补齐训练后可能积累无用梯度、遗漏阶段保存；不能把 --eval 等同于统一处理全部组件。
- **解决状态**：未修复；未来按每阶段明确 requires_grad、optimizer、保存与模式名单，并验证梯度。

## KI-12：默认数据模式访问未定义 B_size

- **分类/发现位置**：项目特有异常；[unaligned_dataset.py](../data/unaligned_dataset.py):19-59。
- **实际表现**：只初始化 A，却在 __len__ 返回 max(A_size,B_size)。
- **证据**：源码事实；第二轮隔离调用确认 AttributeError。
- **影响**：默认 unaligned 路径不可用；当前单域推理需明确 single，Stage 1 也不能直接假定该 Dataset 已具备双域能力。
- **解决状态**：未修复；训练时重建有明确语义的 Dataset，不自行用 Reference 充当空气域。

## KI-13：环境与启动配置不能完整重放

- **分类/发现位置**：实验规范风险；[environment.yml](../environment.yml)；[tasks.json](../.vscode/tasks.json):41、96、100；[launch.json](../.vscode/launch.json):10-25；[base_options.py](../options/base_options.py):25。
- **实际表现**：环境文件为 Linux/PyTorch 1.8.0+cu111，实际本地为 Windows/2.0.1+cu118；任务 JSON 有全角逗号和缺逗号；launch 使用项目内不存在的数据路径；默认 GPU=4。
- **证据**：前两轮环境/JSON/路径实际核验；本轮源码未变。
- **影响**：不能从“有配置文件”认定实验可重放；本机单 GPU 需要明确设备。
- **解决状态**：未修复；本轮不安装包、不编辑配置。P0 先形成可审阅启动方案。

## KI-14：潜在损失与任务数据组件问题

- **分类/发现位置**：项目特有异常；[vggloss.py](../models/vggloss.py):59-63；[CE_Loss](../unet_test/nets/unet_training.py):6-15；[任务 VGG](../unet_test/nets/vgg.py):67-72；[任务数据加载](../unet_test/utils/dataloader.py):8、31-40。
- **实际表现**：Perceptual_loss134 访问不存在的第四个 VGG 输出；CE 默认 ignore_index=21，尺寸判断使用 and；任务 VGG 使用作者绝对路径；数据加载依赖另一套 utils/VOC 路径。
- **证据**：源码事实。这些并非当前增强推理已触发的全部错误。
- **影响**：直接启用对应组件可能越界、错误处理单边尺寸或读取失败；类别数/ignore 规则不能照默认使用。
- **解决状态**：未修复，按实际采用的训练/评价路径处理；不得给未使用模块的缺陷夸大成当前推理失败。

## KI-15：结果来源认证和数值重现不足

- **分类/发现位置**：实验规范风险；results_uvtd_10、results_suim_full 等现有产物与 [保存代码](../util/visualizer.py):22-54。
- **实际表现**：缺少逐 run 的命令、最终配置、源码差异和六份权重绑定记录。第二轮确认 10 张 UVTD 均不同于 Raw/Reference，但不是模型来源证明；3 张 SUIM 试运行与全量结果最大差为 1/255。
- **证据**：第二轮逐像素核验；本轮仅复核产物计数。BaseModel 默认启用 cudnn.benchmark，不据此直接断言微小差异的原因。
- **影响**：不能认证现有产物来自当前权重；更不能由小差异反推 checkpoint 不同。重复 stem/多 visual 还可能覆盖同名 PNG。
- **解决状态**：P0 待认证。用户已说明清理旧 UVTD 混入全量结果；该历史事件不作为当前目录结论，也不重复审计旧目录。

## KI-16：本地 SUIM 总量与论文差 1

- **分类/发现位置**：实验规范风险；D:/Research/SUIM/train_val、TEST、INFO.txt；论文 p.47 IV.B。
- **实际表现**：本轮数得 train_val 图像 1525、Mask 1525，同 stem 1525 对；TEST 110 张，本地合计 1635；论文写 1634。
- **证据**：本轮文件盘点与原文，不是推测。尚未验证跨 split 内容重复、来源版本或那 1 张的原因。
- **影响**：全数据规模和划分可比性待核，不能自动删样本或把 TEST 挪入训练补数。
- **解决状态**：新增待确认，归入 P1；保留原始数据与划分。

## KI-17：UVTD 标注与正式划分未定位

- **分类/发现位置**：实验规范风险；本轮检查 D:/Research/UVTD，仅见 Raw/Reference，未发现另列的语义 Mask、检测标注或划分文件。
- **实际表现**：图像对齐不等于下游标注齐全；Reference 是对应参考图像，不是语义 Mask 或检测框。
- **证据**：文件盘点；论文 p.47 明确 Stage 3 需要 UVTD 语义标签，检测评价需要对应真值。
- **影响**：P4 被语义标注阻塞，P6 被检测标注阻塞；完整任务指标消融也受影响。P0/P2/P3 不因缺下游标注直接阻塞。
- **解决状态**：在已检查路径未找到；其他磁盘/下载包待用户定位，不声称整机绝对不存在。
