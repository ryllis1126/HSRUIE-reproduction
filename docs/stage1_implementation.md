# Stage 1 最小实现方案与验收记录

## 2026-11-10 D-11 反射填充（最新实现与验收）

用户已批准在内存处理UVTD两种尺寸：352×224上下各reflect16，224×352左右各reflect16；其余合法图不填充，随后沿用随机256crop、RGB及[-1,1]。这是工程适配，不是论文明确操作。反射只启用于水下域的上述精确尺寸，不自行扩展到Flickr2K或其他未授权短尺寸。门禁继续拒绝坏图和未授权尺寸，不缩放、不删除、不静默过滤。

本次基线3102bc2d80555ccd86edae00799fe9583a259334，工作区开始时干净。用户报告此前服务器真实GPU Smoke成功，本次不重跑GPU、不执行正式训练。

- [x] 先新增CPU合成测试：两方向尺寸、手算reflect边界、精确crop/值域/原图不变、无需填充、统计、全域门禁与坏图、Smoke/Train一致性。
- [x] 仅修改Dataset预处理和入口门禁/有效配置；原模型、损失、优化器、日程、checkpoint字段结构不改。实际策略进入原config.preprocess和data记录，旧配置不能静默精确续训。
- [x] 保留原27项测试的方法及门禁断言；原短图失败fixture改为未获批准的352×223，新增批准尺寸成功断言。追加D-11与独立测试/diff记录，交付后停止。

**数据与评价边界**：本次Stage1协议使用全部5380张UVTD Raw；预期483+21=504张填充、4876张不填充，属于用户提供的数据统计，需由服务器新版data-check复核。后续在这些5380张上的评价不能称为完全独立的未见测试。

### 实际预处理和data-check变化

`data/stage1_dataset.py:39–54`增加共享`prepare_image`：解码后转RGB，仅对水下域批准尺寸在内存调用`np.pad(...,mode='reflect')`，返回图像及元数据；不重写像素内部内容、不保存文件。`read_crop:86–96`在prepare后使用原`random.randint`坐标、原PIL裁剪与ToTensor/Normalize等价运算。RGB转换与反射操作可交换，此实现先转RGB便于共享CHW契约。

| UVTD原始宽×高 | 左/上/右/下填充 | 裁剪前宽×高 | 用户提供的预期张数 |
|---|---|---|---:|
| 352×224 | 0/16/0/16，reflect | 352×256 | 483 |
| 224×352 | 16/0/16/0，reflect | 256×352 | 21 |
| 其他原本合法尺寸 | 0/0/0/0 | 原尺寸 | 4876 |

旧行为：上述504张作为`too_small`阻止运行。新行为：它们不再进入`too_small`，而是计入`padding_count`和`requires_padding`；其余合格图计入`no_padding_count`。每条可处理图像记录原`width/height/mode`以及`original_size`、`padded_size`、`padding.{left,top,right,bottom}`、`strategy`和`policy`。水下策略标识`uvtd-reflect16-352x224-224x352-v1`；空气域保持`crop256-only`。不可读项仍列入errors并阻断；未批准短尺寸仍列too_small并阻断，不自动推导新填充量。

data-check/train完整Raw/HR根检查、Smoke人工清单范围/每域最少2张/全部清单项真实裁剪检查均保留。两模式Dataset对Xu使用同一反射处理，对Xa保持原处理。训练清单仍显式传入；本次正式协议必须包含全部5380张Raw，不能用过滤清单冒充全量训练。全域统计和训练清单元信息分别存data_check.json/training_data_check.json；有效配置明确记新预处理。checkpoint顶层字段、保存恢复代码和网络state_dict键不变，但预处理数值协议已改变，旧配置精确恢复会被原有严格比较拒绝。

### 本轮修改文件和测试结果

本轮共7个文件：修改`data/stage1_dataset.py`、`train_stage1.py`；新增`tests/test_stage1_reflect.py`；对`tests/test_stage1.py`和`tests/test_stage1_smoke_gate.py`仅调整与已批准规则冲突的非法尺寸fixture及read_crop参数转发；更新本文和`docs/decision_log.md`的D-11。原27个测试方法全部保留，未删除或放松原有断言。stage1.py、models、options、原train.py/test.py无源码变更。

**实验事实**：Run `stage1-reflect-20261110-01`，2026-11-10 21:19:32–21:19:40（Asia/Shanghai；日志UTC），unittest运行6.557秒。Python3.8.20/torch2.0.1+cu118/torchvision0.15.2+cu118/NumPy1.24.4/Pillow9.4.0；模型测试仅CPU，子进程显式CUDA_VISIBLE_DEVICES为空。命令：

```text
D:/Miniconda3/envs/hsruie/python.exe -B -m unittest discover -s tests -p "test_stage1*.py" -v
```

**38通过、0失败、0跳过：原27项回归＋新增11项。** 新增覆盖：两方向精确尺寸/16px量；独立手算边界排除边缘值（区分reflect与replicate/zero）；独立拼接oracle核对裁剪坐标/逐元素值/float32/[-1,1]/原文件字节；合法原图不填充；合成3张统计2填充/1不填充且零过滤；全域检查通过；未列入清单的坏图仍阻止train；Smoke和Train报告/实际Dataset处理一致；不将策略延伸到Flickr2K或未授权尺寸。

先运行新增测试，因新处理API不存在及旧门禁阻止而失败，保存[RED日志](../runs/stage1-reflect-20261110-01/red_output.txt)；再最小实现、运行新增与完整回归。独立只读审查未发现Critical/Important/Minor问题。完整证据：[全部38项测试结果](../runs/stage1-reflect-20261110-01/test_output.txt)、[命令/环境/源码快照/种子记录](../runs/stage1-reflect-20261110-01/record.json)、[完整Diff](../runs/stage1-reflect-20261110-01/changes.patch)。本地只验证合成样本计数，没有生成5380份伪资产或把预期数写成实测数。原27项CPU模型/恢复fixture仍为历史已说明的合成测试，不是正式训练。

### 下一步服务器命令（仅供审查，未执行）

首先只运行CPU完整data-check，复核用户提供的真实计数；失败时保留报告，不过滤图片。下述命令不生成训练/Smoke清单，不运行GPU或训练：

```bash
cd /home/czy/projects/HSRUIE-reproduction
RUN_ID="reflect-data-check-$(date +%Y%m%dT%H%M%S)-$$"
OUTPUT_ROOT=/data/czy/experiments/HSRUIE/stage1
python -B train_stage1.py --mode data-check \
  --underwater-root /data/czy/datasets/UVTD_import/UVTD/Raw \
  --inair-root /data/czy/datasets/Flickr2K \
  --output-root "$OUTPUT_ROOT" --run-id "$RUN_ID"

python - "$OUTPUT_ROOT/$RUN_ID/data_check.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    report = json.load(stream)
u = report['underwater']
print({k: u[k] for k in ('count', 'readable', 'padding_count', 'no_padding_count', 'skipped')})
assert (u['count'], u['readable'], u['padding_count'], u['no_padding_count']) == (5380, 5380, 504, 4876)
assert not u['errors'] and not u['too_small'] and u['skipped'] == 0
PY
```

审查通过后可用文后原Smoke命令、全新run-id和人工选择包含两种短图的水下清单再次做真实组件诊断；其结果属于新反射协议的Smoke，不能用用户报告的旧Smoke日志替代。本轮没有执行该GPU步骤或正式训练，未commit/push/reset。后面的D-09/D-10实现与测试段落是历史证据；其中“所有短图拒绝/无padding/服务器Smoke未执行”的状态由本节和D-11更新，不能覆盖本节最新授权与证据。

更新日期：2026-11-10；分支 repro/stage1；Smoke门禁修复基线55aad4a3e66abb5620669618ea3aa9f8c2bed8a2。以下11月9日实现/测试内容保留为历史证据，最新门禁与测试记录见文末。

## 当前方案（先更新方案，再实施并测试）

本节替代上一版协议 A：取消自动放大、padding、任意 batch 恢复、多卡实现及独立 options/save 框架。只新增独立入口、双域 Dataset 和必要训练函数，复用原样 SAD 与 VGG forward。**最小实现及本地17项合成契约测试已完成；服务器真实数据/预训练VGG/SAD的256×256单Batch验收未执行，正式训练未执行。**

**论文事实**：p.45 Fig.2、III.B 式(1)–(3)，p.46 式(4)–(6)：固定 E，共享可训练 G；L=Lsr+Lrec_a+0.1Lrec_u，Lsr=relu(1-sa+su)，式(5)给出 cosine 示例。p.47 IV.A：随机256裁剪、Adam、1e-4、batch2、50+50。用户本次明确后50轮线性衰减，最后一轮仍需更新。

**作者代码事实（本次公开仓库复查）**：main=397da5e492b309f5c1af4f64df73c5a647e038cd。SAD 在 cycle_gan_model.py:70–71 构造后加载权重，在 :121 把增强特征转成最终 RGB；Mynet 在 DenseNet.py:15,30 使用同一 SAD；公开仓库存在 latest_net_SAD.pth。结合 Fig.2 的生成角色，这比接口相似提供更多对应证据，但没有直接命名声明或完整 Stage1 训练代码，仍不能认证 SAD 与论文 G 逐层相同或作者 checkpoint 来自 Stage1。公开 DenseNet.py 历史只定位到一次上传，README未补充结构/初始化协议；issue3询问完整训练流程，当前无回复。实现按用户要求复用发布 SAD，明确保留未认证边界。

**作者代码事实**：活动 E 是作者公开版本 models/vggloss.py:13–47，vgg19(pretrained=False) 加载完整本地 state_dict，再切片0:4、4:9、9:14，得到 relu1_2/relu2_2/relu3_2；输入[B,3,256,256]，输出[B,64,256,256]/[B,128,128,128]/[B,256,64,64]。输入由 data/base_dataset.py:106–112 映射[-1,1]，E没有额外ImageNet标准化。仅为显式路径/CPU添加构造参数，不改切片/forward/冻结或旧默认调用。当前本地构造适配在 models/vggloss.py:14–25、切片在 :34–43、forward在 :48–55；不要用作者版本旧行号定位改动后文件。

**作者代码事实/未确认**：活动 SAD() 后权重覆盖；networks.define_G('SAD')则走 init_weights(normal,0.02)。没有证据证明哪条用于作者 Stage1 新训，故新入口要求显式选择 constructor 或 author-normal，并记录为初始化假设，绝不自动加载作者 SAD 当作从头训练。

**作者代码事实**：Adam 调用只传 lr 和 betas=(opt.beta1,0.999)，beta1默认0.5；其余沿用已安装torch默认并落盘，不新增优化超参数。原DataLoader batch_size=2，每个样本理论含双域，但实际unaligned只读A且B_size未定义；因此无法证明 Stage1 双域batch含义，也没有Flickr2K读取分支。

**必要工程假设**：新Dataset一个样本包含独立Xu/Xa，batch=2即每域2张；epoch=max(Nu,Na)，水下索引循环、空气随机有放回（沿用未配对模板的简单采样思路），尾批保留。两域各自合法随机裁剪，无翻转/resize/其他增强。损失使用L1 mean、每图CHW cosine（torch默认eps）、逐样本hinge再mean，均在[-1,1]；不把这些归约称为原文明示。

**数据门禁（2026-11-10更新）**：data-check/train始终检查完整Raw/HR域，即使提供manifest也不能绕过。Smoke必填两份人工清单，每域至少2张，仅对清单内全部图片执行路径、解码、尺寸与实际RGB256crop检查；域外小图/坏图不影响Smoke。检查范围内任一问题仍阻止模型构造；不筛除样本、不越界crop、不生成副本。train全域通过后另查实际训练清单，保留原config.data内容和checkpoint字段。禁止把检测split自动当增强split。

**日程**：训练epoch1–50为1e-4，51为1e-4，52为9.8e-5，100为2e-6；epoch100完成后scheduler降到0。每次在该epoch全部optimizer更新完成后step。最后50个非零epoch在边界线性降到0；重复平台端点属于离散边界解释，不是新增论文事实。

**Checkpoint**：仅G、optimizer、scheduler、已完成epoch、global_step、Python/NumPy/torch CPU/CUDA RNG、关键配置。每epoch新文件，恢复至下一个epoch；不保存E（从指定哈希VGG重建），不存batch游标/预取状态。恢复写新run，不覆盖作者和父run；smoke检查包仅供验收，不允许作为正式epoch恢复。

**修改范围及验证**：新增 data/stage1_dataset.py、stage1.py、train_stage1.py、tests/test_stage1.py；仅修改vggloss构造适配。原train.py/test.py/DenseNet/FEM及其他阶段不改；SAD裸键结构保持，完整训练包须取G字段加载；不认证后续指标可比性。先合成数据/微型网络验证读取、shape/值域、loss方向、冻结/梯度/更新、epoch恢复与LR；真实E/SAD单batch由独立smoke入口在服务器后续验收。不执行正式训练。

## 服务器路径与证据

用户最新说明路径已经在其他聊天核实，不再要求本agent具体核实。项目cwd采用 /home/czy/projects/HSRUIE-reproduction；UVTD=/data/czy/datasets/UVTD_import/UVTD/Raw；Flickr2K=/data/czy/datasets/Flickr2K。VGG路径必填，不从cwd猜测；使用用户已核实的完整路径。此前只读SSH尝试因server161无法解析失败，没有进入服务器、没有模型操作。

公开证据：[作者模型](https://github.com/gemyumeng/HSRUIE/blob/397da5e492b309f5c1af4f64df73c5a647e038cd/models/cycle_gan_model.py)、[SAD](https://github.com/gemyumeng/HSRUIE/blob/397da5e492b309f5c1af4f64df73c5a647e038cd/models/DenseNet.py)、[训练缺口issue](https://github.com/gemyumeng/HSRUIE/issues/3)。issue提问不作为作者确认；原论文p.45–47在上一轮直接读取并渲染核对，本次不冒充新论文实验。

## 实际修改清单

| 文件 | 实际内容 |
|---|---|
| 新增 `data/stage1_dataset.py` | 明确两域根/可选清单，目录仅枚举当前层图像，避免自动混入HR/LR子目录；全量data-check记录；RGB、合法256crop、原值域；不筛除小图 |
| 新增 `stage1.py` | 三项损失、冻结E/共享G、复用SAD初始化、原Adam参数、epoch线性调度、必要checkpoint/RNG恢复；不依赖原训练链 |
| 新增 `train_stage1.py` | 必填mode、data-check/smoke/train；新run排他创建；data-check/train全域门禁、Smoke人工清单门禁；模型模式必填VGG路径和初始化选择；正式训练必填两域清单；逐epoch保存/恢复；运行来源记录 |
| 修改 `models/vggloss.py` | 仅可选路径/设备构造适配；旧零参数调用保持原加载规则；新路径CPU map_location后to(device)，strict加载；forward原样 |
| 新增 `tests/test_stage1.py` | 17项CPU合成契约验证，包括真实SAD类、小尺寸合成特征和真实VGG19 feature架构的合成权重fixture |
| 新增/更新本文 | 最小方案、证据、假设、实际结果、服务器命令与剩余问题 |
| 更新5份证据文档 | `reproduction_status.md`、`model_architecture.md`、`known_issues.md`、`experiment_plan.md`、`decision_log.md` 的Stage1状态/旧行号说明；历史证据保留，追加D-09 |

**源码事实**：已用git diff确认原train.py/test.py、DenseNet.py/NEDB_IN.py/FEM.py/cycle_gan_model.py、原data/base_dataset.py/unaligned_dataset.py及options均无变更。Git现有仓库的vggloss.py采用CRLF，使用 `git -c core.whitespace=cr-at-eol diff --check` 核查通过，不为格式检查重写整份作者文件。未执行commit/push/reset；没有覆盖原权重或实验目录。

入口不加载作者SAD checkpoint作为“新训”默认。完整训练包只有G、optimizer、scheduler、epoch、global_step、rng、config；裸权重可取包内`G`字段，键名不改。模型模式保存run.json、data_check.json、effective_config.json，训练另写metrics.jsonl、每epoch唯一checkpoint、end.json或failure.json。run.json包含源码快照（含新增源码）、Git差异、命令、环境；effective_config包含VGG路径/SHA-256、初始G指纹、实际Adam默认参数与Forward说明。短边门禁失败返回非零且保留报告，不产生模型权重。

## 本地测试报告（实验事实，仅限合成契约）

- Run ID：`stage1-contract-20261109-01`。
- 时间：2026-11-09 17:19:54–17:20:00（Asia/Shanghai；日志保存UTC），unittest运行5.045秒。
- cwd：`D:/Research/HSRUIE`。Python：`D:/Miniconda3/envs/hsruie/python.exe` 3.8.20；torch2.0.1+cu118、torchvision0.15.2+cu118、NumPy1.24.4、Pillow9.4.0。模型检查仅CPU，torch线程数2。
- 命令：`D:/Miniconda3/envs/hsruie/python.exe -B -m unittest discover -s tests -p test_stage1.py -v`。
- 结果：**17通过、0失败、0跳过**。初始化、采样、图像生成种子及确定性设置见测试源码和记录；不宣称跨设备逐位一致。
- 独立记录：[record.json](../runs/stage1-contract-20261109-01/record.json)；完整输出：[test_output.txt](../runs/stage1-contract-20261109-01/test_output.txt)。该目录被现有.gitignore排除但实际保留，未覆盖旧run。

| 覆盖 | 结果和范围 |
|---|---|
| Dataset/预处理/Batch | BMP/JPEG/PNG、灰度转RGB；两域batch `[2,3,256,256]` float32/finite/[-1,1]；精确像素核对crop无resize/flip；小图与坏图完整报告且未过滤；manifest去重/越界/空集；不自动递归HR/LR |
| 三项loss | L1手算、权重1/0.1、hinge端点0/2、sa导数-1/su导数+1；逐图hinge与batch均值的区别；零范数finite；错误shape/dtype/范围和NaN拒绝 |
| E冻结/共享G/更新 | 微型E/G两域生成图均有有限非零梯度；optimizer仅G；E参数与状态不变；G实际更新 |
| 原样SAD | 输入为B2、64/128/256通道、32/16/8空间的合成特征，输出B2×3×32×32；实际反向与Adam更新通过；只有预期trans_down_1/2无梯度；strict加载后输出一致。不是真实256图像训练验收 |
| 原样VGG feature架构 | torchvision VGG19实际卷积/pool结构，合成固定权重fixture，省略无关classifier；256 batch核对三个尺度；旧构造分支和新增显式路径分支输出逐元素相等、参数冻结。不是作者预训练权重加载验收 |
| 初始化 | 同种子下新增入口的constructor和author-normal分别与直接SAD()/原init_weights完全一致；不认证作者实际Stage1采用其中哪条 |
| Checkpoint/随机恢复 | 拒绝覆盖、配置失配拒绝、epoch/global_step恢复、Python/NumPy/torch RNG往返；同固定下一批后G与Adam状态相等；另以临时3张/域的微型2epoch集成测试，模拟epoch边界中断，再恢复下一轮shuffle/crop，最终G逐位相等、global_step=4（含尾批）。没有正式训练 |
| LR | 纯调度100次标量更新，前50轮固定、最后一轮仍更新、结束LR=0，恢复位置匹配 |
| CLI/smoke | 小图使整个data-check非零退出且无checkpoint，重用run目录拒绝且旧报告不变；smoke验收函数以合成shape fixture走完整前向/反向/更新/保存恢复；--help正常且不创建run |

上表中真实SAD与VGG feature分别测试，不能合并声称真实预训练E→SAD的256单Batch已经验收。没有读取本地UVTD/Flickr2K，没有加载作者权重，所有图像/训练包来自临时合成fixture；临时fixture由测试框架清理，测试记录保留。此前14项/16项检查通过，补充初始化与入口恢复覆盖后最终为17项；没有删除断言或放宽标准。

## Linux 验收命令（已准备，未执行）

下述项目目录及两域路径按用户确认使用。VGG采用必填环境变量，填入用户已核实的完整权重路径；不根据新cwd拼接或猜测。`--g-init constructor` 在示例中是显式工程假设，可改为 `author-normal` 对应原工厂初始化，不能称已找到作者Stage1新训流程。

```bash
cd /home/czy/projects/HSRUIE-reproduction

# 不加载模型；报告整个当前目录的全部小图/读取失败项。
python -B train_stage1.py --mode data-check \
  --underwater-root /data/czy/datasets/UVTD_import/UVTD/Raw \
  --inair-root /data/czy/datasets/Flickr2K \
  --output-root /data/czy/experiments/HSRUIE/stage1 \
  --run-id "data-check-$(date +%Y%m%dT%H%M%S)-$$"

# VGG_WEIGHTS必须由用户已核实的完整路径赋值，不提供猜测默认。
# SMOKE_U/SMOKE_A指向人工指定的两域诊断清单，各至少2张，路径相对各自root。
# 仅诊断清单不建立正式训练划分；不从data-check自动过滤小图生成该清单。
: "${VGG_WEIGHTS:?set verified VGG weight path}"
: "${SMOKE_U:?set explicit underwater diagnostic manifest}"
: "${SMOKE_A:?set explicit in-air HR diagnostic manifest}"
python -B train_stage1.py --mode smoke \
  --underwater-root /data/czy/datasets/UVTD_import/UVTD/Raw \
  --inair-root /data/czy/datasets/Flickr2K \
  --underwater-manifest "$SMOKE_U" --inair-manifest "$SMOKE_A" \
  --vgg-weights "$VGG_WEIGHTS" --g-init constructor \
  --device cuda:0 --batch-size 2 --num-workers 0 \
  --output-root /data/czy/experiments/HSRUIE/stage1 \
  --run-id "smoke-$(date +%Y%m%dT%H%M%S)-$$"
```

清单一行一个root相对图像路径，不接受域外路径或重复项。Smoke必须提供两份清单，各至少2张；其data_check.json只列清单图片。data-check/train全域检查沿用当前层Raw/HR图像枚举，不受manifest缩小；train另写training_data_check.json验证实际训练清单，checkpoint config.data仍记录训练选择。若Flickr2K实际图像位于HR子目录，应由操作者明确把root设为已核实HR目录，入口不自动选择或递归混入LR。全域data-check若发现小图，仍表示训练数据未就绪；小范围smoke成功不能消除该阻塞。

smoke固定只取一个每域2张的256 batch，检查真实VGG三个尺度、原SAD输出/值域、E全部冻结且不变、两域输出梯度、SAD活动/未调用参数、有限更新、完整包和strict G恢复输出；保存`smoke_checkpoint.pth`及结束报告。该包标为mode=smoke/epoch=0，不允许作为正式train续训。正式训练仅接受mode=train的已完成epoch包，恢复用`--resume <epoch_XXXX.pth>`及全新run-id，关键配置/清单/VGG指纹必须相同；不提供或执行正式100epoch命令。

## 剩余问题与可比性

1. **论文/作者来源未认证**：SAD承担G角色的调用证据已核实，但直接命名声明、逐层MS-UDAG对应、作者Stage1初始化与公开SAD checkpoint阶段来源仍未确认；没有用测试替代这些证据。
2. **工程假设仍明确存在**：每域batch2、epoch/短域采样、L1/cosine/hinge归约、离散LR边界、种子和epoch恢复协议。输入值域沿用发布代码，不是论文原文认证；无额外标准化/增强。
3. **短边与正式split待决定**：不自动放大/pad/丢弃；真实数据全域盘点需在服务器执行。正式训练清单和后续未见测试隔离不能从检测split或文件总数推断，SUIM TEST不参与。
4. **服务器验收尚未执行**：真实预训练VGG权重严格加载/哈希、真实HR文件范围、真实E→SAD的256前向/反向/更新/恢复及3090资源开销仍待记录；本地CUDA未用于模型测试，无多卡实现。
5. **结果边界**：实现可测试不等于论文忠实复现。新训练G不自动与既有NC/HIGH权重组合或论文指标可比。只有真实服务器各项单Batch验收完成并审核后，才能考虑另行授权正式训练。

## 2026-11-10 Smoke 数据门禁修复计划

**用户授权的工程适配**：Smoke 必填两份人工 manifest，每域至少两张，只检查清单中真实存在、可解码、尺寸足够且可合法256裁剪的全部图像。data-check/train 的门禁覆盖完整域，不得通过 manifest 绕过；不改图像、模型、损失、优化器、日程或 checkpoint 格式。

**源码事实（修复前，HEAD 55aad4a）**：`train_stage1.py:171–172` 直接按可选 manifest 选择检查范围，因此 Smoke 已支持部分范围，但未要求清单/最少数量；train 和 data-check 提供清单时也会缩小门禁范围。修复只改入口中检查范围的选择，不重构 Dataset。

实施顺序：

- [x] 新增 `tests/test_stage1_smoke_gate.py`：先验证缺清单/不足两张、全域门禁绕过等缺口；覆盖合法Smoke、清单小图/坏图/路径错误以及data-check/train全域失败。
- [x] 最小修改 `train_stage1.py`：Smoke清单门禁；data-check/train全域门禁；train实际训练清单单独验证并保留原config.data结构。
- [x] 运行原17项及新增全部CPU合成测试，保存独立日志、运行来源、diff；补录结果与D-09边界补充。禁止真实GPU/正式训练、commit/push/reset。

### 本次实际变更与验证（2026-11-10）

**实际修改文件**：`train_stage1.py`、`tests/test_stage1_smoke_gate.py`（新增）、本文、`docs/decision_log.md`（D-10）。生产代码仅入口增加26行、替换3行；没有修改原17项测试。`data/stage1_dataset.py`、`stage1.py`、全部models/options、原train.py/test.py均无差异；checkpoint字段和config.data的实际训练清单含义保持原状。Train增加的`training_data_check.json`是运行报告，不是checkpoint格式变更。

**本地实验事实**：Run `stage1-smoke-gate-20261110-01-final`，2026-11-10 17:23:48–17:23:57（Asia/Shanghai，原记录UTC）；Python3.8.20、torch2.0.1+cu118、torchvision0.15.2+cu118、NumPy1.24.4、Pillow9.4.0；仅CPU，测试子进程显式`CUDA_VISIBLE_DEVICES=''`。命令：

```text
D:/Miniconda3/envs/hsruie/python.exe -B -m unittest discover -s tests -p "test_stage1*.py" -v
```

**结果：27通过、0失败、0跳过；原17项全部保留并通过，新增10项门禁测试。** unittest耗时6.835秒。新增测试在真实临时图像/清单上执行入口；只有模型构造处用异常哨兵停止，不运行真实VGG权重或GPU。原17项的SAD/VGG feature检查范围仍为上文所述的CPU合成fixture，不能写成服务器验收。

| 新增测试覆盖 | 最终结果 |
|---|---|
| 全域data-check（无manifest/提供合法manifest）遇小图 | 两种调用都失败，报告包含域内小图 |
| 合法Smoke清单、清单外有小图和坏图 | 清单内两图/域数据检查通过并到达模型构造边界，报告不混入清单外图片；原图字节不变 |
| Smoke缺任一清单/两份都缺 | 显式拒绝，在run目录创建前失败 |
| 任一域清单只有1张 | 拒绝；不通过重复同图满足数量 |
| 任一域清单包含第3张小图 | 失败并报告，不仅检查将抽到的batch |
| 任一域清单包含第3张不可读图 | 失败并报告 |
| 清单文件不存在的图、越域路径、重复路径 | 全部拒绝 |
| Train提供合法清单，但全域有小图 | 全域门禁仍失败，不构造模型 |
| Smoke每份清单3张（含灰度图）的实际crop调用 | 两域全部6条路径都经原read_crop，输出3×256×256 |
| Train全域通过但清单选中不可读嵌套图 | 第二层清单复核失败，保留全域/清单两份报告 |

**RED→GREEN证据**：首次针对8项测试的修复前运行有7个失败子项（预期暴露缺清单/数量与全域绕过），详见[red_output.txt](../runs/stage1-smoke-gate-20261110-01/red_output.txt)。随后25项完整回归通过。独立审查确认无实现层Critical/Important问题，提示两个检查分支的覆盖仍不足；将其按安全门禁覆盖缺口处理，补2项测试，并临时移除对应检查，观察2项均失败后原样恢复，详见[review_red_output.txt](../runs/stage1-smoke-gate-20261110-01/review_red_output.txt)。最终完整27项通过；临时探针均已移除，无待处理审查项。

**完整交付证据**：[全部测试输出](../runs/stage1-smoke-gate-20261110-01/final_test_output.txt)、[环境/命令/源码快照/种子与指纹](../runs/stage1-smoke-gate-20261110-01/final_record.json)、[完整diff（含新增测试）](../runs/stage1-smoke-gate-20261110-01/changes.patch)。记录目录独立、不覆盖11月9日实验。记录采集曾遇PowerShell环境默认GBK读取中文Git diff失败，在执行最终测试前改为显式UTF-8；不是模型/测试失败。

**本次边界**：不修改图片，不自动生成/过滤任何Smoke清单，不改变实际VGG/SAD、loss、Adam、LR或checkpoint。服务器命令沿用上文Linux Smoke示例，强制两份人工清单各≥2张。504张短边不足是用户提供的服务器全域data-check结果，本次未复查服务器；Smoke数据检查通过不代表这504张图的训练处理方案已解决。未启动真实GPU验收、正式训练、commit、push或reset。
