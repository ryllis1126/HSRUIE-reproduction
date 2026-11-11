# Stage 2 独立训练实现

对应论文 p.45 Fig.2、p.46 III.C 式 (7)–(14)、p.47 IV.A。本实现采用用户确认的结构映射和明确列出的工程假设，不能称为作者完整训练代码的逐行复原。审计原文与固定作者 commit 证据见 [stage2_audit.md](stage2_audit.md)。

## 模型、损失和冻结

`stage2.py:118` 的 Stage2 复用原始 Vgg19_out 和 SAD。C、D 分别独立实例化原始 FE：三个分支通道64/128/256，每分支6个 ResBlock，每个提取器共18块、9,300,480参数；总计18,600,960个可训练参数。C 的源码依据是作者 netNC=FE；D 镜像 FE 是用户批准的工程重建，公开代码没有完整 D 训练路径。保留原始卷积、GN32 和残差 ReLU；不接入 NEDB/FEM/DEM/SEM/LOW/HIGH/U-Net/任务网络。

冻结 E、G*、CLIP 的所有参数，强制 eval，只有 C/D 在 optimizer 内：

```text
Fu = E(Xu)                          # no_grad 原图目标
Fc = C(Fu), Fd = D(Fu)
Xc = G*(Fc)                         # 建图
CLIPu = RN50_234(preprocess(Xu))     # no_grad 原图目标
CLIPc = RN50_234(preprocess(Xc))     # 建图
Fidt = E(Xc)                        # 建图
Xidt = G*(Fidt)                     # 建图

sep  = sum_l mean(abs(Fd_l + Fc_l - Fu_l))
clip = sum_i mean(abs(CLIPu_i - CLIPc_i))
idt1 = sum_l mean(abs(Fidt_l - Fc_l))
idt2 = mean(abs(Xidt - Xc))
total = sep + clip + 2*idt1 + idt2
```

l=1,2,3，i=2,3,4。四项和 total 分别记录。L1 每个 Tensor 元素平均后跨尺度求和是工程假设。相减前严格检查尺度数、完整 shape、batch、设备、FP32 和有限性，不允许广播。没有额外 loss，不 detach 生成图像或恒等损失任意一侧；式 (12) 不再经过 C。D 只由 sep 获得梯度。

VGG 输出为 relu1_2/2_2/3_2；256 crop 对应 `[B,64,256,256]`、`[B,128,128,128]`、`[B,256,64,64]`。VGG/SAD 使用原始 [-1,1]，不增加 ImageNet 标准化。

## RN50 权重与可微输入

`stage2.py:36,64` 使用作者 `CLIP/clip/model.py` 的 ModifiedResNet RN50，只构造和加载 visual 部分，完整保留 attention pooling。通过不 detach 的 hook 获取 visual.layer2/3/4 的空间输出，不拿最终 embedding 替代 layer4；B=2 时为 `[2,512,28,28]`、`[2,1024,14,14]`、`[2,2048,7,7]`。仍执行原始 visual.forward，未为提速删除原始 forward 分支。

显式本地 TorchScript RN50 归档的 SHA-256 必须是作者 loader URL 对应的 `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`；先校验再加载，不自动下载。不使用作者 build_model 的 half 转换，不加载无关 text encoder 到训练模型；归档已有 half 存储值直接复制到 FP32 visual 参数。该精度执行方式来自本次禁止 FP16 的约束。

用户确认的工程预处理为：[-1,1]→[0,1]→可微 bicubic224（align_corners=False、antialias=True）→CLIP mean `(0.48145466,0.4578275,0.40821073)` / std `(0.26862954,0.26130258,0.27577711)`。Xu 和 Xc 同一流程；Xc 全程留在 Tensor/autograd 中。官方 PIL 路径含 uint8 量化/截断，Tensor 双三次可能略超出 [0,1]；没有新增 clamp。两者的本地数值对照详见审计与 `clip_cpu_probe.json`，不宣称逐像素一致或作者完整训练预处理已确认。

## 数据和门禁

`data/stage2_dataset.py` 只读 UVTD Raw，固定 manifest，每图每 epoch 一次，shuffle=True，保留尾 batch，非持久 worker。worker 使用 Stage1 既有 seed_worker；RGB、float32、[-1,1]，无 flip/颜色增强/resize。

用户已单独批准 Stage2 沿用 D-11：352×224 上下各 reflect16，224×352 左右各 reflect16，其余合法图不填充，再统一随机 crop256。复用 Stage1 prepare_image/read_crop，不改原实现或磁盘原图；其他不支持的短图和坏图报错，不过滤。历史 UVTD 盘点5380张、504填充/4876不填充是用户/历史记录，本次没有在服务器复核。

- `data-check`：全域检查，即使提供 manifest 也不能隐藏根目录内坏图。不构建模型、不需要权重。
- `smoke`：必须显式 manifest，至少2个不同成员，检查全部清单图片的路径/解码/尺寸/实际 crop；不自动从全域报告生成或过滤清单。
- `train`：完整域门禁，固定 manifest 必须覆盖完整域，不允许通过子集绕过坏图；再核对 Stage1 G 来源 config 的 underwater records 与该清单一致。目录扫描沿用 Stage1 的非递归规则；用户必须提供实际图像叶目录，manifest 可以显式记录域内嵌套文件。

清单与每个文件的 SHA-256、原始尺寸、padding 四边量、策略、成功/失败/跳过数量及清单指纹保存到独立 run。使用全部5380图进行训练后，在这些图上的评价不能称为完全独立的未见测试。没有使用 Reference 或任何任务标签。

## 训练配置与恢复

正式 train 强制 batch2、Adam lr1e-4、100固定+100衰减。Adam 显式 betas=(0.5,0.999)，其他参数取 PyTorch2.0.1 默认并记入配置；beta 来自作者旧入口而非完整 C/D 训练代码，是工程假设。FE 从固定种子下独立的原始构造器初始化，不自行调用其他 normal 初始化，也不加载作者 NC 来冒充从零预训练。

复用 Stage1 make_scheduler，但不修改它；epoch内先更新参数，整轮结束后才 step。离散边界工程约定：epoch1–100=1e-4、epoch101=1e-4、epoch102=9.9e-5、epoch200=1e-6，完成200后归零。最后一轮有参数更新。

固定随机种子默认20261111；每个 run 记录 Python/torch/torchvision/NumPy/Pillow/CUDA、设备和后端设置。单 GPU、FP32，不使用 AMP/FP16/BF16/TF32/compile/DDP；显式关闭 TF32 和 cudnn.benchmark，记录现有 deterministic 设置，不承诺跨硬件或软件版本逐位一致。

Stage1 G 接口（`stage2.py:181,203`）：必须是原 Stage1 训练包，取 `['G']` 严格加载原 SAD，验证有限性和 VGG SHA。诊断 G 需 `--diagnostic-stage1`，只准 smoke。正式 G 要求来源 mode=train、epoch100、50+50、非零global_step，以及显式 hash-bound Stage1 验收 JSON；不能凭目录名或 checkpoint 文件名自动认证。

验收 JSON 的字段合同是 `stage="stage1"`、`accepted=true`、`completed_epoch=100`、`checkpoint_sha256`、`vgg_sha256`、非空 `basis`（实际验收依据）。该记录必须在真实 Stage1 完成并验收后由用户/审查流程提供，本次没有生成或伪造任何 accepted=true 的真实实验记录。

Stage2 checkpoint 为独立格式：stage、code_version、mode、C/D、optimizer、scheduler、completed_epoch、global_step、Python/NumPy/Torch/CUDA RNG、effective_config。配置含Git commit、实际源码哈希、G/VGG/CLIP路径及哈希、Stage1元数据/验收、manifest指纹、初始C/D参数指纹、软件和训练协议。E/G/CLIP权重通过外部固定来源重载，不重复写入每个 Stage2包。只支持 epoch 边界恢复；拒绝 Stage1 包、smoke 包、配置/权重/代码指纹变化。新 run 记录 resume 原文件路径与 SHA-256，严格比较配置时不将新输出路径纳入。

所有 JSON/日志/checkpoint 都在全新 run 目录下独占创建。没有复用作者 TestOptions，不会触碰 checkpoints/<name>/test_opt.txt；失败写 failure.json，已有文件不覆盖。训练中途未完成 epoch 不生成可续训包，使用上个完整 epoch 重跑。

## Smoke 实际验证内容

`train_stage2.py:104` 只取一个 batch：检查真实 E/G/CLIP 的预期 shape、生成路径的有限非零梯度、C/D均有更新、冻结参数/buffers逐项不变。保存独立 smoke 包后，清除 optimizer 状态、扰动 C/D、LR 和 scheduler，再严格恢复，比较 C/D、optimizer/scheduler 及 G(C(E(Xu))) 输出。smoke 包 completed_epoch=0/global_step=1，明确不是完成了一轮训练，不能用于 train 恢复。

本地 CLI 合成 Smoke 使用匹配这些 shape 合同的微型模块替代 builder；这验证入口/门禁/恢复流程，不能冒充真实权重组合验收。

## 性能处理及正确性范围

| 处理 | 为什么可保持数学行为 | 验证及限制 |
|---|---|---|
| 同 batch 的 E(Xu)、CLIP(Xu) 各算一次、仅原图目标 no_grad | 它们是固定目标；生成路径不关闭图 | 对照保留图的参考流程，四项/total和C/D梯度逐位相同 |
| 多尺度 finite 与梯度标志先在设备聚合 | 不删检查，只减少逐Tensor主机判定；C/D仍分别要求非零 | 非有限/缺失梯度检查保留；损失数值、更新和恢复测试通过 |
| 五个日志 scalar detach后stack，一次cpu().tolist | 只影响日志传输，不参与反传/optimizer | 数值与梯度参考对照；没有每项单独GPU标量读取 |

没有跨随机 crop 缓存特征、改数据顺序、增加 worker/prefetch/persistent_workers、修改网络/损失/精度/分辨率来提速。未执行 GPU benchmark，不能量化节约时间或宣称实际吞吐提升。保留原始 visual.forward，包括未用于损失的 attention pooling。

只加载 visual 是独立训练入口的权重接入选择，FP32是用户要求，不计作已经证明训练结果不变的提速项目。虽然 visual 数值和梯度已对照原始模块，完整 CLIP 与 visual-only 构造器消耗的初始化 RNG 可能不同；本实现固定自身构造顺序、种子和初始 C/D 指纹，不宣称与另一个完整加载器的随机初始化逐位等价。

## 测试与审查

CPU 合成测试命令：

```powershell
$env:CUDA_VISIBLE_DEVICES=''
D:/Miniconda3/envs/hsruie/python.exe -B -m unittest discover -s tests -p 'test_stage*.py' -v
```

新增 Stage2 当前35项：原始 FE 分支/参数/独立性；四项手算/权重/方向/shape/NaN/Inf；逐损失梯度、冻结参数与 BN buffers、optimizer归属及更新；真实 SAD 的训练包 strict 接口；CLIP校验和/空间hook/可微预处理；epoch恢复/RNG/optimizer/LR；真实图像和清单门禁/失败/不覆盖/诊断权重隔离。另有完整shuffle+随机crop的连续/恢复epoch对照，覆盖 num_workers=0和1，比较路径、像素、loss、参数和Adam/scheduler状态。

CLIP 非常量256→224输出及输入梯度与 torchvision Tensor bicubic/antialias参考逐位相同。idt1专门比较包含E(Xc)导数的完整参考梯度；故意detach该分支的内存mutation被断言捕获，避免只凭C仍有梯度就误判通过。

阶段测试记录：`runs/stage2-build-20261111-01/stage2-green.txt` 为35项通过、0失败/跳过，8.178秒。最终完整命令 `python -B -m unittest discover -s tests -v` 报告 **78项通过（Stage1 43+Stage2 35）、0失败、0跳过，15.041秒，退出码0**；开始UTC `2026-11-11T14:28:01.200865+00:00`、结束 `2026-11-11T14:28:17.881408+00:00`。完整输出与命令/环境见同目录 `all-tests.txt` / `all-tests-record.json`。旧 Stage1 43项原样保留，未降低断言。

独立只读审查未发现 Critical/Important 实现缺陷；提出resume来源SHA和上述三项覆盖缺口，均已补足。修复前的resume元数据测试失败日志、detach mutation日志和审查响应记录保存在同目录。完整 Stage2 真实模型 GPU Smoke 尚未执行，不能据此宣称服务器验收通过。

## 服务器准备和 Smoke 命令（未执行）

仅在用户审查后同步代码；不在正在训练的服务器工作区直接修改。项目路径按用户指定。先确认空闲GPU、真实权重和人工Smoke清单，不自动筛选数据。下面变量须填入已核实路径，未猜测服务器VGG/RN50位置，也未假定候选Stage1 run已经完成。

```bash
cd /home/czy/projects/HSRUIE-reproduction
conda activate hsruie
CUDA_VISIBLE_DEVICES='' python -B -m unittest discover -s tests -p 'test_stage*.py' -v

python -B train_stage2.py --mode data-check \
  --underwater-root /data/czy/datasets/UVTD_import/UVTD/Raw \
  --output-root /data/czy/experiments/HSRUIE/stage2 \
  --run-id "data-check-$(date +%Y%m%dT%H%M%S)-$$"

# 设置 SMOKE_GPU、SMOKE_U、VGG_WEIGHTS、CLIP_WEIGHTS、STAGE1_G 后执行。
# SMOKE_U 是人工确认的 >=2张清单，STAGE1_G 是实际存在的 Stage1 训练包。
: "${SMOKE_GPU:?select an idle GPU}" "${SMOKE_U:?}" "${VGG_WEIGHTS:?}" "${CLIP_WEIGHTS:?}" "${STAGE1_G:?}"
CUDA_VISIBLE_DEVICES="$SMOKE_GPU" python -B train_stage2.py --mode smoke \
  --underwater-root /data/czy/datasets/UVTD_import/UVTD/Raw \
  --underwater-manifest "$SMOKE_U" \
  --vgg-weights "$VGG_WEIGHTS" --clip-weights "$CLIP_WEIGHTS" \
  --stage1-checkpoint "$STAGE1_G" --diagnostic-stage1 \
  --device cuda:0 --batch-size 2 --lr 1e-4 --fixed-epochs 100 --decay-epochs 100 \
  --num-workers 0 --seed 20261111 \
  --output-root /data/czy/experiments/HSRUIE/stage2 \
  --run-id "diagnostic-smoke-$(date +%Y%m%dT%H%M%S)-$$"
```

使用已完成且验收的正式 G 做 Smoke 时，去掉 `--diagnostic-stage1`，增加 `--stage1-acceptance "$STAGE1_ACCEPTANCE"`。目前没有提供或执行正式200epoch训练命令。未来 train 还需全量manifest与来源Stage1配置一致；resume 必须使用新的run目录，权重/数据/代码/软件/训练配置保持一致。

## 文件清单与剩余问题

新增：`stage2.py`、`train_stage2.py`、`data/stage2_dataset.py`、`tests/test_stage2.py`、`tests/test_stage2_cli.py`、本文件、实施计划 `docs/superpowers/plans/2026-11-11-stage2.md`。更新：本次先前新增的 `docs/stage2_audit.md`。审计快照/测试/CPU探针/完整diff在独立忽略目录 `runs/stage2-build-20261111-01/`；不修改已有受跟踪文件。

剩余问题：实际 Stage1 epoch100 G及其验收、服务器权重哈希/manifest一致性尚待实际核验；真实模型组合在3090上的显存、单batch前后向、更新和恢复仍须执行上述Smoke后验收。论文与公开代码没有补齐D、层映射、完整CLIP预处理、初始化、mean reduction和精确离散LR等原始训练依据，本实现按用户决定与工程假设记录，不能把这些不确定性写成作者已确认方法。CPU/同环境恢复一致性不构成跨CUDA版本/硬件逐位一致的承诺。
