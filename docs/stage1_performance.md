# Stage 1 性能热路径优化（2026-11-10）

本次只读分析当前实现，不重复论文/模型/数据审计。基线HEAD为062c02bfb6d16f3af7294fd1c4261d47a801c26c；实际测试与差异以运行开始保存的工作区源码为准。已有大量工作区差异完整保留，未清理。未访问服务器、未运行真实GPU或正式训练。未修改论文协议或历史决策文档。

## 证据与实际修改

当前数据预处理已包含D-11反射策略，本次不改。实际修改只有stage1.py；新增tests/test_stage1_performance.py和本文。所有原有其余Python源码及测试逐文件SHA-256与开始快照一致。AST逐项比较确认Stage1类（包括Forward）、stage1_losses、ranking_loss、build_model、Adam、LR、种子、checkpoint/RNG保存恢复函数均不变。

| 代码事实/瓶颈候选 | 分类与处理 | 时间收益的可信边界 |
|---|---|---|
| validate_image对finite/min/max分别作Python条件判断，每次最多3个CUDA主机判断；每步前向两张量、loss再检查四张量 | A：原finite/min/max归约不变，将三个布尔结果先在设备合并，主机一次判断；同样拒绝NaN/Inf/越界 | 最多每步减少12个标量判断；缺CUDA计时，不能换算毫秒 |
| train_step逐梯度isfinite(...).all()再进入Python all，G每个活动参数张量都可能同步 | A：每个原归约照算，将所有finite标志stack后all，一次主机判断；缺失梯度及全零检查保留；不把校验推迟到epoch结束 | N个有限性主机判断降为1；仍有N个isfinite归约，stack有额外开销；净收益需GPU测量 |
| 每步返回6个指标，逐项float(detach())读取设备标量 | A：6个同dtype标量detach后stack、一次cpu().tolist()；日志仍每步记录，字段/数值/顺序不变 | 六次独立标量提取改为一次批量回传；未宣称其占主要计算时间 |
| num_workers=0，当前线程完成读图/crop后才提交GPU；没有下一batch CPU解码预取 | 潜在CPU等待，未证实是生产主要瓶颈；不改变worker配置 | 需生产data-wait计时；不能从21小时总估计反推解码占比 |
| 每个输入域各一次VGG和SAD | 保留：输入不同，共享模块不意味着输出可复用；没有重复加载权重/多余第三次前向 | 未减算量，0项删除 |
| prepare_image的RGB转换、504张图的np.pad→PIL，再crop/NumPy/Torch | 本次不改，保持像素与随机调用完全不变。整库检查还会读全域及训练选择，属于启动成本 | 没有服务器解码profile，不把转换次数当成已测瓶颈；图像缓存会占RAM/改变故障可见性，暂不引入 |
| 每epoch保存完整包及计算哈希 | 保留100次完整保存及恢复语义；不是每batch保存 | 0项降低保存频率；未获得磁盘用时 |

原路径：每域读图/反射/crop→同步to(device)→E两次/G两次→loss→backward→逐参数主机finite判断→Adam→逐指标回传→日志。
新路径：数据/拷贝/E/G/loss/backward不变→设备聚合finite标志→主机判断→同一Adam→合并6指标回传→同一日志。每epoch调度、checkpoint和RNG不变。

## 为什么没有自动打开其他开关

| 候选项 | 判断/潜在幅度 | 当前决定 |
|---|---|---|
| pin_memory + non_blocking | 原则上不改像素/FP32，但num_workers=0通常在取batch线程上执行pin，存在额外pageable→pinned复制；同一CUDA流也不自动实现与下一batch读图重叠。GPU效果/收益未验证 | 不自动启用，也不声称已验证GPU逐位等价；待授权独立H2D测量 |
| num_workers>0、prefetch_factor、persistent_workers | 能在解码受限时改善吞吐，上限受CPU占比约束；改变Python/NumPy/torch worker随机序列，persistent还改变epoch重置/续训状态 | B：仅分析，保留0；现有RNG checkpoint没有保存持久worker状态，不能无记录切换 |
| channels_last、torch.compile、cudnn.benchmark | 可能改变算子/算法/融合与浮点归约顺序；compile还存在编译启动成本 | B：未启用，无法提供可信百分比；不能凭硬件型号承诺加速 |
| AMP/BF16/TF32/梯度累积/改变batch | 本次用户明确禁止 | 不启用；未触碰现有后端开关，未核验服务器当前开关值 |

所有收益均不能在无服务器profile下换算为实际节省小时。按2690步/epoch、100epoch共269000步：若独立测量实际节省1ms/步，才对应约4.48分钟；5ms约22.42分钟；10ms约44.83分钟。每项优化可能相互重叠，不能相加。21小时是用户提供的预计值，不是本次测量结果。A类是数学/控制语义判断，CPU验证不冒充CUDA逐位/速度认证。

## 本地测试与操作证据

运行目录：runs/stage1-performance-20261110-01，开始记录见before.json；新测试先在旧代码失败（red.txt），再修改。Windows hsruie Python3.8.20、torch2.0.1+cu118、torchvision0.15.2+cu118、NumPy1.24.4、Pillow9.4.0。WSL自带python没有torch，故使用已有Windows环境，未安装依赖。沙箱bubblewrap缺失，所有项目访问经显式批准执行；没有利用其他工具绕过权限。

最终命令（实际测试在进程导入torch前设置CUDA_VISIBLE_DEVICES为空，模型只运行CPU）：

```text
D:/Miniconda3/envs/hsruie/python.exe -B -c "import os; os.environ['CUDA_VISIBLE_DEVICES']=''; import unittest; r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_stage1*.py')); raise SystemExit(not r.wasSuccessful())"
```

结果：43通过、0失败、0跳过，8.270秒。包括现有38项（原27项以及后续反射填充11项）及新增5项。没有降低或删除原测试断言。

新增验证：40个梯度张量finite检查只有一次主机布尔读取；缺失、全零、末尾NaN/Inf仍拒绝；固定输入的新旧Loss字典、每个G梯度/更新参数、Adam step/exp_avg/exp_avg_sq逐位相等，E/RNG不变；指标不再逐Tensor调用float；图像finite/值域校验保留。既有像素/反射/裁剪与epoch恢复测试全部通过，Dataset字节指纹不变。

额外CPU profiler使用运行前保存的完整旧stage1模块对照新模块（不只对照一个新helper）：同微型E/G固定输入与seed123，aten::_local_scalar_dense和aten::item均从25次降到10次，isfinite仍为8次；Loss、梯度与更新参数逐位相等。见profile_cpu.json。该计数针对微型CPU网络，不能当作原SAD的CUDA同步时长，且不保证CPU微型模型运行更快。

完整输出：runs/stage1-performance-20261110-01/tests.txt；操作计数：profile_cpu.json；关键diff：changes.patch；全部文件指纹与语义检查：verification.json。记录中的原权重均为未加载，NC/LOW/HIGH/CLIP/任务网不适用；所有图像为临时合成fixture，没有读取训练数据。

## 服务器后续验证命令（未执行）

先运行CPU回归，不与当前GPU训练争用：

```bash
cd /home/czy/projects/HSRUIE-reproduction
CUDA_VISIBLE_DEVICES='' python -B -m unittest discover -s tests -p 'test_stage1*.py' -v
```

当前训练结束、用户批准独立验证后，才考虑以下单batch命令；VGG_WEIGHTS/SMOKE_U/SMOKE_A/G_INIT/SEED均沿用已核实的权重、人工清单和当前初始化/种子，不从data-check自动筛选。每次使用新run，不写当前实验目录：

```bash
: "${VGG_WEIGHTS:?}" "${SMOKE_U:?}" "${SMOKE_A:?}" "${G_INIT:?}" "${SEED:?}"
python -B train_stage1.py --mode smoke \
  --underwater-root /data/czy/datasets/UVTD_import/UVTD/Raw \
  --inair-root /data/czy/datasets/Flickr2K \
  --underwater-manifest "$SMOKE_U" --inair-manifest "$SMOKE_A" \
  --vgg-weights "$VGG_WEIGHTS" --g-init "$G_INIT" --seed "$SEED" \
  --device cuda:1 --batch-size 2 --num-workers 0 \
  --output-root /data/czy/experiments/HSRUIE/stage1-validation \
  --run-id "perf-smoke-$(date +%Y%m%dT%H%M%S)-$$"
```

单batch Smoke是正确性验证，不是可信吞吐基准：启动/data-check/VGG加载不可算进稳态batch时间。未来经授权的性能对照应固定同一输入、权重和种子，warmup后分别记录data wait/H2D/E-G/loss/backward/校验/step/日志以及峰值显存、整步wall-clock；只在测量区间边界同步，避免计时器自己制造每算子同步。本次没有创建GPU benchmark入口或启动它。

## 当前训练建议

不建议中断或从头重启当前训练：没有GPU收益测量，也无法在正在运行的Python进程中热替换为可信新代码。完成当前实验；用户审核后再在隔离运行验证收益。若之后批准在epoch checkpoint续训，本次未修改config字段/加载比较/RNG机制，但仍须使用新run目录并记录代码差异，不能宣称已验证跨GPU完全一致。本文是新的性能记录，不改写论文协议或D-11等历史决策。
