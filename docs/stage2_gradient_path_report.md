# Stage 2 梯度路径专项审计

日期：2026-11-11（Asia/Shanghai）。审计对象为当前 Stage 2 实现及原有78项测试；论文依据为 p.45 Fig.2、p.46 III.C 式 (7)–(14)。本次没有发现需要修正的模型错误，只增强一个现有测试中的 `idt2` 子项，不新增测试方法，不修改训练实现、模型数学行为或 checkpoint 格式。

**结论**：CPU 合成模块上的四项损失到 C/D 的预期路径均有测试证据；新增检查独立验证了 `L_idt2` 的直接 Xc 分支、Xidt 生成分支及两者对 C 的梯度之和。真实 VGG/SAD/FE/RN50 全模型组合的梯度尚未在本次独立验证，不能把合成结果写成真实 GPU 验收。

## 1. 逐项计算图核对

**源码事实**：`stage2.py:137` 仅对 `E(Xu)`、`CLIP(preprocess(Xu))` 使用 no_grad。`G(Fc)`、`E(Xc)`、生成图像的 CLIP、`G(Fidt)` 均在该上下文外。损失定义位于 `stage2.py:107–113`；hook 位于 `stage2.py:36–61`，没有 detach 或 CPU 转换。

以下箭头表示反传方向，G₁/G₂ 是同一个冻结 G* 的两次调用。

| 损失 | 应有反传路径 | C | D | 当前验证结论 |
|---|---|---|---|---|
| L_sep，式 (8) | loss→Fc→C；loss→Fd→D | 有 | 有 | 单项 backward 下 C 有有限非零梯度、D 有非零梯度；总训练步进一步检查 C/D 全部参数梯度存在且有限 |
| L_clip，式 (11) | loss→CLIP层2/3/4→可微预处理→Xc→G₁→Fc→C | 有 | 无 | 单项 loss 能回传到 C，D.grad 全为 None；该 loss 没有绕过 CLIP/G₁ 到 C 的其他分支 |
| L_idt1，式 (13) | 生成侧：loss→Fidt→E→Xc→G₁→Fc→C；直接侧：loss→Fc→C | 有 | 无 | 有完整参考梯度对照，且参考结果与 detach 生成侧后的结果不同；避免仅凭 C 梯度非零误判 |
| L_idt2，式 (13) | 生成侧：loss→Xidt→G₂→Fidt→E→Xc→G₁→Fc→C；直接侧：loss→Xc→G₁→Fc→C | 有 | 无 | 本次补充两侧独立梯度及其和的检查，并用两种 detach mutation 验证断言有效 |

**实验事实与范围**：本表的数值/梯度通过结论来自 TinyE/TinyG/TinyVisual/TinyExtractor 的 CPU 合成图。`tests/test_stage2.py:202` 对四项损失逐项创建新的模型后 backward，不借用之前的梯度；对 clip/idt1/idt2 检查 D.grad 为 None。`tests/test_stage2.py:269` 专门比较 idt1 的完整参考梯度。模块整体非零不表示每个元素、每个参数或任意训练输入上都必须非零。

## 2. L_idt2 原有缺口及最小补充

**原有测试事实**：`test_each_loss_reaches_c_and_only_separation_reaches_d` 已证明单独 idt2 能回传到 C，但只有整体非零断言。`test_raw_target_no_grad_matches_reference_values_and_gradients` 比较的是四项总损失梯度，未将 idt2 两侧独立拆开。CLI Smoke 的总损失梯度检查也不能单独归因到 idt2。

本次先通过内存 patch 分别将 idt2 改为 `L1(Xidt, Xc.detach())` 和 `L1(Xidt.detach(), Xc)`：**旧单项测试对两种错误均通过**。因此确实存在覆盖缺口，不是根据代码外观推测。

**理论计算图依据**：令 z=Xc、y=H(z)=G*(E(z))、N 为图像 Tensor 元素总数，mean L1 的 s=sign(y−z)/N。PyTorch 在差值为0处采用0导数。两侧都参与时：

```text
dL/dy（损失的 y 参数偏导） = s
dL/dz（包含 H 分支的总导数） = J_H(z)^T s - s
dL/dθC = J_z(θC)^T [J_H(z)^T s - s]
```

不能将 `grad(loss, Xc)` 直接叫作“直接分支梯度”，因为 Xidt 本身依赖 Xc，它包含两条分支的贡献。

**本次新增实验断言**（`tests/test_stage2.py:202–252`，保留全部原有断言）：

1. 临时 hook 同一 G 的两次输出，取得真实 forward 中的 Xc/Xidt；两者必须需要梯度。
2. 检查 `grad(loss, Xidt)=s`，不得为 None，验证损失连接了 Xidt 参数。
3. 用 `autograd.grad(Xidt, Xc, grad_outputs=s)` 独立计算生成侧 VJP，要求有限且非零。
4. 检查实际 `grad(loss, Xc)=生成侧VJP−s`，明确捕获直接 Xc 分支缺失。
5. 分别求两侧到 C 的参数 VJP，要求两组均有限且模块整体非零；最终 backward 的 C.grad 必须与两组之和一致。

VJP 和参数梯度相加比较使用 `atol=1e-7, rtol=1e-5`，容纳 FP32 的浮点运算顺序差异；`grad(loss,Xidt)` 与 s 逐位相等。这里的 detach 只用于构造参考导数种子或负例，不改生产计算图。

**断言强度验证**：增强后，detach 直接 Xc 的负例在第4项断言失败；detach Xidt 的负例在第2项断言失败。两次均为预期 assertion failure、无 fixture error；真实实现的定向测试通过。这些负例只在临时内存 patch 中执行，不写回 `stage2.py`。证据在本次 run 的 `mutation-before.txt/.json`、`mutation-after.txt/.json`、`targeted.txt`。负例日志中的 FAIL 是有意注入错误的检测结果，不是最终回归失败。

## 3. 已经通过测试验证的路径与约束

| 项目 | 测试依据 | 已验证的范围 |
|---|---|---|
| 冻结 CLIP/G 仍向 C 传播 L_clip | `tests/test_stage2.py:202` 的 clip 子项 | CPU 合成 E/G/CLIP/C；损失唯一的生成侧路径连通，C 有有限非零梯度，冻结参数无 grad |
| CLIP 空间 hook 不 detach | `tests/test_stage2.py:187` | 三个空间输出与模块顺序调用相同；合成 visual 参数冻结，输入仍有梯度 |
| CLIP 预处理可微 | `tests/test_stage2.py:173` | 非常量256→224结果及输入梯度，与 torchvision Tensor bicubic/antialias参考逐位相同 |
| idt1 生成 E 分支保留 | `tests/test_stage2.py:269` | C 梯度匹配完整参考；仅直接 Fc 分支的梯度不能冒充完整结果 |
| idt2 两侧及对 C 的贡献 | `tests/test_stage2.py:202`，本次增强部分 | 实际 loss 的 Xidt 偏导、Xc 总导数、两侧独立 C 梯度及其和；两个 detach 负例均被捕获 |
| D 只有 sep 能产生梯度 | `tests/test_stage2.py:202` | 新模型分别反传；sep 对 D 非零，其余三个单项的 D.grad 均为 None |
| E/G/CLIP 参数冻结及状态不变 | `tests/test_stage2.py:202,254` | requires_grad=False、eval、无参数梯度；一次真实 optimizer 更新前后冻结 state_dict 全项逐位不变，包括合成 E/CLIP 的 BN running_mean/var/num_batches_tracked |
| optimizer 仅含 C/D | `tests/test_stage2.py:254,285`；`stage2.py:149–158` | 参数对象身份集合等于 C/D；训练步检查数量与集合避免重复；误加入冻结参数会拒绝；C/D均更新 |

上述冻结状态测试中的 TinyG 没有 BN，不将它写成对真实 SAD 所有可能 Buffer 行为的前向认证。原始 SAD 的 GN 等结构依据来自源码；生产 Smoke 入口具备全 state_dict 检查，但本次没有运行真实组合。

**历史实验证据，非本次新实验**：`runs/stage2-build-20261111-01/clip_cpu_probe.json` 曾在官方哈希 RN50 上执行 CPU/FP32 前后向，验证 layer2/3/4 空间输出、输入梯度和冻结 BN buffers。本次核对该记录关联的 `stage2.py` 与 `CLIP/clip/model.py` 指纹仍匹配；未重新运行真实 RN50。该证据只补充真实 CLIP→其输入这一段，不证明真实 G*→C 的组合已经验证。

## 4. 仅有理论计算图依据的结论

- **源码事实＋理论**：冻结参数不会自动关闭网络对输入的 Jacobian；当前生成侧没有 no_grad/detach，因此原始 E/G*/RN50 的组合在代码层面保留反传路径。不能仅凭这点声称某份真实 G* 的实际梯度有限、非零或适合训练。
- **理论**：idt1/idt2 两侧在共享上游处相加，可能部分抵消；L1 在零差值处有不可微点，PyTorch选0。合成 fixture 的非零结果不等于所有真实输入/权重都非零。
- **源码事实**：D 的输出只进入 sep；其余损失不引用 Fd。独立 backward 的合成测试支持这一结论，但未为真实模型每个尺度、每个参数逐项提供梯度数值认证。

## 5. 尚未独立验证的路径

- 已验收 Stage1 G*、真实 VGG、原始 FE C/D、真实 RN50 在256 crop、batch2下的完整四项损失与逐分支梯度组合；本次未做真实权重整图前后向。
- 真实 RN50 每个尺度的 CLIP loss 分别穿过真实 G* 到 C 的数值结果；当前专项逐 loss 测试对三尺度求和验证，不冒充每尺度独立认证。
- 真实完整模型在 RTX3090/CUDA 上的参数/Buffer 不变、显存、前后向与 optimizer 更新；未启动 GPU Smoke 或正式训练。

这些是真实模型后续验收范围，不是本次新增大型测试设计的要求；本次没有为扩大覆盖而改动方法。

## 6. 本次完整回归与修改清单

**新实验事实**：run ID `stage2-gradient-audit-20261111-01`，cwd `D:/Research/HSRUIE`，Windows CPU，Python3.8.20、torch2.0.1+cu118、torchvision0.15.2+cu118。定向 fixture 使用 seed711、torch线程2；完整套件保留各自原有随机种子。没有加载真实预训练权重做本次定向测试。

```powershell
$env:CUDA_VISIBLE_DEVICES=''
D:/Miniconda3/envs/hsruie/python.exe -B -m unittest discover -s tests -v
```

Asia/Shanghai 开始 `2026-11-11 22:43:36.805761 +08:00`，结束 `2026-11-11 22:44:05.223268 +08:00`。unittest 报告 **78项全部通过，0失败、0跳过，26.872秒，进程退出码0**。原有43项Stage1及35项Stage2测试方法均保留，只增强现有一个方法中的 idt2 检查。

- 新增报告：`docs/stage2_gradient_path_report.md`。
- 增强测试：`tests/test_stage2.py` 的 `test_each_loss_reaches_c_and_only_separation_reaches_d`。
- 不修改任何生产实现、原有其他测试或已有实验记录，不改变 checkpoint/指标可比性/训练协议。

完整日志：[all-tests.txt](../runs/stage2-gradient-audit-20261111-01/all-tests.txt)；命令与时间：[all-tests-record.json](../runs/stage2-gradient-audit-20261111-01/all-tests-record.json)；本次 diff 与文件指纹分别保存为同目录 `changes.patch`、`final_record.json`。没有报告或估计梯度范数。本次未操作服务器、GPU、正式训练或 Git commit/push/reset/分支切换。
