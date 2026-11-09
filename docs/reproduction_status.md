# 复现状态

更新日期：2026-11-02。源码基线：21ff69916b5ce95a1f4a1aea89b23d15c0a15d4b。

## 状态与证据定义

- **已核验**：已有指定范围的文件、源码或实际运行检查；不自动等于论文复现成功。
- **产物存在、待认证**：文件已生成，但缺少绑定源码、权重、配置的独立运行记录。
- **未完成 / 阻塞 / 暂缓**：均不属于已执行实验。
- **历史实验事实**来自前两轮会话审计；会话结论不是完整原始运行日志。本轮仅进行文档整理、源码阅读及轻量文件盘点，没有加载模型、执行 GPU 操作或重跑实验。

## 已有进展与局限

| 工作 | 状态 | 证据与局限 |
|---|---|---|
| Windows + Conda hsruie 环境搭建 | 已核验，历史实验事实 | 第一轮查询：Python 3.8.20、torch 2.0.1+cu118、torchvision 0.15.2+cu118、RTX 4060 Laptop 约 8 GB、CUDA 可用；本轮未重新初始化 CUDA |
| 学校双 RTX 3090 | 待确认 | 用户描述，前两轮未连接验证 |
| 五组自定义 checkpoint 结构检查 | 已核验，历史实验事实 | 第二轮 CPU 只读检查：参数键、shape 匹配，无非有限值；本轮仅复核文件存在与大小 |
| VGG19 权重存在与哈希 | 已核验，历史实验事实 | 第二轮记录哈希；本轮未重新执行 VGG 或加载该权重 |
| 历史 checkpoint 推理 | 用户报告成功，存在样本产物 | 仍缺少绑定全部权重和源码的完整运行记录；不等于已认证复现 |
| SUIM 110 张增强结果 | 产物完整、推理来源待认证 | 本轮 results_suim_full 内计数 110；前两轮验证同名匹配、填充尺寸和 HTML 110 条记录 |
| SUIM 尺寸恢复 | 已核验，历史实验事实 | 110 张恢复图逐像素等于增强结果左上角裁剪，与原始图像/Mask 尺寸相同；本轮复核仍有 110 张 |
| UVTD 数据准备 | 基础文件已盘点 | Raw/Reference 各 5380；全部 Raw 宽高为 16 倍数；标注和官方划分未定位 |
| UVTD 10 张样本 | 产物存在、待认证 | 第二轮逐张检查均不同于 Raw 和 Reference，尺寸一致；不能据像素不同证明来自当前 checkpoint |
| UVTD 全量推理 | 尚未认证，当前暂缓 | 用户说明旧混入结果已清理且尚未重推；不沿用旧目录计数作为完成证据，本轮不检查该目录内容 |
| Stage 1 / Stage 2 / Stage 3 | 均未完成 | 当前训练计算图缺失，见 known_issues；没有执行新的训练 |
| SUIM 分割、UVTD 检测及其他下游评价 | 未完成 | 没有经过核验的本项目评价结果，不能把数据集附带 Benchmark_Evaluation 当成我们的实验 |
| DEM、SEM、损失消融 | 未完成 | 需先建立正式基线、训练和任务评价 |

SUIM TEST 110 张是测试数据，任何“110 张推理完成”均只指增强图像产物，不表示进行过任务训练、预测 Mask 或得到分割指标。

## 现有资产位置

| 资产 | 路径 | 本轮文件数 |
|---|---|---:|
| SUIM 填充输入 | D:/Research/SUIM/TEST_padded_images | 110 |
| SUIM 尺寸记录 | D:/Research/SUIM/TEST_padded_sizes.csv | 前两轮核验为 110 条 |
| SUIM 增强产物 | results_suim_full/maps_cyclegan/test_latest/images | 110 |
| SUIM 恢复产物 | results_suim_restored | 110 |
| UVTD 小样本 | results_uvtd_10/maps_cyclegan/test_latest/images | 10 |

输入与输出路径来自 [prepare_suim.py](../prepare_suim.py):6-14 和 [restore_suim_size.py](../restore_suim_size.py):5-18。恢复逻辑在 restore_suim_size.py:43-55。数据细节见 [dataset_inventory](dataset_inventory.md)。

## 历史 checkpoint 指纹

以下是第二轮审计记录，不代表本轮重新加载或重新计算，也不能替代未来每个 run 的权重核验。

| 文件 | SHA-256 |
|---|---|
| checkpoints/maps_cyclegan/latest_net_NC.pth | 8a530b5960698b78e638fad2ae2d3b09f789b090cc21d505abeffdf39da57fed |
| checkpoints/maps_cyclegan/latest_net_SAD.pth | 0d66726fbf23257dead2328bd41afa90fcc145cdc5a41b9d4585a80315c1b63b |
| checkpoints/maps_cyclegan/latest_net_LOW.pth | 2be0932616dc6b09fd78a333af00ac574c7f468da922bf5e11f4967b2371fa19 |
| checkpoints/maps_cyclegan/latest_net_HIGH2.pth | a464cff319a17e8cebfbc200f70846925b9964640f79386505e7c6cde7b3f2af |
| checkpoints/maps_cyclegan/latest_net_HIGH3.pth | ddbb0f17a1881a14fd4c8823a84e18e8a3c8d181fad3401b2ed0dea179a13eb5 |
| model_data/vgg19-dcbb9e9d.pth | dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44 |

权重兼容性不证明训练阶段、数据来源或论文指标可比性。两轮审计没有建立“当前模型输出与正式作者输出一致”的证据链。

## 运行记录和工作区边界

- 当前唯一发现的参数快照是 checkpoints/maps_cyclegan/test_opt.txt，内容属于 SUIM 设置，不能用于证明 UVTD 小样本的配置。
- [BaseOptions.print_options](../options/base_options.py):108-114 在解析时覆写参数文件；[test.py](../test.py):48-54 随后又修改 batch_size、线程数等。因此该快照既不保证运行成功，也不是完整的最终生效配置。
- 文档任务开始前已有 17 个被跟踪的 .pyc 文件处于修改状态。它们不是本轮文档变更，不应清理、回退或归入本轮科研修改。
- environment.yml 记录 Linux/PyTorch 1.8.0+cu111 等，与已核验本地 Windows 环境不同。尚未产出精确可重建的本地环境锁定文件。

## 与前两轮报告的更新关系

1. 延续第二轮的 LOW-off 基线、权重兼容性、UVTD 无须填充及三阶段未完成判断。
2. 第一轮旧 UVTD 全量目录问题已由用户说明并清理；正式状态为“尚未认证、暂缓”，不把该问题继续描述为当前产物结论。
3. 新增盘点：SUIM train_val 确认为 1525 张图像和 1525 张同 stem Mask；合计本地 1635 张与论文 p.47 的 1634 张差 1，原因待核，没有删除或调换样本。
4. 明确“产物完整”与“推理来源认证”区别。第二轮 3 张 SUIM 试运行与完整结果存在最大 1/255 的差异，没有足够证据归因为不同 checkpoint。

下一步只按 [experiment_plan](experiment_plan.md) 规划；用户当前尚未授权重新推理或训练。
