Task-Driven Underwater Image Enhancement via Hierarchical Semantic Refinement
====

## 本地论文复现文档

当前以忠实、可追溯的论文复现为目标。后续 Agent 先阅读 [AGENTS.md](AGENTS.md)。用户当前暂不进行 UVTD 全量推理；本次文档整理没有执行推理或训练。

| 文档 | 主要内容 |
|---|---|
| [论文原始协议](docs/paper_protocol.md) | 三阶段目标、损失、超参数、数据用途与指标 |
| [当前复现状态](docs/reproduction_status.md) | 已核验事项、已有产物和未认证工作 |
| [已知问题](docs/known_issues.md) | 论文/代码差异、加载与记录风险及处理状态 |
| [实验计划](docs/experiment_plan.md) | P0-P7、数据阻塞与验收条件；不是执行授权 |
| [实际模型结构](docs/model_architecture.md) | Forward、LOW 旁路、Tensor shape 与权重 |
| [数据资产清单](docs/dataset_inventory.md) | UVTD/SUIM/Flickr2K、Mask 与检测标注状态 |
| [实验决策日志](docs/decision_log.md) | LOW-off、HDE、测试集隔离和暂缓全量等决定 |

项目目录中 models/ 为网络与训练包装，data/ 为数据加载，options/ 为参数，util/ 为保存/可视化，unet_test/ 为未完整接入的任务组件，docs/ 为复现记录。checkpoints/、model_data/ 和 results*/ 分别保存权重与产物，不能自动覆盖。

注意：下方保留原仓库说明，不代表已经在本地认证可重放。当前存在 LOW 旁路、默认数据加载问题、cwd 敏感和 test_opt.txt 覆盖风险，运行前先阅读已知问题与 P0 计划。

## 原仓库说明

- Set up a conda environment
```
conda env create -f environment.yml
conda activate hsruie
```

- checkpoints
```
download latest_net_NC.pth (./checkpoints/maps_cyclegan/) at: https://drive.google.com/file/d/1SgiK6fBHXd2yICBhNNfoncUWjSUxslYe/view?usp=sharing

download vgg19-dcbb9e9d.pth (./model_data/maps_cyclegan/) at: https://drive.google.com/file/d/1eRwpDSR8PlJKDuoG4JvBN9fOGfasaN51/view?usp=sharing

```

- Test:
```
test datasets path: ./datasets/maps/testA
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
