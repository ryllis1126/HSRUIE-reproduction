# 数据资产清单

> **最新结论见文末「2026-11-09 UVTD 轻量读取审计」**。以下 2026-11-02 内容保留为历史证据，其中“UVTD 未定位标注”等状态已被本次本地检查更新；历史尺寸统计不是本次重新进行的全库检查。SUIM 等非本次对象未重新核验。

盘点日期：2026-11-02。范围为已知 D:/Research 数据目录及本项目已有产物；没有全盘搜索其他硬盘，没有生成、移动或修改图像。本轮只枚举文件并读取 UVTD Raw 图像尺寸，未重新进行全库像素/损坏检测。

## 资产与检查深度

| 资产 | 已知位置 | 已确认数量/结构 | 检查边界与状态 |
|---|---|---|---|
| UVTD Raw | D:/Research/UVTD/Raw | 5380 张 BMP，直接位于顶层 | 本轮计数及全部尺寸；第二轮已核验与 Reference stem 配对 |
| UVTD Reference | D:/Research/UVTD/Reference | 5380 张 BMP | 本轮计数；参考图像不等于语义/检测真值 |
| UVTD 语义标签 | 已检查 UVTD 根目录中未定位 | 待提供路径/归档 | 缺少当前可用资产，阻塞 Stage 3 正式数据路径 |
| UVTD 检测标注 | 已检查 UVTD 根目录中未定位 | 未找到独立 XML/JSON/TXT 等标注文件 | 格式不限于这些扩展名；仍需官方标注、类别和划分，阻塞检测评价 |
| UVTD train/val/test 列表 | 未定位 | 待确认 | 5380 张全集不能直接称为测试集 |
| SUIM TEST/images | D:/Research/SUIM/TEST/images | 110 张 JPG | 本轮计数；前两轮已验证同名 RGB Mask 和尺寸 |
| SUIM TEST/masks 顶层 | D:/Research/SUIM/TEST/masks | 110 张 RGB BMP | 正式多类标签资产；颜色值前两轮已检查 |
| SUIM TEST/masks 子目录 | FV、HD、PF、RI、RO、Saliency、SR、WR | 每目录 110，共 880 张 | 是同一 TEST 样本的派生二值标签，不是额外图像样本 |
| SUIM train_val/images | D:/Research/SUIM/train_val/images | 1525 张 JPG | 本轮枚举并按 stem 配对；未检查全部像素、标签颜色、损坏和跨 split 重复 |
| SUIM train_val/masks | D:/Research/SUIM/train_val/masks | 1525 张 BMP | 同 stem 1525 对，无仅图像/仅 Mask 的文件；正式 train/val 再划分未建立 |
| Flickr2K | 未提供路径 | 待确认 | 已查看 D:/Research 顶层未见该目录，不代表其他磁盘没有；不得写成已准备 |
| USOD10K | 未提供路径 | 待确认 | 论文额外显著性数据，当前计划不包含已完成实验 |
| 下游任务模型 | SUIM/Checkpoint_Data 等历史资产 | 第一轮见多份 .hdf5 | 与论文采用模型/权重的对应尚未认证，不能直接用于 PyTorch U-Net |

**文件事实**：本轮 UVTD 根目录只有 Raw 和 Reference；未定位标注或划分。这里的“缺失”仅指当前可用项目资产，不等于全机无文件。不能用 Reference 代替语义 Mask、检测框或 Stage 1 的空气域。

## UVTD Raw 的全部尺寸分布

**本轮文件统计**：宽×高；共 20 种，全部宽高为 16 的倍数，需要填充的图像数为 0。由 [当前 Forward](model_architecture.md) 的 shape 约束可知空间拼接兼容；显存可行性仍未验证。

| 宽×高 | 数量 |
|---|---:|
| 1280×704 | 1807 |
| 640×480 | 1556 |
| 704×576 | 655 |
| 352×224 | 483 |
| 1920×1056 | 443 |
| 2560×1600 | 206 |
| 256×256 | 110 |
| 1056×1920 | 25 |
| 1600×2560 | 23 |
| 224×352 | 21 |
| 576×704 | 18 |
| 704×1280 | 12 |
| 416×416 | 5 |
| 480×640 | 4 |
| 1024×1024 | 3 |
| 2048×2048 | 3 |
| 480×480 | 2 |
| 1280×1280 | 2 |
| 640×640 | 1 |
| 960×960 | 1 |
| **合计** | **5380** |

## SUIM 标签与数量边界

**数据说明事实**：D:/Research/SUIM/INFO.txt 声明 train_val=1525、TEST=110，并列出下列 RGB 颜色编码。类别索引、ignore 规则以及论文评价实际采用的类别集合仍须单独确定。

| RGB bit 编码 | RGB 值 | 说明文件中的类别 |
|---|---|---|
| 000 | (0,0,0) | BW，背景水体 |
| 001 | (0,0,255) | HD，潜水员 |
| 010 | (0,255,0) | PF，植物/海草 |
| 011 | (0,255,255) | WR，沉船/遗迹 |
| 100 | (255,0,0) | RO，机器人/仪器 |
| 101 | (255,0,255) | RI，礁石与无脊椎动物 |
| 110 | (255,255,0) | FV，鱼与脊椎动物 |
| 111 | (255,255,255) | SR，沙地/海床/岩石 |

TEST/masks 递归计数 990=110 顶层 RGB + 880 子目录二值图，不能写成 990 个测试样本。Saliency 是派生目标，不是第九个多类类别。

本轮确认本地总量为 1525+110=1635；**论文事实 p.47 IV.B 写 1634**。原因待确认，保持原始资产不变，不能为凑论文数字删除一张。尚未检查 train_val 与 TEST 的内容级重叠；不得宣称已完成去重审计。

SUIM TEST 的 110 张仅用于评价，不能训练、微调、早停或选择结构。train_val 也不能不经划分就声称已有正式独立训练/验证集。

## 预处理衍生资产

| 资产 | 路径 | 状态 |
|---|---|---|
| SUIM 反射填充输入 | D:/Research/SUIM/TEST_padded_images | 110 张 PNG，本轮计数 |
| 尺寸 manifest | D:/Research/SUIM/TEST_padded_sizes.csv | 前两轮已核验 110 条，11 张实际填充 |
| SUIM 增强结果 | results_suim_full/maps_cyclegan/test_latest/images | 110 张，本轮计数；推理来源仍待认证 |
| SUIM 恢复结果 | results_suim_restored | 110 张，本轮计数；前两轮已逐像素验证裁剪关系 |
| UVTD 10 张样本 | results_uvtd_10/maps_cyclegan/test_latest/images | 10 张，本轮计数；第二轮确认非 Raw/Reference 直接副本 |

来源：[prepare_suim.py](../prepare_suim.py):30-69；[restore_suim_size.py](../restore_suim_size.py):24-55。右/下 reflect 填充及裁剪是本地工程适配，没有证据把它归为论文原始测试协议。UVTD 当前无需该适配。

## 正式数据准备仍需完成

以下均为**计划/待确认**，没有被本轮盘点完成：

- 官方数据来源、归档版本、授权说明、输入文件清单及内容哈希。
- UVTD 语义与检测标签、类别集合、官方划分、丢失/空标签规则。
- Flickr2K 的实际路径、版本、数量及 Stage 1 采样方案。
- SUIM 数量差异原因、train/val 列表、跨 split 重复核验和全部标签有效性。
- 任务网络 checkpoint 与数据/类别协议对应关系；输入归一化和评价图像尺寸。
- 每个实验引用唯一数据版本和划分哈希，避免以后仅凭目录名认定相同数据。

---

## 2026-11-09 UVTD 轻量读取审计

**实验事实（只读数据审计，非模型实验）**：审计 ID `UVTD-local-readiness-20261109-01`；cwd `D:/Research/HSRUIE`；数据根 `D:/Research/UVTD`。服务器目标根 `/data/czy/datasets/UVTD_import/UVTD` 来自用户提供，本次未连接服务器。用户报告的 Linux 部署、PyTorch 2.0.1+cu118、作者 checkpoint 和单图推理成功属于会话背景，不作为本次重测结果。

审计使用 PowerShell、`rg` 和通过 `@'…'@ | python -B -` 执行的内存脚本；Python `D:/Python/Python312/python.exe` 3.12.10，Pillow 11.3.0；仅 CPU 文件枚举、Pillow 抽样解码/内存裁剪、标准库 ElementTree 解析小型 XML。未加载 torch、模型或权重，未训练、推理、生成数据副本、全量哈希或全库像素检查。初次导入 NumPy 2.3.5 因 C 扩展缺失失败，后续检查已用标准库/Pillow 完成，未修复环境。开始时刻未单独记录；末次数据核验时间 `2026-11-09T09:17:15+08:00`。抽样种子 `20261109`，不涉及模型随机种子或 GPU 确定性设置。

源码 HEAD 仍为 `21ff69916b5ce95a1f4a1aea89b23d15c0a15d4b`；开始时已有 README、若干已跟踪 `.pyc` 改动及未跟踪 `AGENTS.md`、`docs/`，未清理或回退。本次只更新本文；无源码修改，不改变 Forward、checkpoint 兼容性或指标计算，也不产生新的模型评价结果。

### 1. 实际结构、计数与配对

**实验事实**：`os.walk` 枚举全部目录项；下表为各目录直接包含的文件数，不把 XML 的两个子集重复计算成额外图像样本。根目录和 `object_detection_labels/` 本身均为 0 个文件，其余均无更深子目录。

| 相对 `D:/Research/UVTD` 的目录 | 文件数 | 扩展名 | 内容/对应方式 |
|---|---:|---|---|
| `Raw/` | 5380 | `.bmp` | 原始输入 |
| `Reference/` | 5380 | `.bmp` | 同名参考图 |
| `semantic_labels/` | 5380 | `.bmp` | 同名语义标签 |
| `salient_label/` | 5380 | `.bmp` | 同名显著性标签；实际容器存在 PNG，见下文 |
| `object_detection_labels/5380_object_detection_label(4377_have_objects)/` | 5380 | `.xml` | 检测主标注集 |
| `object_detection_labels/train_3380/` | 3380 | `.xml` | 标注子集，无图像、无 TXT 列表 |
| `object_detection_labels/test_2000/` | 2000 | `.xml` | 标注子集，无图像、无 TXT 列表 |

**实验事实**：Raw、Reference、semantic_labels、salient_label 的完整文件名集合相等，均为 5380 个唯一 stem；检测主目录 XML stem 集合也与 Raw 相等。命名示例：`Raw/1.bmp` ↔ `Reference/1.bmp` ↔ `semantic_labels/1.bmp` ↔ 主目录 `1.xml`。数字 ID 范围为 1–5500，中间缺 120 个编号；**不能用 `range(1,5381)` 构建清单，应枚举真实文件名后按 stem 连接**。未做 Raw/Reference 全量像素比对，文件对应不代表参考图质量或监督用途已认证。

### 2. Raw 抽样与 256×256 裁剪条件

**实验事实**：按数字排序真实 stem，取 `random.Random(20261109).sample(stems,30)`，再加入 `1,5380`，得到 32 个样本：

`1,73,406,730,800,1173,1365,1366,1537,1548,1581,1724,1795,1826,2469,2571,2704,3133,3140,3151,3375,3533,3570,3573,3915,4112,4139,4863,5180,5192,5380,5458`。

32 组 Raw/Reference/语义/显著性图均成功解码、组内尺寸相同；Raw、Reference 均为 RGB BMP。Raw 抽样尺寸覆盖 `256×256、640×480、704×576、1280×704、1056×1920、1920×1056、2560×1600、1600×2560`。这 32 张均可直接裁剪，使用每张种子 `20261109+int(stem)` 随机选择合法坐标，已在内存得到 256×256 patch，未保存。

补充边界抽查按数字排序 Raw 后每隔 34 项读取图头，遇到首个短边不足 256 的样本即停止：**`Raw/39.bmp` 为 RGB BMP、352×224**，三个同名配套图尺寸相同。它无法进行不越界的原尺寸 256×256 随机裁剪。历史全量尺寸表中 `352×224` 483 张、`224×352` 21 张，合计 504 张存在相同限制；该数量是 **2026-11-02 历史证据，本次未重数**。宽高满足 16 倍数与能裁出 256 方块是两项不同条件。

**计划/待确认**：Stage 1 的 UVTD 原图读取条件成立，但完整裁剪策略尚不成立；短边不足时如何缩放/填充/采样须另行确定并记录，不能静默越界裁剪引入黑边，也不能自行删除这些样本。论文 p.47 IV.A 记载随机裁剪，但本次未获得短边处理的原始协议，不实施任何预处理变更。

### 3. Stage 3 语义编码与 ignore 现状

**实验事实**：上述 32 张 `semantic_labels/<stem>.bmp` 全为真正 BMP、RGB 三通道图，非单通道类别索引、非 P 模式调色板图。用 Pillow `getcolors(10000)` 读取原始像素，颜色并集恰为以下八种；每个通道只见 0/255，未见其他颜色。此结论只覆盖抽样，未证明全库不存在其他编码。

| RGB 颜色 | RGB bit 表示 | 类别含义/整数 ID 状态 |
|---|---|---|
| `(0,0,0)` | `000` | UVTD 官方映射未确认 |
| `(0,0,255)` | `001` | 同上 |
| `(0,255,0)` | `010` | 同上 |
| `(0,255,255)` | `011` | 同上 |
| `(255,0,0)` | `100` | 同上 |
| `(255,0,255)` | `101` | 同上 |
| `(255,255,0)` | `110` | 同上 |
| `(255,255,255)` | `111` | 同上；不能因含 255 就判作 ignore |

证据示例：`semantic_labels/1.bmp` 包含黑、蓝、青、品红、黄、白；`1366.bmp`、`3533.bmp` 有绿色，`1365.bmp` 有红色。颜色集合与本文历史 SUIM 表一致，**推测**可能采用相关编码；这不证明 UVTD 的类别含义、类别次序、背景或忽略规则与 SUIM 相同，不能直接移用。

**未确认**：UVTD 官方 RGB→类别→连续整数 ID 表、背景是否参与评价、ignore index、未知色处理、任务模型输出通道和 checkpoint 的类别顺序。RGB 文件没有可直接读出的标量 ignore index；黑色/白色均不能擅自视为 ignore。未来 CE 读取需明确得到整数 `[H,W]` 标签并与任务网络协议一致；颜色变换应精确匹配，几何变换与 Raw 同步，标签插值需保持离散颜色。此处仅列兼容性要求，未修改或执行训练代码。

### 4. 检测 XML、类别与划分依据

**实验事实**：只读解析全部 5380 份主 XML 及两个子目录共 5380 份 XML，全部语法解析成功（10760 份处理、0 解析失败）。格式为 Pascal VOC 风格：`annotation/size`、`object/name`、`object/bndbox/{xmin,ymin,xmax,ymax}`，附 `difficult`、`truncated`；存的是类别字符串，未找到数值 class ID 字段或官方整数映射。没有 COCO JSON、YOLO TXT 或相关 YAML。

| XML 中原样类别名 | 主目录框数 | train_3380 框数 | test_2000 框数 |
|---|---:|---:|---:|
| `HumanDivers` | 1882 | 1131 | 751 |
| `Wrecks/ruins` | 1085 | 652 | 433 |
| `Fish/vertebrates` | 12901 | 8136 | 4765 |
| `Plant/sea-grass` | 760 | 488 | 272 |
| `Robots` | 591 | 381 | 210 |
| `Fish_group` | 731 | 449 | 282 |
| `a` | 2 | 2 | 0 |
| **总框数** | **17952** | **11239** | **6713** |

- **框格式**：整数、绝对像素 `xyxy`，并非归一化中心宽高。主目录所有框满足 `0≤xmin<xmax≤XML宽`、`0≤ymin<ymax≤XML高`。例 `1.xml` 的 HumanDivers 框为 `(853,239,1236,624)`、尺寸 1280×704。62 个框含 0 坐标，2789 个框的 xmax/ymax 等于 XML 宽/高；例 `1041.xml` 为 `(0,0,351,223)`。因此 VOC 的命名不足以证明统一的 1-based/0-based 或端点包含规则，转换时不能统一先减 1。未做全库 XML 尺寸与实际图像尺寸对照。
- **空/占位标注**：主目录 1077 份没有 object，其中 **1003 份同时为 `<filename>0.bmp</filename>`、宽高 0**；例 `1005.xml`，其对应 `Raw/1005.bmp` 实为 640×480。其余 74 份无 object 但尺寸非零。有 object 的实际文件数为 **4303**，与目录名 `4377_have_objects` 不同；4377 恰等于 5380−1003，但这只是数值关系，不能据此认定作者定义。空标签是否表示确认无目标或缺失标注仍待说明，不能自动合并处理。
- **子集问题分布**：train 中 712 份无 object（669 份零尺寸占位、43 份非零尺寸空标注）；test 中 365 份无 object（334 份零尺寸占位、31 份非零尺寸空标注）。不能按 XML 内 filename/path 连接图像：1003 份会错误指向不存在于实际集合的 `0.bmp`；应核对真实 XML stem 与 Raw stem，实际尺寸从图像取得。XML 中绝对 path 是旧 Windows 标注机路径，不是服务器路径。
- **异常类别**：`1477.xml` 的两个框 name 为 `a`，主目录与 train 子目录均存在，框分别为 `(1125,738,1180,768)`、`(1122,578,1148,660)`。不能自动纠正、删除或映射成背景；`Fish_group` 与 `Fish/vertebrates` 是否合并同样未确认。全部 difficult 为 0；主目录 truncated=1 有 4404 个框。

**划分的已知依据**：两个目录确实分别含 3380/2000 个 XML，stem 交集为 0、并集恰为 Raw 全集。每目录固定种子抽取 8 份与主目录同名 XML 做字节比较，16/16 相同；train 样本 `1235,334,3317,1952,4141,919,2899,2556`，test 样本 `1050,260,2719,1625,3476,780,2338,2043`。只证明这些是本地标注子集且抽样内容一致，未全量比较副本。未找到来源说明/独立官方 split 清单来认证它们的生成方法、随机种子、官方地位或 HSRUIE 的实际使用方式；**不得把此检测划分直接认作三阶段增强训练或语义分割划分**。

### 5. 显著性标注抽样

**实验事实**：同一组 32 张 `salient_label/*.bmp` 成功解码：26 张为 L 模式 BMP，1 张为 RGB BMP（`1826.bmp`），5 张实际为 RGB PNG（`73,1173,3570,3915,4863.bmp`，文件头均为 `89504e470d0a1a0a`）。L 图像素仅 `{0,255}`；RGB 图仅黑白且通道相同，抽样中均与 Raw 尺寸相同。**扩展名不是实际容器格式的充分证据，读取应依赖解码器，不能手工硬读 BMP 字节布局**。二值 mask 的前景/背景语义与评价阈值仍需官方确认，本次不批量改后缀、不转换数据。

### 6. README、官方类别与 split 搜索边界

**实验事实**：本地 UVTD 全部文件只见 `.bmp/.xml`；根及全部子目录未找到 README、INFO、类别表、TXT/JSON/YAML/CSV split。仓库内也未定位 UVTD 专用类别表或数据划分列表。搜索边界限于此 UVTD 根和当前仓库，没有全盘搜索，不能推断其他位置不存在。

**源码事实（官方公开说明，2026-11-09 查阅）**：[UVTD 官方 README](https://github.com/Underwater-Lab-SHU/UVTD/blob/main/README.md)（[纯文本](https://raw.githubusercontent.com/Underwater-Lab-SHU/UVTD/main/README.md)，行 1–10）仅说明公开部分为 Raw/UIE 标签及下载联系方式，没有语义颜色表、ignore 或检测划分协议。[HSRUIE 官方 README](https://github.com/gemyumeng/HSRUIE/blob/main/README.md) 提供环境、权重和测试路径，也未给出这些数据读取协议。本次搜索包括 `UVTD semantic classes`、`UVTD color segmentation labels`、`UVTD HumanDivers`、`UVTD 3380 2000`、`UVTD train_3380`、`UVTD ignore label`，未取得能确认上述映射/划分用途的官方说明。搜索无结果不等于官方从未发布；本地标注归档来源、版本及随附说明仍需追溯。

### 7. 当前 Dataset 代码兼容性（静态审查，未执行 DataLoader）

**源码事实**：以下行号以本次本地文件为准；仅报告问题，没有修正。

| 代码证据 | 对当前 UVTD 的影响 |
|---|---|
| `data/image_folder.py:12–32`、`data/single_dataset.py:19–36` | 能枚举 `.bmp` 并用 Pillow 转 RGB；应指向 `…/UVTD/Raw`。递归指向 UVTD 根会混入 Reference 和两类 mask（共 21520 张候选图），不是 5380 张 Raw。single 只返回 A/路径，不提供空气域或标签。 |
| `data/unaligned_dataset.py:26–30,46–59` | 实际只读 `<phase>A`，未读取 Flickr2K/B 域；`__len__` 却引用未定义 `self.B_size`。与注释宣称的不配对双域读取不同，不能直接满足 Stage 1。 |
| `data/base_dataset.py:81–112`、`data/__init__.py:75–79` | resize/crop/flip 被注释，活动变换仅 ToTensor 和 Normalize(0.5)。默认 collate 无尺寸适配，混合尺寸 batch=2 不能保证堆叠；设置 `crop_size=256` 并不会让当前代码裁剪。 |
| `data/aligned_dataset.py:21–22,40–46` | 假设单文件左右拼接 AB，不支持 Raw/Reference 两目录；更不能因“配对”接口就把 Reference 当 Stage 1 空气域。 |
| `unet_test/utils/dataloader.py:31–47,56–72` | 硬编码 `VOC2007/JPEGImages/<id>.bmp` 与 `SegmentationClass/<id>.png`；本地目录/后缀不同；期望单通道整数标签，没有 RGB→类别解码。粘贴到 L 图会发生灰度转换，破坏颜色编码；只改后缀不足以兼容。 |
| `unet_test/utils/dataloader.py:40`、`unet_test/nets/unet_training.py:6–15` | 把超出类别数的值设为 num_classes，并以 num_classes 为 ignore；这是源码约定，不能当作 UVTD 官方规则。RGB 的 255 值不能直接进入这条逻辑。 |
| `unet_test/utils/dataloader_medical.py:31–47` | Images/Labels PNG 路径及二值阈值编码，不能替代八色语义标签读取。 |
| `models/cycle_gan_model.py:97–107,125–128` | set_input 只收 A/路径，训练损失引用 unet_out/label；现有 Dataset 未向该路径提供语义标签。没有现成的 VOC XML 检测或显著性专用读取链。 |

### 8. 三阶段所需数据与当前缺口

**论文事实（历史原文核查）**：HSRUIE p.45 III.B、式(1)–(6)，p.46 III.C，p.47 III.D/IV.A；详见 [论文协议](paper_protocol.md)。以下“可读”只指资产与格式，不表示训练实现或复现已完成。

| 阶段 | 需要的数据 | 当前条件与缺失内容 |
|---|---|---|
| Stage 1 | UVTD Raw 水下域 + Flickr2K 空气域，不配对；256×256 随机训练裁剪 | Raw 5380 张已枚举、RGB 抽样可读；本次 `D:/Research` 顶层仍只有 HSRUIE/SUIM/UVTD，未提供 Flickr2K 路径、版本或数量（不代表全机不存在）。尚缺 Flickr2K、实际双域 Dataset、采样/划分及短边不足裁剪策略。**Reference 不替代 Flickr2K，也不默认参与配对监督。** |
| Stage 2 | UVTD Raw；自监督解耦不需要 Reference/任务标签 | Raw 资产已具备；尚缺该阶段确认的样本清单、变换与实际读取分支；继承 Stage 1 所需模型组件的认证属于模型准备，不是额外图像标签。 |
| Stage 3 | UVTD Raw + 同名 semantic_labels | 文件级 5380 对已具备，RGB 八色抽样已确认；尚缺官方类别/整数映射、ignore、任务模型通道/权重协议、独立 train/val/test 清单及同步变换规则。检测与显著性标签不是 CE 语义监督替代品。 |

Stage 1/2/3 使用哪些 UVTD ID、与下游测试集如何隔离仍待论文/作者协议确认；不能自行把 5380 全部纳入训练后又将其中 2000 张宣称为独立未见测试。SUIM TEST 110 张的最终评价边界继续有效。本次只确认读取前提和阻塞，未建立新 split。

### 9. Linux 服务器最少补充检查（计划，未执行）

1. **路径和数量**：在 `/data/czy/datasets/UVTD_import/UVTD` 核对表中各目录的大小写/层级及计数，实际枚举 stem 比较 Raw/Reference/两类 mask/主 XML 集合、检测 3380/2000 交并集；确认训练配置最终指向 `Raw` 而非整个 UVTD 根。补录 Flickr2K 的真实根路径、数量和两个可读 RGB 样本；不需全量哈希。
2. **代表性标签读取**：CPU 解码 `Raw/39.bmp`、`semantic_labels/1.bmp,1366.bmp`、`salient_label/73.bmp,1826.bmp`；确认尺寸、RGB 八色保真、伪 BMP 后缀的 PNG 能读。解析 `1.xml,1005.xml,1041.xml,1477.xml`，确认旧 path/占位 filename 不被用来找图，空标签、`a` 和坐标约定会明确报告；映射/ignore 决策缺失时不能静默进入训练。
3. **首个 Batch**：未来 Dataset 开发完成且数据协议确定后，使用 CPU、`num_workers=0`、固定 seed，分别只取一个 batch=2：Stage 1 返回独立水下/空气域张量及真实路径；Stage 2 返回 Raw；Stage 3 返回 Raw 与整数语义 `[B,H,W]`（long）并核对唯一 ID/ignore、同步裁剪。目标训练图像 `[B,3,256,256]` 的 dtype/数值范围按批准的阶段协议检查，显式纳入短边样本 39 验证既定策略；不能靠刚好相同尺寸的前两张通过。只检查 Dataset/DataLoader，不实例化模型、不调用 GPU，不运行现有会写 `test_opt.txt` 的测试入口。

**结束状态**：本地轻量审计完成；图像固定抽样 32 组及边界样本可读，XML 语法检查全部完成。中间一次汇总因混排 L 整数与 RGB 元组触发 TypeError，已统一为 RGB 元组后重跑汇总成功；不属于数据解码失败。未做全库像素有效性/质量/重复审计。Stage 1 尚未完整就绪（Flickr2K、双域读取与小尺寸裁剪阻塞）；Stage 3 文件资产已找到但编码语义/ignore 和划分仍未认证；检测划分已确认本地内容及集合关系，官方用途未认证。无训练/推理产物、无权重修改；Linux 首 Batch 检查尚未执行。
