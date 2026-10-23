# feng-30m 文档

一个 ~30M 参数中文对话模型的完整记录：从零预训练、用 27B 教师蒸馏、量化到 Q4 后
在 **ESP32-S3（32MB flash / 16MB PSRAM）上离线跑**。所有数字都是本机实测
（RTX 5060 Ti 16GB、i7-12700KF、ESP32-S3-WROOM-2-N32R16V）。

## 从哪里开始

| 你想干什么 | 看这个 |
|---|---|
| 先看看这是什么 | 根 [`README.md`](../README.md) |
| **把这个模型跑起来**（PC / 板端 / 烧录） | [`USAGE.md`](USAGE.md) |
| 数据跑完了，要把 4.0 训出来 | [`v4-pipeline.md`](v4-pipeline.md) |
| 看版本演进与每次改动的实测数字 | [`CHANGELOG.md`](CHANGELOG.md) |
| **想知道哪些坑已经踩过、哪些是修不了的** | [`v3x-lesson.md`](v3x-lesson.md) |
| 三代模型横向对比 | [`COMPARISON.md`](COMPARISON.md) |
| 交付物清单与验收口径 | [`DELIVERY.md`](DELIVERY.md) |
| ESP32 固件工程细节 | [`esp32-firmware.md`](esp32-firmware.md) |
| 改代码前的工作约定 | [`../AGENTS.md`](../AGENTS.md) |

## 全部文档

| 文件 | 内容 |
|---|---|
| [`USAGE.md`](USAGE.md) | 下载 / 推理 / 烧录 / 串口协议，以及本地一键验收 |
| [`v4-pipeline.md`](v4-pipeline.md) | **4.0 训练链路**：数据合并 → 全参 SFT → 检索回放 → 名字补丁 → 板端 QAT → 导出 → 四道验收关卡（含每步该出现什么数字、对不上说明什么） |
| [`CHANGELOG.md`](CHANGELOG.md) | 按时间倒序的完整更新日志（**失败尝试也保留**，供避免重复踩坑） |
| [`v3x-lesson.md`](v3x-lesson.md) | **v3.x 迭代复盘**：二十余次更新按问题类型归类——症状 / 根因 / 尝试 / 解法 / 教训，附**已证伪、不要再试**的路线清单 |
| [`COMPARISON.md`](COMPARISON.md) | v1 / v2 / v3 / v4 四代的架构、语料、评测横向对比 |
| [`DELIVERY.md`](DELIVERY.md) | 交付清单、验收口径、硬件约束与踩过的坑 |
| [`RELEASE.md`](RELEASE.md) | **发布产物与备份**：sha256 清单、备份什么、恢复步骤、从 PC 权重复现导出 |
| [`esp32-firmware.md`](esp32-firmware.md) | 固件工程：硬件要求、引擎结构、编译烧录、串口协议、实测结果 |
| [`board-qat.md`](board-qat.md) | **4.0-embed 的发布记录**：H 候选（4.0B + v3.19 embedding）的 QAT/soup 失败对照与最终指标 |

根目录另外两个：
- [`../README.md`](../README.md) —— 仓库首页（GitHub 会渲染它，所以留在根）
- [`../AGENTS.md`](../AGENTS.md) —— AI 编码代理的工作约定与硬性约束

模型卡跟着模型目录走：`v4/base/MODEL_CARD.md` 描述 HF 仓库（PC 4.0 + esp32 4.0-embed）；
本地最终 PC 权重在 `v4/patch_name_e5_lr5e-6/`，SFT 起点在 `v4/base/`。
v1/v2/v3 的历史模型卡随目录一并移除。

## 数字的可追溯性

核心数字（参数量、固件体积、v4 留出、SciQ/PIQA）必须能追到 `eval/*.json`、
`*/summary.json` 或 `logs/`；其余历史数字以 CHANGELOG 原文为准。
两个自检脚本守住其中可自动核对的部分：

```powershell
python tools\check_md.py      # 反引号路径是否存在、代码围栏是否成对、有没有过时表述
python tools\check_docs.py    # 关键数字（参数量 / 固件体积 / 各版评测分数）与真实产物对齐
```

改完文档两个都要跑（`python tools\check_all.py` 会一起跑，并附带引擎构建与回归）。
