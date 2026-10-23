# 发布产物与备份（RELEASE）

> **一句话**：权重不进 git（14 GB），但**哈希和恢复步骤必须进**。
> 没有这份文档，换台机器或硬盘坏掉之后，项目就只剩推理代码——模型没了，
> 而且无法判断手上那份是不是当初发布的那份。

## 1. 什么是「发布产物」

| 产物 | 路径 | 大小 | 用途 |
|---|---|---|---|
| 手机 / ESP32 权重 | `board/model_export_4.0-embed/model.bin` | 14.9 MB | 板端烧 `0x110000`；手机 `adb push` |
| 分词器 | `board/model_export_4.0-embed/tokenizer.bin` | 0.4 MB | 板端烧 `0x1000000`；手机同目录 |
| 对拍基准 | `board/model_export_4.0-embed/ref_logits.bin` | 64 KB | `check_all` 的 `pc_check` 项 |
| PC 权重 | `v4/patch_name_e5_lr5e-6/model.safetensors` | 112 MB | 复现导出、重跑基准与长文评测 |

权威清单在 [`weights/manifest.json`](../weights/manifest.json)（含 sha256、字节数、角色）。
校验：

```bash
python tools/verify_weights.py            # 缺文件只警告（干净 clone 的正常状态）
python tools/verify_weights.py --strict   # 发版前 / 备份校验：缺文件即失败
```

当前发布口径（2026-10-23 校验一致）：

```
b159fd8e…e48c0  model.bin
7c4eff87…651f0  tokenizer.bin
3ef57300…ac49   ref_logits.bin
f69fc934…ccc0   v4/patch_name_e5_lr5e-6/model.safetensors
```

## 2. 必须备份什么

按「丢了会痛」排序：

1. **`board/model_export_4.0-embed/`** —— 15 MB，丢了要重跑整条导出链（还依赖 PC 权重）。
2. **`v4/patch_name_e5_lr5e-6/`** —— 114 MB，PC 发布权重，是导出的上游。
3. **`v4/base/`、`v4/sft_pc2/`** —— 训练底座与第一次全参 SFT（各 ~1.8 GB）。
   丢了不影响推理，但**无法再复现 4.0 的训练过程**。
4. `data/`（1.4 GB 语料）—— 同上，只在重训时需要。

`v4/` 里 14 GB 的 `4.0-20261018A~H`、`qat32k_trial`、`moe_*` 都是**实验中间产物**，
不进备份也能接受（结论已写进 `CHANGELOG.md`）。

推荐做法（任选其一，关键是**离线**一份）：

```bash
# 打包最小可发布集（约 130 MB）
tar -czf feng-4.0-release-$(date +%Y%m%d).tar.gz \
    board/model_export_4.0-embed v4/patch_name_e5_lr5e-6 weights/manifest.json

# 全量（约 17 GB，含训练中间产物）
tar -czf feng-4.0-full-$(date +%Y%m%d).tar.gz board/model_export_4.0-embed v4 data
```

放到外置硬盘 / 对象存储 / HuggingFace 私有仓库后，**在下面登记位置**（别只存在脑子里）：

| 备份位置 | 内容 | 日期 |
|---|---|---|
| （待填） | 最小发布集 | |
| （待填） | 全量 | |

## 3. 恢复后怎么确认

```bash
python tools/verify_weights.py --strict     # 4/4 一致才算恢复成功
cd pc && ./pc_check.exe ../board/model_export_4.0-embed
# 期望：logits max|diff| <= 3.50，argmax MATCH，续写「你好！有什么我可以帮你的吗？」
```

板端恢复（COM19 烧固件、COM20 跑验收）：

```bash
# 固件
python board/tools/idf_launch.py --port COM19 flash
# 模型 + 分词器
python -m esptool --chip esp32s3 --port COM19 --baud 921600 write_flash \
    0x110000 board/model_export_4.0-embed/model.bin \
    0x1000000 board/model_export_4.0-embed/tokenizer.bin
# 验收（注意：应用层 I/O 在 UART0 = COM20，COM19 只有 ROM 引导输出）
python scripts/esp32_multi.py --port COM20
```

## 4. 从 PC 权重复现导出

```bash
# fp16 safetensors -> 板端 Q4 打包格式（含 export_info.json 与 ref_logits.bin）
python tools/export_board.py --model v4/patch_name_e5_lr5e-6 \
    --out board/model_export_4.0-embed --ctx 2048 --keep 200
python tools/verify_weights.py --strict
```

导出的 `export_info.json` 会自己记下 `model_sha256` / `tokenizer_sha256`；
如果与 `weights/manifest.json` 不一致，说明权重或导出参数变了——**不要直接发布**，
先按 `docs/v4-pipeline.md` 的四道关卡重跑验收。

## 5. 已知缺口（如实记录）

- 权重目前**没有任何远端副本**，只有本机磁盘。这份文档是补救的第一步，
  真正的备份动作还需要人去做（上面的表格待填）。
- 训练用的教师数据（36,916 条蒸馏数据）与 `data/` 语料也没有远端副本。
- 本清单只覆盖 4.0 / 4.0-embed；更早的 v3.x 产物在 `archive/`，未做哈希登记。
