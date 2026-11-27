# Qwen2.5-1.5B 蒸馏 token 选择对照实验

三组实验从同一 Qwen2.5-1.5B-Instruct 快照分别训练，共用 v5 的 21,005 条 GRPO 训练样本、相同的奖励与过滤规则、蒸馏系数 **μ=0.03**。14,981 条多跳 QA 中有 **10,320 条**通过反事实认证。三组都只在通过认证的 QA 回答上计算蒸馏，使用金证据教师和相同的损失分母（符合条件的有效回答 token）。正式比较使用各组的 **step 41**。[数据清单](data/full/manifest.json)

| 启动参数 | 每条认证回答的蒸馏位置 | 教师前向 |
|---|---|---|
| `random` | 用 CF 门限计算该回答的 K，均匀、无放回地选 K 个有效位置 | 金证据与反事实各一次 |
| `kl-topk` | 用同一规则计算 K，按学生对金证据教师的逐位置反向 KL 从大到小选 K 个；同分按位置先后排序 | 金证据与反事实各一次 |
| `certified-dense` | 该回答的全部有效位置 | 仅金证据 |

两个稀疏实验都在本组当前生成的回答上计算 K。反向 KL 沿用原蒸馏实现，在教师概率最高的 100 个词和一个合并的尾部词表桶上计算。词表大小 100 和回答中选出的位置数 K 是两个不同的量。

## 训练

在 OBX 上运行时，按[英文提交指南](obx/README.md)为每组实验提交一个单节点 8×H200 作业。作业会训练到 step 41，合并 checkpoint，完成长、短上下文评测，并将结果写入 S3。

| 参数 | 正式实验 `full` | 小样本试跑 `pilot` |
|---|---:|---:|
| GPU / rollout tensor parallel | 8 / 2 | 4 / 1 |
| 训练 / 验证文件中的样本数 | 21,005 / 20 | 12 / 3 |
| 每步问题数 × 每题采样数 | 512 × 8 | 8 × 8 |
| PPO mini-batch（过滤前的问题数） | 512 | 8 |
| actor micro-batch / 每卡 | 2 | 1 |
| vLLM GPU memory utilization | 0.70 | 0.50 |
| 训练步数 | 41（1 epoch） | 1 |
| 保存检查点 | 10、20、30、40、41 | 关闭 |

正式训练使用 FSDP、FP32 主权重、学习率 2e-6、KL 系数 0.001、蒸馏系数 μ=0.03、门限 τ=0.1、教师 top-100、每 5 次优化器更新同步一次快照。输入 / 输出上限为 16,416 / 8,192 token。每个 rollout DP 组处理 128 道题。8 张卡分为 4 个 TP=2 的组，每步生成 4,096 条回答。每题全对或全错时丢弃该组，整步都被过滤时跳过更新。[训练配置](configs/random.yaml)

### 安装与启动

先在 GitHub 获得本私有仓库的读取权限，在 Linux x86_64、带 CUDA 12.8 toolkit 和 nvcc 的训练机上执行下列命令。训练环境沿用已通过四卡试跑的 Python 3.12、PyTorch 2.8.0、vLLM 0.11.0。正式档恢复历史训练参数；八卡机器的型号、单卡显存、CPU 和互联会影响实际耗时与容量。

```bash
git clone https://github.com/FYYFU/qwen25-random-token-baseline.git
cd qwen25-random-token-baseline
git lfs install --local
git lfs pull
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements-linux-cu128.lock
MAX_JOBS=4 python -m pip install --no-build-isolation flash-attn==2.8.1
python -m pip check
python scripts/prepare.py full
```

三组可以分发到三台机器，每位协作者在完成安装后运行分配给自己的那一条。共用一台八卡机器时，依次执行三条命令。

```bash
./run.sh full random
./run.sh full kl-topk
./run.sh full certified-dense
```

正式档默认使用 GPU 0–7，也可用 `CUDA_VISIBLE_DEVICES` 指定八张卡。每次启动会创建独立运行目录，如 `runs/full/kl-topk_<时间戳>.<ID>`，其中的 `trial.yaml` 记录实际配置。旧命令 `./run.sh full` 等价于 `./run.sh full random`。

首次运行自动下载并校验基础模型 revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。离线运行时，可设置 `MODEL_PATH` 指向该 revision 的本地快照。训练器和三个对照的实现均随仓库交付。正式档使用历史分片顺序、shuffle 和原 data seed，默认由 Ray 分配对象存储。每组训练会保存 5 个分布式检查点。

如需先验证环境，可将上述命令中的 `full` 换成 `pilot`，例如 `CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh pilot kl-topk`。试跑独立从基础模型开始，固定 data seed 为 42。历史 Random 四卡 RTX 6000 Ada 试跑中，单步耗时 70.54 秒，8 题中有 1 题通过过滤，4 个 rank 均测到参数变化。[试跑指标](reference/pilot_metrics.jsonl) · [各卡显存与权重变化](reference/pilot_hardware.json)

### 数据与输出

- `data/full/` 含 v5 counterfactual 的 10 个训练 Parquet、1 个验证 Parquet、原 manifest 与校验和。训练包括 HotpotQA、MuSiQue、2Wiki 的六个 QA 分片，以及 MathQA、DAPO、RULER multikey 和 multivalue。每个 epoch 包含 41 个完整 batch，末尾 13 条按原规则丢弃。验证文件含 20 条 MathQA，历史配置最多加载 1 条，训练中关闭验证。
- `data/pilot/` 含三类 QA 各 4 条认证训练样本及合计 3 条验证样本。正式训练和全部评测数据均随 Git LFS 交付。
- 启动器会打印形如 `runs/full/<run-id>` 的目录。运行 `python scripts/report.py runs/full/<run-id>` 可查看每步耗时、过滤后保留回答数、各 rank 实际测到的参数变化次数，以及每张卡的显存峰值。显存按 0.25 秒采样，同时记录本任务及整卡用量。
- 交付结果时保留 `trial.yaml`、`metrics.jsonl`、`gpu.csv`、各 rank 的 `optimizer_rank*.jsonl` 和 step 41 actor，并注明 GPU 型号。交付时标明实验名，并分别保存三个 step 41 的评测结果。

## 评测第 41 步

[完整评测步骤与参数](eval/README.md) 包含 FSDP 合并、五项 LongBench QA、LongReason 8K–64K 和六项短上下文评测的启动命令。LongBench QA 使用 AWS Bedrock GPT-OSS-120B 判分。LongReason 与短上下文评测均可用本地数据打分。

作为核对，历史 1.5B CF 的 step 41 在 LongBench QA 上均分为 **46.64**，LongReason 均分为 **43.38**。这两组数值已由历史逐条记录回算。[历史 CF 各项成绩](reference/step41_cf_scores.json)
