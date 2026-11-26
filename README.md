# Experiment 1 · Qwen2.5-1.5B Random token selection

本实验训练一个与 CF-OPSDL 蒸馏 token 数量匹配的 Random baseline。对每条符合认证条件的回答，用原 CF 判据计算应选的 token 数 K，再从该回答的有效位置中均匀、无放回地抽取 K 个位置做蒸馏。损失沿用原实验的分母。两次教师前向都参与计算 K，训练只产生 Random 这一组模型。

| 设置 | `pilot` 四卡试跑 | `full` 正式实验 |
|---|---:|---:|
| 训练 / 验证样本 | 12 / 3 | 21,005 / 20 |
| 每步问题数 × 每题回答数 | 8 × 8 | 512 × 8 |
| PPO mini-batch（按问题计，过滤前） | 8 | 512 |
| 训练步数 | 1 | 41，1 个 epoch |
| 检查点 | 关闭 | 10、20、30、40、41 |

两档均使用 4 张 GPU、FSDP、rollout TP=1、学习率 2e-6，输入 / 输出上限为 16,416 / 8,192 token。蒸馏系数 μ=0.03，门限 τ=0.1，参考策略 KL 系数 0.001，教师每 5 次优化器更新同步一次。每题全对或全错时丢弃该组回答，整个 batch 都被过滤时，该训练步跳过参数更新。正式配置采用历史分片顺序、shuffle 和原始 data seed 设置，pilot 固定 data seed 为 42。历史 CF 使用 8 卡、TP=2；两者的正式全局 batch 和训练步数相同。参数详见 [random.yaml](configs/random.yaml)。

## 安装

进入本目录。推荐在一台 Linux x86_64 机器使用 Python 3.12、支持 CUDA 12.8 的 NVIDIA 驱动和带 `nvcc` 的 CUDA toolkit。小样本试跑使用 4 张 48 GiB RTX 6000 Ada。正式档默认给 Ray 对象存储分配 32 GiB，共享内存和主机内存应能容纳此分配及训练进程。可用 `RAY_OBJECT_STORE_BYTES` 覆盖。将运行目录放在有空间保存 5 个分布式检查点的磁盘上。

```bash
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

最后一条命令核对完整数据的 SHA-256，可在启动 GPU 前执行。锁文件固定了 Linux 的 PyTorch 2.8.0 / CUDA 12.8、vLLM 0.11.0、Transformers 4.57.6 等依赖。CuPy 和 OpenCV 选择了与 NumPy 1.26 兼容的版本。FlashAttention 在安装 PyTorch 后构建。启动器从仓库内的 `RLCSD/third_party/verl` 导入定制训练代码。

## 运行

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
./run.sh pilot
./run.sh full
```

两条命令分别从同一基础模型开始。首次运行自动下载并校验 `Qwen/Qwen2.5-1.5B-Instruct` 的 revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。离线机器可将 `MODEL_PATH` 指向这个 revision 的本地快照；程序校验权重、配置和 tokenizer 内容。正式任务可在 tmux 或作业调度器中运行。

在 **mlnlp** 上，使用以下已通过四卡试跑的环境设置，再执行上面的启动命令。

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_HOME=/usr/lib/cuda
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo
```

## 数据和结果

- `data/full/` 含 v5 counterfactual 数据的全部 11 个原始 Parquet、原始 manifest 和 SHA-256 清单。训练由 HotpotQA、MuSiQue、2Wiki 的六个 QA 分片，以及 MathQA、DAPO、RULER multikey 和 multivalue 组成。验证分片是 MathQA 的 20 条记录；历史配置最多加载 1 条，训练中关闭验证。21,005 条训练样本形成 41 个完整的 512 题 batch，最后 13 条按原训练规则丢弃。
- `data/pilot/` 含 HotpotQA、MuSiQue、2Wiki 各 4 条认证训练样本和合计 3 条验证样本。它们用于检查四卡通信、回答生成、奖励过滤和参数更新。
- 启动器打印本次运行目录。运行 `python scripts/report.py runs/full/<运行目录>` 可查看各步耗时、过滤后保留的回答数、四个 rank 的优化器记录和各卡显存峰值。显存按 0.25 秒采样，分别统计本任务进程与整卡用量。原始记录位于该目录下的 `gpu.csv`、`optimizer_rank*.jsonl` 和下一级时间戳目录的 `metrics.jsonl`。最终检查点在同一时间戳目录的 `global_step_41/actor/`。
- 交付实验结果时，请带上 step 41 检查点、`trial.yaml`、`metrics.jsonl`、`gpu.csv` 和四份优化器记录，并注明 GPU 型号。五项 LongBench QA 历史 CF 均分为 46.64，对应 step 41；本目录交付的是 Random 训练任务。

[已完成的 pilot 指标](reference/pilot_metrics.jsonl) 和 [参数变化及显存记录](reference/pilot_hardware.json) 来自一次四卡单步实测，单步 70.54 秒，8 题中 1 题通过组过滤，4 个 rank 均记录到参数变化。正式 Random 的 41 步仍待运行。

## 上传到 GitHub

本目录可以单独作为仓库根目录，也可以作为已有仓库的 `experiment1/` 子目录。完整 Parquet 约 1.43 GiB，已配置 Git LFS。首次在目标 Git 仓库执行 `git lfs install --local`，添加文件后用 `git lfs ls-files` 确认列出 **11 个**完整数据分片，再提交和推送。小样本直接存入 Git。模型缓存、训练输出和虚拟环境已由本目录的 `.gitignore` 排除。RLCSD 和 veRL 的上游许可保存在各自源码目录。
