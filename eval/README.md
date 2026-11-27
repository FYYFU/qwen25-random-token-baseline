# Qwen2.5-1.5B step 41 评测

从仓库根目录运行。长上下文分数按 0–100 报告，短上下文的原始 lm-eval 指标按 0–1 保存。

## 1. 合并训练检查点

在训练环境中，将 `RUN_DIR` 设为 对应实验的 `./run.sh full <实验名>` 打印的运行目录。合并读取八个 FSDP rank 的 actor 分片，在 CPU 上生成供 vLLM 加载的模型和 tokenizer。

```bash
source .venv/bin/activate
RUN_DIR="runs/full/<run-id>"
ACTOR=$(find "$RUN_DIR" -type d -path '*/global_step_41/actor' -print -quit)
MODEL="$RUN_DIR/merged-step41"
RESULTS="$RUN_DIR/eval-step41"
python eval/run.py merge "$ACTOR" "$MODEL"
```

分别评测三个实验，每次将 `<run-id>` 替换为带有该实验名前缀的实际目录名。后续命令沿用同一个 shell 中的 `MODEL` 和 `RESULTS`。

## 2. 准备 LongBench QA 评测环境

本次 OBX 作业只评测 LongBench QA。单独建立 Python 3.11 环境，使用历史生成版本 vLLM 0.12.0 和 Bedrock 判分依赖。

```bash
python3.11 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r eval/requirements-generate.txt -r eval/requirements-judge.txt
python -m pip check
(cd data/eval && awk '$2 ~ /^longbench\//' checksums.sha256 | sha256sum -c -)
```

## 3. LongBench QA

```bash
CUDA_VISIBLE_DEVICES=0 python eval/run.py generate --longbench-only "$MODEL" "$RESULTS"
python eval/run.py judge "$RESULTS"
python eval/run.py report --longbench-only "$RESULTS"
```

- **生成。** LongBench 使用一张 GPU、bf16、TP=1、vLLM 显存比例 0.92、temperature=1、top-p=1、每题 5 个样本、基础 seed=0、每题 seed 加题号、生成上限 8,192 token、16 个客户端线程。重复运行会复用同一模型已完成且通过样本检查的生成记录。
- **LongBench QA。** 2WikiMQA、HotpotQA、MuSiQue、NarrativeQA、Qasper 各 200 题，共 5,000 条回答。沿用 `think` + `passages` 模板、32,768 token 输入预算和中间截断。每题从回答中解析最终答案，对样本取 `max(sub_EM, ORM)`，先对任务内样本平均，再等权平均五项。
- **LongBench 判分。** `judge` 会用当前 AWS 身份调用 Bedrock，在 `ap-south-1` 使用 `openai.gpt-oss-120b-1:0`，会产生调用费用。需要能够调用该模型的 AWS profile、环境凭据或实例角色。仅将需要判分的题目、模型答案和参考答案发给判分模型。temperature=0，输出上限 512 token，64 个线程。中断或凭据失效时可重新执行，跳过已完成的判定。未完成判分时 `report` 会报错。
- **结果。** 逐条回答和判分记录位于 `$RESULTS/longbench/model/`，五项得分及等权均分写入 `$RESULTS/scores.json`。脚本检查每个题号与采样号是否齐全、参考答案是否对齐，以及是否存在请求失败或未解决的判分。

### 历史 LongReason 复现

历史全套长上下文复现可使用完整评测数据，运行上述 `generate` 和 `report` 命令时去掉 `--longbench-only`。该模式同时生成 LongBench 和 LongReason，最多并行使用五张 GPU。LongReason 固定修订 `2b080b67248d3faa575a143c2d300e6fa1bb21c9`。8K、16K、32K、64K 各有相同的 794 个题号，每档 3,970 条回答。使用 `think_mcq` 提示和未编号的上下文，按 lenient 规则提取选项，等权平均四档准确率。保留 1,024 token 余量，超预算样本按历史规则记零。

| 评测单元 | 服务窗口 | max-num-seqs | YaRN |
|---|---:|---:|---:|
| LongBench QA | 41,984 | 32 | ×2 |
| LongReason 8K | 24,576 | 32 | 原生 |
| LongReason 16K | 32,768 | 32 | 原生 |
| LongReason 32K | 41,984 | 32 | ×2 |
| LongReason 64K | 73,728 | 16 | ×3 |

## 4. 历史六项短上下文评测

```bash
python -m pip install -r eval/short/requirements.txt
python -m nltk.downloader punkt_tab
CUDA_VISIBLE_DEVICES=0 python eval/short/run.py "$MODEL" "$RESULTS/short"
```

这个入口直接读取随仓库提供的离线数据，使用标准 Hugging Face chat template、bf16、TP=1、窗口 12,288、显存比例 0.90 和 seed=0。分别运行 MMLU 与其余五项，并验证 MMLU 的全部 57 个科目。原始 lm-eval 结果、样本和 `scores.json` 均在 `$RESULTS/short` 下。[历史数据版本](short/dataset_revisions.tsv) · [历史软件版本](short/software_versions.tsv)

| 任务 | Few-shot | 报告指标 | 测试样本 |
|---|---:|---|---:|
| MMLU | 5 | acc | 14,042 |
| ARC-Challenge | 25 | acc_norm | 1,172 |
| HellaSwag | 10 | acc_norm | 10,042 |
| WinoGrande | 5 | acc | 1,267 |
| Minerva MATH | 4 | math_verify | 546 |
| IFEval | 0 | prompt-level strict / instruction-level strict | 541 |

历史结果中的 Minerva MATH 实际对应 `precalculus` 的 546 道题，生成采用 greedy、上限 3,072 token，以 `Problem:` 为停止串。入口显式固定这个子集。历史各项分数见[step 41 CF 对照](../reference/step41_cf_scores.json)。
