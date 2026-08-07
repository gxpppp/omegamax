# omigamax

**一个从零手写的 AlphaGo Zero 风格围棋 AI 引擎。** 规则引擎、策略-价值神经网络、MCTS 树搜索、自我对弈训练闭环全部原创实现，无任何开源棋手代码/权重作为本体；KataGo 仅作评测对手。训练过程中 pygame 窗口实时展示黑 vs 白自对弈棋局与训练指标，训练出的模型可通过标准 GTP 引擎协议对外对弈（为日后联机平台接入预留标准接口）。

- 语言/栈：Python 3.12 + PyTorch（CUDA 13）+ uv
- 硬件基线：单卡 RTX 3060 Laptop GPU 6GB（b10c128，约 3M 参数网络可训）
- 路径：`omigamax/` 包（rules / network / mcts / train / viz / gtp / cli 六组件）
- 文档：配置详解 → [`docs/config.md`](docs/config.md) ｜ 训练运维 → [`docs/ops.md`](docs/ops.md)

## 当前状态（诚实说明）

以下为 todo 21 端到端冒烟实测结果（见 `logs/e2e_report.md`）：

| 项目 | 状态 |
| --- | --- |
| 全链路可运行 | ✅ 自对弈→训练→评估门控→可视化→GTP→评测 完整跑通 |
| 训练循环冒烟 | ✅ ≥2 checkpoint（`models/latest.pt` + `models/best.pt`）、日志含 step/loss/elo、中断恢复 step 接续 |
| 可视化 | ✅ pygame 实时窗口 + `--capture` 无头截图（`SDL_VIDEODRIVER=dummy`） |
| GTP 引擎 | ✅ 完整 GTP v2 命令集（`protocol_version`…`kgs-time_settings`/`final_score`/`printsgf`/`undo`） |
| 评测 | ✅ vs-random 可出胜率；vs-KataGo 自动对弈（GTP 管道）可出胜率/SGF |
| **棋力** | ⚠️ **弱信号：vs 随机落子 9/20 = 45%（sims=40，3200 步训练后实测）**——早期里程碑"vs-random >80%"未达成（不阻塞；1000 局为长训目标） |
| **todo-12 门控（模拟数-棋力单调）** | ⚠️ **未确认**——点估计方向性支持（P(200>40)=1.0, P(800>200)=1.0）但样本远低于 60 局/配对 的关键门槛，95% 置信区间跨 0.5；正式判定需 `uv run python -m omigamax.cli.mcts_strength --games 60`（详见 [`docs/ops.md`](docs/ops.md)） |

## 安装

前置：Windows 10、Python 3.12、[uv](https://docs.astral.sh/uv/)、NVIDIA 驱动（CUDA 13 兼容，RTX 3060 Laptop GPU 实测驱动 581.08）。

```powershell
cd E:\AAAhuancun\betamaster
uv sync                       # 创建 .venv 并安装依赖（torch==2.13.0+cu130 / numpy / pygame / pyyaml / pytest）
```

## 快速开始（四条命令）

> 下述四条命令即 todo 22 验收命令：环境 / 训练冒烟 / 可视化冒烟 / 评测 1 局，逐一执行成功（exit 0）。

**1. 环境验证**（确认 CUDA 可用、pygame/yaml 可导入）：

```powershell
uv run python -c "import torch,pygame,yaml;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
# 期望输出：2.x.x+cu130 True NVIDIA GeForce RTX 3060 Laptop GPU
```

**2. 训练冒烟**（低配置完整跑一轮：自对弈→训练→评估门控，产出 `latest.pt` + `best.pt`）：

```powershell
uv run python -m omigamax.train.loop --smoke
```

**3. 可视化冒烟**（无头截图一帧，验证渲染链路）：

```powershell
uv run python -m omigamax.cli.viz_smoke --capture logs/viz_capture.png
```

**4. 评测 1 局**（omigamax vs 随机落子引擎 1 局，输出胜率）：

```powershell
uv run python -m omigamax.cli.match --engine2 random --games 1 --sims 40
```

完整测试：`uv run pytest -q`（431 用例，全绿）。

## CLI 一览

| 命令 | 用途 |
| --- | --- |
| `python -m omigamax.train.loop [--smoke] [--resume] [--cycles N]` | 训练主循环（可中断恢复） |
| `python -m omigamax.train.selfplay --games N --simulations S` | 自对弈生成 (s, π, z) 样本 |
| `python -m omigamax.cli.match --engine2 katago\|random --games N [--sims S]` | 自动对弈评测（胜率/ELO/SGF） |
| `python -m omigamax.cli.play --model models/best.pt [--vs omigamax\|random] [--max-moves N]` | 终端人机对弈（两连 pass 终局结算；`--max-moves` 达上限强制结算，默认 300） |
| `python -m omigamax.cli.gtp_main [--model models/best.pt] [--simulations S]` | 标准 GTP 引擎（stdin/stdout 协议） |
| `python -m omigamax.cli.e2e_smoke` | 端到端全链路冒烟（零到可对弈模型） |
| `python -m omigamax.cli.viz_smoke [--capture png]` | 可视化冒烟/无头截图 |
| `python -m omigamax.cli.mcts_strength --games 60` | 模拟数-棋力单调验证（todo-12 门控） |
| `python -m omigamax.cli.rule_consistency --games 100` | vs KataGo 规则一致性验证 |
| `python -m omigamax.cli.smoke_net [--mcts-sims 200]` | 网络冒烟训练与显存验证 |

## 架构简述（六组件 + 数据流）

```
          ┌────────────── 自对弈 → 训练 → 评估 闭环 ──────────────┐
          │                                                      │
  rules   │   network    mcts            train                   │
 ┌──────┐ │  ┌────────┐ ┌─────────┐   ┌───────────┐   viz        │
 │棋盘/气│ │  │ResNet  │ │UCB选择  │   │ selfplay  │ ┌─────────┐ │
 │劫/计分│◄─┼─ │17平面  │ │Dirichlet│◄─ │ replay    │ │pygame  │ │
 │SGF   │ │  │policy/ │ │温度/虚拟 │   │ buffer    │ │实时棋盘 │ │
 └──────┘ │  │value   │ │损失/批推 │   │ train/eval│ │+指标    │ │
          │  └────────┘ │理      │   │ loop(可恢复)│ └─────────┘ │
          │             └─────────┘   └───────────┘              │
          │                                                      │
          └──────────────► gtp / cli (对弈评测接口) ◄────────────┘
```

- **`rules/`**：19×19 棋盘、落子合法性（气/提子/自杀禁止）、simple-ko、两连 pass 终局 + Tromp-Taylor 计分（komi=7.5）、SGF FF[4] 导出；尺寸参数化，9×9/13×13 供测试。
- **`network/`**：ResNet 策略-价值网络（b10c128：10 残差块 × 128 通道，policy 头 362 logits + value 头 tanh），输入 AGZ 17 平面特征；`create_model(blocks, channels, board_size)` 工厂。
- **`mcts/`**：AGZ 风格 MCTS——UCB（c_puct 可配）、Dirichlet 根噪声、温度调度、虚拟损失、叶子批推理（`leaf_batch=16`）；`batched_evaluator` 批前向。
- **`train/`**：自对弈生成器（每步记录 17 平面特征 + 搜索策略 π + 结果 z，`data/selfplay/*.npz`）→ 回放缓冲（`replay_buffer_games=1000` 磁盘后备）→ 训练步（SGD 动量 0.9 / lr 0.2 分段调度 / L2 1e-4 / 8 向对称增强 / FP16 可选）→ 评估门控（55% 阈值替换 `best.pt`，ELO 记入 `logs/eval_history.jsonl`）→ 主循环（100 局自对弈 → 1000 训练步 → cycle 末评估；Ctrl+C/SIGBREAK 优雅中断，`--resume` 续跑）。
- **`viz/`**：pygame 独立线程窗口——19×19 棋盘、当前局面/落子/手数/胜率估计 + 右侧指标面板（对局数/训练步/loss/ELO 趋势）；无 GUI 自动降级纯日志；`--capture` 无头截图。
- **`gtp/` + `cli/`**：完整 GTP v2 引擎（含 `kgs-time_settings` 解析、`final_score`、`printsgf`、`undo`、`loadsgf`、handicap 序列；畸形输入健壮性表驱动测试保障）；CLI 对弈/评测/冒烟入口。

## 联机平台预留说明

- **GTP 已就绪**：`python -m omigamax.cli.gtp_main` 实现标准 GTP v2（GNU Go 规范），覆盖 KGS 类平台 bot 接入所需命令集（`boardsize`/`clear_board`/`play`/`genmove`/`kgs-time_settings`/`time_left`/`final_score` 等），未知命令返回 error 不崩溃、对弈中 `quit` 随时可退。
- **后期接入路径**：平台侧（KGS/OGS/野狐/弈城类，具体平台未定）通过 GTP 进程/管道驱动 `gtp_main.py`；`kgs-time_settings` 当前实现为简化搜索预算映射 stub，**byo-yomi 完整读秒时钟为明确延展项**，留待平台接入阶段实现。

## 配置

全部超参与默认值见 [`config/default.yaml`](config/default.yaml)，逐键说明（类型/默认值/含义）见 [`docs/config.md`](docs/config.md)。运行时可用 `--config <path>` 传自定义 YAML。

## 运维

24/7 长训注意事项、中断/恢复、checkpoint/日志位置、可视化技巧、硬件速度预期（实测 ~88-160 sims/s vs 计划 300-600）、故障排查（CUDA/首步延迟/Windows GTP 管道）→ 见 [`docs/ops.md`](docs/ops.md)。
