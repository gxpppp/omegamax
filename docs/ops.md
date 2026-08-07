# omigamax 训练运维手册（Windows）

面向 24/7 长训的实操手册：训练如何启动/中断/续跑、产物位置、可视化、对弈评测、硬件速度预期与故障排查。对应配置键详见 [`config.md`](config.md)。

---

## 1. 训练循环：启动 / 中断 / 续跑

### 启动

```powershell
# 默认节奏：每 cycle 100 局自对弈 → 1000 训练步 → 评估门控（cycle 末必评估）
uv run python -m omigamax.train.loop

# 只跑 1 个 cycle（测试）
uv run python -m omigamax.train.loop --cycles 1

# 低配置冒烟（sims=40, batch=32, 强制最终评估；验收用）
uv run python -m omigamax.train.loop --smoke

# 显式关闭可视化线程（--viz off）；config viz_enabled=false 亦生效
uv run python -m omigamax.train.loop --viz off
```

### 中断（优雅停止）

- **Ctrl+C（KeyboardInterrupt）或 Ctrl+Break（SIGBREAK）**：均被捕获，立即把 `models/latest.pt` 落盘——权重、SGD 优化器状态、`global_step`、缓冲采样 RNG 状态、进行中 cycle 进度（已生成局数/已训步数），并 flush JSONL 日志后退出。
- 硬杀（`taskkill /PID <pid>`）最多丢失当前 cycle 未 checkpoint 的部分。

### 续跑

```powershell
# 从 latest.pt + data/selfplay 续跑，step 精确接续（确定性续跑）
uv run python -m omigamax.train.loop --resume --cycles 1
```

> Windows 提示：程序化向子进程发 SIGINT 不可靠，优雅停止路径依赖控制台 Ctrl+C/Ctrl+Break（已在代码中同时捕获 `KeyboardInterrupt` 与 `SIGBREAK`）。

### 产物位置

| 产物 | 路径 | 说明 |
| --- | --- | --- |
| 训练日志 | `logs/train.jsonl` | 每训练步 1 行：`step/loss/lr/games(缓冲局数)/elo/timestamp` |
| 评估历史 | `logs/eval_history.jsonl` | 每次评估：step/胜率/ELO 增量/是否替换 best |
| 最新权重 | `models/latest.pt` | 网络 + 优化器 + global_step + RNG + cycle 进度（恢复用） |
| 最佳权重 | `models/best.pt` | 评估门控（胜率 ≥ 0.55）后更新；对弈/评测默认加载它 |
| 自对弈样本 | `data/selfplay/*.npz` | 每局 (features 17 平面, π 搜索策略, z)；`replay_buffer_games=1000` 局内保留 |
| 对弈 SGF | `logs/matches/*.sgf` | `match` 自动对弈产物 |

---

## 2. 可视化（pygame）

训练时默认挂载 pygame 线程：19×19 棋盘 + 黑/白自对弈 + 当前局面/落子/手数/胜率估计，右侧指标面板（对局数/训练步/loss/ELO 趋势）。

- **关窗不中断训练**：点 X / ESC 只停可视化线程，训练继续。
- **无 GUI 自动降级**：无显示器/pygame 初始化失败 → 纯日志模式，不崩溃。
- **无头截图（agent/CI/RDP 验证）**：`SDL_VIDEODRIVER=dummy` 离屏渲染一帧：

```powershell
uv run python -m omigamax.cli.viz_smoke --capture logs/viz_capture.png
# 交互式冒烟（50 快照渲染 5 秒后程序化关窗，断言生产者无异常）
uv run python -m omigamax.cli.viz_smoke --frames 50 --seconds 5
```

---

## 3. 对弈 / 评测 / GTP

```powershell
# omigamax vs 随机落子（vs-random 里程碑命令；--sims 默认取 config simulations=200）
uv run python -m omigamax.cli.match --engine2 random --games 20 --sims 40

# omigamax vs KataGo（官方权重，GTP 子进程驱动；需 tools/katago 下的二进制+权重）
uv run python -m omigamax.cli.match --engine2 katago --games 9

# 终端人机对弈
uv run python -m omigamax.cli.play --model models/best.pt --vs random

# vs 引擎（MCTS）人机对弈；--max-moves 达上限强制终局结算
uv run python -m omigamax.cli.play --model models/best.pt --vs omigamax --board-size 19 --max-moves 60

# 标准 GTP 引擎（stdin/stdout；供平台/外部 GUI 驱动）
uv run python -m omigamax.cli.gtp_main --model models/best.pt --simulations 200
```

对局以两连 pass 正常终局（Tromp-Taylor 结算）。人类 pass 后若引擎仍落子（弱引擎几乎不主动 pass），会提示
`[info] engine did not pass -- game continues (pass twice in a row to end)`，再 pass 一次即可终局；
`--max-moves`（默认 300）兜底：达到上限强制结算并输出得分，避免弱引擎对局无限拖到 1000+ 手。

KataGo 启动依赖 `tools/katago/` 下的官方 Windows 二进制与权重（todo 5/20 已下载；.gitignore 排除 tools/ 不提交）。

---

## 4. 硬件速度预期（重要）

**实测与本机性能直接挂钩，与计划预估有偏差：**

- **实测吞吐 ~88-160 sims/s**（计划预估 300-600 sims/s）。纯 Python MCTS + 6GB GPU 网络前向是本机瓶颈。
- **单局时长**：19×19 自对弈弱模型对局实测约 2.9 分钟/局（弱模型局常跑到 `--max-moves` 上限，默认 300）。
- **训练时间预期**：
  - `loop --smoke`（2 局 + 25 步 + 强制评估）：分钟级，可在 3060 上完整跑通。
  - 默认 cycle（100 局 + 1000 步 + 评估）：以 ~100 sims/s 估算，单 cycle 数小时级（`simulations=200` 时更慢）。
  - todo 21 端到端冒烟（16 局 + 3200 步 + 5 次门控 + 限时 todo-12 重跑 + 20 局 vs-random）：实测 **179.5 分钟**（峰值显存仅 1.05 GB）。
- **显存**：6GB 上限下 b10c128 + batch 128 训练峰值 ≤ 5.5GB（todo 8 冒烟验证）；自对弈/推理峰值远低于训练。
- **长训收敛以月计**：19×19 从零训练棋力增长慢，"看到它成长"是首要目标；vs-random 早期里程碑 >80% 是弱信号检查，不达不阻塞。

### 24/7 长训 Windows 运维要点

- **电源计划**：禁用睡眠/休眠，避免训练被系统挂起（`powercfg /change standby-timeout-ac 0`，`powercfg /change hibernate-timeout-ac 0`）。
- **Windows Defender 实时扫描**：将 `models/`、`data/`、`logs/` 目录加入排除项（大 checkpoint/npz 反复扫描会拖慢磁盘 I/O）。
- **驱动状态刷新**：建议每 24h 重启一次训练以刷新驱动/显存状态（训练支持 `--resume`，重启成本低）。
- 电费估算：3060 满载 ~170W，24h ≈ 4 kWh/天 ≈ 60-100 元/月。

---

## 5. 故障排查

### CUDA 不可用 / 显存不足

```powershell
uv run python -c "import torch;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
```

- `torch.cuda.is_available()` 为 False → 核对 NVIDIA 驱动版本（CUDA 13 兼容驱动；RTX 3060 实测 581.08）与 `torch==2.13.0+cu130` 是否随 `uv sync` 装成 cu130 wheel（PyPI 默认即 cu130）。
- OOM → 降 `batch_size`（二分）或开 `fp16: true`；`leaf_batch` 可调 8-32。

### 首步（first-search）延迟大

首手 19×19 全空盘搜索 200 sims 在纯 Python MCTS 下会有明显延迟（批推理 batch 建立 + 首轮前向）。这是基线行为，非故障；`--sims` 调低或等 model 训练更强后（早停/更多 pass）局数变短即缓解。

### Windows 下 GTP 管道异常

- 引擎作为子进程驱动时（`match --engine2 katago` / 平台 bot），对端输出行结束符为 `\r\n`：omigamax 的 GTP 客户端已做 CRLF-tolerant 读取；若自写驱动脚本遇到解析错，确认按空白行终止帧、rstrip `\r`。
- `gtp_main.py` 以 stdin/stdout 交互：管道驱动时确保读取方不回显混入 stdout（stderr 单独收集）。
- 子进程挂起 → 120s/手超时 kill（match 内置）。

### todo-12 门控（模拟数-棋力单调）未确认

todo 12 关键门槛为 40/200/800 sims 两两互弈各 ≥60 局、逐级胜率 >50%。目前仅点估计方向性支持（P(200>40)=1.0、P(800>200)=1.0，样本 2-7 局/配对，Wilson 95% CI 跨 0.5），**判定为 UNCONFIRMED**。

重跑正式门控：

```powershell
# 完整版：每配对 60 局（预计 11-22 小时 @ 300-600 sims/s；本机 ~100 sims/s 更久——时间预算见 todo 12）
uv run python -m omigamax.cli.mcts_strength --games 60
# 快速版（800-sim 配对降为 30 局，节省时间）
uv run python -m omigamax.cli.mcts_strength --games 60 --quick
```

胜率矩阵写入 `.omo/evidence/omigamax-go/task-12-strength.json`。判定单调成立需 `P(800>200)>0.5 且 P(200>40)>0.5`。

---

## 相关文档

- 项目总览与快速开始 → [`../README.md`](../README.md)
- 配置详解 → [`config.md`](config.md)
