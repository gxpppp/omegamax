# omigamax 配置详解

所有配置集中在一个 YAML 文件：[`config/default.yaml`](../config/default.yaml)。运行时通过 `omigamax.config.load_config` 加载，各 CLI/脚本用 `--config <path>` 传入自定义文件；不传则使用默认值。表中默认值逐一对应 `config/default.yaml` 的实际内容（25 个键，0 缺失 / 0 多余）。

## 配置表

| # | 键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| 1 | `board_size` | int | `19` | 棋盘边长（19×19 全棋盘训练；9/13 供测试，规则引擎尺寸参数化） |
| 2 | `komi` | float | `7.5` | 贴目（白方贴目，AGZ 论文值；两连 pass 终局后 Tromp-Taylor 计分判定胜负） |
| 3 | `resign_threshold` | float | `0.0` | 认输阈值。`0.0` = 关闭认输（AGZ 的 0.5% 阈值优化为延展项，文档已注明）；非 0 时自对弈中胜率低于该值则认输 |
| 4 | `blocks` | int | `10` | ResNet 残差块数量（b10c128：10 block × 128 channel，~3M 参数，6GB 可训） |
| 5 | `channels` | int | `128` | 卷积通道数 |
| 6 | `fp16` | bool | `false` | FP16 半精度训练开关（autocast；显存紧张时开启，冒烟已验证） |
| 7 | `c_puct` | float | `2.5` | MCTS UCB 探索常数（AGZ 论文用 5；6GB 低模拟数下 2.5 更平衡探索/利用，可配置） |
| 8 | `dirichlet_alpha` | float | `0.03` | 根节点 Dirichlet 噪声 α（AGZ 值；仅自对弈使用，评估不加噪） |
| 9 | `dirichlet_eps` | float | `0.25` | 根节点噪声混合比例 ε（AGZ 值） |
| 10 | `temperature_threshold` | int | `30` | 温度 τ=1.0 的手数阈值；此后 τ→0（argmax） |
| 11 | `virtual_loss` | int | `3` | 虚拟损失（批推理时叶子占位，防止同一叶子被重复选择） |
| 12 | `simulations` | int | `200` | 每手 MCTS 模拟数（默认对局/自对弈；评估用 `eval_sims`） |
| 13 | `leaf_batch` | int | `16` | 批推理叶子并行度（6GB 上可调 8-32） |
| 14 | `batch_size` | int | `128` | 训练批大小（6GB 显存冒烟验证的上限；OOM 时二分下调） |
| 15 | `lr` | float | `0.2` | 学习率（AGZ 论文值；SGD + momentum） |
| 16 | `momentum` | float | `0.9` | SGD 动量（AGZ 论文值） |
| 17 | `l2` | float | `1e-4` | L2 权重衰减 |
| 18 | `lr_schedule_steps` | list[int] | `[50000, 100000]` | 分段学习率调度步数：0.2 前 50K 步 → 0.02 前 100K 步 → 0.002（AGZ 论文 300K/500K 按 batch 128 缩比） |
| 19 | `replay_buffer_games` | int | `1000` | 回放缓冲保留的最近自对弈局数（磁盘后备；超出清理旧局） |
| 20 | `symmetry_aug` | bool | `true` | 8 向对称增强（AGZ 论文用；训练时对样本做 8 种对称变换） |
| 21 | `eval_interval_steps` | int | `2000` | 长训时评估门控的可选稀化间隔（每 cycle 末必评估，不受此键覆盖） |
| 22 | `eval_games` | int | `21` | 每次评估门控对局数（候选 vs 当前 best，先后手交替、无噪声、τ=0） |
| 23 | `eval_sims` | int | `200` | 评估局每手模拟数 |
| 24 | `replace_threshold` | float | `0.55` | 评估门控替换阈值：胜率 ≥ 0.55 时用候选替换 `best.pt`（含等于） |
| 25 | `viz_enabled` | bool | `true` | 训练循环挂载 pygame 可视化线程开关（无 GUI 自动降级纯日志；`--viz off` 强制关闭） |

## 键对齐校验

由 `uv run python -c "import yaml;print(len(yaml.safe_load(open('config/default.yaml',encoding='utf-8'))))"` 核对：`config/default.yaml` 实际键数 = **25**，本表行数 = **25**，键名逐一相等（todo 22 验收以脚本核对为准）。

## 覆盖方式

```powershell
# 自定义配置（示例：降低模拟数加速冒烟）
uv run python -m omigamax.train.loop --smoke --config my_config.yaml
# 部分 CLI 参数可直接覆盖对应键（如 match --sims、--board-size、--komi、--config）
uv run python -m omigamax.cli.match --engine2 random --games 1 --sims 40
```

## 相关文档

- 项目总览与快速开始 → [`../README.md`](../README.md)
- 训练运维/硬件/故障排查 → [`ops.md`](ops.md)
