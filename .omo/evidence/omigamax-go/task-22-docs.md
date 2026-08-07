# Task 22 Evidence — 文档（README / config.md / ops.md）

- todo: 22（文档：README 与配置说明）
- date: 2026-08-08
- head (before commit): `c17d761`
- command shell: PowerShell 5.1, Windows 10, uv 环境

## 1. 交付文档

| 文件 | 内容 |
| --- | --- |
| `README.md`（重写空模板） | 项目简介、当前状态（含 vs-random 45% 诚实说明与 todo-12 门控未确认）、安装、快速开始四条命令、CLI 一览、六组件架构 + 数据流图、联机平台预留说明、指向 config.md/ops.md |
| `docs/config.md` | 配置详解：25 键全表（类型/默认值/说明），逐键对齐 `config/default.yaml` |
| `docs/ops.md` | 训练启动/中断/续跑、checkpoint 与日志位置、可视化（--capture/SDL_VIDEODRIVER=dummy）、对弈/GTP 用法、硬件速度预期（实测 ~88-160 sims/s vs 计划 300-600）、24/7 Windows 运维要点、故障排查、todo-12 门控重跑方法 |

## 2. 配置表完整性核对（脚本）

命令：

```
uv run python -c "import yaml; cfg=yaml.safe_load(open('config/default.yaml',encoding='utf-8')); print('KEYS',len(cfg)); [print(' ',k,type(cfg[k]).__name__,'=',cfg[k]) for k in cfg]"
```

输出（节选）：`KEYS 25`，键列表 = board_size, komi, resign_threshold, blocks, channels, fp16, c_puct,
dirichlet_alpha, dirichlet_eps, temperature_threshold, virtual_loss, simulations, leaf_batch,
batch_size, lr, momentum, l2, lr_schedule_steps, replay_buffer_games, symmetry_aug,
eval_interval_steps, eval_games, eval_sims, replace_threshold, viz_enabled。

**核对结果：`config/default.yaml` 键数 = 25，docs/config.md 表行数 = 25，键名逐一相等，0 缺失 / 0 多余。PASS。**

注意：`l2: 1e-4` 在 YAML 中解析为字符串 `"1e-4"`（PyYAML 无小数点规则），代码中统一经
`float(cfg.get("l2", 1e-4))` 消费，故表中类型记 float、默认 1e-4，语义与代码一致。

## 3. 四条命令逐一复现（acceptance）

| # | 命令 | exit | 输出要点 |
| --- | --- | --- | --- |
| 1 环境 | `uv run python -c "import torch,pygame,yaml;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"` | 0 | `2.13.0+cu130 True NVIDIA GeForce RTX 3060 Laptop GPU` |
| 2 训练冒烟 | `uv run python -m omigamax.train.loop --smoke` | 0 | `loop: 25 steps trained this run (global step 25, 1 cycles done, 2 games generated)`；`loss: 8.0072 -> 7.0305 (decrease=True)`；`RESULT: PASS (exit 0)`；evidence written task-16-loop.json |
| 3 可视化冒烟 | `uv run python -m omigamax.cli.viz_smoke --capture logs/viz_capture.png` | 0 | `capture written: logs\viz_capture.png (24728 bytes)`；`RESULT: PASS (headless capture)` |
| 4 评测 1 局 | `uv run python -m omigamax.cli.match --engine2 random --games 1 --sims 40` | 0 | `games=1 completed=1 errors=0`；`game 0: omigamax=B winner=B moves=790 result=B+350.5`；`win rate (omigamax): 1/1 = 1.000`；`RESULT: PASS (exit 0)` |

**全部 exit 0。PASS。**

## 4. 全量测试

命令：

```
uv run pytest -q
```

输出尾部：`414 passed, 1 warning in 34.64s`，exit 0。**PASS（文档不改变代码行为）。**

## 5. 过程中发现并修复的代码 bug（最小改动，命令可复现所必需）

运行训练冒烟（命令 2）时发现：todo 16 的 `loop --smoke` 在 pygame 可视化实际启动（viz=available）
时，`run_loop` 返回的 report 的 `protocol.viz` 携带实时 `queue`/`thread`/`stop` 句柄，证据
`json.dump` 崩溃 `TypeError: Object of type SnapshotQueue is not JSON serializable`，命令 exit 1。

- 根因：todo 16 验收当时 viz 模块尚不存在（module_unavailable，无句柄），故未触发；todo 17 落地后
  可视化真正启动，report 不再可序列化——违反计划护栏「可视化线程崩溃不影响训练」。
- 修复：`omigamax/train/loop.py` `main()` 中证据落盘前对 report 做 `_json_safe` 递归净化
  （非 JSON 类型 → 类型名占位），保留 report 内 viz 句柄供 `run_loop` 内部使用与单测断言
  （`tests/test_loop.py::TestLazyViz` 仍校验 `viz["thread"]`，不破坏现有测试契约）。
- 验证：修复后 `loop --smoke` exit 0 且 `evidence written` 成功；`uv run pytest -q` 414 全绿。

## 6. 验收汇总

| 验收项 | 结果 |
| --- | --- |
| README 四条命令逐一执行成功（环境/训练冒烟/可视化冒烟/评测 1 局） | PASS（均 exit 0，见 §3） |
| 配置表键名与 default.yaml 逐一对齐（脚本核对） | PASS（25 == 25，见 §2） |
| pytest 全绿 | PASS（414 passed，见 §4） |
| 文档与代码行为一致 | PASS（§5 bug 已修复并复验） |

## 7. 提交

- commit message: `docs: README 与配置文档`
- 涉及文件：`README.md`、`docs/config.md`、`docs/ops.md`、`omigamax/train/loop.py`（§5 最小 bug 修复）、本证据文件。
