# Base 1v1 诊断训练阶段记录

这份文档用于记录我们为了排查 PPO 收敛问题，在 `BASE` 步兵 1v1 简化场景中做过的阶段化训练实验。

## 通用设置

- 环境：`GameType.BASE`
- 阵容：红方 `STANDARD_3` vs 蓝方 `STANDARD_3`
- 决策频率：通常为 2 Hz
- 初始位置：固定起点，或使用 `--random-start` 随机有效起点
- 红方策略：`PPOAgent`
- 评估方式：指标评估通常使用确定性策略；录制视频时可以使用采样策略
- 训练脚本：`stage_diagnostic.py`

当前诊断奖励刻意保持简单，主要包括：

- 靠近对方机器人给奖励。
- 对敌方造成伤害给奖励。
- 自己受到伤害给惩罚。
- 有无遮挡射线时给少量奖励。
- 红方胜利/失败给终局奖励或惩罚。
- 对动作头进行轻量塑形：鼓励 `navigation_set=1` 和 `attack_target=STANDARD_3`。

这些奖励不是最终版本，只是为了快速诊断“最小局部策略”是否能学出来。

## Stage A：脚本对手 / 基础动作头训练

目标：

- 验证智能体能否在最简单条件下学到有用行为。
- 验证 `attack_target=STANDARD_3` 是否能通过奖励学出来。
- 验证 `navigation_set=1` 是否能通过奖励学出来。
- 在静止或脚本蓝方条件下测试导航点输出。

典型命令：

```powershell
D:\miniconda\envs\rmcommander\python.exe stage_diagnostic.py --stage a --episodes 600 --eval-interval 100 --eval-episodes 10 --blue-mode idle --random-start --no-force-actions
```

已有较有参考价值的产物：

- `models\stage_a_agent_20260605_160807.pt`
- `logs\stage_a_20260605_160807.json`
- `videos\stage_a_agent_20260605_160807.mp4`

本次复测产物（使用统一后的 `stage_diagnostic.py` 入口）：

- 命令：`stage_diagnostic.py --stage a --episodes 600 --eval-interval 100 --eval-episodes 10 --blue-mode idle --random-start --no-force-actions --save-interval 100`
- 最终模型：`models\stage_a_agent_20260605_171048.pt`
- 日志：`logs\stage_a_20260605_171048.json`
- 最终评估：10 局中红方 9 胜、1 平，平均造成伤害 135，平均蓝方剩余血量 15。
- 观察：结果与之前 Stage A 结论一致，动作头稳定学到 `navigation_set=1` 和 `attack_target=3`，并能在静止蓝方下形成接敌击毁行为。

阶段观察：

- 离散动作头可以学到 `navigation_set=1` 和 `attack_target=3`。
- 使用确定性策略评估时，连续导航点会退化为策略均值，容易表现为固定点移动。
- 使用采样策略评估时行为更有变化，但仍不能说明已经学到稳定战术。
- 这个阶段更适合验证 PPO 和动作头是否能工作，而不是验证真实比赛策略。

## Stage B：同策略自博弈

目标：

- 不再只打静止或脚本蓝方。
- 红蓝双方使用同一个当前策略。
- 蓝方使用蓝方视角观测，再通过 `mirror_navigation_target_actions` 将归一化导航点镜像回世界坐标。
- 观察策略是否能在动态对手下学到更强的接敌和交火行为。

典型命令：

```powershell
D:\miniconda\envs\rmcommander\python.exe stage_diagnostic.py --stage b --base-model models\stage_a_agent_20260605_160807.pt --episodes 500 --eval-interval 100 --eval-episodes 20 --random-start --save-interval 100
```

首次 Stage B 产物：

- `models\stage_a_agent_20260605_163446_episode_100.pt`
- `models\stage_a_agent_20260605_163446_episode_200.pt`
- `models\stage_a_agent_20260605_163446_episode_300.pt`
- `models\stage_a_agent_20260605_163446_episode_400.pt`
- `models\stage_a_agent_20260605_163446_episode_500.pt`
- `models\stage_a_agent_20260605_163446.pt`
- `logs\stage_a_20260605_163446.json`
- `videos\stage_a_agent_20260605_163446_episode_100.mp4`

注：这次运行发生在脚本重命名前，所以文件名仍是 `stage_a_agent_...`。后续使用 `stage_diagnostic.py --stage b` 会输出 `stage_b_agent_...`。

本次复测产物（从本次 Stage A 模型继续训练）：

- 命令：`stage_diagnostic.py --stage b --base-model models\stage_a_agent_20260605_171048.pt --episodes 500 --eval-interval 100 --eval-episodes 20 --random-start --save-interval 100`
- 最终模型：`models\stage_b_agent_20260605_171440.pt`
- 日志：`logs\stage_b_20260605_171440.json`
- 最终评估：20 局中红方 16 胜、蓝方 4 胜，平均造成伤害 147.5，平均红方剩余血量 21，平均蓝方剩余血量 2.5。
- 观察：结果仍符合“同策略自博弈能运行但更激烈、更不稳定”的判断；这次没有明显退化成大量平局，但红方自身血量显著降低，说明策略主要是快速贴近互换伤害，还没有学出更细的换血节奏。

阶段观察：

- 同策略自博弈可以正常运行。
- 但红蓝双方同时使用当前策略时，训练容易不稳定。
- 中后期经常漂移到平局较多的模式：双方动作头正确，但没有稳定创造有效交火。
- 第一次运行中，`episode_100` 是相对更有参考价值的中间模型。

## Stage C：固定导航攻击脚本对手

目标：

- 不再面对静止蓝方，也不直接进入不稳定的同策略自博弈。
- 蓝方使用随机初始点、固定导航点、固定攻击目标。
- 观察红方是否会因为对手主动接敌而学出不同于 Stage A/B 的行为。

典型命令：

```powershell
D:\miniconda\envs\rmcommander\python.exe stage_diagnostic.py --stage c --base-model models\stage_a_agent_20260605_171048.pt --episodes 500 --eval-interval 100 --eval-episodes 20 --save-interval 100
```

默认设置：

- `--random-start` 在 Stage C 中默认开启。
- 蓝方默认使用 `fixed_nav_attack`。
- 蓝方固定导航世界坐标默认为 `(1.0, 2.5)`，可以通过 `--blue-fixed-nav x y` 覆盖。
- 蓝方固定 `navigation_set=1`，固定 `attack_target=STANDARD_3`。
- 红方动作头不再强制钳制，继续靠奖励学习。

阶段观察：

- 2026-06-05 复测：
  - smoke 产物 `models\stage_c_agent_20260605_173359.pt` 只跑了 2 episode，用于验证脚本流程，不建议评估策略。
  - 正式模型：`models\stage_c_agent_20260605_173416.pt`
  - 日志：`logs\stage_c_20260605_173416.json`
  - 最终评估：20 局中红方 8 胜、10 平、蓝方 2 胜，平均造成伤害 99，平均红方剩余血量 61.5，平均蓝方剩余血量 51。
  - 中间评估：`episode_400` 表现最好，20 局中红方 16 胜、3 平、蓝方 1 胜，平均造成伤害 131.5，平均蓝方剩余血量 18.5。
- 初步结论：固定导航攻击蓝方确实制造了不同于 Stage A 的压力，但训练仍会漂移；最终模型不一定优于中间 checkpoint，`stage_c_agent_20260605_173416_episode_400.pt` 更值得优先录视频观察。

## 下一步候选：Stage D / 对手池

目标：

- 避免“当前策略打当前策略”导致双方一起退化到无效均衡。
- 让红方训练时面对多个历史版本的蓝方策略。

候选设计：

- 维护一个历史 checkpoint 池。
- 每个 rollout 中，蓝方从最近若干历史模型里随机抽取。
- 偶尔让蓝方使用当前模型，以保留自博弈压力。
- 中间模型按固定间隔保存，而不是只保存最后一个模型。
- 评估时分别对多个对手快照测试，而不是只看当前同策略胜率。

这个阶段应该比纯 self-play 更接近后续 RMUL 规则训练需求。
