# 基于 Gymnasium 的输电线路覆冰处置强化学习

用气象/运行数据驱动覆冰仿真环境，训练 PyTorch DQN 智能体学习“无操作/预警/融冰”的最优处置策略，并在多场景下与传统阈值策略对比评估。

## 文件说明

- `ice_env.py`：覆冰仿真环境 `IcingEnv`（状态/动作空间 + 多目标加权奖励 + 场景扰动）
- `dqn_torch.py`：PyTorch DQN（Q 网络、经验回放、目标网络）
- `generate_weather_data.py`：生成可复现实验用的 `cold_wave_data.csv`
- `train_dqn.py`：训练并保存模型 `dqn_icing_model.pt`
- `evaluate_and_plot.py`：多场景评估 + 与基线策略对比；可选绘图复盘

## 快速运行（示例）

```powershell
# 1) 生成数据
python generate_weather_data.py --out cold_wave_data.csv --hours 168 --seed 0

# 2) 训练 DQN（输出 dqn_icing_model.pt）
python train_dqn.py --data-path cold_wave_data.csv --total-steps 100000 --save-path dqn_icing_model.pt

# 3) 多场景评估 + 对比基线策略
python evaluate_and_plot.py --data-path cold_wave_data.csv --model-path dqn_icing_model.pt

# 4) 画一条场景曲线（默认 baseline）
python evaluate_and_plot.py --data-path cold_wave_data.csv --model-path dqn_icing_model.pt --plot --plot-scenario baseline

```

## 指标口径

- `mean_reward`：总奖励（单回合）在场景上的平均值
- `hazard_rate`：是否发生“灾害/失效”（覆冰厚度达到 `critical_ice_thickness`）的比例
- `mean_cost`：总处置成本（预警/融冰成本加总）的平均值
- `resp_time`：首次进入预警阈值后，首次做出非 `0` 动作的响应时延（单位：步/小时；无法计算则为 `N/A`）

