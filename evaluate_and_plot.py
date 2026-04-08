import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ice_env import IcingEnv


@dataclass
class EpisodeResult:
    """
    使用数据类 (dataclass) 结构化存储单次评估（Episode）的结果。
    这比返回一个巨大的 Tuple 更具可读性，便于后续的统计与绘图。
    """
    total_reward: float  # 累计奖励（在本环境中主要体现为负数的惩罚值）
    total_cost: float  # 累计动作成本（能耗/通讯等资源消耗）
    hazard: bool  # 是否触发了灾害（即覆冰厚度超过临界值断线）
    response_time: Optional[int]  # 响应时间：从首次达到预警阈值到系统首次采取动作的步数差
    steps: int  # 存活总步数

    # 供 matplotlib 绘图使用的全生命周期时间序列数据
    temps: List[float]
    humidities: List[float]
    ice_thicknesses: List[float]
    actions: List[int]
    step_rewards: List[float]


def parse_args():
    """
    解析评估阶段的命令行参数。
    """
    p = argparse.ArgumentParser(description="在多种场景下评估 DQN 模型与基线策略的表现。")
    p.add_argument("--data-path", type=str, default="cold_wave_data.csv")
    p.add_argument("--model-path", type=str, default="dqn_icing_model.pt")  # 待评估的模型权重文件
    p.add_argument("--device", type=str, default="cpu")
    # 是否开启绘图模式，以及指定要绘制的具体场景
    p.add_argument("--plot", action="store_true", help="是否绘制 DQN 策略在特定场景下的运行曲线。")
    p.add_argument("--plot-scenario", type=str, default="baseline")
    return p.parse_args()


def load_dqn_policy(model_path: str, device: str) -> Callable[[np.ndarray], int]:
    """
    加载已训练好的 DQN 模型，并将其封装为一个标准的「策略函数」。
    策略函数签名：接收环境观测值 (obs) -> 返回要执行的动作 (action)
    """
    import torch
    from dqn_torch import QNetwork

    # 1. 尝试加载保存的字典状态
    try:
        state = torch.load(model_path, map_location=device)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"找不到模型文件: {model_path}。请先运行 train_dqn.py 生成 .pt 模型。"
        ) from e

    # 2. 从状态字典中动态获取网络结构参数，并实例化网络
    obs_dim = int(state["obs_dim"])
    action_dim = int(state["action_dim"])
    net = QNetwork(obs_dim, action_dim).to(device)
    net.load_state_dict(state["q"])

    # 3. 关键步骤：将网络设置为评估模式，关闭 Dropout/BatchNorm 的随机性
    net.eval()

    # 4. 定义具体的推断逻辑
    @torch.no_grad()  # 评估时不计算梯度，大幅节省显存并提升速度
    def policy(obs: np.ndarray) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q = net(x)
        # 选择具有最大 Q 值的动作（纯利用，无 Epsilon 随机探索）
        return int(torch.argmax(q, dim=1).item())

    return policy


def policy_noop(_obs: np.ndarray, _env: IcingEnv) -> int:
    """
    「消极基线」：无操作策略（Do Nothing）。
    用于评估如果不做任何干预，环境自然恶化的情况，作为最差表现的兜底参考。
    """
    return 0


def policy_threshold(obs: np.ndarray, env: IcingEnv) -> int:
    """
    「业务基线」：阈值规则策略（Rule-based）。
    这是现实工业场景中最常见的做法：达到一定阈值就报警，再达到一定阈值就融冰。
    如果 DQN 的表现不能打败这个简单的规则，说明强化学习模型没有学到有价值的前瞻性策略。
    """
    ice = float(obs[5])  # 索引 5 是覆冰厚度
    if ice >= env.critical_ice_thickness:
        return 2  # 达到临界值，立刻融冰
    if ice >= env.warn_ice_thickness:
        return 1  # 达到预警值，发布预警
    return 0


def run_episode(env: IcingEnv, policy: Callable[[np.ndarray], int]) -> EpisodeResult:
    """
    在给定环境中，使用给定策略运行完整的一集，并详细记录交互过程。
    """
    obs, _ = env.reset()

    # 初始化统计指标
    total_reward = 0.0
    total_cost = 0.0
    hazard = False

    # 用于计算「响应时间」的追踪变量
    first_warning_step: Optional[int] = None
    first_response_step: Optional[int] = None

    # 初始化时序数据记录列表
    temps: List[float] = []
    humidities: List[float] = []
    ice_thicknesses: List[float] = []
    actions: List[int] = []
    step_rewards: List[float] = []

    steps = 0
    while True:
        # 记录当前状态（供绘图使用）
        temps.append(float(obs[0]))
        humidities.append(float(obs[3]))
        ice_thicknesses.append(float(obs[5]))

        # 捕获首次达到预警阈值的时刻
        if first_warning_step is None and float(obs[5]) >= env.warn_ice_thickness:
            first_warning_step = steps

        # 策略前向推断，决定动作
        action = int(policy(obs))
        actions.append(action)

        # 捕获首次实际采取应对动作的时刻
        if first_warning_step is not None and first_response_step is None and action != 0:
            first_response_step = steps

        # 与环境交互
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        # 收集回报与惩罚数据
        step_rewards.append(float(reward))
        total_reward += float(reward)
        total_cost += float(info.get("cost", 0.0))
        hazard = hazard or bool(info.get("hazard", False))

        steps += 1
        if done:
            break

    # 计算预警响应时间（反应越快通常越好，但也可能带来更高的抢跑成本）
    response_time: Optional[int] = None
    if first_warning_step is not None:
        if first_response_step is None:
            response_time = None  # 有预警但一直未响应
        else:
            response_time = int(first_response_step - first_warning_step)

    return EpisodeResult(
        total_reward=float(total_reward),
        total_cost=float(total_cost),
        hazard=bool(hazard),
        response_time=response_time,
        steps=int(steps),
        temps=temps,
        humidities=humidities,
        ice_thicknesses=ice_thicknesses,
        actions=actions,
        step_rewards=step_rewards,
    )


def mean_optional_int(values: List[Optional[int]]) -> Optional[float]:
    """辅助函数：计算包含 None 值的列表的平均值。"""
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return float(np.mean(xs))


def format_optional(x: Optional[float], *, digits: int = 2) -> str:
    """辅助函数：格式化可选浮点数。"""
    if x is None:
        return "N/A"
    return f"{x:.{digits}f}"


def summarize(results: List[EpisodeResult]) -> Dict[str, Optional[float]]:
    """将一集或多集的结果汇总为平均评估指标。"""
    return {
        "mean_reward": float(np.mean([r.total_reward for r in results])) if results else 0.0,
        "mean_cost": float(np.mean([r.total_cost for r in results])) if results else 0.0,
        "hazard_rate": float(np.mean([1.0 if r.hazard else 0.0 for r in results])) if results else 0.0,
        "mean_response_time": mean_optional_int([r.response_time for r in results]),
    }


def print_report(title: str, rows: List[Tuple[str, Dict[str, Optional[float]]]]) -> None:
    """打印控制台测试报告，展示不同策略的对比表格。"""
    print("=" * 80)
    print(title)
    print("-" * 80)
    print(f"{'policy':18s} | {'mean_reward':>10s} | {'hazard_rate':>10s} | {'mean_cost':>10s} | {'resp_time':>9s}")
    print("-" * 80)
    for name, m in rows:
        print(
            f"{name:18s} | {m['mean_reward']:10.2f} | {m['hazard_rate']:10.2f} | {m['mean_cost']:10.2f} | {format_optional(m['mean_response_time'], digits=1):>9s}"
        )
    print("=" * 80)


def main():
    args = parse_args()

    # 定义「鲁棒性评估场景集」。
    # 优秀的 RL 模型不应只在训练数据上表现好，还必须能应对没见过的极端恶劣天气。
    scenarios: Dict[str, Dict] = {
        "baseline": {},  # 原始气象数据
        "colder": {"temp_bias": -3.0},  # 更冷：气温整体下降 3 度
        "wetter": {"humidity_bias": 7.0},  # 更湿：相对湿度整体提高 7%
        "windier": {"wind_speed_scale": 1.3},  # 风更大：风速放大 1.3 倍
        "faster_growth": {"ice_growth_scale": 1.5},  # 覆冰微地形恶化：覆冰增长速率直接翻倍
    }

    # 加载已训练的主角策略
    dqn_policy = load_dqn_policy(args.model_path, device=args.device)

    # 将各策略打包，统一调用接口
    policies: List[Tuple[str, Callable[[np.ndarray, IcingEnv], int]]] = [
        ("noop", lambda o, e: policy_noop(o, e)),
        ("threshold", lambda o, e: policy_threshold(o, e)),
        ("dqn", lambda o, _e: dqn_policy(o)),
    ]

    # 对每个评估场景分别进行遍历测试
    for scenario_name, scenario_kwargs in scenarios.items():
        rows: List[Tuple[str, Dict[str, Optional[float]]]] = []
        for policy_name, pol in policies:
            # 动态实例化带有扰动参数的测试环境
            env = IcingEnv(data_path=args.data_path, **scenario_kwargs)
            result = run_episode(env, lambda obs: pol(obs, env))
            rows.append((policy_name, summarize([result])))

            # 如果用户指定了要画图，且当前正是用户关心的策略和场景，则调用绘图函数
            if args.plot and policy_name == "dqn" and scenario_name == args.plot_scenario:
                plot_episode(result, scenario_name=scenario_name)

        # 打印当前场景下，三个策略的角逐结果
        print_report(f"Scenario: {scenario_name}", rows)


def plot_episode(result: EpisodeResult, scenario_name: str) -> None:
    """
    使用 Matplotlib 绘制复杂的多轴时序图，直观复盘智能体的每一步决策与环境变化的关系。
    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    # 解决 matplotlib 在图表中显示中文和负号的问题
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    steps = list(range(len(result.ice_thicknesses)))

    # 创建主画布
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制第一纵轴：气温曲线
    color_temp = "tab:blue"
    ax1.set_xlabel("时间步 (小时)")
    ax1.set_ylabel("气温 (℃)", color=color_temp)
    ax1.plot(steps, result.temps, color=color_temp, label="气温 (℃)", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color_temp)
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)  # 0度冰点参考线

    # 绘制第二纵轴（共享横轴）：覆冰厚度曲线
    ax2 = ax1.twinx()
    color_ice = "tab:red"
    ax2.set_ylabel("覆冰厚度 (mm)", color=color_ice)
    ax2.plot(steps, result.ice_thicknesses, color=color_ice, label="覆冰厚度", linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color_ice)

    # 绘制第三纵轴（偏移到右侧外围）：智能体累计得分曲线
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))  # 将第三个轴向右推移，防止重叠
    color_score = "tab:green"

    # 业务逻辑调整：因为奖励主要是持续的惩罚值（负数），取相反数 (-np.cumsum) 可以让曲线呈现「逐步扣分」的直观感受
    score_curve = -np.cumsum(result.step_rewards)
    ax3.set_ylabel("智能体累计得分(越高越好)", color=color_score)
    ax3.plot(
        steps,
        score_curve,
        color=color_score,
        linestyle="--",
        label="智能体累计得分",
        linewidth=2,
    )
    ax3.tick_params(axis="y", labelcolor=color_score)

    # 绘制散点标记：将智能体的离散动作（预警、融冰）打在覆冰厚度的曲线上
    for s, a, ice in zip(steps, result.actions, result.ice_thicknesses):
        if a == 1:
            # 动作 1（预警）：橙色向下三角形
            ax2.scatter(s, ice, color="orange", marker="v", s=80, zorder=5)
        elif a == 2:
            # 动作 2（融冰）：紫色五角星（尺寸更大，更醒目）
            ax2.scatter(s, ice, color="purple", marker="*", s=150, zorder=5)

    # 手动创建自定义图例，因为散点是循环生成的
    warn_marker = mlines.Line2D([], [], color="orange", marker="v", linestyle="None", markersize=8, label="动作: 预警")
    melt_marker = mlines.Line2D([], [], color="purple", marker="*", linestyle="None", markersize=12, label="动作: 融冰")

    # 组装图例并展示图表
    fig.legend(handles=[warn_marker, melt_marker], loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.title(f"DQN 策略复盘 (scenario={scenario_name})")
    fig.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()