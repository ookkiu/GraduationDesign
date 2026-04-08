import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True, precision=2)


class IcingEnv(gym.Env):
    """
    更新后的输电线路覆冰仿真环境（加入预警状态追踪与合规考核）
    """

    def __init__(
            self,
            data_path: str = "cold_wave_data.csv",
            *,
            warn_ice_thickness: float = 3.0,
            critical_ice_thickness: float = 6.0,
            deice_amount: float = 2.0,
            random_start: bool = False,
            random_start_max_hours: int = 24,
            temp_bias: float = 0.0,
            humidity_bias: float = 0.0,
            wind_speed_scale: float = 1.0,
            ice_growth_scale: float = 1.0,
            w_risk: float = 100.0,
            w_reliability: float = 10.0,
            w_cost: float = 1.0,
    ):
        super().__init__()

        self.action_space = spaces.Discrete(3)

        # 【修改点 1】：状态空间从 6 维增加到 7 维
        # 新增最后一维：预警状态 (0.0 表示未预警，1.0 表示已预警)
        low = np.array([-30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([40.0, 35.0, 90.0, 100.0, 1000.0, 50.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.warn_ice_thickness = float(warn_ice_thickness)
        self.critical_ice_thickness = float(critical_ice_thickness)
        self.deice_amount = float(deice_amount)
        self.random_start = bool(random_start)
        self.random_start_max_hours = int(random_start_max_hours)

        self.temp_bias = float(temp_bias)
        self.humidity_bias = float(humidity_bias)
        self.wind_speed_scale = float(wind_speed_scale)
        self.ice_growth_scale = float(ice_growth_scale)

        self.w_risk = float(w_risk)
        self.w_reliability = float(w_reliability)
        self.w_cost = float(w_cost)

        try:
            self.weather_data = pd.read_csv(data_path)
            self.max_steps = len(self.weather_data) - 1
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"找不到气象数据文件: {data_path}。请先生成或提供 CSV 数据。"
            ) from e

        self.current_step = 0
        self.current_ice_thickness = 0.0
        # 【修改点 2】：增加环境内部的预警状态变量
        self.is_warned = False
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.random_start:
            max_start = min(self.random_start_max_hours, self.max_steps)
            self.current_step = int(self.np_random.integers(0, max_start + 1))
        else:
            self.current_step = 0

        self.current_ice_thickness = 0.0
        # 每集重置时，解除预警状态
        self.is_warned = False
        self.state = self._get_observation()
        return self.state, {}

    def _get_observation(self) -> np.ndarray:
        row = self.weather_data.iloc[self.current_step]
        temp = float(row["temperature"]) + self.temp_bias
        wind_speed = float(row["wind_speed"]) * self.wind_speed_scale
        wind_angle = float(row["wind_angle"])
        humidity = float(row["humidity"]) + self.humidity_bias
        line_current = float(row["line_current"])

        # 【修改点 3】：将预警状态加入到返回的观测张量中
        return np.array(
            [
                temp,
                wind_speed,
                wind_angle,
                humidity,
                line_current,
                self.current_ice_thickness,
                float(self.is_warned),
            ],
            dtype=np.float32,
        )

    def _action_cost(self, action: int, line_current: float) -> float:
        # 预警成本。如果是重复预警，依然扣费（防止智能体一直无脑刷动作 1）
        if action == 1:
            return 1.0
        if action == 2:
            current_factor = 0.5 + (max(0.0, line_current) / 500.0)
            return 10.0 * current_factor
        return 0.0

    def step(self, action):
        action = int(action)

        # 【修改点 4】：智能体执行了预警动作，更新环境状态
        if action == 1:
            self.is_warned = True

        row = self.weather_data.iloc[self.current_step]
        temp = float(row["temperature"]) + self.temp_bias
        humidity = float(row["humidity"]) + self.humidity_bias
        wind_speed = float(row["wind_speed"]) * self.wind_speed_scale
        wind_angle = float(row["wind_angle"])
        line_current = float(row["line_current"])

        RHO_ICE = 0.9
        D_0 = 27.6
        ice_growth = 0.0

        if temp < 0.0 and humidity > 85.0:
            v_effective = wind_speed * np.sin(np.radians(wind_angle))
            lwc = 0.5 + ((humidity - 85.0) / 15.0) * 2.0
            current_D = D_0 + 2.0 * self.current_ice_thickness
            alpha = 0.8
            ice_growth = (alpha * lwc * v_effective) / (RHO_ICE * np.pi * current_D) * 3.6
        elif temp > 2.0:
            ice_growth = -0.5

        ice_growth *= self.ice_growth_scale

        self.current_ice_thickness += ice_growth

        deice_effect = 0.0
        if action == 2:
            deice_effect = -self.deice_amount
            self.current_ice_thickness += deice_effect
            # 融冰后通常意味着危机解除，这里我们可以选择自动重置预警状态
            # self.is_warned = False

        self.current_ice_thickness = max(0.0, self.current_ice_thickness)

        self.current_step += 1
        self.state = self._get_observation()

        hazard = bool(self.current_ice_thickness >= self.critical_ice_thickness)

        # 【修改点 5】：方案 A 核心落地 —— 预警减免灾害惩罚
        if hazard:
            # 如果断线了，但之前预警过，给它一个“宽大处理” (-0.3)；毫无准备的断线则重罚 (-1.0)
            risk_component = -0.3 if self.is_warned else -1.0
        else:
            risk_component = 0.0

        reliability_component = -float(self.current_ice_thickness / self.critical_ice_thickness)
        reliability_component = float(np.clip(reliability_component, -2.0, 0.0))

        # 【修改点 6】：合规惩罚 —— 倒逼开口
        compliance_penalty = 0.0
        if self.current_ice_thickness >= self.warn_ice_thickness and not self.is_warned:
            # 到了该报告的厚度却装死，每步额外扣分！
            compliance_penalty = -0.5

        cost = float(self._action_cost(action, line_current))

        reward = (
                self.w_risk * risk_component
                + self.w_reliability * reliability_component
                + self.w_reliability * compliance_penalty  # 将合规扣分计入总奖励
                - self.w_cost * cost
        )

        terminated = bool(hazard)
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "ice_thickness": float(self.current_ice_thickness),
            "ice_growth": float(ice_growth),
            "deice_effect": float(deice_effect),
            "hazard": bool(hazard),
            "is_warned": bool(self.is_warned),  # 记录状态供分析
            "risk_component": float(risk_component),
            "reliability_component": float(reliability_component),
            "compliance_penalty": float(compliance_penalty),
            "cost": float(cost),
            "temp": float(temp),
            "humidity": float(humidity),
            "wind_speed": float(wind_speed),
            "wind_angle": float(wind_angle),
            "line_current": float(line_current),
        }

        return self.state, float(reward), terminated, truncated, info


if __name__ == "__main__":
    env = IcingEnv(data_path="cold_wave_data.csv")
    obs, _ = env.reset()
    print(f"初始状态: {np.round(obs, 2)}")
    print("-" * 50)
    for i in range(48):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"步 {i + 1:3d} | 动作: {action} | 预警: {info['is_warned']} | 覆冰: {info['ice_thickness']:.2f}mm | 奖励: {reward:.2f}"
        )
        if terminated or truncated:
            print("环境结束 (terminated/truncated)。")
            break