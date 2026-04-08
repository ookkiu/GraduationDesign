import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True, precision=2)


class IcingEnv(gym.Env):
    """
    更新后的输电线路覆冰仿真环境（加入预警状态追踪、合规考核、安全互锁与存活奖励）
    """

    def __init__(
            self,
            data_path: str = "cold_wave_data.csv",
            *,
            # 【关键修改 1】：阈值反转。必须先达到预警线 (1.0)，才有可能达到融冰线 (3.0)
            warn_ice_thickness: float = 1.0,
            critical_ice_thickness: float = 6.0,
            deice_amount: float = 2.0,
            min_deice_thickness: float = 3.0,  # 冰厚达到 3.0 才能开启大功率融冰
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

        # 状态空间增加到 7 维：最后一位是预警状态 (0.0/1.0)
        low = np.array([-30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([40.0, 35.0, 90.0, 100.0, 1000.0, 50.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.warn_ice_thickness = float(warn_ice_thickness)
        self.critical_ice_thickness = float(critical_ice_thickness)
        self.deice_amount = float(deice_amount)
        self.min_deice_thickness = float(min_deice_thickness)
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

    # 这里的旧 _action_cost 方法已经被弃用，成本计算移到了 step 函数内，可以保留也可以删掉。
    def _action_cost(self, action: int, line_current: float) -> float:
        if action == 1:
            return 1.0
        if action == 2:
            current_factor = 0.5 + (max(0.0, line_current) / 500.0)
            return 10.0 * current_factor
        return 0.0

    def step(self, action):
        action = int(action)

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

        # 【关键修改 2】：安全互锁与动作成本解耦
        """
        当它在没冰的时候瞎按，系统只会轻轻拍一下它的手（扣 1 分），告诉它现在按没用。
        这种极其温和的惩罚，保护了它的探索欲。
        等到冰真的结到 3.0 毫米以上时，它依然有胆量去尝试融冰，
        并最终发现融冰能帮它活到最后、赚取丰厚的存活奖励。
        """
        deice_effect = 0.0
        current_factor = 0.5 + (max(0.0, line_current) / 500.0)
        base_melt_cost = 10.0 * current_factor

        if action == 2:
            if self.current_ice_thickness >= self.min_deice_thickness:
                deice_effect = -self.deice_amount
                self.current_ice_thickness += deice_effect
                action_cost = base_melt_cost # 融冰成功，扣除高额成本
                # self.is_warned = False # 可选：融冰后重置预警状态
            else:
                deice_effect = 0.0
                action_cost = 1.0  # 触发安全互锁，只象征性扣 1 分误操作费
        elif action == 1:
            action_cost = 1.0      # 预警扣 1 分
        else:
            action_cost = 0.0      # 无操作扣 0 分

        self.current_ice_thickness = max(0.0, self.current_ice_thickness)

        self.current_step += 1
        self.state = self._get_observation()

        hazard = bool(self.current_ice_thickness >= self.critical_ice_thickness)

        # 【关键修改 3】：大幅度提高死亡惩罚
        if hazard:
            risk_component = -5.0 if self.is_warned else -20.0
        else:
            risk_component = 0.0

        reliability_component = -float(self.current_ice_thickness / self.critical_ice_thickness)
        reliability_component = float(np.clip(reliability_component, -2.0, 0.0))

        compliance_penalty = 0.0
        if self.current_ice_thickness >= self.warn_ice_thickness and not self.is_warned:
            compliance_penalty = -0.5

        # 使用我们上面单独计算的 action_cost，而不是去调旧函数
        cost = float(action_cost)

        # 【关键修改 4】：存活奖励
        survival_bonus = 0.0 if hazard else 5.0

        # 【关键修改 5】：把所有奖励全部加起来
        reward = (
                self.w_risk * risk_component
                + self.w_reliability * reliability_component
                + self.w_reliability * compliance_penalty
                - self.w_cost * cost
                + survival_bonus  # 加上存活奖励
        )

        terminated = bool(hazard)
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "ice_thickness": float(self.current_ice_thickness),
            "ice_growth": float(ice_growth),
            "deice_effect": float(deice_effect),
            "hazard": bool(hazard),
            "is_warned": bool(self.is_warned),
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