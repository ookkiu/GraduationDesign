import argparse

import numpy as np
import pandas as pd


def generate_mock_weather(filename: str, hours: int = 168, seed: int = 0) -> None:
    """
    生成一段“冷空气过程”的模拟气象/运行数据，用作训练与评估的可复现实验数据。

    CSV 字段:
      timestamp, temperature, wind_speed, wind_angle, humidity, line_current
    """
    rng = np.random.default_rng(seed)
    time_index = pd.date_range(start="2026-01-01 00:00", periods=hours, freq="h")

    # 气温：从 5℃ 逐步降到 -8℃，叠加日变化
    base_temp = np.linspace(5.0, -8.0, hours)
    diurnal_temp = 3.0 * np.sin(np.linspace(0, 7 * 2 * np.pi, hours))
    temperature = base_temp + diurnal_temp + rng.normal(0.0, 0.3, hours)

    # 风速：冷空气增强时逐步增大，带随机扰动
    wind_speed = np.linspace(3.0, 10.0, hours) + rng.normal(0.0, 1.5, hours)
    wind_speed = np.clip(wind_speed, 0.0, 35.0)

    # 风向夹角：0-90°
    wind_angle = rng.uniform(30.0, 90.0, hours)

    # 湿度：降温通常伴随湿度增加
    humidity = np.linspace(60.0, 95.0, hours) + rng.normal(0.0, 5.0, hours)
    humidity = np.clip(humidity, 0.0, 100.0)

    # 线路电流：围绕某个均值波动
    line_current = rng.normal(250.0, 20.0, hours)
    line_current = np.clip(line_current, 0.0, 1000.0)

    df = pd.DataFrame(
        {
            "timestamp": time_index,
            "temperature": np.round(temperature, 2),
            "wind_speed": np.round(wind_speed, 2),
            "wind_angle": np.round(wind_angle, 2),
            "humidity": np.round(humidity, 2),
            "line_current": np.round(line_current, 2),
        }
    )
    df.to_csv(filename, index=False)


def parse_args():
    p = argparse.ArgumentParser(description="Generate mock weather/operation data for the icing RL env.")
    p.add_argument("--out", type=str, default="cold_wave_data.csv")
    p.add_argument("--hours", type=int, default=168)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    generate_mock_weather(filename=args.out, hours=args.hours, seed=args.seed)
    print(f"数据已生成: {args.out} (hours={args.hours}, seed={args.seed})")


if __name__ == "__main__":
    main()
