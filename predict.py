"""
用训练好的模型预测房价。先运行 house_price.py 生成 output/model.json。

示例：
  python predict.py --area 89 --room 2 --living 1 --floor 15 --subway 0.8
  python predict.py            # 不带参数则进入交互式输入
"""
import argparse
import json
from pathlib import Path

import numpy as np

MODEL_FILE = Path(__file__).parent / "output" / "model.json"
ARGS = [("area", "建筑面积（㎡）"), ("room", "卧室数量"), ("living", "客厅数量"),
        ("floor", "所在楼层"), ("subway", "距地铁站距离（km）")]


def load_model(path=MODEL_FILE):
    m = json.loads(path.read_text(encoding="utf-8"))
    s = m["standardized"]
    return s["w0"], np.array(s["w"]), np.array(s["mean"]), np.array(s["std"])


def predict_price(x, model):
    w0, w, mean, std = model
    return float(w0 + ((np.asarray(x, float) - mean) / std) @ w)


def main():
    ap = argparse.ArgumentParser(description="城市二手房房价预测")
    for name, desc in ARGS:
        ap.add_argument(f"--{name}", type=float, help=desc)
    a = vars(ap.parse_args())
    x = []
    for name, desc in ARGS:
        x.append(a[name] if a[name] is not None else float(input(f"{desc}：")))
    if x[0] > 200:
        print("提示：面积超过 200㎡，超出训练数据范围，预测结果可能不可靠。")
    print(f"预测总价：{predict_price(x, load_model()):.1f} 万元")


if __name__ == "__main__":
    main()
