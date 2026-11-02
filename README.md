# 城市二手房房价预测系统

基于多元线性回归的二手房房价预测。线性回归模型、MSE 损失和梯度下降（BGD / SGD / MBGD）均用 numpy 手写实现，没有使用 sklearn。

## 运行

```bash
pip install -r requirements.txt
python house_price.py      # 预处理、训练、评估、特征分析，结果输出到 output/
python predict.py --area 89 --room 2 --living 1 --floor 15 --subway 0.8
```

## 结果

| 指标（测试集） | 值 |
|---|---|
| R² | 0.991 |
| MAE | 5.26 万元 |
| MSE | 65.3 |

完整的推导、图表和分析见 [项目报告.md](项目报告.md)。

## 文件

- `house_price.py`：主程序
- `predict.py`：用训练好的模型预测房价
- `项目报告.md`：实验报告
- `output/`：图表、模型参数（model.json）、指标（results.json）、运行日志
- `城市二手房房价预测（1000 条完整数据）(1).txt`：数据集
- `项目（一）(1).docx`：项目要求
