"""
项目一：城市二手房房价预测系统
多元线性回归 + MSE 损失 + 梯度下降（BGD / SGD / MBGD），核心算法仅用 numpy 手写实现。

运行：python house_price.py
输出：output/ 目录下的图表、指标与模型参数
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "城市二手房房价预测（1000 条完整数据）(1).txt"
OUT_DIR = Path(__file__).parent / "output"
SEED = 42

FEATURES = ["area", "room_num", "living_room_num", "floor", "distance_subway"]
CONTINUOUS = ["area", "distance_subway"]          # 连续型：中位数填充
DISCRETE = ["room_num", "living_room_num", "floor"]  # 离散型：众数填充
TARGET = "price"

MISSING_RATE = 0.02     # 模拟 2% 缺失率
AREA_LIMIT = 200        # 面积 > 200㎡ 视为异常
TEST_RATIO = 0.2
LR = 0.01
EPOCHS = 1000
BATCH_SIZE = 32
LR_GRID = [0.001, 0.01, 0.1, 0.3]   # BGD 学习率实验
FINAL_LR = 0.1                     # 最终模型使用的学习率

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "Noto Sans SC",
                                   "Microsoft YaHei", "PingFang SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------------
# 1. 数据加载与预处理
# ----------------------------------------------------------------------------
def load_data(path=DATA_FILE):
    # 原始文件中部分行带有 "# 异常值" / "# 缺失值" 注释，用 comment 参数去掉
    df = pd.read_csv(path, comment="#", skipinitialspace=True)
    return df.apply(pd.to_numeric, errors="coerce")


def simulate_missing(df, rate=MISSING_RATE, seed=SEED):
    """在特征列中随机置空，使特征单元格的总缺失率达到 rate（已有缺失计入在内）。"""
    rng = np.random.default_rng(seed)
    df = df.copy()
    values = df[FEATURES].to_numpy(dtype=float, copy=True)
    target_missing = int(round(values.size * rate))
    already = int(np.isnan(values).sum())
    need = max(target_missing - already, 0)
    candidates = np.argwhere(~np.isnan(values))
    picked = candidates[rng.choice(len(candidates), size=need, replace=False)]
    values[picked[:, 0], picked[:, 1]] = np.nan
    df[FEATURES] = values
    return df, already, need


def remove_outliers(df, limit=AREA_LIMIT):
    """删除面积 > limit 的异常样本（面积缺失的行保留，后续填充）。"""
    mask = df["area"] > limit
    return df[~mask].reset_index(drop=True), df[mask]


def train_test_split(df, test_ratio=TEST_RATIO, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(round(len(df) * test_ratio))
    return df.iloc[idx[n_test:]].reset_index(drop=True), df.iloc[idx[:n_test]].reset_index(drop=True)


class Preprocessor:
    """缺失值填充 + Z-score 标准化。统计量只在训练集上拟合，避免数据泄漏。"""

    def fit(self, df):
        self.fill_values = {}
        for c in CONTINUOUS:
            self.fill_values[c] = float(df[c].median())
        for c in DISCRETE:
            self.fill_values[c] = float(df[c].mode().iloc[0])
        filled = df[FEATURES].fillna(self.fill_values)
        self.mean = filled.mean().to_numpy()
        self.std = filled.std(ddof=0).to_numpy()
        return self

    def transform(self, df):
        X = df[FEATURES].fillna(self.fill_values).to_numpy(dtype=float)
        return (X - self.mean) / self.std


def add_bias(X):
    """在特征矩阵前加一列 1，对应偏置项 w0。"""
    return np.hstack([np.ones((X.shape[0], 1)), X])


# ----------------------------------------------------------------------------
# 2. 模型：y = w0 + w1*area + w2*room_num + w3*living_room_num + w4*floor + w5*distance_subway
#    损失：L(w) = 1/(2m) * Σ (y_i - ŷ_i)^2
#    梯度：∂L/∂w_j = 1/m * Σ (ŷ_i - y_i) * x_ij      （x_i0 = 1）
#    向量形式：∇L = 1/m * Xᵀ(Xw - y)
# ----------------------------------------------------------------------------
def predict(Xb, w):
    return Xb @ w


def mse_loss(Xb, y, w):
    r = predict(Xb, w) - y
    return float(r @ r) / (2 * len(y))


def gradient(Xb, y, w):
    return Xb.T @ (predict(Xb, w) - y) / len(y)


def gradient_descent(Xb, y, lr=LR, epochs=EPOCHS, batch_size=None, seed=SEED):
    """
    batch_size=None -> BGD（每次用全部样本）
    batch_size=1    -> SGD（每次 1 个样本）
    batch_size=k    -> MBGD（每次 k 个样本）
    每个 epoch 结束后在整个训练集上记录一次损失；同时记录每次参数更新后的损失（用于按更新次数对比）。
    """
    rng = np.random.default_rng(seed)
    m, n = Xb.shape
    w = np.zeros(n)
    bs = m if batch_size is None else batch_size
    epoch_loss = [mse_loss(Xb, y, w)]
    update_loss = [epoch_loss[0]]
    t0 = time.perf_counter()
    for _ in range(epochs):
        idx = rng.permutation(m) if bs < m else np.arange(m)
        for start in range(0, m, bs):
            b = idx[start:start + bs]
            w -= lr * gradient(Xb[b], y[b], w)
            if len(update_loss) < 20000:  # 只保留前 2 万次更新，够画图
                update_loss.append(mse_loss(Xb, y, w))
        epoch_loss.append(mse_loss(Xb, y, w))
    elapsed = time.perf_counter() - t0
    return w, np.array(epoch_loss), np.array(update_loss), elapsed


def normal_equation(Xb, y):
    """解析解 w = (XᵀX)⁻¹Xᵀy，仅作为梯度下降结果的参照。"""
    return np.linalg.solve(Xb.T @ Xb, Xb.T @ y)


# ----------------------------------------------------------------------------
# 3. 评估指标
# ----------------------------------------------------------------------------
def metrics(y_true, y_pred):
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    r2 = 1 - float(np.sum(err ** 2)) / float(np.sum((y_true - y_true.mean()) ** 2))
    return {"R2": r2, "MAE": mae, "MSE": mse, "RMSE": mse ** 0.5}


def to_original_scale(w, prep):
    """把标准化空间的权重换算回原始单位：price = b + Σ coef_j * x_j。"""
    coef = w[1:] / prep.std
    bias = w[0] - float(np.sum(w[1:] * prep.mean / prep.std))
    return bias, coef


def epochs_to_converge(loss, ref_loss, tol=0.01):
    """损失首次进入 ref_loss*(1+tol) 以内的 epoch；未达到返回 None。"""
    hit = np.where(loss <= ref_loss * (1 + tol))[0]
    return int(hit[0]) if len(hit) else None


def ols_t_stats(Xb, y, w):
    """OLS 系数的标准误与 t 值：SE = sqrt(σ² · diag((XᵀX)⁻¹))，σ² = RSS/(m-n)。"""
    m, n = Xb.shape
    r = y - Xb @ w
    sigma2 = float(r @ r) / (m - n)
    se = np.sqrt(sigma2 * np.diag(np.linalg.inv(Xb.T @ Xb)))
    return se, w / se


def compute_vif(X):
    """方差膨胀因子：VIF_j = 1/(1-R_j²)，R_j² 为第 j 个特征对其余特征回归的决定系数。"""
    out = []
    for j in range(X.shape[1]):
        others = add_bias(np.delete(X, j, axis=1))
        beta = np.linalg.lstsq(others, X[:, j], rcond=None)[0]
        r = X[:, j] - others @ beta
        r2 = 1 - (r @ r) / np.sum((X[:, j] - X[:, j].mean()) ** 2)
        out.append(1 / (1 - r2))
    return np.array(out)


# ----------------------------------------------------------------------------
# 4. 可视化
# ----------------------------------------------------------------------------
COLORS = {"BGD": "#2563eb", "SGD": "#dc2626", "MBGD": "#16a34a"}


def plot_bgd_curve(loss, ref, path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(loss, color=COLORS["BGD"], lw=2, label="BGD 训练损失")
    ax.axhline(ref, color="gray", ls="--", lw=1, label=f"解析解最优损失 = {ref:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("MSE 损失 L(w)（对数轴）")
    ax.set_title(f"批量梯度下降损失收敛曲线（lr={LR}, 迭代 {EPOCHS} 次）")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_compare(results, ref, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    for name, r in results.items():
        el = r["epoch_loss"]
        ax.plot(np.arange(1, len(el)), el[1:], color=COLORS[name], lw=1.8, label=name)
    ax.axhline(ref, color="gray", ls="--", lw=1, label="解析解最优损失")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch（遍历训练集的轮数，对数轴）")
    ax.set_ylabel("训练集 MSE 损失（对数轴）")
    ax.set_title("按 Epoch 对比")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    styles = {"BGD": dict(lw=2.2, ls="--", zorder=3), "SGD": dict(lw=1.0, zorder=1), "MBGD": dict(lw=1.4, zorder=2)}
    for name, r in results.items():
        ax.plot(r["update_loss"][:3000], color=COLORS[name], label=name, **styles[name])
    ax.axhline(ref, color="gray", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("参数更新次数（前 3000 次）")
    ax.set_ylabel("训练集 MSE 损失（对数轴）")
    ax.set_title("按参数更新次数对比")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle(f"BGD / SGD / MBGD(batch={BATCH_SIZE}) 收敛速度对比（lr={LR}）")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pred_vs_true(y_true, y_pred, m, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(y_true, y_pred, s=22, alpha=0.75, color=COLORS["BGD"], edgecolor="white", lw=0.5)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="gray", ls="--", lw=1, label="y = x（理想预测）")
    ax.set_xlabel("真实房价（万元）")
    ax.set_ylabel("预测房价（万元）")
    ax.set_title(f"测试集预测值 vs 真实值\nR²={m['R2']:.4f}  MAE={m['MAE']:.2f}  MSE={m['MSE']:.2f}")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    res = y_true - y_pred
    ax.scatter(y_pred, res, s=22, alpha=0.75, color="#9333ea", edgecolor="white", lw=0.5)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("预测房价（万元）")
    ax.set_ylabel("残差 = 真实 - 预测（万元）")
    ax.set_title("测试集残差分布")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_weights(w_std, corr, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    order = np.argsort(np.abs(w_std))[::-1]
    names = [FEATURES[i] for i in order]
    vals = w_std[order]
    ax = axes[0]
    bars = ax.barh(names[::-1], vals[::-1], color=["#16a34a" if v > 0 else "#dc2626" for v in vals[::-1]])
    for b, v in zip(bars, vals[::-1]):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:+.2f} ", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("标准化权重（特征每增加 1 个标准差，房价变化的万元数）")
    ax.set_title("模型权重系数（标准化后可直接比较大小）")
    ax.grid(alpha=0.3, axis="x")
    lim = np.abs(vals).max() * 1.35
    ax.set_xlim(-lim if vals.min() < 0 else -lim * 0.15, lim)

    ax = axes[1]
    c = corr[order]
    ax.barh(names[::-1], c[::-1], color=["#16a34a" if v > 0 else "#dc2626" for v in c[::-1]])
    for i, v in enumerate(c[::-1]):
        ax.text(v, i, f" {v:+.3f} ", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-1.3, 1.3)
    ax.set_xlabel("与房价的 Pearson 相关系数")
    ax.set_title("单特征与房价的相关性")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lr_sweep(lr_results, ref, path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    palette = ["#94a3b8", "#2563eb", "#f59e0b", "#16a34a"]
    for (lr, r), col in zip(lr_results.items(), palette):
        conv = r["conv_epoch"]
        ax.plot(r["epoch_loss"], color=col, lw=1.8,
                label=f"lr={lr}（{'第 %d 次收敛' % conv if conv is not None else '1000 次内未收敛'}）")
    ax.axhline(ref, color="gray", ls="--", lw=1, label=f"最优损失 L*={ref:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("训练集 MSE 损失（对数轴）")
    ax.set_title("BGD 不同学习率的收敛情况")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_corr_matrix(df, path):
    cols = FEATURES + [TARGET]
    c = df[cols].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(c[i, j]) > 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("特征相关系数矩阵（训练集）")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    # ---- 1. 预处理 ----
    raw = load_data()
    say("=" * 70)
    say("一、数据预处理")
    say("=" * 70)
    say(f"原始样本数：{len(raw)}，字段：{list(raw.columns)}")
    df, n_orig_missing, n_sim = simulate_missing(raw)
    n_cells = len(df) * len(FEATURES)
    say(f"原始缺失单元格：{n_orig_missing}；为模拟 {MISSING_RATE:.0%} 缺失率额外随机置空：{n_sim}；"
        f"当前缺失率 {df[FEATURES].isna().sum().sum() / n_cells:.2%}")
    say("各特征缺失数：" + ", ".join(f"{c}={int(v)}" for c, v in df[FEATURES].isna().sum().items()))

    df, outliers = remove_outliers(df)
    say(f"删除面积 > {AREA_LIMIT}㎡ 的异常样本 {len(outliers)} 条：" +
        "; ".join(f"area={a}, price={p}" for a, p in zip(outliers["area"], outliers["price"])))
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    say(f"清洗后样本数：{len(df)}")

    train_df, test_df = train_test_split(df)
    say(f"训练集 {len(train_df)} 条（80%），测试集 {len(test_df)} 条（20%）")

    prep = Preprocessor().fit(train_df)
    say("缺失值填充（训练集统计量）：连续型用中位数，离散型用众数 -> " +
        ", ".join(f"{k}={v:g}" for k, v in prep.fill_values.items()))
    say("Z-score 标准化：x' = (x - μ) / σ")
    for c, mu, sd in zip(FEATURES, prep.mean, prep.std):
        say(f"    {c:<16} μ={mu:8.3f}  σ={sd:7.3f}")

    X_train, X_test = add_bias(prep.transform(train_df)), add_bias(prep.transform(test_df))
    y_train, y_test = train_df[TARGET].to_numpy(float), test_df[TARGET].to_numpy(float)

    # ---- 2/3. 训练 ----
    say()
    say("=" * 70)
    say("二、模型训练（梯度下降）")
    say("=" * 70)
    w_ne = normal_equation(X_train, y_train)
    ref_loss = mse_loss(X_train, y_train, w_ne)
    say(f"参照：正规方程解析解的最优训练损失 L* = {ref_loss:.4f}")

    configs = {"BGD": None, "SGD": 1, "MBGD": BATCH_SIZE}
    results = {}
    for name, bs in configs.items():
        w, el, ul, t = gradient_descent(X_train, y_train, batch_size=bs)
        conv = epochs_to_converge(el, ref_loss)
        updates_per_epoch = int(np.ceil(len(y_train) / (bs or len(y_train))))
        results[name] = dict(w=w, epoch_loss=el, update_loss=ul, time=t, conv_epoch=conv,
                             updates_per_epoch=updates_per_epoch)
        say(f"{name:<5} batch={bs or len(y_train):<4} 每轮更新 {updates_per_epoch:>3} 次 | "
            f"最终训练损失 {el[-1]:.4f} | 损失降到 L*×1.01 以内用了 "
            f"{conv if conv is not None else '未达到'} 轮"
            f"{'' if conv is None else f'（{conv * updates_per_epoch} 次更新）'} | 耗时 {t:.2f}s")

    say("（三者使用相同学习率 lr=0.01，便于公平比较收敛速度）")

    # ---- 学习率实验：解释 BGD 在 lr=0.01 下为何 1000 次未收敛 ----
    say()
    eig = np.linalg.eigvalsh(X_train.T @ X_train / len(y_train))
    say(f"Hessian XᵀX/m 特征值：{np.round(eig, 3).tolist()}，条件数 κ = {eig[-1] / eig[0]:.1f}")
    say(f"BGD 稳定的学习率上界 2/λmax = {2 / eig[-1]:.3f}；lr=0.01 时最慢方向每步只收缩 "
        f"(1 - lr·λmin) = {1 - LR * eig[0]:.5f}，1000 步后误差仍剩 {(1 - LR * eig[0]) ** EPOCHS:.1%}")
    lr_results = {}
    for lr in LR_GRID:
        w, el, _, _ = gradient_descent(X_train, y_train, lr=lr)
        conv = epochs_to_converge(el, ref_loss)
        lr_results[lr] = dict(w=w, epoch_loss=el, conv_epoch=conv)
        say(f"BGD lr={lr:<6} 最终训练损失 {el[-1]:10.4f} | 收敛轮数 {conv if conv is not None else '未达到'}"
            f" | 与解析解权重最大偏差 {np.abs(w - w_ne).max():.4f}")
    w_final = lr_results[FINAL_LR]["w"]
    say(f"最终模型：BGD，lr={FINAL_LR}，迭代 {EPOCHS} 次（已收敛到解析解）")

    # ---- 4. 评估 ----
    say()
    say("=" * 70)
    say("三、模型评估")
    say("=" * 70)
    all_metrics = {}
    candidates = [(f"BGD(lr={FINAL_LR})", w_final), (f"BGD(lr={LR})", results["BGD"]["w"]),
                  (f"SGD(lr={LR})", results["SGD"]["w"]), (f"MBGD(lr={LR})", results["MBGD"]["w"]),
                  ("正规方程", w_ne)]
    say(f"{'模型':<16}{'数据集':<6}{'R²':>9}{'MAE':>9}{'MSE':>11}{'RMSE':>9}")
    for name, w in candidates:
        all_metrics[name] = {}
        for split, Xs, ys in [("train", X_train, y_train), ("test", X_test, y_test)]:
            mt = metrics(ys, predict(Xs, w))
            all_metrics[name][split] = mt
            say(f"{name:<16}{'训练' if split == 'train' else '测试':<6}"
                f"{mt['R2']:>9.4f}{mt['MAE']:>9.3f}{mt['MSE']:>11.3f}{mt['RMSE']:>9.3f}")

    # ---- 5. 特征分析 ----
    say()
    say("=" * 70)
    say("四、特征分析（最终模型）")
    say("=" * 70)
    bias, coef = to_original_scale(w_final, prep)
    corr = train_df.fillna(prep.fill_values)[FEATURES + [TARGET]].corr()[TARGET][FEATURES].to_numpy()
    se, t = ols_t_stats(X_train, y_train, w_final)
    vif = compute_vif(X_train[:, 1:])
    say(f"{'特征':<16}{'标准化权重':>10}{'原始单位系数':>12}{'t 值':>9}{'相关系数':>10}{'VIF':>8}")
    say(f"{'w0 偏置':<16}{w_final[0]:>12.3f}{bias:>16.3f}{t[0]:>11.1f}")
    for i, c in enumerate(FEATURES):
        say(f"{c:<16}{w_final[i + 1]:>12.3f}{coef[i]:>16.3f}{t[i + 1]:>11.2f}{corr[i]:>12.3f}{vif[i]:>9.1f}")
    say("原始单位方程：price = {:.3f} ".format(bias) +
        " ".join(f"{'+' if v >= 0 else '-'} {abs(v):.3f}×{c}" for c, v in zip(FEATURES, coef)))
    rank = np.argsort(np.abs(w_final[1:]))[::-1]
    say("影响力排序（|标准化权重|）：" + " > ".join(FEATURES[i] for i in rank))
    say("注：|t| > 1.96 表示在 5% 水平上显著；VIF > 10 表示存在严重多重共线性。")

    # ---- 图表 ----
    plot_bgd_curve(results["BGD"]["epoch_loss"], ref_loss, OUT_DIR / "1_bgd_loss_curve.png")
    plot_compare(results, ref_loss, OUT_DIR / "2_gd_comparison.png")
    plot_lr_sweep(lr_results, ref_loss, OUT_DIR / "3_bgd_learning_rate.png")
    plot_pred_vs_true(y_test, predict(X_test, w_final), all_metrics[f"BGD(lr={FINAL_LR})"]["test"],
                      OUT_DIR / "4_pred_vs_true.png")
    plot_weights(w_final[1:], corr, OUT_DIR / "5_feature_weights.png")
    plot_corr_matrix(train_df.fillna(prep.fill_values), OUT_DIR / "6_correlation_matrix.png")

    # ---- 保存模型与结果 ----
    model = {
        "features": FEATURES,
        "optimizer": f"BGD lr={FINAL_LR} epochs={EPOCHS}",
        "standardized": {"w0": float(w_final[0]), "w": w_final[1:].tolist(),
                         "mean": prep.mean.tolist(), "std": prep.std.tolist()},
        "original_scale": {"bias": bias, "coef": dict(zip(FEATURES, coef.tolist()))},
        "fill_values": prep.fill_values,
    }
    (OUT_DIR / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "n_raw": len(raw), "n_clean": len(df), "n_train": len(train_df), "n_test": len(test_df),
        "outliers_removed": len(outliers), "ref_loss": ref_loss, "hessian_eigenvalues": eig.tolist(),
        "optimizers_lr_0.01": {k: {"final_loss": float(v["epoch_loss"][-1]), "conv_epoch": v["conv_epoch"],
                                   "updates_per_epoch": v["updates_per_epoch"], "time_s": v["time"],
                                   "weights": v["w"].tolist()} for k, v in results.items()},
        "bgd_lr_sweep": {str(k): {"final_loss": float(v["epoch_loss"][-1]), "conv_epoch": v["conv_epoch"]}
                         for k, v in lr_results.items()},
        "metrics": all_metrics,
        "feature_analysis": {c: {"w_std": float(w_final[i + 1]), "coef": float(coef[i]), "t": float(t[i + 1]),
                                 "corr": float(corr[i]), "vif": float(vif[i])} for i, c in enumerate(FEATURES)},
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "run_log.txt").write_text("\n".join(log), encoding="utf-8")
    say()
    say(f"图表与结果已保存到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
