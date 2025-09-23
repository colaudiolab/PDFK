import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ====== 将你的表格原样贴到这里 ======
txt = """range\tstd\tmean\tmethod\ttask
58.18790658\t7.863589165\t10.56800671\tER\t1
35.28519472\t3.529921582\t8.019396737\tER\t2
8.369369249\t0.975936486\t6.892280769\tER\t3
15.48588489\t1.761424028\t7.442720825\tER\t4
30.39187525\t4.059150985\t9.486205283\tER\t5
51.61461145\t3.854476079\t7.174975226\tOurs\t1
14.64589685\t1.537132966\t6.842954747\tOurs\t2
10.24672566\t0.763711763\t6.689668908\tOurs\t3
11.34916326\t0.967891319\t6.66140407\tOurs\t4
6.149572858\t0.602280602\t6.657972045\tOurs\t5
"""

# 读入与预处理
df = pd.read_csv(io.StringIO(txt), sep=r"\s+")
df["task"] = df["task"].astype(int)

# 方法重命名以贴近示例图图例
label_map = {"ER": "Baseline", "Ours": "Ours"}
df["label"] = df["method"].map(label_map)

# 颜色（示例图风格：蓝/红）
colors = {"Baseline": "#4C78A8", "Ours": "#E45756"}  # 可按需替换 HEX

# 画图
plt.figure(figsize=(3.2, 2.2), dpi=200)
ax = plt.gca()

for name, g in df.sort_values("task").groupby("label"):
    x = g["task"].values
    y = g["mean"].values
    s = g["std"].values

    # 带状（±std）
    ax.fill_between(x, y - s, y + s,
                    color=colors[name], alpha=0.18, linewidth=0)

    # 折线 + 圆点
    ax.plot(x, y, "-x", lw=2.0, ms=4.0,
            color=colors[name], label=name)

# 轴与网格（仿示例图）
ax.set_xlabel("Task")
ax.set_ylabel("Flatness(mean ± std)")  # 如需别的单位可改
ax.set_xticks(sorted(df["task"].unique()))
ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.4)

# 图例
ax.legend(frameon=False, loc="best")
plt.savefig("flatness_metrics_plot.pdf",bbox_inches="tight")
plt.tight_layout()
plt.show()

