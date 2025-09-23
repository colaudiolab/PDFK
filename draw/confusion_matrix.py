import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# 加载你的特征和标签
features = np.load("../tsne/features_ER.npy")  # shape: (N, D)
labels = np.load("../tsne/labels_ER.npy")      # shape: (N,)

# === 2. 获取预测类别 ===
preds = np.argmax(features, axis=1)

# === 3. 构建并归一化混淆矩阵 ===
cm = confusion_matrix(labels, preds)
cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # 按行归一化

# === 4. 绘制图像 ===
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap='jet')
plt.title('Confusion Matrix')
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.colorbar()

# 可选：不显示坐标轴刻度（适合类别很多时）
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
