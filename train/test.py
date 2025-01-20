import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import load_img, img_to_array, to_categorical
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# 評価用ライブラリ
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ===========
# 設定
# ===========
test_data_path = "datasets"  # テストに使用するデータセット(例: 同じフォルダを流用)
model_path = "model/model_100x100.keras"  # train.py で保存したモデル

image_width = 100
image_height = 100
color_setting = 1  # 1:モノクロ、3:カラー

folder = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "blank",
]
class_number = len(folder)

# ===========
# テストデータ読み込み
# ===========
X_test = []
Y_test = []

# 必要に応じて以下を変更: 例として全データから再度 10% をテスト用に利用
for index, name in enumerate(folder):
    print(f"データセットを {test_data_path}/{name} から読み込みます。")
    read_data = os.path.join(test_data_path, name)
    files = glob.glob(read_data + "/*.jpg")
    for file in files:
        if color_setting == 1:
            img = load_img(
                file, color_mode="grayscale", target_size=(image_width, image_height)
            )
        elif color_setting == 3:
            img = load_img(
                file, color_mode="rgb", target_size=(image_width, image_height)
            )
        array = img_to_array(img)
        X_test.append(array)
        Y_test.append(index)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_test = X_test.astype("float32") / 255.0
Y_test = to_categorical(Y_test, class_number)

# ここで全データからさらに test_split で分割し、
# 本当にテストしたいデータだけを抜き出して使うことも可能
_, X_test_data, _, Y_test_data = train_test_split(
    X_test, Y_test, test_size=0.1, random_state=42
)

# ===========
# モデル読み込み
# ===========
print(f"学習済みモデル {model_path} を読み込みます。")
model = load_model(model_path)

# ===========
# 評価
# ===========
score = model.evaluate(X_test_data, Y_test_data, verbose=0)
print("===== テストデータでの評価 =====")
print(f"Loss: {score[0]:.4f}")
print(f"Accuracy: {score[1]*100:.2f}%")

# ===========
# 予測と結果確認
# ===========
y_pred = model.predict(X_test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(Y_test_data, axis=1)

# 混同行列
cm = confusion_matrix(y_true, y_pred_classes)
print("\n=== Confusion Matrix ===")
print(cm)

# 分類レポート
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred_classes, target_names=folder))

# ヒートマップ表示
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, cmap="Blues", fmt="d", xticklabels=folder, yticklabels=folder
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
