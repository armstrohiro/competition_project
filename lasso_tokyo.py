# lasso_tokyo.py
# Lasso回帰による特徴量選択

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# CSVファイル読み込み（1行目を列名として読み込む）
df = pd.read_csv("tokyo_data1.csv", header=0)

# 目的変数と説明変数の抽出（変数14を除外）
y = df["学力(%)"]
X = df.loc[:, "1":"13"]  # 変数1〜13まで
#y = df["学力(%)"]
#X = df.loc[:, "1":"13"]  # 変数1〜13まで(データが確定次第14追加)

# 数値に変換
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

#########################################################
# 欠損値があれば除去（行単位で）
#data = pd.concat([X, y], axis=1).dropna()
#X = data.drop(columns="学力(%)")
#y = data["学力(%)"]
#########################################################

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LassoCVの学習（3分割交差検証）
lasso = LassoCV(cv=3, max_iter=10000, random_state=0)
lasso.fit(X_scaled, y)

# 非ゼロ係数（選ばれた特徴量）を表示
coef = pd.Series(lasso.coef_, index=X.columns)
print("==== Lasso回帰 選ばれた説明変数（非ゼロ係数） ====")
print(coef[coef != 0])

# 最適なalpha値の表示
print(f"\n最適な正則化パラメータ (alpha): {lasso.alpha_}")

# 係数をグラフ表示
coef.plot(kind="barh", title="Lasso Feature Selection", figsize=(10, 6))
plt.xlabel("coefficient")
plt.ylabel("Explanatory variable")
plt.tight_layout()
plt.savefig("lasso_tokyo_result.png")  # 結果を画像ファイルとして保存
plt.show()
