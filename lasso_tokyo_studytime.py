import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# データの読み込み（パスは自身の環境に合わせて変更）
df = pd.read_csv("tokyo_data.csv")

# 実データ部分（2行目以降）を取得
data = df.iloc[1:].copy()

# 目的変数（「学力(%)」列）を float に変換
y = data["目的変数"].astype(float)

# 説明変数：列番号で選択（14個の学習時間データなど）
X = data.iloc[:, 4:13].astype(float)

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso回帰（3分割交差検証に設定）
lasso = LassoCV(cv=3, random_state=0).fit(X_scaled, y)

# 回帰係数と決定係数の出力
print("回帰係数:", lasso.coef_)
print("決定係数 R²:", lasso.score(X_scaled, y))
