import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# CSVファイルを読み込む（前回の修正コードと同様）
df = pd.read_csv("tokyo_data1.csv", header=None)

# 目的変数と説明変数の抽出
y = df[0]
X = df.loc[:, 1:13] # 変数1〜13まで

# 数値に変換 (エラーが発生した場合はNaNにする)
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# 目的変数または説明変数にNaNが含まれる行を削除
data_for_correlation = pd.concat([X, y], axis=1).dropna()

# --- ここから相関係数計算と表示 ---

print("\n==== 全ての説明変数と目的変数の相関係数 ====")
# 目的変数と各説明変数との相関係数を計算
# df.corr() を使うと全ての列間の相関行列が得られるので、そこから目的変数の行/列を抽出します
correlation_with_target = data_for_correlation.corr()[0].drop(0)
print(correlation_with_target.sort_values(ascending=False))

print("\n==== 説明変数同士の相関係数（上位/下位10組） ====")
# 説明変数同士の相関行列を計算
corr_matrix_X = data_for_correlation.drop(columns=0).corr()

# 重複を避けて、絶対値が大きい順にソートして表示
# 上三角行列を取得（自身との相関と重複を除くため）
corr_pairs = corr_matrix_X.unstack()
sorted_corr_pairs = corr_pairs.sort_values(key=abs, ascending=False)

# 自分自身との相関 (1) と重複するペアを除外
unique_sorted_corr_pairs = sorted_corr_pairs[sorted_corr_pairs != 1]
unique_sorted_corr_pairs = unique_sorted_corr_pairs.drop_duplicates() # 重複するペア (A-BとB-A) を削除

print("絶対値が高い順（上位10組）:")
print(unique_sorted_corr_pairs.head(10))

print("\n==== 相関ヒートマップ（説明変数同士） ====")
plt.figure(figsize=(14, 12)) # グラフサイズを調整
sns.heatmap(corr_matrix_X, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Independent Variables')
plt.tight_layout()
plt.savefig("independent_variables_correlation_heatmap.png")
plt.show()

# --- ここまで相関係数計算と表示 ---

# 以下、Lasso回帰のコード（省略、上記分析後に実行）
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LassoCV
# from sklearn.metrics import mean_squared_error, r2_score
# ... (前回のLasso回帰コードを続けて実行)