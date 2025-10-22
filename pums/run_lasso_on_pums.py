import pandas as pd
import copy
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats
import time  # --- 変更点 1: timeライブラリをインポート ---

# --- CSVファイルを読み込む設定 ---
# 前処理済みのPUMSデータファイル名
INPUT_CSV_PATH = "pums_wa_cleaned_full.csv"
# 目的変数（予測したいもの）の列名
TARGET_COLUMN = 'WAGP'

def soft_th (lam, x):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

# --- 高速化版の linear_lasso 関数 ---
def linear_lasso(X, y, lam=0, beta=None, eta=None):
    n, p = X.shape
    if beta is None:
        beta = np.zeros(p)
    X_std, y_std, X_bar, X_sd, y_bar = centralize(X, y)
    
    if eta is None:
        eta = np.zeros((p, n))

    max_iter = 500
    
    r_std = y_std - np.dot(X_std, beta)
    
    for i in range(max_iter):
        beta_old = copy.copy(beta)
        
        for j in range(p):
            r_std += X_std[:, j] * beta[j]
            z = np.dot(X_std[:, j], r_std - eta[j]) / n
            beta[j] = soft_th(lam, z)
            r_std -= X_std[:, j] * beta[j]
            
        eps = np.linalg.norm(beta - beta_old, 2)
        if eps < 0.0001:
            break
            
    beta = beta / X_sd
    beta_0 = y_bar - np.dot(X_bar, beta)
    return beta, beta_0

def centralize(X0, y0, standardize=True):
    X = copy.copy(X0)
    y = copy.copy(y0)
    n, p = X.shape
    X_bar = np.zeros(p)
    X_sd = np.zeros(p)
    for j in range(p):
        X_bar[j] = np.mean(X[:, j])
        X[:, j] = X[:, j] - X_bar[j]
        sd_val = np.std(X[:, j])
        X_sd[j] = sd_val if sd_val > 0 else 1
        if standardize is True:
            X[:, j] = X[:, j] / X_sd[j]
            
    y_bar = np.mean(y)
    y = y - y_bar
    return X, y, X_bar, X_sd, y_bar

# --- データをCSVから読み込む ---
try:
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"'{INPUT_CSV_PATH}' を読み込みました。({len(df)}行, {len(df.columns)}列)")
except FileNotFoundError:
    print(f"エラー: ファイル '{INPUT_CSV_PATH}' が見つかりません。")
    exit()

# 説明変数 X と 目的変数 y に分割
y_pd = df[TARGET_COLUMN]
X_pd = df.drop(columns=[TARGET_COLUMN])

# 特徴量名を取得
feature_names = X_pd.columns.tolist()

# Pandas DataFrameをNumPy配列に変換
x = X_pd.values
y = y_pd.values

n, p = x.shape
print(f"データ準備完了: n={n}, p={p}")


# --- LASSOパスの計算 ---

lambda_seq = np.logspace(0, 12, 1000)
r = len(lambda_seq)

# 加えるノイズの分散を決める
noise_variance_value = 0.5
noise_std_dev = np.sqrt(noise_variance_value)

# 一度だけノイズを生成する
unique_eta = np.random.normal(loc=0, scale=noise_std_dev, size=(p, n))

# 計算結果を保存する配列
coef_seq_noise = np.zeros((r, p))

# --- 変更点 2: 計算開始前に時刻を記録 ---
print(f"Calculating LASSO path WITH unique noise (Variance Σ={noise_variance_value})...")
start_time = time.time()

for i in range(r):
    coef_seq_noise[i, :], _ = linear_lasso(x, y, lambda_seq[i], eta=unique_eta)

# --- 変更点 3: 処理が終わったら経過時間を出力 ---
end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = elapsed_time % 60

print("Calculation finished.")
print(f"Elapsed Time: {minutes} minutes and {seconds:.2f} seconds.") # 経過時間を表示


# --- グラフの描画 ---
plt.figure(figsize=(12, 8))

# 各係数のパスをプロットする
for j in range(p):
    plt.plot(np.log(lambda_seq), coef_seq_noise[:, j], label=feature_names[j])

plt.xlabel(r"$\log(\lambda)$")
plt.ylabel(r"Coefficients $\beta$")
plt.title(f"LASSO Path for PUMS Washington Data (Variance Σ={noise_variance_value})")
plt.grid(True)
# 特徴量が多すぎるため、凡例は表示しない方が見やすい
# plt.legend(loc="upper right")
plt.show()