import pandas as pd
import copy
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats
import time

# --- CSVファイルを読み込む設定 ---
INPUT_CSV_PATH = "pums_wa_cleaned_full.csv"
TARGET_COLUMN = 'WAGP'

# --- 関数定義 (ユーザー提供のもの) ---
def soft_th (lam, x):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

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

y_pd = df[TARGET_COLUMN]
X_pd = df.drop(columns=[TARGET_COLUMN])
feature_names = X_pd.columns.tolist()
x = X_pd.values
y = y_pd.values
n, p = x.shape
print(f"データ準備完了: n={n}, p={p}")

# --- GCV（LOOCVの高速近似）計算 ---

lambda_seq = np.logspace(0, 4, 100)
r = len(lambda_seq)

# --- 変更点 1: ノイズあり・なし両方の結果を保存する配列を用意 ---
gcv_errors_noise = np.zeros(r)
dfs_noise = np.zeros(r)
gcv_errors_no_noise = np.zeros(r)
dfs_no_noise = np.zeros(r)

# ノイズの設定
noise_variance_value = 0.5
noise_std_dev = np.sqrt(noise_variance_value)
unique_eta = np.random.normal(loc=0, scale=noise_std_dev, size=(p, n))

print(f"Calculating GCV (Approximate LOOCV) Error...")
start_time = time.time()

# --- 変更点 2: 「ノイズあり」と「ノイズなし」の両方を計算 ---
for i, lam in enumerate(lambda_seq):
    
    # --- 1. ノイズありの場合 ---
    beta_n, beta_0_n = linear_lasso(x, y, lam, eta=unique_eta)
    y_pred_n = np.dot(x, beta_n) + beta_0_n
    mse_n = np.mean((y - y_pred_n)**2)
    df_n = np.sum(np.abs(beta_n) > 1e-6)
    
    denominator_n = (1.0 - df_n / n)
    if denominator_n <= 0:
        gcv_errors_noise[i] = np.nan
    else:
        gcv_errors_noise[i] = mse_n / (denominator_n**2)
    dfs_noise[i] = df_n
    
    # --- 2. ノイズなしの場合 (eta=None) ---
    beta_nn, beta_0_nn = linear_lasso(x, y, lam, eta=None)
    y_pred_nn = np.dot(x, beta_nn) + beta_0_nn
    mse_nn = np.mean((y - y_pred_nn)**2)
    df_nn = np.sum(np.abs(beta_nn) > 1e-6)
    
    denominator_nn = (1.0 - df_nn / n)
    if denominator_nn <= 0:
        gcv_errors_no_noise[i] = np.nan
    else:
        gcv_errors_no_noise[i] = mse_nn / (denominator_nn**2)
    dfs_no_noise[i] = df_nn
    
end_time = time.time()
elapsed_time = end_time - start_time
print(f"GCV Calculation finished in {elapsed_time:.2f} seconds.")

# --- 変更点 3: グラフ描画を修正し、両方のGCV誤差をプロット ---
plt.figure(figsize=(12, 8))

# GCV誤差 (ノイズあり)
plt.plot(np.log(lambda_seq), gcv_errors_noise, 'r-', 
         label=f'GCV Error (With Noise, Σ={noise_variance_value})')

# GCV誤差 (ノイズなし)
plt.plot(np.log(lambda_seq), gcv_errors_no_noise, 'g--', 
         label='GCV Error (No Noise)')

plt.xlabel(r"$\log(\lambda)$")
plt.ylabel('GCV Error (Approx. LOOCV)')
plt.title(f"GCV (Approximate LOOCV) Error Comparison")
plt.grid(True)
plt.legend()
plt.show()

# --- 変更点 4: 両方のシナリオで最も良かったλを表示 ---

# ノイズありのベスト
min_gcv_index_n = np.nanargmin(gcv_errors_noise)
best_lambda_n = lambda_seq[min_gcv_index_n]
min_gcv_n = gcv_errors_noise[min_gcv_index_n]
print(f"\n--- With Noise (Σ={noise_variance_value}) ---")
print(f"Best GCV Error: {min_gcv_n:.4f}")
print(f"  at log(lambda) = {np.log(best_lambda_n):.4f} (lambda = {best_lambda_n:.4f})")
print(f"  with Effective Degrees of Freedom (df) = {dfs_noise[min_gcv_index_n]}")

# ノイズなしのベスト
min_gcv_index_nn = np.nanargmin(gcv_errors_no_noise)
best_lambda_nn = lambda_seq[min_gcv_index_nn]
min_gcv_nn = gcv_errors_no_noise[min_gcv_index_nn]
print(f"\n--- No Noise ---")
print(f"Best GCV Error: {min_gcv_nn:.4f}")
print(f"  at log(lambda) = {np.log(best_lambda_nn):.4f} (lambda = {best_lambda_nn:.4f})")
print(f"  with Effective Degrees of Freedom (df) = {dfs_no_noise[min_gcv_index_nn]}")
