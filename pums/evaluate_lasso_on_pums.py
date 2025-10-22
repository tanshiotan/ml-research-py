import pandas as pd
import copy
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import time  # --- 変更点 1: timeライブラリをインポート ---

# --- CSVファイルを読み込む設定 ---
INPUT_CSV_PATH = "pums_wa_cleaned_full.csv"
TARGET_COLUMN = 'WAGP'

# --- 関数定義（変更なし） ---
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
    for i in range(max_iter):
        beta_old = copy.copy(beta)
        
        for j in range(p):
            r_j = y_std - (np.dot(X_std, beta) - X_std[:, j] * beta[j])
            z = np.dot(X_std[:, j], r_j - eta[j]) / n
            beta[j] = soft_th(lam, z)
            
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
    X_bar = np.mean(X, axis=0)
    X = X - X_bar
    
    X_sd = np.std(X, axis=0)
    X_sd[X_sd == 0] = 1 # 標準偏差が0の場合のゼロ除算を防ぐ
    if standardize:
        X = X / X_sd
        
    y_bar = np.mean(y)
    y = y - y_bar
    return X, y, X_bar, X_sd, y_bar

# --- メイン処理 ---

# 1. データ準備
try:
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"'{INPUT_CSV_PATH}' を読み込みました。({len(df)}行, {len(df.columns)}列)")
except FileNotFoundError:
    print(f"エラー: ファイル '{INPUT_CSV_PATH}' が見つかりません。")
    exit()

y_pd = df[TARGET_COLUMN]
X_pd = df.drop(columns=[TARGET_COLUMN])
X = X_pd.values
y = y_pd.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"データを訓練用({len(X_train)}件)とテスト用({len(X_test)}件)に分割しました。")

lambda_seq = np.logspace(0, 4, 100)
r = len(lambda_seq)
rmse_no_noise = np.zeros(r)
rmse_noise = np.zeros(r)

# --- 変更点 2: 計算開始前に時刻を記録 ---
print("\nStarting RMSE calculation for both scenarios...")
start_time = time.time()

# 2. 「ノイズなし」の計算と評価
print("Calculating and evaluating WITHOUT noise...")
for i, lam in enumerate(lambda_seq):
    beta, beta_0 = linear_lasso(X_train, y_train, lam=lam, eta=None)
    y_pred = np.dot(X_test, beta) + beta_0
    rmse_no_noise[i] = np.sqrt(np.mean((y_test - y_pred)**2))

# 3. 「ノイズあり」の計算と評価
noise_variance_value = 0.5
noise_std_dev = np.sqrt(noise_variance_value)
n_train, p = X_train.shape
unique_eta = np.random.normal(loc=0, scale=noise_std_dev, size=(p, n_train))

print(f"Calculating and evaluating WITH unique noise (Variance Σ={noise_variance_value})...")
for i, lam in enumerate(lambda_seq):
    beta, beta_0 = linear_lasso(X_train, y_train, lam=lam, eta=unique_eta)
    y_pred = np.dot(X_test, beta) + beta_0
    rmse_noise[i] = np.sqrt(np.mean((y_test - y_pred)**2))

# --- 変更点 3: 処理が終わったら経過時間を出力 ---
end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = elapsed_time % 60

print("Calculation finished.")
print(f"Total Elapsed Time: {minutes} minutes and {seconds:.2f} seconds.") # 経過時間を表示

# 4. グラフ描画
plt.figure(figsize=(10, 7))
plt.plot(np.log(lambda_seq), rmse_no_noise, 'r-', label='RMSE (No Noise)')
plt.plot(np.log(lambda_seq), rmse_noise, 'b--', label=f'RMSE (With Unique Noise, Σ={noise_variance_value})')
plt.xlabel(r"$\log(\lambda)$")
plt.ylabel("RMSE (Prediction Error)")
plt.title("Model Error (RMSE) vs. Regularization Strength for PUMS Data")
plt.grid(True)
plt.legend()
plt.show()
