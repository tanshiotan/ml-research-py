import pandas as pd
import copy
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats
from sklearn.model_selection import train_test_split
import time

# --- CSVファイルを読み込む設定 ---
INPUT_CSV_PATH = "pums_wa_cleaned_full.csv"
TARGET_COLUMN = 'WAGP'

# --- 関数定義 (変更なし) ---
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
    X_bar = np.mean(X, axis=0)
    X = X - X_bar
    
    X_sd = np.std(X, axis=0)
    X_sd[X_sd == 0] = 1
    if standardize:
        X = X / X_sd
        
    y_bar = np.mean(y)
    y = y- y_bar
    return X, y, X_bar, X_sd, y_bar

# --- 1. データをCSVから読み込み、分割 ---
try:
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"'{INPUT_CSV_PATH}' を読み込みました。({len(df)}行, {len(df.columns)}列)")
except FileNotFoundError:
    print(f"エラー: ファイル '{INPUT_CSV_PATH}' が見つかりません。")
    exit()

y_pd = df[TARGET_COLUMN]
X_pd = df.drop(columns=[TARGET_COLUMN])
x = X_pd.values
y = y_pd.values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
n_train, p = X_train.shape
n_test = X_test.shape[0]
print(f"データを訓練用({n_train}件)とテスト用({n_test}件)に分割しました。")


# --- 2. GCVとRMSEの計算準備 ---
lambda_seq = np.logspace(0, 4, 100)
r = len(lambda_seq)

rmse_no_noise = np.zeros(r)
gcv_no_noise_mse = np.zeros(r)
rmse_noise = np.zeros(r)
gcv_noise_mse = np.zeros(r)

noise_variance_value = 10000000000
noise_std_dev = np.sqrt(noise_variance_value)
unique_eta = np.random.normal(loc=0, scale=noise_std_dev, size=(p, n_train))

print(f"Calculating RMSE (on test set) and GCV (on train set)...")
start_time = time.time()

# --- 3. 「ノイズあり」と「ノイズなし」の両方を計算 ---
for i, lam in enumerate(lambda_seq):
    
    # --- A. ノイズなし ---
    beta_nn, beta_0_nn = linear_lasso(X_train, y_train, lam, eta=None)
    y_pred_test_nn = np.dot(X_test, beta_nn) + beta_0_nn
    rmse_no_noise[i] = np.sqrt(np.mean((y_test - y_pred_test_nn)**2))
    
    y_pred_train_nn = np.dot(X_train, beta_nn) + beta_0_nn
    mse_train_nn = np.mean((y_train - y_pred_train_nn)**2)
    df_nn = np.sum(np.abs(beta_nn) > 1e-6)
    
    denominator_nn = (1.0 - df_nn / n_train)
    if denominator_nn <= 0:
        gcv_no_noise_mse[i] = np.nan
    else:
        gcv_no_noise_mse[i] = mse_train_nn / (denominator_nn**2)

    # --- B. ノイズあり ---
    beta_n, beta_0_n = linear_lasso(X_train, y_train, lam, eta=unique_eta)
    y_pred_test_n = np.dot(X_test, beta_n) + beta_0_n
    rmse_noise[i] = np.sqrt(np.mean((y_test - y_pred_test_n)**2))
    
    y_pred_train_n = np.dot(X_train, beta_n) + beta_0_n
    mse_train_n = np.mean((y_train - y_pred_train_n)**2)
    df_n = np.sum(np.abs(beta_n) > 1e-6)
    
    denominator_n = (1.0 - df_n / n_train)
    if denominator_n <= 0:
        gcv_noise_mse[i] = np.nan
    else:
        gcv_noise_mse[i] = mse_train_n / (denominator_n**2)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Calculation finished in {elapsed_time:.2f} seconds.")

# GCVをRMSEスケールに変換
gcv_no_noise_rmse = np.sqrt(gcv_no_noise_mse)
gcv_noise_rmse = np.sqrt(gcv_noise_mse)

# --- 4. グラフ描画 (変更なし) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
ax1.plot(np.log(lambda_seq), rmse_no_noise, 'r-', label='RMSE (Actual Test Error)')
ax1.plot(np.log(lambda_seq), gcv_no_noise_rmse, 'g--', label='Sqrt(GCV) (Estimated Error)')
ax1.set_ylabel('Error (RMSE Scale)')
ax1.set_title('Error Comparison (No Noise) - RMSE Scale')
ax1.grid(True)
ax1.legend()
min_val_nn = np.nanmin([np.nanmin(rmse_no_noise), np.nanmin(gcv_no_noise_rmse)])
max_val_nn = np.nanmax([np.nanmax(rmse_no_noise), np.nanmax(gcv_no_noise_rmse)])
if not np.isnan(min_val_nn) and not np.isnan(max_val_nn):
    ax1.set_ylim(min_val_nn * 0.98, max_val_nn * 1.02)

ax2.plot(np.log(lambda_seq), rmse_noise, 'r-', label='RMSE (Actual Test Error)')
ax2.plot(np.log(lambda_seq), gcv_noise_rmse, 'g--', label='Sqrt(GCV) (Estimated Error)')
ax2.set_xlabel(r"$\log(\lambda)$")
ax2.set_ylabel(f'Error (RMSE Scale)')
ax2.set_title(f'Error Comparison (With Noise, Σ={noise_variance_value}) - RMSE Scale')
ax2.grid(True)
ax2.legend()
min_val_n = np.nanmin([np.nanmin(rmse_noise), np.nanmin(gcv_noise_rmse)])
max_val_n = np.nanmax([np.nanmax(rmse_noise), np.nanmax(gcv_noise_rmse)])
if not np.isnan(min_val_n) and not np.isnan(max_val_n):
    ax2.set_ylim(min_val_n * 0.98, max_val_n * 1.02)

plt.tight_layout()
plt.show()

# --- 5. ズレの数値化 (変更なし) ---
#    (GCV vs RMSE のズレ)

print("\n--- 'ズレ'の分析 (No Noise) ---")
min_gcv_nn_idx = np.nanargmin(gcv_no_noise_rmse)
lambda_chosen_by_gcv_nn = lambda_seq[min_gcv_nn_idx]
estimated_error_at_gcv_choice = gcv_no_noise_rmse[min_gcv_nn_idx]
print(f" [GCVの予測]")
print(f"   GCVが選んだlog(λ): {np.log(lambda_chosen_by_gcv_nn):.4f} (推定誤差: {estimated_error_at_gcv_choice:.2f})")

min_rmse_nn_idx = np.nanargmin(rmse_no_noise)
lambda_best_actual_nn = lambda_seq[min_rmse_nn_idx]
best_actual_error = rmse_no_noise[min_rmse_nn_idx]
print(f" [実際の最適性能]")
print(f"   実際の最適なlog(λ): {np.log(lambda_best_actual_nn):.4f} (実際の最小誤差: {best_actual_error:.2f})")

actual_error_at_gcv_choice = rmse_no_noise[min_gcv_nn_idx]
print(f" [GCVを信じた結果]")
print(f"   GCVが選んだλでの実際の誤差: {actual_error_at_gcv_choice:.2f}")

performance_gap_nn = actual_error_at_gcv_choice - best_actual_error
lambda_gap_nn = np.log(lambda_chosen_by_gcv_nn) - np.log(lambda_best_actual_nn)
print(f" [ズレの数値化]")
print(f"   ▶︎ 最適なlog(λ)のズレ (GCV予測 - 実際): {lambda_gap_nn:.4f}")
print(f"   ▶︎ 性能のズレ (GCVを信じた場合の誤差の増加): {performance_gap_nn:.2f}")


print(f"\n--- 'ズレ'の分析 (With Noise, Σ={noise_variance_value}) ---")
min_gcv_n_idx = np.nanargmin(gcv_noise_rmse)
lambda_chosen_by_gcv_n = lambda_seq[min_gcv_n_idx]
estimated_error_at_gcv_choice_n = gcv_noise_rmse[min_gcv_n_idx]
print(f" [GCVの予測]")
print(f"   GCVが選んだlog(λ): {np.log(lambda_chosen_by_gcv_n):.4f} (推定誤差: {estimated_error_at_gcv_choice_n:.2f})")

min_rmse_n_idx = np.nanargmin(rmse_noise)
lambda_best_actual_n = lambda_seq[min_rmse_n_idx]
best_actual_error_n = rmse_noise[min_rmse_n_idx]
print(f" [実際の最適性能]")
print(f"   実際の最適なlog(λ): {np.log(lambda_best_actual_n):.4f} (実際の最小誤差: {best_actual_error_n:.2f})")

actual_error_at_gcv_choice_n = rmse_noise[min_gcv_n_idx]
print(f" [GCVを信じた結果]")
print(f"   GCVが選んだλでの実際の誤差: {actual_error_at_gcv_choice_n:.2f}")

performance_gap_n = actual_error_at_gcv_choice_n - best_actual_error_n
lambda_gap_n = np.log(lambda_chosen_by_gcv_n) - np.log(lambda_best_actual_n)
print(f" [ズレの数値化]")
print(f"   ▶︎ 最適なlog(λ)のズレ (GCV予測 - 実際): {lambda_gap_n:.4f}")
print(f"   ▶︎ 性能のズレ (GCVを信じた場合の誤差の増加): {performance_gap_n:.2f}")


# --- 6. 「ノイズ有無」の性能比較 (★★ ここが追加点 ★★) ---
print(f"\n--- 'ノイズ有無' の最終性能比較 ---")
print(f" ノイズなし の最小RMSE (実際の最適性能): {best_actual_error:.2f}")
print(f" ノイズあり(Σ={noise_variance_value}) の最小RMSE (実際の最適性能): {best_actual_error_n:.2f}")

performance_difference = best_actual_error_n - best_actual_error
print(f"\n   ▶︎ 性能の差 (ノイズあり - ノイズなし): {performance_difference:.2f}")

if performance_difference < 0:
    print(f"   (結論: ノイズを加えたことで、実際の予測誤差が {abs(performance_difference):.2f} 改善しました。)")
elif performance_difference > 0:
    print(f"   (結論: ノイズを加えたことで、実際の予測誤差が {performance_difference:.2f} 悪化しました。)")
else:
    print(f"   (結論: ノイズを加えても、性能は変わりませんでした。)")

