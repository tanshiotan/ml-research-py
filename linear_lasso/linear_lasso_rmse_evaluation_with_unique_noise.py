import copy
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# ----------------- 関数定義 -----------------
def soft_th (lam, x):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

# linear_lasso関数を修正: noise_variance の代わりに eta を直接受け取る
def linear_lasso(X, y, lam=0, beta=None, eta=None):
    n, p = X.shape
    if beta is None:
        beta = np.zeros(p)
    
    X_std, y_std, X_bar, X_sd, y_bar = centralize(X, y)
    
    # etaがNoneの場合（ノイズなしを意図する場合）は0の配列を生成
    if eta is None:
        eta = np.zeros((p, n))

    max_iter = 500
    for i in range(max_iter):
        beta_old = copy.copy(beta)
        
        for j in range(p):
            r_j = y_std - (np.dot(X_std, beta) - X_std[:, j] * beta[j])
            # eta[j] を使用してノイズを適用
            z = np.dot(X_std[:, j], r_j - eta[j]) / n
            beta[j] = soft_th(lam, z)
            
        eps = np.linalg.norm(beta - beta_old, 2)
        if eps < 0.0001:
            break
            
    beta = beta / X_sd
    beta_0 = y_bar - np.dot(X_bar, beta)
    return beta, beta_0, np.sum(eta) # np.sum(eta) はデバッグ用

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

# ----------------- ここからメイン処理 -----------------

# 1. データ準備
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lambda_seq = np.logspace(-3, 2, 100)
r = len(lambda_seq)

# RMSEを保存するための配列
rmse_no_noise = np.zeros(r)
rmse_noise = np.zeros(r)

# 2. 「ノイズなし」の計算と評価
print("Calculating and evaluating WITHOUT noise...")
# ノイズなしの場合は eta を None として渡す（linear_lasso内で0の配列が使われる）
for i, lam in enumerate(lambda_seq):
    beta, beta_0, total_noise = linear_lasso(X_train, y_train, lam=lam, eta=None)
    y_pred = np.dot(X_test, beta) + beta_0
    rmse_no_noise[i] = np.sqrt(np.mean((y_test - y_pred)**2))
    if i == 0:
        print(f"  [DEBUG] 'No Noise' loop used total noise: {total_noise}")

# 3. 「ノイズあり」の計算と評価
noise_variance_value = 0.5
noise_std_dev = np.sqrt(noise_variance_value)
# 回帰の外で一度だけノイズを生成する
n_train, p = X_train.shape
unique_eta = np.random.normal(loc=0, scale=noise_std_dev, size=(p, n_train))

print(f"Calculating and evaluating WITH unique noise (Variance Σ={noise_variance_value})...")
for i, lam in enumerate(lambda_seq):
    # 生成した一意なノイズ 'unique_eta' を linear_lasso 関数に渡す
    beta, beta_0, total_noise = linear_lasso(X_train, y_train, lam=lam, eta=unique_eta)
    y_pred = np.dot(X_test, beta) + beta_0
    rmse_noise[i] = np.sqrt(np.mean((y_test - y_pred)**2))
    
    if i == 0:
        print(f"  [DEBUG] 'With Noise' loop used total noise: {total_noise}")

print("Calculation finished.")

# 4. グラフ描画
plt.figure(figsize=(10, 7))
plt.plot(np.log(lambda_seq), rmse_no_noise, 'r-', label='RMSE (No Noise)')
plt.plot(np.log(lambda_seq), rmse_noise, 'b--', label=f'RMSE (With Unique Noise, Σ={noise_variance_value})')
plt.xlabel(r"$\log(\lambda)$")
plt.ylabel("RMSE (Prediction Error)")
plt.title("Model Error (RMSE) vs. Regularization Strength")
plt.grid(True)
plt.legend()
plt.show()