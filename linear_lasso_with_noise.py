import copy
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats
from sklearn.datasets import fetch_california_housing
# ボストンデータセットが削除されていたため、代わりにカリフォルニア住宅データセットを使用

def soft_th (lam, x):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def linear_lasso(X, y, lam=0, beta=None, sigma=0):
    # Args:
        # X (np.ndarray): 説明変数
        # y (np.ndarray): 目的変数
        # lam (float): 正則化パラメータλ
        # beta (np.ndarray, optional): βの初期値. Defaults to None.
        # sigma (float, optional): zに加えるノイズの標準偏差. Defaults to 0.
    n, p = X.shape
    if beta is None:
        beta = np.zeros(p)
    X, y, X_bar, X_sd, y_bar = centralize(X, y)
    
    # 標準化したXでは、各列の二乗和/n は1になる
    # X_col_norm_sq = np.sum(X**2, axis=0) / n  # <- これは常に1になるはず
    
    max_iter = 500  # 最大反復回数
    for i in range(max_iter):
        beta_old = copy.copy(beta)
        
        for j in range(p):
            # 現在の係数betaを使って、j番目の特徴量の影響を除いた予測を計算
            # これが最も効率的な残差の計算方法
            r_j = y - (np.dot(X, beta) - X[:, j] * beta[j])
            
            # zの計算
            z = np.dot(X[:, j], r_j) / n
            
            # ノイズを加える場合
            if sigma > 0:
                noise = np.random.normal(loc=0, scale=sigma)
                z = z + noise
                
            beta[j] = soft_th(lam, z)
            
        # 収束判定
        eps = np.linalg.norm(beta - beta_old, 2)
        if eps < 0.0001:
            break
            
    # 最後に係数を元のスケールに戻す
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
        X_sd[j] = np.std(X[:, j])
        if standardize is True:
            X[:, j] = X[:, j] / X_sd[j]
    if np.ndim(y) == 2:
        K = y.shape[1]
        y_bar = np.zeros(K)
        for k in range(K):
            y_bar[k] = np.mean(y[:, k])
            y[:, k] = y[:, k] - y_bar[k]
    else:
        y_bar = np.mean(y)
        y = y- y_bar
    return X, y, X_bar, X_sd, y_bar

housing = fetch_california_housing()
x = housing.data
y = housing.target
n, p = x.shape

lambda_seq = np.logspace(-2, 1, 100) # 0.01から10までを対数スケールで100個
r = len(lambda_seq)

# 各ラムダでの係数を保存するための配列を用意
beta_path = np.zeros((r, p))

# 加えるノイズの大きさを決める
sigma_value = 0.1 

# 1. 各ラムダでLassoを実行し、係数を記録する
print("Calculating LASSO path...")
for i, lam in enumerate(lambda_seq):
    beta, beta_0 = linear_lasso(x, y, lam=lam, sigma=sigma_value)
    beta_path[i, :] = beta
print("Calculation finished.")

# 2. グラフの準備
plt.figure(figsize=(10, 6)) # グラフのサイズを指定

# 3. 各係数のパスをプロットする
for j in range(p):
    # x軸をlogスケールにすると見やすい
    plt.plot(np.log(lambda_seq), beta_path[:, j], label=housing.feature_names[j])

# グラフ描画
plt.figure(figsize=(12, 8)) # グラフサイズを少し大きく

for j in range(p):
    plt.plot(np.log(lambda_seq), beta_path[:, j], label=housing.feature_names[j])

plt.xlabel(r"$\log(\lambda)$")
plt.ylabel(r"Coefficients $\beta$")
plt.title(f"LASSO Path With Noise (sigma={sigma_value}) for California Housing Dataset")
plt.grid(True)
plt.legend(loc="upper right")
plt.show()