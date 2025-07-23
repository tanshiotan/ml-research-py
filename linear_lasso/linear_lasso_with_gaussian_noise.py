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

def linear_lasso(X, y, lam=0, beta=None, noise_variance=0):
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
    
    #ノイズη(eta)を事前に生成
    # 分散(variance)から標準偏差(standard deviation)を計算
    noise_std_dev = np.sqrt(noise_variance)
    # η を 平均0、標準偏差noise_std_devのガウス分布から生成
    # shapeは (p, n) とし、各特徴jごとに異なるノイズベクトルη[j]を使えるようにする
    eta = np.random.normal(loc=0, scale=noise_std_dev, size=(p, n))

    max_iter = 500
    for i in range(max_iter):
        beta_old = copy.copy(beta)
        
        for j in range(p):
            r_j = y - (np.dot(X, beta) - X[:, j] * beta[j])
            
            # ノイズη[j]を残差r_jから引く
            z = np.dot(X[:, j], r_j - eta[j]) / n
            
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

# 加えるノイズの分散を決める
noise_variance_value = 0.5

# 計算結果を保存する配列
coef_seq_noise = np.zeros((r, p))
print(f"Calculating LASSO path WITH noise (Variance Σ={noise_variance_value})...")
for i in range(r):
    # 新しいLasso関数を呼び出す
    coef_seq_noise[i, :], _ = linear_lasso(x, y, lambda_seq[i], noise_variance=noise_variance_value)
print("Calculation finished.")

plt.figure(figsize=(12, 8)) # グラフサイズを指定

# 各係数のパスをプロットする
for j in range(p):
    plt.plot(np.log(lambda_seq), coef_seq_noise[:, j], label=housing.feature_names[j])

plt.xlabel(r"$\log(\lambda)$")
plt.ylabel(r"Coefficients $\beta$")
plt.title(f"LASSO Path With Noise (Variance Σ={noise_variance_value}) for California Housing Dataset")
plt.grid(True)
plt.legend(loc="upper right")
plt.show()