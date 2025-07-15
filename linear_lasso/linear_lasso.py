import copy
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats
from sklearn.datasets import fetch_california_housing

def soft_th (lam, x):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def linear_lasso(X, y, lam=0, beta=None, sigma=0):
    n, p = X.shape
    if beta is None:
        beta = np.zeros(p)
    X, y, X_bar, X_sd, y_bar = centralize(X, y)
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
y = housing.target
X = housing.data
p = X.shape[1]

#特徴量ラベルをデータセットから取得
labels = housing.feature_names

#lambdaの範囲をデータセットに合わせて調整
# 対数スケールでラムダを生成すると、係数の変化が滑らかに観察できる
lambda_seq = np.logspace(-2, 1, 100)
r = len(lambda_seq)

coef_seq = np.zeros((r, p))
print("Calculating LASSO path...")
for i in range(r):
    coef_seq[i, :], _ = linear_lasso(X, y, lambda_seq[i])
print("Calculation finished.")

# グラフ描画
plt.figure(figsize=(12, 8)) # グラフサイズを少し大きく

# X軸をlogスケールにすると見やすい
for j in range(p):
    plt.plot(np.log(lambda_seq), coef_seq[:, j], label=labels[j])


plt.xlabel(r"$\log(\lambda)$")
plt.ylabel(r"Coefficients $\beta$")
plt.title(r"LASSO Path for California Housing Dataset")
plt.grid(True)
plt.legend(loc="upper right")
plt.show()