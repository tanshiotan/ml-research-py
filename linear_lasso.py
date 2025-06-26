import copy
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats

def soft_th (lam, x):
    return np.sign(x) * np.maximum(np.abs(x) - lam, np.zeros(1))

def linear_lasso(X, y, lam=0, beta=None):
    n, p = X.shape
    if beta is None:
        beta = np.zeros(p)
    X, y, X_bar, X_sd, y_bar = centralize(X, y)
    eps = 1
    while eps > 0.00001:
        beta_old = copy.copy(beta)
        for j in range(p):
            r = y
            for k in range(p):
                if j!= k:
                    r = r - X[:, k] * beta[k]
            z = (np.dot(r, X[:, j]) / n) / (np.dot(X[:, j], X[:, j]) / n)
            beta[j] = soft_th(lam, z)
        eps = np.linalg.norm(beta - beta_old, 2)
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

# 読み込みたい列を指定 (yが0列目, Xが2,3,4,5,6列目)
# usecolsで必要な列だけを直接読み込む
# これにより、数値でないデータや列数が異なる行の問題を回避できる
data = np.loadtxt("crime.txt", delimiter="\t", usecols=(0, 2, 3, 4, 5, 6))

# 読み込んだデータからyとXを分割
y = data[:, 0]  # 読み込んだデータの最初の列がy
X = data[:, 1:] # 読み込んだデータの2番目以降がX
p = X.shape[1]
lambda_seq = np.arange(0, 200, 0.1)

plt.xlim(0, 200)
plt.ylim(-10, 20)
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\beta$")
plt.title(r"各$\lambda$についての各係数の値")
labels = ["警察への年間資金", "25歳以上で高校を卒業した人の割合", 
          "16-19歳で高校に通っていない人の割合", "18-24歳で大学生の割合", "25歳以上で4年生大学を卒業した人の割合"]
r = len(lambda_seq)
coef_seq = np.zeros((r,p))
for i in range(r):
    coef_seq[i, :], _ = linear_lasso(X, y, lambda_seq[i])
for j in range(p):
    plt.plot(lambda_seq, coef_seq[:, j], label=labels[j])

plt.legend(loc="upper right")

plt.show()