import copy
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats

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

def ridge(X, y, lam=0):
    n, p = X.shape
    X, y, X_bar, X_sd, y_bar = centralize(X, y)
    beta = np.dot(
        np.linalg.inv(np.dot(X.T, X) + n * lam * np.eye(p)),
        np.dot(X.T, y)
    )
    beta = beta / X_sd
    beta_0 = y_bar - np.dot(X_bar, beta)
    return beta, beta_0

# 読み込みたい列を指定 (yが0列目, Xが2,3,4,5,6列目)
# usecolsで必要な列だけを直接読み込む
# これにより、数値でないデータや列数が異なる行の問題を回避できる
data = np.loadtxt("crime.txt", delimiter="\t", usecols=(0, 2, 3, 4, 5, 6))

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
beta = np.zeros(p)
coef_seq = np.zeros((r,p))
for i in range(r):
    beta, beta_0 = ridge(X, y, lambda_seq[i])
    coef_seq[i, :] = beta
for j in range(p):
    plt.plot(lambda_seq, coef_seq[:, j], label=labels[j])

plt.legend(loc="upper right")

plt.show()