import copy
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats
from PIL import Image

def soft_svd(lambd, z):
    n = z.shape[1]
    u, s, vh = np.linalg.svd(z)
    # ソフト閾値処理を適用した新しい特異値を計算
    s_thresh = np.maximum(s - lambd, 0)
    
    # 閾値処理後の特異値を使って対角行列を再構築
    sigma_thresh = np.zeros((z.shape[0], z.shape[1]))
    # 小さい方の次元（特異値の数）だけループを回す
    min_dim = min(z.shape[0], z.shape[1])
    for i in range(min_dim):
        sigma_thresh[i, i] = s_thresh[i]
        
    # 再構築した行列を返す
    return np.dot(u, np.dot(sigma_thresh, vh))

def mat_lasso(lambd, z, mask):
    m = z.shape[0]; n = z.shape[1]
    guess = np.random.normal(size=m*n).reshape(m, -1)
    for i in range(20):
        guess = soft_svd(lambd, mask * z + (1 - mask) * guess)
    return guess

image = np.array(Image.open("lion.jpg"))
m = image[:, :, 1].shape[0]; n =image[:, :, 1].shape[1]
p = 0.5; lambd = 0.5
mat = np.zeros((m, n, 3))
mask = np.random.binomial(1, p, size=m*n).reshape(-1, n)
for i in range(3):
    mat[:, :, i] = mat_lasso(lambd, image[:, :, i], mask)
Image.fromarray(np.uint8(mat)).save("compressed/lion3_compressed_mat_soft.jpg")
i = Image.open("compressed/lion3_compressed_mat_soft.jpg")
imshow(i)
plt.show()