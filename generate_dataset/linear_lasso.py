import copy
import time
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ===================================================================
# 1. 人工データ生成 関数
# ===================================================================
def generate_synthetic_data(n, p, rho, seed=None):
    """
    差分プライバシー評価用の人工データを生成
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.normal(loc=0, scale=np.sqrt(1/p), size=(n, p))
    
    non_zero_mask = np.random.binomial(1, rho, size=p).astype(bool)
    
    beta0 = np.zeros(p)
    beta0[non_zero_mask] = np.random.normal(0, 1, size=np.sum(non_zero_mask))
    
    y_true = X @ beta0
    noise = np.random.normal(0, 1, size=n)
    y = y_true + noise
    
    return X, y, beta0

# ===================================================================
# 2. LASSOアルゴリズム 
# ===================================================================
def soft_th(lam, x):
    """ソフトしきい値作用素"""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def centralize(X0, y0, standardize=True):
    """データの中央化・標準化"""
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

def linear_lasso(X, y, lam=0, beta=None, eta=None):
    """目的関数摂動法を用いたLASSO"""
    n, p = X.shape
    if beta is None:
        beta = np.zeros(p)
    
    # データを標準化
    X_std, y_std, X_bar, X_sd, y_bar = centralize(X, y)
    
    # ノイズ eta が指定されていない場合は、ゼロ行列として扱う
    if eta is None:
        eta = np.zeros((p, n))

    max_iter = 500
    for i in range(max_iter):
        beta_old = copy.copy(beta)
        
        for j in range(p):
            # j番目の特徴量の影響を除いた残差を計算
            r_j = y_std - (np.dot(X_std, beta) - X_std[:, j] * beta[j])
            
            # 残差にプライバシーノイズを加味して係数を更新
            z = np.dot(X_std[:, j], r_j - eta[j]) / n
            beta[j] = soft_th(lam, z)
            
        # 収束判定
        eps = np.linalg.norm(beta - beta_old, 2)
        if eps < 0.0001:
            break
            
    # 係数を元のスケールに戻す
    beta = beta / X_sd
    beta_0 = y_bar - np.dot(X_bar, beta)
    return beta, beta_0

# ===================================================================
# 3. メイン処理
# ===================================================================
if __name__ == "__main__":
    # --- 3.1. 人工データの設定と生成 ---
    n = 100      # データ数
    p = 200      # 特徴量の次元 (n < p の高次元)
    rho = 0.1    # 真の係数が非ゼロである確率 (スパース率)
    
    X, y, beta0_true = generate_synthetic_data(n, p, rho, seed=42)
    print(f"人工データを生成しました (n={n}, p={p})")
    print(f"真の非ゼロ係数の数: {np.sum(beta0_true != 0)} / {p}")

    # --- 3.2. LASSOのパラメータ設定 ---
    lambda_seq = np.logspace(-2, 1, 50) # λの範囲 (0.01から10まで50個)
    r = len(lambda_seq)
    
    # 結果を格納する配列
    mse_no_noise = np.zeros(r)
    mse_with_noise = np.zeros(r)

    # --- 3.3. 「ノイズあり」の場合のガウシアンノイズを事前に生成 ---
    noise_variance_value = 0.1 # プライバシーノイズの分散Σ
    noise_std_dev = np.sqrt(noise_variance_value)
    # ノイズ行列etaの形状は (p, n)
    eta = np.random.normal(loc=0, scale=noise_std_dev, size=(p, n))

    # --- 3.4. 計算実行 ---
    print("\nLASSO計算を開始します...")
    start_time = time.time()

    for i, lam in enumerate(lambda_seq):
        # (1) ノイズなし (標準的なLASSO)
        beta_est_no_noise, _ = linear_lasso(X, y, lam=lam, eta=None)
        mse_no_noise[i] = np.mean((beta_est_no_noise - beta0_true)**2)
        
        # (2) ノイズあり (目的関数摂動法)
        beta_est_with_noise, _ = linear_lasso(X, y, lam=lam, eta=eta)
        mse_with_noise[i] = np.mean((beta_est_with_noise - beta0_true)**2)

    end_time = time.time()
    print(f"計算が完了しました。(経過時間: {end_time - start_time:.2f}秒)")

    # --- 3.5. 結果のグラフ描画 ---
    plt.figure(figsize=(10, 7))
    plt.plot(np.log(lambda_seq), mse_no_noise, 'r-o', markersize=4, label='係数誤差 (ノイズなし)')
    plt.plot(np.log(lambda_seq), mse_with_noise, 'b--^', markersize=4, label=f'係数誤差 (ノイズあり, Σ={noise_variance_value})')
    
    # 最も誤差が小さかった点を見つけてプロット
    min_mse_no_noise = np.min(mse_no_noise)
    best_lam_no_noise = np.log(lambda_seq[np.argmin(mse_no_noise)])
    plt.axvline(x=best_lam_no_noise, color='r', linestyle=':', alpha=0.7, label=f'最適λ (ノイズなし, MSE={min_mse_no_noise:.4f})')

    min_mse_with_noise = np.min(mse_with_noise)
    best_lam_with_noise = np.log(lambda_seq[np.argmin(mse_with_noise)])
    plt.axvline(x=best_lam_with_noise, color='b', linestyle=':', alpha=0.7, label=f'最適λ (ノイズあり, MSE={min_mse_with_noise:.4f})')
    
    plt.xlabel(r"正則化パラメータ $\log(\lambda)$")
    plt.ylabel(r"係数誤差 (MSE): $E[||\hat{\beta} - \beta_0||^2]$")
    plt.title("LASSOにおけるパラメータ復元性能の比較")
    plt.grid(True)
    plt.legend()
    plt.show()