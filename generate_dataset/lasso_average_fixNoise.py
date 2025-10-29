import copy
import time
import os
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# ===================================================================
# 1. 人工データ生成 関数
# ===================================================================
def generate_synthetic_data(n, p, beta0, seed=None):
    """
    指定されたβ0を使用して人工データを生成
    
    Parameters:
    -----------
    n : int
        サンプル数
    p : int
        特徴量の次元
    beta0 : ndarray
        真の係数ベクトル
    seed : int, optional
        乱数シード
    
    Returns:
    --------
    X, y : ndarray
        生成されたデータ
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.normal(loc=0, scale=np.sqrt(1/p), size=(n, p))
    
    y_true = X @ beta0
    noise = np.random.normal(0, 1, size=n)
    y = y_true + noise
    
    return X, y

def generate_true_beta(p, rho, seed=None):
    """
    真のβ0を生成
    
    Parameters:
    -----------
    p : int
        特徴量の次元
    rho : float
        スパース率（非ゼロ係数の割合）
    seed : int, optional
        乱数シード
    
    Returns:
    --------
    beta0 : ndarray
        真の係数ベクトル
    """
    if seed is not None:
        np.random.seed(seed)
    
    non_zero_mask = np.random.binomial(1, rho, size=p).astype(bool)
    
    beta0 = np.zeros(p)
    beta0[non_zero_mask] = np.random.normal(0, 1, size=np.sum(non_zero_mask))
    
    return beta0

# ===================================================================
# 2. 改良版LASSOアルゴリズム（Coordinate Descent法）
# ===================================================================
def soft_th(lam, x):
    """
    ソフトしきい値作用素
    S(x, λ) = sign(x) * max(|x| - λ, 0)
    """
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def linear_lasso_cd(X, y, lam1=0, lam2=0, beta=None, eta=None, 
                    max_iter=1000, tol=1e-6, verbose=False):
    """
    Cyclic Coordinate Descent法によるElastic Net LASSO
    
    目的関数:
    minimize: (1/2)||y - X*β||² + λ₁||β||₁ + (λ₂/2)||β||²
    
    プライバシーノイズ込みの更新式:
    β[j] = S(b[j] - η[j], λ₁*n) / (||X[:,j]||² + λ₂)
    
    Parameters:
    -----------
    X : ndarray (n, p)
        特徴量行列
    y : ndarray (n,)
        目的変数
    lam1 : float
        L1正則化パラメータ（LASSO項）
    lam2 : float
        L2正則化パラメータ（Ridge項）。lam2=0でLASSO
    beta : ndarray (p,), optional
        初期値
    eta : ndarray (p, n), optional
        プライバシーノイズ
    max_iter : int
        最大反復回数
    tol : float
        収束判定の閾値
    verbose : bool
        デバッグ情報を表示
    
    Returns:
    --------
    beta : ndarray (p,)
        推定された係数ベクトル
    converged : bool
        収束したかどうか
    """
    n, p = X.shape
    
    # 初期化
    if beta is None:
        beta = np.zeros(p)
    else:
        beta = beta.copy()
    
    if eta is None:
        eta = np.zeros((p, n))
    
    # 各特徴量の正規化定数を事前計算
    # norm_x[j] = ||X[:,j]||² + λ₂
    norm_x = np.sum(X**2, axis=0) + lam2
    
    # 初期残差 r = y - X*β
    r = y - X @ beta
    
    # 係数の履歴（収束判定用）
    beta_old = beta.copy()
    
    # 反復処理
    converged = False
    for iteration in range(max_iter):
        # ランダムシャッフル（C実装と同様）
        indices = np.random.permutation(p)
        
        # 収束した係数のカウント
        n_converged = 0
        
        for j in indices:
            # Cavity残差: r_cavity = r + X[:,j] * β[j]
            # （j番目の寄与を除いた残差）
            r_cavity = r + X[:, j] * beta[j]
            
            # 勾配計算（ノイズ込み）
            # b = X[:,j]^T * (r_cavity + η[j])
            b = np.dot(X[:, j], r_cavity + eta[j])
            
            # 古い値を保存
            beta_old_j = beta[j]
            
            # Soft-thresholding更新
            # β[j] = S(b - λ₁*n, λ₁*n) / norm_x[j]
            if norm_x[j] > 1e-10:
                beta[j] = soft_th(lam1 * n, b) / norm_x[j]
            else:
                beta[j] = 0
            
            # 収束判定（各係数の変化量）
            if np.abs(beta[j] - beta_old_j) < tol:
                n_converged += 1
            
            # 残差の更新
            # r = r - X[:,j] * (β[j] - β_old[j])
            r = r - X[:, j] * (beta[j] - beta_old_j)
        
        # 全係数が収束したかチェック
        if n_converged == p:
            converged = True
            if verbose:
                print(f"収束しました (反復回数: {iteration + 1})")
            break
    
    if not converged and verbose:
        print(f"警告: 最大反復回数 {max_iter} に到達しました")
    
    return beta, converged

# ===================================================================
# 3. 誤差計算関数
# ===================================================================
def compute_training_error(X, y, beta):
    """
    訓練誤差を計算: ||y - X*beta||^2 / n （平均二乗誤差）
    """
    n = len(y)
    y_pred = X @ beta
    return np.sum((y - y_pred)**2) / n

def compute_test_error(X_test, y_test, beta):
    """
    テストデータでの予測誤差を計算: ||y_test - X_test*beta||^2 / n_test
    """
    n_test = len(y_test)
    y_pred = X_test @ beta
    return np.sum((y_test - y_pred)**2) / n_test

def compute_sparsity_ratio(beta, p):
    """
    スパース率を計算: ρ̂ = ||β||_0 / p
    （推定されたモデルの非ゼロ係数の割合）
    """
    non_zero_count = np.sum(np.abs(beta) > 1e-10)
    return non_zero_count / p

# ===================================================================
# 4. 単一実験の実行（固定ノイズ使用、改良版LASSO）
# ===================================================================
def run_single_experiment_fixed_noise(n, p, beta0_true, lambda_seq, fixed_eta, 
                                      lam2=0, n_test=1000):
    """
    固定されたノイズetaを使用して、1つのランダムデータセットに対してLASSO実験を実行
    
    Parameters:
    -----------
    n : int
        訓練データのサンプル数
    p : int
        特徴量の次元
    beta0_true : ndarray
        真の係数ベクトル
    lambda_seq : ndarray
        正則化パラメータのシーケンス
    fixed_eta : ndarray
        固定されたプライバシーノイズ (p, n)
    lam2 : float
        L2正則化パラメータ（0でLASSO）
    n_test : int
        テストデータのサンプル数
    
    Returns:
    --------
    各λに対する誤差とスパース率比の配列のタプル
    """
    # 訓練データ生成
    X_train, y_train = generate_synthetic_data(n, p, beta0_true, seed=None)
    
    # テストデータ生成（同じβ0_trueを使用）
    X_test, y_test = generate_synthetic_data(n_test, p, beta0_true, seed=None)
    
    r = len(lambda_seq)
    train_error_no_noise = np.zeros(r)
    train_error_with_noise = np.zeros(r)
    test_error_no_noise = np.zeros(r)
    test_error_with_noise = np.zeros(r)
    sparsity_ratio_no_noise = np.zeros(r)
    sparsity_ratio_with_noise = np.zeros(r)
    convergence_no_noise = np.zeros(r, dtype=bool)
    convergence_with_noise = np.zeros(r, dtype=bool)
    
    # 各λについて計算
    for i, lam in enumerate(lambda_seq):
        # ノイズなし
        beta_est_no_noise, conv_no = linear_lasso_cd(
            X_train, y_train, lam1=lam, lam2=lam2, eta=None
        )
        train_error_no_noise[i] = compute_training_error(X_train, y_train, beta_est_no_noise)
        test_error_no_noise[i] = compute_test_error(X_test, y_test, beta_est_no_noise)
        sparsity_ratio_no_noise[i] = compute_sparsity_ratio(beta_est_no_noise, p)
        convergence_no_noise[i] = conv_no
        
        # 固定ノイズあり
        beta_est_with_noise, conv_with = linear_lasso_cd(
            X_train, y_train, lam1=lam, lam2=lam2, eta=fixed_eta
        )
        train_error_with_noise[i] = compute_training_error(X_train, y_train, beta_est_with_noise)
        test_error_with_noise[i] = compute_test_error(X_test, y_test, beta_est_with_noise)
        sparsity_ratio_with_noise[i] = compute_sparsity_ratio(beta_est_with_noise, p)
        convergence_with_noise[i] = conv_with
    
    return (train_error_no_noise, train_error_with_noise, 
            test_error_no_noise, test_error_with_noise,
            sparsity_ratio_no_noise, sparsity_ratio_with_noise,
            convergence_no_noise, convergence_with_noise)

# ===================================================================
# 5. 固定ノイズで複数回実験の平均計算
# ===================================================================
def run_averaged_experiments_fixed_noise(n, p, rho, lambda_seq, noise_variance_value, 
                                         lam2=0, n_experiments=50, n_test=1000):
    """
    1つの固定されたノイズetaに対して、n_experiments個の異なるデータセットで実験を実行し、結果を平均する
    
    Returns:
    --------
    平均された誤差とスパース率比配列のタプル
    """
    r = len(lambda_seq)
    
    # 固定ノイズを1回だけ生成
    noise_std_dev = np.sqrt(noise_variance_value)
    fixed_eta = np.random.normal(loc=0, scale=noise_std_dev, size=(p, n))
    print(f"\n固定プライバシーノイズを生成しました (分散Σ={noise_variance_value})")
    print(f"アルゴリズム: Cyclic Coordinate Descent法（MATLAB/C実装準拠）")
    print(f"L2正則化: λ₂={lam2} {'(LASSO)' if lam2 == 0 else '(Elastic Net)'}")
    
    # 累積用の配列
    sum_train_error_no_noise = np.zeros(r)
    sum_train_error_with_noise = np.zeros(r)
    sum_test_error_no_noise = np.zeros(r)
    sum_test_error_with_noise = np.zeros(r)
    sum_sparsity_ratio_no_noise = np.zeros(r)
    sum_sparsity_ratio_with_noise = np.zeros(r)
    total_non_converged_no_noise = np.zeros(r)
    total_non_converged_with_noise = np.zeros(r)
    
    print(f"{n_experiments}個の異なるデータセットで実験を開始します...")
    print(f"  各実験で訓練データ{n}サンプル、テストデータ{n_test}サンプルを使用")
    start_time = time.time()
    
    for exp_num in range(n_experiments):
        # 進捗表示
        if (exp_num + 1) % 10 == 0:
            print(f"  実験 {exp_num + 1}/{n_experiments} 完了...")
        
        # 各実験で新しい真のβを生成
        beta0_true = generate_true_beta(p, rho, seed=None)
        
        # 固定ノイズを使って単一実験を実行
        (train_no, train_with, test_no, test_with, 
         sparse_no, sparse_with, conv_no, conv_with) = run_single_experiment_fixed_noise(
            n, p, beta0_true, lambda_seq, fixed_eta, lam2, n_test
        )
        
        # 累積
        sum_train_error_no_noise += train_no
        sum_train_error_with_noise += train_with
        sum_test_error_no_noise += test_no
        sum_test_error_with_noise += test_with
        sum_sparsity_ratio_no_noise += sparse_no
        sum_sparsity_ratio_with_noise += sparse_with
        total_non_converged_no_noise += ~conv_no
        total_non_converged_with_noise += ~conv_with
    
    end_time = time.time()
    print(f"全実験が完了しました。(経過時間: {end_time - start_time:.2f}秒)")
    
    # 収束状況のレポート
    if np.any(total_non_converged_no_noise > 0):
        print(f"警告: ノイズなしで収束しなかったケース数: {np.sum(total_non_converged_no_noise)}/{r * n_experiments}")
    if np.any(total_non_converged_with_noise > 0):
        print(f"警告: ノイズありで収束しなかったケース数: {np.sum(total_non_converged_with_noise)}/{r * n_experiments}")
    
    # 平均を計算
    avg_train_error_no_noise = sum_train_error_no_noise / n_experiments
    avg_train_error_with_noise = sum_train_error_with_noise / n_experiments
    avg_test_error_no_noise = sum_test_error_no_noise / n_experiments
    avg_test_error_with_noise = sum_test_error_with_noise / n_experiments
    avg_sparsity_ratio_no_noise = sum_sparsity_ratio_no_noise / n_experiments
    avg_sparsity_ratio_with_noise = sum_sparsity_ratio_with_noise / n_experiments
    
    return (avg_train_error_no_noise, avg_train_error_with_noise, 
            avg_test_error_no_noise, avg_test_error_with_noise,
            avg_sparsity_ratio_no_noise, avg_sparsity_ratio_with_noise)

# ===================================================================
# 6. メイン処理
# ===================================================================
if __name__ == "__main__":
    # --- 6.1. パラメータ設定 ---
    n = 100              # 訓練データ数
    p = 500              # 特徴量の次元
    rho = 0.05            # スパース率
    n_experiments = 50   # 実験回数（データセット数）
    n_test = 1000        # テストデータ数
    noise_variance_value = 1.0  # プライバシーノイズの分散
    lam2 = 0.0           # L2正則化（0でLASSO）
    
    lambda_seq = np.logspace(-2, 1, 50)
    
    print(f"実験設定 (改良版Coordinate Descent LASSO):")
    print(f"  訓練データサイズ: n={n}, p={p}")
    print(f"  テストデータサイズ: n_test={n_test}")
    print(f"  スパース率: rho={rho}")
    print(f"  実験回数（データセット数）: {n_experiments}")
    print(f"  ノイズ分散: Σ={noise_variance_value}")
    print(f"  L2正則化: λ₂={lam2}")
    print(f"  モデル: 切片なし (y = X*β)")
    print(f"  アルゴリズム: Cyclic Coordinate Descent (MATLAB/C実装準拠)")
    print(f"  ※ 1つのノイズを固定して、{n_experiments}個の異なるデータで平均")
    
    # --- 6.2. 固定ノイズ方式での平均実験の実行 ---
    (avg_train_no, avg_train_with, avg_test_no, avg_test_with,
     avg_sparse_no, avg_sparse_with) = run_averaged_experiments_fixed_noise(
        n, p, rho, lambda_seq, noise_variance_value, lam2, n_experiments, n_test
    )
    
    # --- 6.3. 誤差比を計算（ゼロ除算を回避） ---
    error_ratio_no_noise = np.where(avg_train_no > 1e-10, 
                                     avg_test_no / avg_train_no, 
                                     np.nan)
    error_ratio_with_noise = np.where(avg_train_with > 1e-10, 
                                       avg_test_with / avg_train_with, 
                                       np.nan)
    
    # --- 6.4. 最適λでの統計情報を表示 ---
    print("\n" + "="*70)
    print("【平均結果: ノイズなしの場合】")
    idx_opt_no_noise = np.argmin(avg_test_no)
    print(f"  最適λ = {lambda_seq[idx_opt_no_noise]:.4f}")
    print(f"  平均訓練誤差 = {avg_train_no[idx_opt_no_noise]:.4f}")
    print(f"  平均テスト誤差 = {avg_test_no[idx_opt_no_noise]:.4f}")
    print(f"  誤差比 (テスト/訓練) = {error_ratio_no_noise[idx_opt_no_noise]:.4f}")
    print(f"  スパース率比 ρ̂ = {avg_sparse_no[idx_opt_no_noise]:.4f}")
    
    print("\n【平均結果: 固定ノイズありの場合】")
    idx_opt_with_noise = np.argmin(avg_test_with)
    print(f"  最適λ = {lambda_seq[idx_opt_with_noise]:.4f}")
    print(f"  平均訓練誤差 = {avg_train_with[idx_opt_with_noise]:.4f}")
    print(f"  平均テスト誤差 = {avg_test_with[idx_opt_with_noise]:.4f}")
    print(f"  誤差比 (テスト/訓練) = {error_ratio_with_noise[idx_opt_with_noise]:.4f}")
    print(f"  スパース率比 ρ̂ = {avg_sparse_with[idx_opt_with_noise]:.4f}")
    print("="*70)
    
    # --- 6.4.5. 図5と図6の特定点の値を出力 ---
    print("\n" + "="*70)
    print("【図5: λ=10付近での誤差比】")
    idx_lambda_10 = np.argmin(np.abs(lambda_seq - 10))
    print(f"  λ = {lambda_seq[idx_lambda_10]:.4f} (最も10に近い値)")
    print(f"  ノイズなしの誤差比 = {error_ratio_no_noise[idx_lambda_10]:.6f}")
    print(f"  ノイズありの誤差比 = {error_ratio_with_noise[idx_lambda_10]:.6f}")
    
    print("\n【図6: スパース率=0付近での誤差比（切片）】")
    idx_sparse_0_no = np.argmin(np.abs(avg_sparse_no))
    print(f"  [ノイズなし] スパース率 ρ̂ = {avg_sparse_no[idx_sparse_0_no]:.6f}")
    print(f"  [ノイズなし] 誤差比 = {error_ratio_no_noise[idx_sparse_0_no]:.6f}")
    
    idx_sparse_0_with = np.argmin(np.abs(avg_sparse_with))
    print(f"  [ノイズあり] スパース率 ρ̂ = {avg_sparse_with[idx_sparse_0_with]:.6f}")
    print(f"  [ノイズあり] 誤差比 = {error_ratio_with_noise[idx_sparse_0_with]:.6f}")
    print("="*70)
    
    # --- 6.5. 結果のグラフ描画（3行2列）---
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    # (1) 訓練誤差の平均比較
    axes[0, 0].plot(np.log(lambda_seq), avg_train_no, 'r-o', markersize=4, label='ノイズなし')
    axes[0, 0].plot(np.log(lambda_seq), avg_train_with, 'b--^', markersize=4, label=f'固定ノイズあり (Σ={noise_variance_value})')
    axes[0, 0].axvline(x=np.log(lambda_seq[idx_opt_no_noise]), color='r', linestyle=':', alpha=0.5)
    axes[0, 0].axvline(x=np.log(lambda_seq[idx_opt_with_noise]), color='b', linestyle=':', alpha=0.5)
    axes[0, 0].set_xlabel(r"$\log(\lambda)$")
    axes[0, 0].set_ylabel(r"平均訓練誤差: $\mathbb{E}[||y - X\hat{\beta}||^2/n]$")
    axes[0, 0].set_title("平均訓練誤差の比較 (Coordinate Descent)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # (2) テスト誤差の平均比較
    axes[0, 1].plot(np.log(lambda_seq), avg_test_no, 'r-o', markersize=4, label='ノイズなし')
    axes[0, 1].plot(np.log(lambda_seq), avg_test_with, 'b--^', markersize=4, label=f'固定ノイズあり (Σ={noise_variance_value})')
    axes[0, 1].axvline(x=np.log(lambda_seq[idx_opt_no_noise]), color='r', linestyle=':', alpha=0.5)
    axes[0, 1].axvline(x=np.log(lambda_seq[idx_opt_with_noise]), color='b', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel(r"$\log(\lambda)$")
    axes[0, 1].set_ylabel(r"平均テスト誤差: $\mathbb{E}[||y_{test} - X_{test}\hat{\beta}||^2/n_{test}]$")
    axes[0, 1].set_title("平均テスト誤差の比較 (Coordinate Descent)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # (3) 訓練誤差 vs テスト誤差（ノイズなし）
    axes[1, 0].plot(np.log(lambda_seq), avg_train_no, 'g-o', markersize=4, label='平均訓練誤差')
    axes[1, 0].plot(np.log(lambda_seq), avg_test_no, 'r--s', markersize=4, label='平均テスト誤差')
    axes[1, 0].axvline(x=np.log(lambda_seq[idx_opt_no_noise]), color='k', linestyle=':', alpha=0.5, label='最適λ')
    axes[1, 0].set_xlabel(r"$\log(\lambda)$")
    axes[1, 0].set_ylabel("平均誤差")
    axes[1, 0].set_title("ノイズなし: 平均訓練誤差 vs 平均テスト誤差")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # (4) 訓練誤差 vs テスト誤差（ノイズあり）
    axes[1, 1].plot(np.log(lambda_seq), avg_train_with, 'g-o', markersize=4, label='平均訓練誤差')
    axes[1, 1].plot(np.log(lambda_seq), avg_test_with, 'b--s', markersize=4, label='平均テスト誤差')
    axes[1, 1].axvline(x=np.log(lambda_seq[idx_opt_with_noise]), color='k', linestyle=':', alpha=0.5, label='最適λ')
    axes[1, 1].set_xlabel(r"$\log(\lambda)$")
    axes[1, 1].set_ylabel("平均誤差")
    axes[1, 1].set_title(f"固定ノイズあり (Σ={noise_variance_value}): 平均訓練誤差 vs 平均テスト誤差")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # (5) 誤差比 vs λ
    axes[2, 0].plot(lambda_seq, error_ratio_no_noise, 'r-o', markersize=4, label='ノイズなし')
    axes[2, 0].plot(lambda_seq, error_ratio_with_noise, 'b--^', markersize=4, label=f'固定ノイズあり (Σ={noise_variance_value})')
    axes[2, 0].axvline(x=lambda_seq[idx_opt_no_noise], color='r', linestyle=':', alpha=0.5)
    axes[2, 0].axvline(x=lambda_seq[idx_opt_with_noise], color='b', linestyle=':', alpha=0.5)
    axes[2, 0].set_xlabel(r"$\lambda$")
    axes[2, 0].set_ylabel(r"誤差比: テスト誤差 / 訓練誤差")
    axes[2, 0].set_title("誤差比の比較 (λに対して)")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # (6) 誤差比 vs スパース率比
    axes[2, 1].plot(avg_sparse_no, error_ratio_no_noise, 'r-o', markersize=4, label='ノイズなし')
    axes[2, 1].plot(avg_sparse_with, error_ratio_with_noise, 'b--^', markersize=4, label=f'固定ノイズあり (Σ={noise_variance_value})')
    axes[2, 1].set_xlabel(r"スパース率 $\hat{\rho} = ||\hat{\beta}||_0 / p$")
    axes[2, 1].set_xlim(0, 0.5)
    axes[2, 1].set_ylabel(r"誤差比: テスト誤差 / 訓練誤差")
    axes[2, 1].set_title("誤差比の比較 (スパース率に対して)")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()
    
    plt.suptitle(f"改良版Coordinate Descent LASSO - 切片なし ({n_experiments}個のデータセット)", fontsize=14, y=0.997)
    plt.tight_layout()
    
    # --- 6.6. 結果を保存 ---
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # グラフを保存
    plt.savefig(f'results/CD_LASSO_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nグラフを保存しました: results/CD_LASSO_comparison_{timestamp}.png")
    
    # 結果データをCSVで保存
    results_df = pd.DataFrame({
        'log_lambda': np.log(lambda_seq),
        'lambda': lambda_seq,
        'avg_train_error_no_noise': avg_train_no,
        'avg_test_error_no_noise': avg_test_no,
        'avg_train_error_with_noise': avg_train_with,
        'avg_test_error_with_noise': avg_test_with,
        'error_ratio_no_noise': error_ratio_no_noise,
        'error_ratio_with_noise': error_ratio_with_noise,
        'sparsity_ratio_no_noise': avg_sparse_no,
        'sparsity_ratio_with_noise': avg_sparse_with
    })
    results_df.to_csv(f'results/CD_LASSO_results_{timestamp}.csv', index=False)
    print(f"平均結果データを保存しました: results/CD_LASSO_results_{timestamp}.csv")
    
    # --- 6.7. アルゴリズムの詳細を記録 ---
    with open(f'results/CD_LASSO_config_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("改良版Coordinate Descent LASSO 実験設定\n")
        f.write("="*70 + "\n\n")
        f.write("【アルゴリズム】\n")
        f.write("  - Cyclic Coordinate Descent法（MATLAB/C実装準拠）\n")
        f.write("  - ランダムシャッフル付き反復更新\n")
        f.write("  - Soft-thresholding演算子による疎な解の生成\n")
        f.write(f"  - 収束判定閾値: {1e-8}\n")
        f.write(f"  - 最大反復回数: {1000}\n\n")
        f.write("【データ設定】\n")
        f.write(f"  - 訓練データ: n={n}, p={p}\n")
        f.write(f"  - テストデータ: n_test={n_test}\n")
        f.write(f"  - 真のスパース率: rho={rho}\n")
        f.write(f"  - 実験回数: {n_experiments}\n\n")
        f.write("【正則化】\n")
        f.write(f"  - L1正則化 (LASSO): λ₁ ∈ [{lambda_seq[0]:.4f}, {lambda_seq[-1]:.4f}]\n")
        f.write(f"  - L2正則化 (Ridge): λ₂ = {lam2}\n")
        f.write(f"  - 正則化タイプ: {'LASSO' if lam2 == 0 else 'Elastic Net'}\n\n")
        f.write("【プライバシー】\n")
        f.write(f"  - ノイズ分散: Σ={noise_variance_value}\n")
        f.write(f"  - ノイズ生成: 固定（全実験で同一）\n")
        f.write(f"  - ノイズ次元: (p, n) = ({p}, {n})\n\n")
        f.write("【最適結果（ノイズなし）】\n")
        f.write(f"  - 最適λ: {lambda_seq[idx_opt_no_noise]:.4f}\n")
        f.write(f"  - 訓練誤差: {avg_train_no[idx_opt_no_noise]:.4f}\n")
        f.write(f"  - テスト誤差: {avg_test_no[idx_opt_no_noise]:.4f}\n")
        f.write(f"  - 誤差比: {error_ratio_no_noise[idx_opt_no_noise]:.4f}\n")
        f.write(f"  - スパース率: {avg_sparse_no[idx_opt_no_noise]:.4f}\n\n")
        f.write("【最適結果（ノイズあり）】\n")
        f.write(f"  - 最適λ: {lambda_seq[idx_opt_with_noise]:.4f}\n")
        f.write(f"  - 訓練誤差: {avg_train_with[idx_opt_with_noise]:.4f}\n")
        f.write(f"  - テスト誤差: {avg_test_with[idx_opt_with_noise]:.4f}\n")
        f.write(f"  - 誤差比: {error_ratio_with_noise[idx_opt_with_noise]:.4f}\n")
        f.write(f"  - スパース率: {avg_sparse_with[idx_opt_with_noise]:.4f}\n")
        f.write("="*70 + "\n")
    
    print(f"実験設定を保存しました: results/CD_LASSO_config_{timestamp}.txt")
    print("\n実験が完了しました！")