import copy
import time
import os
import warnings
import logging

# すべての警告を抑制
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import matplotlib
matplotlib.use('Agg')

# フォント設定
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Sans', 'Hiragino Maru Gothic Pro', 
                                'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 
                                'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
rcParams['axes.unicode_minus'] = False

import numpy as np
import pandas as pd
from datetime import datetime

# ===================================================================
# C実装完全互換のCoordinate Descent LASSO
# ===================================================================

def soft_threshold_c_exact(b, eta_i, lam1, norm_x_i):
    """
    C実装の正確な再現
    x[i] = (b - eta[i] - lambda_1*sgn(b - eta[i])) * (|b - eta[i]| > lambda_1) / norm_x[i]
    
    これは以下と等価:
    S(b - eta[i], lambda_1) / norm_x[i]
    where S(z, λ) = sgn(z) * max(|z| - λ, 0)
    """
    z = b - eta_i
    if np.abs(z) > lam1:
        return np.sign(z) * (np.abs(z) - lam1) / norm_x_i
    else:
        return 0.0

def lasso_cd_c_exact(X, y, lam1=0, lam2=0, beta_ini=None, eta=None,
                     max_iter=1000, tol=1e-8, verbose=False):
    """
    C実装(CCD_EN_Private.c)の正確な再現
    
    重要な修正点:
    1. eta は p次元ベクトル（各係数に1つのスカラー）
    2. 勾配計算: b = X[:,j]^T * r_cavity （ノイズなし）
    3. Soft-thresholding: S(b - eta[j], lambda_1) / norm_x[j]
    
    Parameters:
    -----------
    X : ndarray (n, p)
        特徴量行列（正規化済みを想定: F'/sqrt(N)）
    y : ndarray (n,)
        目的変数
    lam1 : float
        L1正則化パラメータ（lambda_1）
    lam2 : float
        L2正則化パラメータ（lambda_2）
    beta_ini : ndarray (p,)
        初期値
    eta : ndarray (p,)  ← 重要: p次元ベクトル！
        プライバシーノイズベクトル（各係数に1つのスカラー）
    max_iter : int
        最大反復回数
    tol : float
        収束判定閾値
    verbose : bool
        デバッグ出力
    
    Returns:
    --------
    beta : ndarray (p,)
        推定係数
    converged : bool
        収束フラグ
    n_iter : int
        反復回数
    """
    n, p = X.shape
    
    # 初期化
    if beta_ini is None:
        beta = np.zeros(p)
    else:
        beta = beta_ini.copy()
    
    if eta is None:
        eta = np.zeros(p)  # p次元ベクトル
    
    # norm_x[i] = lambda_2 + Σ(X[:,i]^2) の計算
    norm_x = lam2 + np.sum(X**2, axis=0)
    
    # 初期残差の計算
    y_hat = X @ beta
    r = y - y_hat
    
    beta_old = beta.copy()
    converged = False
    
    for t in range(max_iter):
        n_converged = 0
        
        # ランダム順序で更新（Cバージョンのshuffle関数に対応）
        indices = np.random.permutation(p)
        
        for j in indices:
            # cavity residual: r_cavity = r + X[:,j] * beta[j]
            r_cavity = r + X[:, j] * beta[j]
            
            # b = X[:,j]^T * r_cavity （ノイズはここでは加えない）
            b = np.dot(X[:, j], r_cavity)
            
            # 古い値を保存
            beta_old_j = beta[j]
            
            # Soft thresholding（C実装準拠）
            # x[i] = S(b - eta[i], lambda_1) / norm_x[i]
            beta[j] = soft_threshold_c_exact(b, eta[j], lam1, norm_x[j])
            
            # 収束判定
            if np.abs(beta[j] - beta_old_j) < tol:
                n_converged += 1
            
            # 残差の更新
            r = r - X[:, j] * (beta[j] - beta_old_j)
        
        # 全係数が収束したかチェック
        if n_converged == p:
            converged = True
            if verbose:
                print(f"収束: 反復回数={t+1}")
            break
    
    if not converged and verbose:
        print(f"警告: 最大反復回数{max_iter}に到達")
    
    return beta, converged, t+1

# ===================================================================
# データ生成関数
# ===================================================================

def generate_synthetic_data(n, p, beta0, seed=None):
    """人工データ生成"""
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.normal(0, 1.0, size=(n, p))
    y_true = X @ beta0
    noise = np.random.normal(0, 1, size=n)
    y = y_true + noise
    
    return X, y

def generate_true_beta(p, rho, seed=None):
    """真のβを生成（スパース率rho）"""
    if seed is not None:
        np.random.seed(seed)
    
    non_zero_mask = np.random.binomial(1, rho, size=p).astype(bool)
    beta0 = np.zeros(p)
    beta0[non_zero_mask] = np.random.normal(0, 1, size=np.sum(non_zero_mask))
    
    return beta0

# ===================================================================
# 誤差計算
# ===================================================================

def compute_training_error(X, y, beta):
    """訓練誤差: MSE"""
    y_pred = X @ beta
    return np.mean((y - y_pred)**2)

def compute_test_error(X_test, y_test, beta):
    """テスト誤差: MSE"""
    y_pred = X_test @ beta
    return np.mean((y_test - y_pred)**2)

def compute_sparsity_ratio(beta, p):
    """スパース率: 非ゼロ係数数 / p"""
    return np.sum(np.abs(beta) > 1e-10) / p

# ===================================================================
# 実験実行（C実装互換版）
# ===================================================================

def run_single_experiment_c_exact(n, p, beta0_true, lambda_seq, 
                                  fixed_eta, lam2=0, n_test=1000,
                                  normalize=True):
    """
    C実装完全互換の単一実験
    
    Parameters:
    -----------
    fixed_eta : ndarray (p,)  ← 重要: p次元ベクトル
        固定プライバシーノイズ
    normalize : bool
        Trueの場合、MATLABコードのようにXを sqrt(p) で正規化
    """
    # 訓練データ生成
    X_train, y_train = generate_synthetic_data(n, p, beta0_true)
    X_test, y_test = generate_synthetic_data(n_test, p, beta0_true)
    
    # 正規化（MATLABコード: F'/sqN）
    if normalize:
        sqrt_p = np.sqrt(p)
        X_train_norm = X_train / sqrt_p
        X_test_norm = X_test / sqrt_p
    else:
        X_train_norm = X_train
        X_test_norm = X_test
    
    r = len(lambda_seq)
    train_error_no = np.zeros(r)
    train_error_with = np.zeros(r)
    test_error_no = np.zeros(r)
    test_error_with = np.zeros(r)
    sparsity_no = np.zeros(r)
    sparsity_with = np.zeros(r)
    
    for i, lam in enumerate(lambda_seq):
        # ノイズなし
        beta_no, _, _ = lasso_cd_c_exact(
            X_train_norm, y_train, lam1=lam, lam2=lam2, eta=None
        )
        train_error_no[i] = compute_training_error(X_train_norm, y_train, beta_no)
        test_error_no[i] = compute_test_error(X_test_norm, y_test, beta_no)
        sparsity_no[i] = compute_sparsity_ratio(beta_no, p)
        
        # ノイズあり
        beta_with, _, _ = lasso_cd_c_exact(
            X_train_norm, y_train, lam1=lam, lam2=lam2, eta=fixed_eta
        )
        train_error_with[i] = compute_training_error(X_train_norm, y_train, beta_with)
        test_error_with[i] = compute_test_error(X_test_norm, y_test, beta_with)
        sparsity_with[i] = compute_sparsity_ratio(beta_with, p)
    
    return (train_error_no, train_error_with,
            test_error_no, test_error_with,
            sparsity_no, sparsity_with)

def run_averaged_experiments_fixed_noise(n, p, rho, lambda_seq, 
                                          noise_variance, lam2=0,
                                          n_experiments=20, n_test=1000,
                                          normalize=True):
    """
    固定ノイズで複数実験を平均（C実装完全互換版）
    """
    r = len(lambda_seq)
    
    # 固定ノイズ生成（p次元ベクトル）← 重要な修正点
    fixed_eta = np.random.normal(0, np.sqrt(noise_variance), size=p)
    
    print(f"\n{'='*70}")
    print(f"C実装完全互換版 Coordinate Descent LASSO")
    print(f"{'='*70}")
    print(f"固定ノイズ生成: Sigma={noise_variance}, eta.shape={fixed_eta.shape}")
    print(f"正規化: {'あり (X/sqrt(p))' if normalize else 'なし'}")
    print(f"実験回数: {n_experiments}")
    
    # 累積用配列
    sum_train_no = np.zeros(r)
    sum_train_with = np.zeros(r)
    sum_test_no = np.zeros(r)
    sum_test_with = np.zeros(r)
    sum_sparse_no = np.zeros(r)
    sum_sparse_with = np.zeros(r)
    
    start_time = time.time()
    
    for exp_num in range(n_experiments):
        if (exp_num + 1) % 5 == 0:
            print(f"  実験 {exp_num + 1}/{n_experiments} 完了")
        
        beta0_true = generate_true_beta(p, rho)
        
        (train_no, train_with, test_no, test_with,
         sparse_no, sparse_with) = run_single_experiment_c_exact(
            n, p, beta0_true, lambda_seq, fixed_eta, lam2, n_test, normalize
        )
        
        sum_train_no += train_no
        sum_train_with += train_with
        sum_test_no += test_no
        sum_test_with += test_with
        sum_sparse_no += sparse_no
        sum_sparse_with += sparse_with
    
    elapsed = time.time() - start_time
    print(f"完了: {elapsed:.2f}秒")
    
    # 平均
    avg_train_no = sum_train_no / n_experiments
    avg_train_with = sum_train_with / n_experiments
    avg_test_no = sum_test_no / n_experiments
    avg_test_with = sum_test_with / n_experiments
    avg_sparse_no = sum_sparse_no / n_experiments
    avg_sparse_with = sum_sparse_with / n_experiments
    
    return (avg_train_no, avg_train_with, avg_test_no, avg_test_with,
            avg_sparse_no, avg_sparse_with)

# ===================================================================
# メイン処理
# ===================================================================
if __name__ == "__main__":
    # --- パラメータ設定 ---
    n = 500              # 訓練データ数
    p = 1000             # 特徴量の次元
    rho = 0.1            # スパース率
    n_experiments = 10   # 実験回数（データセット数）
    n_test = 1000        # テストデータ数
    lam2 = 0.0           # L2正則化（0でLASSO）
    tol = 1e-8           # 収束判定閾値
    max_iter = 1000      # 最大反復回数
    
    # 複数のノイズレベルを設定
    noise_variance_values = [1.0, 0.1]
    
    lambda_seq = np.logspace(0, 1.2, 35)  # λ = 1.0 ～ 15.8
    
    print("="*70)
    print("実験設定 (C/MATLAB完全互換版 Coordinate Descent LASSO)")
    print("="*70)
    print(f"  訓練データサイズ: n={n}, p={p}")
    print(f"  テストデータサイズ: n_test={n_test}")
    print(f"  スパース率: rho={rho}")
    print(f"  実験回数: {n_experiments}")
    print(f"  ノイズ分散レベル: Sigma={noise_variance_values}")
    print(f"  L2正則化: lambda_2={lam2}")
    print(f"  lambda範囲: [{lambda_seq[0]:.4f}, {lambda_seq[-1]:.4f}]")
    print(f"  モデル: 切片なし (y = X*beta)")
    print(f"  アルゴリズム: Cyclic Coordinate Descent (C/MATLAB完全互換)")
    print(f"  ※ 重要: eta は p次元ベクトル（C実装準拠）")
    print("="*70)
    
    # 結果を格納する辞書
    all_results = {}
    
    # --- 各ノイズレベルで実験 ---
    for noise_var in noise_variance_values:
        print(f"\n{'='*70}")
        print(f"ノイズ分散 Sigma = {noise_var} での実験を開始")
        print('='*70)
        
        # 固定ノイズ方式での平均実験の実行
        (avg_train_no, avg_train_with, avg_test_no, avg_test_with,
         avg_sparse_no, avg_sparse_with) = run_averaged_experiments_fixed_noise(
            n, p, rho, lambda_seq, noise_var, lam2, n_experiments, n_test
        )
        
        # 誤差比を計算
        error_ratio_no_noise = np.where(avg_train_no > 1e-10, 
                                         avg_test_no / avg_train_no, 
                                         np.nan)
        error_ratio_with_noise = np.where(avg_train_with > 1e-10, 
                                           avg_test_with / avg_train_with, 
                                           np.nan)
        
        # 結果を保存
        all_results[noise_var] = {
            'avg_train_no': avg_train_no,
            'avg_train_with': avg_train_with,
            'avg_test_no': avg_test_no,
            'avg_test_with': avg_test_with,
            'avg_sparse_no': avg_sparse_no,
            'avg_sparse_with': avg_sparse_with,
            'error_ratio_no': error_ratio_no_noise,
            'error_ratio_with': error_ratio_with_noise
        }
        
        # 最適λでの統計情報を表示
        print("\n" + "-"*70)
        print(f"【Sigma = {noise_var}: ノイズなしの場合】")
        idx_opt_no_noise = np.argmin(avg_test_no)
        print(f"  最適lambda = {lambda_seq[idx_opt_no_noise]:.4f}")
        print(f"  平均訓練誤差 = {avg_train_no[idx_opt_no_noise]:.4f}")
        print(f"  平均テスト誤差 = {avg_test_no[idx_opt_no_noise]:.4f}")
        print(f"  誤差比 (テスト/訓練) = {error_ratio_no_noise[idx_opt_no_noise]:.4f}")
        print(f"  スパース率比 = {avg_sparse_no[idx_opt_no_noise]:.4f}")
        
        print(f"\n【Sigma = {noise_var}: 固定ノイズありの場合】")
        idx_opt_with_noise = np.argmin(avg_test_with)
        print(f"  最適lambda = {lambda_seq[idx_opt_with_noise]:.4f}")
        print(f"  平均訓練誤差 = {avg_train_with[idx_opt_with_noise]:.4f}")
        print(f"  平均テスト誤差 = {avg_test_with[idx_opt_with_noise]:.4f}")
        print(f"  誤差比 (テスト/訓練) = {error_ratio_with_noise[idx_opt_with_noise]:.4f}")
        print(f"  スパース率比 = {avg_sparse_with[idx_opt_with_noise]:.4f}")
        
        # ノイズの影響度を計算
        noise_impact = ((avg_test_with[idx_opt_with_noise] - avg_test_no[idx_opt_no_noise]) / 
                       avg_test_no[idx_opt_no_noise] * 100)
        print(f"\n  ノイズによるテスト誤差の増加率: {noise_impact:+.2f}%")
        print("-"*70)
    
    # --- 全体のサマリー ---
    print("\n" + "="*70)
    print("【全体サマリー：ノイズレベル間の比較】")
    print("="*70)
    print(f"{'ノイズ分散':>12} | {'最適lambda(無)':>13} | {'テスト誤差(無)':>13} | "
          f"{'最適lambda(有)':>13} | {'テスト誤差(有)':>13} | {'影響率(%)':>10}")
    print("-"*70)
    
    for noise_var in noise_variance_values:
        res = all_results[noise_var]
        idx_no = np.argmin(res['avg_test_no'])
        idx_with = np.argmin(res['avg_test_with'])
        impact = ((res['avg_test_with'][idx_with] - res['avg_test_no'][idx_no]) / 
                 res['avg_test_no'][idx_no] * 100)
        
        print(f"Sigma = {noise_var:>5.1f} | "
              f"{lambda_seq[idx_no]:>13.4f} | "
              f"{res['avg_test_no'][idx_no]:>13.4f} | "
              f"{lambda_seq[idx_with]:>13.4f} | "
              f"{res['avg_test_with'][idx_with]:>13.4f} | "
              f"{impact:>+9.2f}")
    print("="*70)
    
    print("\n実験完了！C/MATLAB実装との完全互換性を確保しました。")
    print("重要な修正点:")
    print("  1. eta は p次元ベクトル（各係数に1つのスカラー）")
    print("  2. 勾配計算でノイズは直接加えない")
    print("  3. Soft-thresholding時に b - eta[j] を使用")
    
    # --- グラフ描画（各ノイズレベルごと）---
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nグラフを生成中...")
    
    for noise_var in noise_variance_values:
        res = all_results[noise_var]
        idx_opt_no = np.argmin(res['avg_test_no'])
        idx_opt_with = np.argmin(res['avg_test_with'])
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        
        # (1) 訓練誤差の平均比較
        axes[0, 0].plot(np.log(lambda_seq), res['avg_train_no'], 'r-o', 
                       markersize=4, label='Noise-free')
        axes[0, 0].plot(np.log(lambda_seq), res['avg_train_with'], 'b--^', 
                       markersize=4, label=f'With noise (Sigma={noise_var})')
        axes[0, 0].axvline(x=np.log(lambda_seq[idx_opt_no]), color='r', 
                          linestyle=':', alpha=0.5)
        axes[0, 0].axvline(x=np.log(lambda_seq[idx_opt_with]), color='b', 
                          linestyle=':', alpha=0.5)
        axes[0, 0].set_xlabel(r"$\log(\lambda)$")
        axes[0, 0].set_ylabel(r"Average Training Error")
        axes[0, 0].set_title("Training Error Comparison")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # (2) テスト誤差の平均比較
        axes[0, 1].plot(np.log(lambda_seq), res['avg_test_no'], 'r-o', 
                       markersize=4, label='Noise-free')
        axes[0, 1].plot(np.log(lambda_seq), res['avg_test_with'], 'b--^', 
                       markersize=4, label=f'With noise (Sigma={noise_var})')
        axes[0, 1].axvline(x=np.log(lambda_seq[idx_opt_no]), color='r', 
                          linestyle=':', alpha=0.5)
        axes[0, 1].axvline(x=np.log(lambda_seq[idx_opt_with]), color='b', 
                          linestyle=':', alpha=0.5)
        axes[0, 1].set_xlabel(r"$\log(\lambda)$")
        axes[0, 1].set_ylabel(r"Average Test Error")
        axes[0, 1].set_title("Test Error Comparison")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # (3) 訓練誤差 vs テスト誤差（ノイズなし）
        axes[1, 0].plot(np.log(lambda_seq), res['avg_train_no'], 'g-o', 
                       markersize=4, label='Training Error')
        axes[1, 0].plot(np.log(lambda_seq), res['avg_test_no'], 'r--s', 
                       markersize=4, label='Test Error')
        axes[1, 0].axvline(x=np.log(lambda_seq[idx_opt_no]), color='k', 
                          linestyle=':', alpha=0.5, label='Optimal lambda')
        axes[1, 0].set_xlabel(r"$\log(\lambda)$")
        axes[1, 0].set_ylabel("Average Error")
        axes[1, 0].set_title("Noise-free: Train vs Test Error")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # (4) 訓練誤差 vs テスト誤差（ノイズあり）
        axes[1, 1].plot(np.log(lambda_seq), res['avg_train_with'], 'g-o', 
                       markersize=4, label='Training Error')
        axes[1, 1].plot(np.log(lambda_seq), res['avg_test_with'], 'b--s', 
                       markersize=4, label='Test Error')
        axes[1, 1].axvline(x=np.log(lambda_seq[idx_opt_with]), color='k', 
                          linestyle=':', alpha=0.5, label='Optimal lambda')
        axes[1, 1].set_xlabel(r"$\log(\lambda)$")
        axes[1, 1].set_ylabel("Average Error")
        axes[1, 1].set_title(f"With Noise (Sigma={noise_var}): Train vs Test")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        # (5) 誤差比 vs λ
        axes[2, 0].plot(lambda_seq, res['error_ratio_no'], 'r-o', 
                       markersize=4, label='Noise-free')
        axes[2, 0].plot(lambda_seq, res['error_ratio_with'], 'b--^', 
                       markersize=4, label=f'With noise (Sigma={noise_var})')
        axes[2, 0].axvline(x=lambda_seq[idx_opt_no], color='r', 
                          linestyle=':', alpha=0.5)
        axes[2, 0].axvline(x=lambda_seq[idx_opt_with], color='b', 
                          linestyle=':', alpha=0.5)
        axes[2, 0].set_xlabel(r"$\lambda$")
        axes[2, 0].set_ylabel(r"Error Ratio: Test/Train")
        axes[2, 0].set_title("Error Ratio Comparison")
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].legend()
        
        # (6) 誤差比 vs スパース率比
        axes[2, 1].plot(res['avg_sparse_no'], res['error_ratio_no'], 'r-o', 
                       markersize=4, label='Noise-free')
        axes[2, 1].plot(res['avg_sparse_with'], res['error_ratio_with'], 'b--^', 
                       markersize=4, label=f'With noise (Sigma={noise_var})')
        axes[2, 1].set_xlabel(r"Sparsity Ratio")
        axes[2, 1].set_xlim(0, 0.5)
        axes[2, 1].set_ylabel(r"Error Ratio: Test/Train")
        axes[2, 1].set_title("Error Ratio vs Sparsity")
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend()
        
        plt.suptitle(f"CD LASSO (Sigma={noise_var}) - No Intercept ({n_experiments} datasets)", 
                    fontsize=14, y=0.997)
        plt.tight_layout()
        
        # グラフを保存（ノイズレベルごと）
        filename = f'results/CD_LASSO_Sigma{noise_var}_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    # --- 比較グラフ（全ノイズレベル重ね合わせ）---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['b', 'r', 'g', 'm']
    markers = ['o', '^', 's', 'd']
    
    for i, noise_var in enumerate(noise_variance_values):
        res = all_results[noise_var]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # (1) テスト誤差の比較（ノイズなし）
        axes[0, 0].plot(np.log(lambda_seq), res['avg_test_no'], 
                       color=color, marker=marker, linestyle='-',
                       markersize=3, alpha=0.7,
                       label=f'Sigma={noise_var} (noise-free)')
        
        # (2) テスト誤差の比較（ノイズあり）
        axes[0, 1].plot(np.log(lambda_seq), res['avg_test_with'], 
                       color=color, marker=marker, linestyle='--',
                       markersize=3, alpha=0.7,
                       label=f'Sigma={noise_var} (with noise)')
        
        # (3) 誤差比の比較（ノイズなし）
        axes[1, 0].plot(lambda_seq, res['error_ratio_no'], 
                       color=color, marker=marker, linestyle='-',
                       markersize=3, alpha=0.7,
                       label=f'Sigma={noise_var} (noise-free)')
        
        # (4) 誤差比の比較（ノイズあり）
        axes[1, 1].plot(lambda_seq, res['error_ratio_with'], 
                       color=color, marker=marker, linestyle='--',
                       markersize=3, alpha=0.7,
                       label=f'Sigma={noise_var} (with noise)')
    
    axes[0, 0].set_xlabel(r"$\log(\lambda)$")
    axes[0, 0].set_ylabel("Average Test Error")
    axes[0, 0].set_title("Noise-free: Test Error Comparison")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_xlabel(r"$\log(\lambda)$")
    axes[0, 1].set_ylabel("Average Test Error")
    axes[0, 1].set_title("With Noise: Test Error Comparison")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].set_xlabel(r"$\lambda$")
    axes[1, 0].set_ylabel("Error Ratio")
    axes[1, 0].set_title("Noise-free: Error Ratio Comparison")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].set_xlabel(r"$\lambda$")
    axes[1, 1].set_ylabel("Error Ratio")
    axes[1, 1].set_title("With Noise: Error Ratio Comparison")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.suptitle(f"All Noise Levels Comparison ({n_experiments} datasets)", 
                fontsize=14, y=0.995)
    plt.tight_layout()
    
    filename_comp = f'results/CD_LASSO_AllSigma_comparison_{timestamp}.png'
    plt.savefig(filename_comp, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename_comp}")
    
    # --- CSV保存 ---
    print("\nSaving data to CSV...")
    for noise_var in noise_variance_values:
        res = all_results[noise_var]
        results_df = pd.DataFrame({
            'log_lambda': np.log(lambda_seq),
            'lambda': lambda_seq,
            'avg_train_error_no_noise': res['avg_train_no'],
            'avg_test_error_no_noise': res['avg_test_no'],
            'avg_train_error_with_noise': res['avg_train_with'],
            'avg_test_error_with_noise': res['avg_test_with'],
            'error_ratio_no_noise': res['error_ratio_no'],
            'error_ratio_with_noise': res['error_ratio_with'],
            'sparsity_ratio_no_noise': res['avg_sparse_no'],
            'sparsity_ratio_with_noise': res['avg_sparse_with']
        })
        csv_filename = f'results/CD_LASSO_Sigma{noise_var}_{timestamp}.csv'
        results_df.to_csv(csv_filename, index=False)
        print(f"  Saved: {csv_filename}")
    
    # --- 設定ファイル保存 ---
    config_filename = f'results/CD_LASSO_config_{timestamp}.txt'
    with open(config_filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("C/MATLAB Complete Compatible Coordinate Descent LASSO Experiment Config\n")
        f.write("="*80 + "\n\n")
        
        f.write("【Data Configuration】\n")
        f.write(f"  Training data size    : n = {n}, p = {p} (n/p = {n/p:.2f})\n")
        f.write(f"  Test data size        : n_test = {n_test}\n")
        f.write(f"  Sparsity ratio        : rho = {rho}\n")
        f.write(f"  Number of experiments : {n_experiments}\n\n")
        
        f.write("【Regularization Configuration】\n")
        f.write(f"  L1 regularization     : lambda in [{lambda_seq[0]:.4f}, {lambda_seq[-1]:.4f}] ({len(lambda_seq)} points)\n")
        f.write(f"  L2 regularization     : lambda_2 = {lam2}\n\n")
        
        f.write("【Noise Configuration】\n")
        f.write(f"  Noise variance levels : Sigma = {noise_variance_values}\n")
        f.write(f"  Noise generation      : Fixed (same for all experiments)\n")
        f.write(f"  Noise shape           : p-dimensional vector (C/MATLAB compatible)\n\n")
        
        f.write("【Algorithm Configuration】\n")
        f.write(f"  Method                : Cyclic Coordinate Descent (C/MATLAB exact)\n")
        f.write(f"  Model                 : No intercept (y = X*beta)\n")
        f.write(f"  Normalization         : Yes (X/sqrt(p))\n")
        f.write(f"  Convergence tolerance : {tol}\n")
        f.write(f"  Max iterations        : {max_iter}\n\n")
        
        f.write("【Important Implementation Details】\n")
        f.write(f"  1. eta is p-dimensional vector (one scalar per coefficient)\n")
        f.write(f"  2. Gradient: b = X[:,j]^T * r_cavity (no noise added here)\n")
        f.write(f"  3. Soft-threshold: S(b - eta[j], lambda_1) / norm_x[j]\n")
        f.write(f"  4. Random shuffling of coefficient indices per iteration\n\n")
        
        f.write("【Results Summary】\n")
        f.write(f"{'Noise Var':>12} | {'Opt lambda(no)':>14} | {'Test Err(no)':>12} | ")
        f.write(f"{'Opt lambda(yes)':>15} | {'Test Err(yes)':>13} | {'Impact(%)':>10}\n")
        f.write("-"*80 + "\n")
        
        for noise_var in noise_variance_values:
            res = all_results[noise_var]
            idx_no = np.argmin(res['avg_test_no'])
            idx_with = np.argmin(res['avg_test_with'])
            impact = ((res['avg_test_with'][idx_with] - res['avg_test_no'][idx_no]) / 
                     res['avg_test_no'][idx_no] * 100)
            
            f.write(f"Sigma = {noise_var:>5.1f} | ")
            f.write(f"{lambda_seq[idx_no]:>14.4f} | ")
            f.write(f"{res['avg_test_no'][idx_no]:>12.4f} | ")
            f.write(f"{lambda_seq[idx_with]:>15.4f} | ")
            f.write(f"{res['avg_test_with'][idx_with]:>13.4f} | ")
            f.write(f"{impact:>+9.2f}\n")
        
        f.write("="*80 + "\n\n")
        
        f.write("【Timestamp】\n")
        f.write(f"  Experiment date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  File timestamp : {timestamp}\n")
    
    print(f"  Saved: {config_filename}")
    
    print("\n" + "="*70)
    print("All experiments completed successfully!")
    print("="*70)