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
    S(z, λ) = sgn(z) * max(|z| - λ, 0) where z = b - eta[i]
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
    
    重要:
    - eta は p次元ベクトル（各係数に1つのスカラー）
    - 勾配計算: b = X[:,j]^T * r_cavity （ノイズなし）
    - Soft-thresholding: S(b - eta[j], lambda_1) / norm_x[j]
    """
    n, p = X.shape
    
    # 初期化
    if beta_ini is None:
        beta = np.zeros(p)
    else:
        beta = beta_ini.copy()
    
    if eta is None:
        eta = np.zeros(p)  # p次元ベクトル
    
    # norm_x[i] = lambda_2 + Σ(X[:,i]^2)
    norm_x = lam2 + np.sum(X**2, axis=0)
    
    # 初期残差
    y_hat = X @ beta
    r = y - y_hat
    
    converged = False
    
    for t in range(max_iter):
        n_converged = 0
        
        # ランダム順序で更新
        indices = np.random.permutation(p)
        
        for j in indices:
            # cavity residual: r_cavity = r + X[:,j] * beta[j]
            r_cavity = r + X[:, j] * beta[j]
            
            # 勾配: b = X[:,j]^T * r_cavity
            b = np.dot(X[:, j], r_cavity)
            
            # 古い値を保存
            beta_old_j = beta[j]
            
            # Soft thresholding: S(b - eta[j], lambda_1) / norm_x[j]
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

def generate_synthetic_data(n, p, beta0, sigma_y=1.0, seed=None):
    """
    人工データ生成
    y = (X @ beta0) / sqrt(p) + noise
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.normal(0, 1.0, size=(n, p))
    sqrt_p = np.sqrt(p)
    y_true = (X @ beta0) / sqrt_p
    noise = np.random.normal(0, sigma_y, size=n)
    y = y_true + noise
    
    return X, y

def generate_true_beta(p, rho, seed=None):
    """
    真のβを生成（スパース率rho）
    rho = 真の非ゼロ係数の割合
    """
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
    """
    スパース率: 推定された非ゼロ係数の割合
    
    定義: ρ^ = ||β^||_0 / p
    
    ここで:
    - ||β^||_0 = 非ゼロ係数の数
    - p = 全係数の数
    - ρ^ は 0 から 1 の値をとる
    """
    return np.sum(np.abs(beta) > 1e-10) / p

# ===================================================================
# 実験実行（C実装互換版）
# ===================================================================

def run_single_experiment(n, p, beta0_true, lambda_seq, 
                         fixed_eta, lam2=0, n_test=1000,
                         sigma_y=1.0):
    """
    C実装完全互換の単一実験
    
    Parameters:
    -----------
    fixed_eta : ndarray (p,)
        固定プライバシーノイズ（p次元ベクトル）
    """
    # 訓練データ生成
    X_train, y_train = generate_synthetic_data(n, p, beta0_true, sigma_y)
    X_test, y_test = generate_synthetic_data(n_test, p, beta0_true, sigma_y)
    
    # X を sqrt(p) で正規化
    sqrt_p = np.sqrt(p)
    X_train_norm = X_train / sqrt_p
    X_test_norm = X_test / sqrt_p
    
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
        
        y_pred_train_no = X_train_norm @ beta_no
        y_pred_test_no = X_test_norm @ beta_no
        
        train_error_no[i] = np.mean((y_train - y_pred_train_no)**2)
        test_error_no[i] = np.mean((y_test - y_pred_test_no)**2)
        sparsity_no[i] = compute_sparsity_ratio(beta_no, p)  # ← 修正: p を使用
        
        # ノイズあり
        beta_with, _, _ = lasso_cd_c_exact(
            X_train_norm, y_train, lam1=lam, lam2=lam2, eta=fixed_eta
        )
        
        y_pred_train_with = X_train_norm @ beta_with
        y_pred_test_with = X_test_norm @ beta_with
        
        train_error_with[i] = np.mean((y_train - y_pred_train_with)**2)
        test_error_with[i] = np.mean((y_test - y_pred_test_with)**2)
        sparsity_with[i] = compute_sparsity_ratio(beta_with, p)  # ← 修正: p を使用
    
    return (train_error_no, train_error_with,
            test_error_no, test_error_with,
            sparsity_no, sparsity_with)


def run_averaged_experiments_fixed_noise(n, p, rho, lambda_seq, 
                                          noise_variance, lam2=0,
                                          n_experiments=20, n_test=1000,
                                          sigma_y=1.0):
    """
    固定ノイズで複数実験を平均
    """
    r = len(lambda_seq)
    
    # 固定ノイズ生成（p次元ベクトル）
    fixed_eta = np.random.normal(0, noise_variance, size=p)
    
    print(f"\n{'='*70}")
    print(f"Coordinate Descent LASSO")
    print(f"{'='*70}")
    print(f"固定ノイズ生成: Sigma={noise_variance}, eta.shape={fixed_eta.shape}")
    print(f"スパース率の定義: ρ^ = ||β^||_0 / p")
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
         sparse_no, sparse_with) = run_single_experiment(
            n, p, beta0_true, lambda_seq, fixed_eta, lam2, n_test, sigma_y
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
    rho = 0.1            # 真のスパース率（10%が非ゼロ）
    n_experiments = 1   # 実験回数
    n_test = 1000        # テストデータ数
    lam2 = 0.0           # L2正則化（0でLASSO）
    sigma_y = 1.0        # データノイズ
    
    # 複数のノイズレベルを設定
    noise_variance_values = [0.1, 1.0]
    
    lambda_seq = np.logspace(-3, 0, 50)  # [0.001, 1.0]
    
    print("="*70)
    print("実験設定 (Coordinate Descent LASSO)")
    print("="*70)
    print(f"  訓練データサイズ: n={n}, p={p} (n/p={n/p:.2f})")
    print(f"  テストデータサイズ: n_test={n_test}")
    print(f"  真のスパース率: rho={rho} ({int(p*rho)}個の非ゼロ係数)")
    print(f"  実験回数: {n_experiments}")
    print(f"  ノイズ分散レベル: Sigma={noise_variance_values}")
    print(f"  L2正則化: lambda_2={lam2}")
    print(f"  lambda範囲: [{lambda_seq[0]:.4f}, {lambda_seq[-1]:.4f}]")
    print(f"  モデル: y = (X/sqrt(p))*beta + noise")
    print(f"  スパース率の定義: ρ^ = ||β^||_0 / p")
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
            n, p, rho, lambda_seq, noise_var, lam2, n_experiments, n_test, sigma_y
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
        print(f"  スパース率 ρ^ = {avg_sparse_no[idx_opt_no_noise]:.4f} "
              f"({int(avg_sparse_no[idx_opt_no_noise]*p)}/{p}個)")
        
        print(f"\n【Sigma = {noise_var}: 固定ノイズありの場合】")
        idx_opt_with_noise = np.argmin(avg_test_with)
        print(f"  最適lambda = {lambda_seq[idx_opt_with_noise]:.4f}")
        print(f"  平均訓練誤差 = {avg_train_with[idx_opt_with_noise]:.4f}")
        print(f"  平均テスト誤差 = {avg_test_with[idx_opt_with_noise]:.4f}")
        print(f"  誤差比 (テスト/訓練) = {error_ratio_with_noise[idx_opt_with_noise]:.4f}")
        print(f"  スパース率 ρ^ = {avg_sparse_with[idx_opt_with_noise]:.4f} "
              f"({int(avg_sparse_with[idx_opt_with_noise]*p)}/{p}個)")
        
        # ノイズの影響度を計算
        noise_impact = ((avg_test_with[idx_opt_with_noise] - avg_test_no[idx_opt_no_noise]) / 
                       avg_test_no[idx_opt_no_noise] * 100)
        print(f"\n  ノイズによるテスト誤差の増加率: {noise_impact:+.2f}%")
        print("-"*70)
    
    # --- 全体のサマリー ---
    print("\n" + "="*70)
    print("【全体サマリー：ノイズレベル間の比較】")
    print("="*70)
    print(f"{'ノイズ分散':>12} | {'最適λ(無)':>10} | {'テスト誤差(無)':>13} | "
          f"{'スパース率(無)':>13} | {'最適λ(有)':>10} | {'テスト誤差(有)':>13} | "
          f"{'スパース率(有)':>13}")
    print("-"*70)
    
    for noise_var in noise_variance_values:
        res = all_results[noise_var]
        idx_no = np.argmin(res['avg_test_no'])
        idx_with = np.argmin(res['avg_test_with'])
        
        print(f"Sigma={noise_var:>6.1f} | "
              f"{lambda_seq[idx_no]:>10.4f} | "
              f"{res['avg_test_no'][idx_no]:>13.4f} | "
              f"{res['avg_sparse_no'][idx_no]:>13.4f} | "
              f"{lambda_seq[idx_with]:>10.4f} | "
              f"{res['avg_test_with'][idx_with]:>13.4f} | "
              f"{res['avg_sparse_with'][idx_with]:>13.4f}")
    print("="*70)
    
    print("\n実験完了")
    print("※ スパース率 ρ^ = ||β^||_0 / p は 0 から 1 の値をとります")
    print(f"  真のスパース率 rho = {rho} ({int(p*rho)}個の非ゼロ係数)")
    
    # --- グラフ描画 ---
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*70)
    print("グラフとデータを保存中...")
    print("="*70)
    
    # 各ノイズレベルごとにグラフを生成
    for noise_var in noise_variance_values:
        res = all_results[noise_var]
        idx_opt_no = np.argmin(res['avg_test_no'])
        idx_opt_with = np.argmin(res['avg_test_with'])
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        
        # (1) 訓練誤差の比較
        axes[0, 0].plot(np.log(lambda_seq), res['avg_train_no'], 'r-o', 
                       markersize=4, label='Noise-free')
        axes[0, 0].plot(np.log(lambda_seq), res['avg_train_with'], 'b--^', 
                       markersize=4, label=f'With noise (Σ={noise_var})')
        axes[0, 0].axvline(x=np.log(lambda_seq[idx_opt_no]), color='r', 
                          linestyle=':', alpha=0.5)
        axes[0, 0].axvline(x=np.log(lambda_seq[idx_opt_with]), color='b', 
                          linestyle=':', alpha=0.5)
        axes[0, 0].set_xlabel(r"$\log(\lambda)$", fontsize=11)
        axes[0, 0].set_ylabel("Training Error", fontsize=11)
        axes[0, 0].set_title("Training Error vs Lambda", fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # (2) テスト誤差の比較
        axes[0, 1].plot(np.log(lambda_seq), res['avg_test_no'], 'r-o', 
                       markersize=4, label='Noise-free')
        axes[0, 1].plot(np.log(lambda_seq), res['avg_test_with'], 'b--^', 
                       markersize=4, label=f'With noise (Σ={noise_var})')
        axes[0, 1].axvline(x=np.log(lambda_seq[idx_opt_no]), color='r', 
                          linestyle=':', alpha=0.5)
        axes[0, 1].axvline(x=np.log(lambda_seq[idx_opt_with]), color='b', 
                          linestyle=':', alpha=0.5)
        axes[0, 1].set_xlabel(r"$\log(\lambda)$", fontsize=11)
        axes[0, 1].set_ylabel("Test Error", fontsize=11)
        axes[0, 1].set_title("Test Error vs Lambda", fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # (3) 訓練誤差 vs テスト誤差（ノイズなし）
        axes[1, 0].plot(np.log(lambda_seq), res['avg_train_no'], 'g-o', 
                       markersize=4, label='Training Error')
        axes[1, 0].plot(np.log(lambda_seq), res['avg_test_no'], 'r--s', 
                       markersize=4, label='Test Error')
        axes[1, 0].axvline(x=np.log(lambda_seq[idx_opt_no]), color='k', 
                          linestyle=':', alpha=0.5, label='Optimal λ')
        axes[1, 0].set_xlabel(r"$\log(\lambda)$", fontsize=11)
        axes[1, 0].set_ylabel("Error", fontsize=11)
        axes[1, 0].set_title("Noise-free: Train vs Test", fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # (4) 訓練誤差 vs テスト誤差（ノイズあり）
        axes[1, 1].plot(np.log(lambda_seq), res['avg_train_with'], 'g-o', 
                       markersize=4, label='Training Error')
        axes[1, 1].plot(np.log(lambda_seq), res['avg_test_with'], 'b--s', 
                       markersize=4, label='Test Error')
        axes[1, 1].axvline(x=np.log(lambda_seq[idx_opt_with]), color='k', 
                          linestyle=':', alpha=0.5, label='Optimal λ')
        axes[1, 1].set_xlabel(r"$\log(\lambda)$", fontsize=11)
        axes[1, 1].set_ylabel("Error", fontsize=11)
        axes[1, 1].set_title(f"With Noise (Σ={noise_var}): Train vs Test", fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        # (5) 誤差比 vs λ
        axes[2, 0].plot(lambda_seq, res['error_ratio_no'], 'r-o', 
                       markersize=4, label='Noise-free')
        axes[2, 0].plot(lambda_seq, res['error_ratio_with'], 'b--^', 
                       markersize=4, label=f'With noise (Σ={noise_var})')
        axes[2, 0].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        axes[2, 0].axvline(x=lambda_seq[idx_opt_no], color='r', 
                          linestyle=':', alpha=0.5)
        axes[2, 0].axvline(x=lambda_seq[idx_opt_with], color='b', 
                          linestyle=':', alpha=0.5)
        axes[2, 0].set_xlabel(r"$\lambda$", fontsize=11)
        axes[2, 0].set_ylabel("Error Ratio (Test/Train)", fontsize=11)
        axes[2, 0].set_title("Error Ratio vs Lambda", fontsize=12)
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].legend()
        
        # (6) 誤差比 vs スパース率
        axes[2, 1].plot(res['avg_sparse_no'], res['error_ratio_no'], 'r-o', 
                       markersize=4, label='Noise-free')
        axes[2, 1].plot(res['avg_sparse_with'], res['error_ratio_with'], 'b--^', 
                       markersize=4, label=f'With noise (Σ={noise_var})')
        axes[2, 1].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        axes[2, 1].set_xlabel(r"Sparsity Ratio $\hat{\rho}$", fontsize=11)
        axes[2, 1].set_xlim(0, 1.0)
        axes[2, 1].set_ylabel("Error Ratio (Test/Train)", fontsize=11)
        axes[2, 1].set_title("Error Ratio vs Sparsity", fontsize=12)
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend()
        
        plt.suptitle(f"CD LASSO (Σ={noise_var}, n={n}, p={p}, experiments={n_experiments})", 
                    fontsize=14, y=0.997)
        plt.tight_layout()
        
        # グラフを保存
        png_filename = f'results/CD_LASSO_Sigma{noise_var}_{timestamp}.png'
        plt.savefig(png_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: {png_filename}")
    
    # --- 比較グラフ（全ノイズレベル重ね合わせ）---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = {'0.1': 'blue', '1.0': 'red'}
    
    for noise_var in noise_variance_values:
        res = all_results[noise_var]
        color = colors[str(noise_var)]
        label_base = f'Σ={noise_var}'
        
        # (1) テスト誤差比較（ノイズなし）
        axes[0, 0].plot(np.log(lambda_seq), res['avg_test_no'], 
                       color=color, marker='o', linestyle='-',
                       markersize=3, alpha=0.7, label=f'{label_base} (no noise)')
        
        # (2) テスト誤差比較（ノイズあり）
        axes[0, 1].plot(np.log(lambda_seq), res['avg_test_with'], 
                       color=color, marker='^', linestyle='--',
                       markersize=3, alpha=0.7, label=f'{label_base} (with noise)')
        
        # (3) スパース率比較（ノイズなし）
        axes[0, 2].plot(np.log(lambda_seq), res['avg_sparse_no'], 
                       color=color, marker='o', linestyle='-',
                       markersize=3, alpha=0.7, label=f'{label_base} (no noise)')
        
        # (4) 誤差比比較（ノイズなし）
        axes[1, 0].plot(lambda_seq, res['error_ratio_no'], 
                       color=color, marker='o', linestyle='-',
                       markersize=3, alpha=0.7, label=f'{label_base} (no noise)')
        
        # (5) 誤差比比較（ノイズあり）
        axes[1, 1].plot(lambda_seq, res['error_ratio_with'], 
                       color=color, marker='^', linestyle='--',
                       markersize=3, alpha=0.7, label=f'{label_base} (with noise)')
        
        # (6) スパース率比較（ノイズあり）
        axes[1, 2].plot(np.log(lambda_seq), res['avg_sparse_with'], 
                       color=color, marker='^', linestyle='--',
                       markersize=3, alpha=0.7, label=f'{label_base} (with noise)')
    
    axes[0, 0].set_xlabel(r"$\log(\lambda)$", fontsize=11)
    axes[0, 0].set_ylabel("Test Error", fontsize=11)
    axes[0, 0].set_title("Test Error (Noise-free)", fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_xlabel(r"$\log(\lambda)$", fontsize=11)
    axes[0, 1].set_ylabel("Test Error", fontsize=11)
    axes[0, 1].set_title("Test Error (With Noise)", fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[0, 2].set_xlabel(r"$\log(\lambda)$", fontsize=11)
    axes[0, 2].set_ylabel(r"Sparsity $\hat{\rho}$", fontsize=11)
    axes[0, 2].set_title("Sparsity (Noise-free)", fontsize=12)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    axes[1, 0].set_xlabel(r"$\lambda$", fontsize=11)
    axes[1, 0].set_ylabel("Error Ratio", fontsize=11)
    axes[1, 0].set_title("Error Ratio (Noise-free)", fontsize=12)
    axes[1, 0].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].set_xlabel(r"$\lambda$", fontsize=11)
    axes[1, 1].set_ylabel("Error Ratio", fontsize=11)
    axes[1, 1].set_title("Error Ratio (With Noise)", fontsize=12)
    axes[1, 1].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    axes[1, 2].set_xlabel(r"$\log(\lambda)$", fontsize=11)
    axes[1, 2].set_ylabel(r"Sparsity $\hat{\rho}$", fontsize=11)
    axes[1, 2].set_title("Sparsity (With Noise)", fontsize=12)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    plt.suptitle(f"All Noise Levels Comparison (n={n}, p={p}, experiments={n_experiments})", 
                fontsize=14, y=0.997)
    plt.tight_layout()
    
    png_comp_filename = f'results/CD_LASSO_AllSigma_comparison_{timestamp}.png'
    plt.savefig(png_comp_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: {png_comp_filename}")
    
    # --- CSV保存 ---
    print("\nCSVファイルを保存中...")
    for noise_var in noise_variance_values:
        res = all_results[noise_var]
        results_df = pd.DataFrame({
            'lambda': lambda_seq,
            'log_lambda': np.log(lambda_seq),
            'train_error_no_noise': res['avg_train_no'],
            'test_error_no_noise': res['avg_test_no'],
            'train_error_with_noise': res['avg_train_with'],
            'test_error_with_noise': res['avg_test_with'],
            'sparsity_no_noise': res['avg_sparse_no'],
            'sparsity_with_noise': res['avg_sparse_with'],
            'error_ratio_no_noise': res['error_ratio_no'],
            'error_ratio_with_noise': res['error_ratio_with']
        })
        csv_filename = f'results/CD_LASSO_Sigma{noise_var}_{timestamp}.csv'
        results_df.to_csv(csv_filename, index=False)
        print(f"  保存: {csv_filename}")
    
    # --- 設定ファイル保存 ---
    print("\n設定ファイルを保存中...")
    config_filename = f'results/CD_LASSO_config_{timestamp}.txt'
    with open(config_filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Coordinate Descent LASSO Experiment Configuration\n")
        f.write("="*80 + "\n\n")
        
        f.write("【Data Configuration】\n")
        f.write(f"  Training data size    : n = {n}, p = {p} (n/p = {n/p:.2f})\n")
        f.write(f"  Test data size        : n_test = {n_test}\n")
        f.write(f"  True sparsity ratio   : rho = {rho} ({int(p*rho)} non-zero coefficients)\n")
        f.write(f"  Number of experiments : {n_experiments}\n")
        f.write(f"  Data noise variance   : sigma_y = {sigma_y}\n\n")
        
        f.write("【Regularization Configuration】\n")
        f.write(f"  L1 regularization     : lambda in [{lambda_seq[0]:.4f}, {lambda_seq[-1]:.4f}] ({len(lambda_seq)} points)\n")
        f.write(f"  L2 regularization     : lambda_2 = {lam2}\n\n")
        
        f.write("【Privacy Noise Configuration】\n")
        f.write(f"  Noise variance levels : Sigma = {noise_variance_values}\n")
        f.write(f"  Noise generation      : Fixed (same for all experiments)\n")
        f.write(f"  Noise dimension       : p-dimensional vector (one scalar per coefficient)\n\n")
        
        f.write("【Algorithm Configuration】\n")
        f.write(f"  Method                : Cyclic Coordinate Descent (C/MATLAB compatible)\n")
        f.write(f"  Model                 : y = (X/sqrt(p))*beta + noise (no intercept)\n")
        f.write(f"  Convergence tolerance : 1e-8\n")
        f.write(f"  Max iterations        : 1000\n")
        f.write(f"  Random shuffling      : Yes (per iteration)\n\n")
        
        f.write("【Sparsity Definition】\n")
        f.write(f"  Formula               : rho_hat = ||beta||_0 / p\n")
        f.write(f"  Range                 : 0 <= rho_hat <= 1\n")
        f.write(f"  Interpretation        : Proportion of non-zero coefficients\n\n")
        
        f.write("【Results Summary】\n")
        f.write(f"{'Noise Var':>12} | {'Opt λ(no)':>10} | {'Test Err(no)':>12} | "
                f"{'Sparsity(no)':>12} | {'Opt λ(yes)':>10} | {'Test Err(yes)':>13} | "
                f"{'Sparsity(yes)':>13} | {'Impact(%)':>10}\n")
        f.write("-"*100 + "\n")
        
        for noise_var in noise_variance_values:
            res = all_results[noise_var]
            idx_no = np.argmin(res['avg_test_no'])
            idx_with = np.argmin(res['avg_test_with'])
            impact = ((res['avg_test_with'][idx_with] - res['avg_test_no'][idx_no]) / 
                     res['avg_test_no'][idx_no] * 100)
            
            f.write(f"Sigma={noise_var:>6.1f} | ")
            f.write(f"{lambda_seq[idx_no]:>10.4f} | ")
            f.write(f"{res['avg_test_no'][idx_no]:>12.4f} | ")
            f.write(f"{res['avg_sparse_no'][idx_no]:>12.4f} | ")
            f.write(f"{lambda_seq[idx_with]:>10.4f} | ")
            f.write(f"{res['avg_test_with'][idx_with]:>13.4f} | ")
            f.write(f"{res['avg_sparse_with'][idx_with]:>13.4f} | ")
            f.write(f"{impact:>+9.2f}\n")
        
        f.write("="*80 + "\n\n")
        
        f.write("【Detailed Results for Each Noise Level】\n\n")
        for noise_var in noise_variance_values:
            res = all_results[noise_var]
            idx_no = np.argmin(res['avg_test_no'])
            idx_with = np.argmin(res['avg_test_with'])
            
            f.write(f"Sigma = {noise_var}\n")
            f.write("-" * 40 + "\n")
            f.write("Noise-free:\n")
            f.write(f"  Optimal lambda        : {lambda_seq[idx_no]:.4f}\n")
            f.write(f"  Training error        : {res['avg_train_no'][idx_no]:.4f}\n")
            f.write(f"  Test error            : {res['avg_test_no'][idx_no]:.4f}\n")
            f.write(f"  Error ratio           : {res['error_ratio_no'][idx_no]:.4f}\n")
            f.write(f"  Sparsity ratio        : {res['avg_sparse_no'][idx_no]:.4f} "
                   f"({int(res['avg_sparse_no'][idx_no]*p)}/{p} coefficients)\n\n")
            
            f.write("With noise:\n")
            f.write(f"  Optimal lambda        : {lambda_seq[idx_with]:.4f}\n")
            f.write(f"  Training error        : {res['avg_train_with'][idx_with]:.4f}\n")
            f.write(f"  Test error            : {res['avg_test_with'][idx_with]:.4f}\n")
            f.write(f"  Error ratio           : {res['error_ratio_with'][idx_with]:.4f}\n")
            f.write(f"  Sparsity ratio        : {res['avg_sparse_with'][idx_with]:.4f} "
                   f"({int(res['avg_sparse_with'][idx_with]*p)}/{p} coefficients)\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"File timestamp: {timestamp}\n")
    
    print(f"  保存: {config_filename}")
    
    print("\n" + "="*70)
    print("全てのファイルを保存しました！")
    print("="*70)
    print(f"\n保存先: results/")
    print(f"  - PNG: CD_LASSO_Sigma*_{timestamp}.png (各ノイズレベル)")
    print(f"  - PNG: CD_LASSO_AllSigma_comparison_{timestamp}.png (比較)")
    print(f"  - CSV: CD_LASSO_Sigma*_{timestamp}.csv (各ノイズレベル)")
    print(f"  - TXT: CD_LASSO_config_{timestamp}.txt (設定ファイル)")
    print("="*70)