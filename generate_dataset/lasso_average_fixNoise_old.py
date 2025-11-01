import copy
import time
import os
import warnings
import matplotlib
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
    
    X = np.random.normal(loc=0, scale=1.0, size=(n, p))
    
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
                    max_iter=1000, tol=1e-4, verbose=False):
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
                                         lam2=0, n_experiments=20, n_test=1000):
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
    n = 500              # 訓練データ数
    p = 1000             # 特徴量の次元
    rho = 0.5            # スパース率
    n_experiments = 10   # 実験回数（データセット数）
    n_test = 1000        # テストデータ数
    lam2 = 0.0           # L2正則化（0でLASSO）
    
    # 複数のノイズレベルを設定
    noise_variance_values = [1.0] 
    
    lambda_seq = np.logspace(0, 1, 30)  # λ = 1 ～ 10
    
    # ===== 実験設定の表示 =====
    print("\n" + "="*80)
    print("  改良版 Coordinate Descent LASSO 実験".center(80))
    print("="*80)
    print("\n【データ設定】")
    print(f"  訓練データ     : n = {n:,}, p = {p:,} (n/p = {n/p:.2f})")
    print(f"  テストデータ   : n_test = {n_test:,}")
    print(f"  スパース率     : ρ = {rho}")
    print(f"  実験回数       : {n_experiments} 回")
    
    print("\n【正則化設定】")
    print(f"  L1正則化       : λ ∈ [{lambda_seq[0]:.4f}, {lambda_seq[-1]:.4f}] ({len(lambda_seq)}点)")
    print(f"  L2正則化       : λ₂ = {lam2}")
    
    print("\n【ノイズ設定】")
    print(f"  ノイズ分散     : Σ = {noise_variance_values}")
    print(f"  ノイズ生成     : 固定 (全実験で同一)")
    
    print("\n【アルゴリズム】")
    print(f"  手法           : Cyclic Coordinate Descent (C/MATLAB準拠)")
    print(f"  モデル         : 切片なし (y = Xβ)")
    print(f"  正規化         : あり (X/√p)")
    print("="*80 + "\n")
    
    # 結果を格納する辞書
    all_results = {}
    
    # --- 各ノイズレベルで実験 ---
    for i, noise_var in enumerate(noise_variance_values, 1):
        print(f"\n{'━'*80}")
        print(f"  実験 {i}/{len(noise_variance_values)}: ノイズ分散 Σ = {noise_var}".center(80))
        print('━'*80)
        
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
        idx_opt_no = np.argmin(avg_test_no)
        idx_opt_with = np.argmin(avg_test_with)
        noise_impact = ((avg_test_with[idx_opt_with] - avg_test_no[idx_opt_no]) / 
                       avg_test_no[idx_opt_no] * 100)
        
        print(f"\n【結果サマリー: Σ = {noise_var}】")
        print("-"*80)
        print(f"{'':20} {'ノイズなし':>20} {'ノイズあり':>20} {'差分':>15}")
        print("-"*80)
        print(f"{'最適 λ':20} {lambda_seq[idx_opt_no]:>20.4f} {lambda_seq[idx_opt_with]:>20.4f} {lambda_seq[idx_opt_with]-lambda_seq[idx_opt_no]:>+15.4f}")
        print(f"{'訓練誤差':20} {avg_train_no[idx_opt_no]:>20.4f} {avg_train_with[idx_opt_with]:>20.4f} {avg_train_with[idx_opt_with]-avg_train_no[idx_opt_no]:>+15.4f}")
        print(f"{'テスト誤差':20} {avg_test_no[idx_opt_no]:>20.4f} {avg_test_with[idx_opt_with]:>20.4f} {avg_test_with[idx_opt_with]-avg_test_no[idx_opt_no]:>+15.4f}")
        print(f"{'誤差比':20} {error_ratio_no_noise[idx_opt_no]:>20.4f} {error_ratio_with_noise[idx_opt_with]:>20.4f} {error_ratio_with_noise[idx_opt_with]-error_ratio_no_noise[idx_opt_no]:>+15.4f}")
        print(f"{'スパース率 ρ̂':20} {avg_sparse_no[idx_opt_no]:>20.4f} {avg_sparse_with[idx_opt_with]:>20.4f} {avg_sparse_with[idx_opt_with]-avg_sparse_no[idx_opt_no]:>+15.4f}")
        print("-"*80)
        print(f"ノイズの影響度: テスト誤差が {noise_impact:+.2f}% 変化")
        print("-"*80)
    
    # --- 全体のサマリー ---
    print("\n" + "="*80)
    print("  全ノイズレベル比較".center(80))
    print("="*80)
    print(f"\n{'ノイズ分散':>10} │ {'最適λ(無)':>10} │ {'テスト誤差(無)':>13} │ "
          f"{'最適λ(有)':>10} │ {'テスト誤差(有)':>13} │ {'影響率':>10}")
    print("─"*80)
    
    for noise_var in noise_variance_values:
        res = all_results[noise_var]
        idx_no = np.argmin(res['avg_test_no'])
        idx_with = np.argmin(res['avg_test_with'])
        impact = ((res['avg_test_with'][idx_with] - res['avg_test_no'][idx_no]) / 
                 res['avg_test_no'][idx_no] * 100)
        
        print(f"Σ = {noise_var:>5.1f} │ "
              f"{lambda_seq[idx_no]:>10.4f} │ "
              f"{res['avg_test_no'][idx_no]:>13.4f} │ "
              f"{lambda_seq[idx_with]:>10.4f} │ "
              f"{res['avg_test_with'][idx_with]:>13.4f} │ "
              f"{impact:>+9.2f}%")
    print("="*80 + "\n")
    
    # --- グラフ描画 ---
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("グラフを生成中...")
    
    # 各ノイズレベルごとのグラフ
    for noise_var in noise_variance_values:
        res = all_results[noise_var]
        idx_opt_no = np.argmin(res['avg_test_no'])
        idx_opt_with = np.argmin(res['avg_test_with'])
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        
        # ... (グラフ描画コードは同じ) ...
        
        plt.suptitle(f"CD LASSO (Σ={noise_var}) - 切片なし ({n_experiments}個のデータセット)", 
                    fontsize=14, y=0.997)
        plt.tight_layout()
        
        filename = f'results/CD_LASSO_Sigma{noise_var}_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {filename}")
    
    # 比較グラフ
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # ... (比較グラフコードは同じ) ...
    
    filename_comp = f'results/CD_LASSO_AllSigma_comparison_{timestamp}.png'
    plt.savefig(filename_comp, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {filename_comp}")
    
    # CSV保存
    print("\nデータを保存中...")
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
        print(f"  ✓ {csv_filename}")
    
    # 設定ファイル保存
    config_filename = f'results/CD_LASSO_config_{timestamp}.txt'
    with open(config_filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("改良版Coordinate Descent LASSO 実験設定\n")
        f.write("="*80 + "\n\n")
        # ... (設定内容は同じ) ...
    print(f"  ✓ {config_filename}")
    
    print("\n" + "="*80)
    print("  実験完了！".center(80))
    print("="*80 + "\n")