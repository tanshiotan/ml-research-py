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
    X_sd[X_sd == 0] = 1
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
    
    X_std, y_std, X_bar, X_sd, y_bar = centralize(X, y)
    
    # ノイズを標準化に合わせて変換
    if eta is None:
        eta_scaled = np.zeros((p, n))
    else:
        # 元のスケールのノイズを標準化後のスケールに変換
        eta_scaled = eta / X_sd[:, np.newaxis]

    max_iter = 500
    for i in range(max_iter):
        beta_old = copy.copy(beta)
        
        for j in range(p):
            r_j = y_std - (np.dot(X_std, beta) - X_std[:, j] * beta[j])
            z = np.dot(X_std[:, j], r_j - eta_scaled[j]) / n
            beta[j] = soft_th(lam, z)
            
        eps = np.linalg.norm(beta - beta_old, 2)
        if eps < 0.0001:
            break
            
    beta = beta / X_sd
    beta_0 = y_bar - np.dot(X_bar, beta)
    return beta, beta_0

# ===================================================================
# 3. 誤差計算関数
# ===================================================================
def compute_training_error(X, y, beta, beta_0):
    """
    訓練誤差を計算: ||y - X*beta - beta_0||^2
    """
    y_pred = X @ beta + beta_0
    return np.sum((y - y_pred)**2)

def compute_prediction_error(beta_true, beta_est):
    """
    予測誤差（係数誤差）を計算: ||beta_true - beta_est||^2
    """
    return np.sum((beta_true - beta_est)**2)

# ===================================================================
# 4. 単一実験の実行
# ===================================================================
def run_single_experiment(n, p, rho, lambda_seq, noise_variance_value):
    """
    単一のランダムデータセットに対してLASSO実験を実行
    
    Returns:
    --------
    各λに対する4つの誤差配列のタプル
    """
    # データ生成
    X, y, beta0_true = generate_synthetic_data(n, p, rho, seed=None)
    
    r = len(lambda_seq)
    train_error_no_noise = np.zeros(r)
    train_error_with_noise = np.zeros(r)
    pred_error_no_noise = np.zeros(r)
    pred_error_with_noise = np.zeros(r)
    
    # プライバシーノイズ生成
    noise_std_dev = np.sqrt(noise_variance_value)
    eta = np.random.normal(loc=0, scale=noise_std_dev, size=(p, n))
    
    # 各λについて計算
    for i, lam in enumerate(lambda_seq):
        # ノイズなし
        beta_est_no_noise, beta_0_no_noise = linear_lasso(X, y, lam=lam, eta=None)
        train_error_no_noise[i] = compute_training_error(X, y, beta_est_no_noise, beta_0_no_noise)
        pred_error_no_noise[i] = compute_prediction_error(beta0_true, beta_est_no_noise)
        
        # ノイズあり
        beta_est_with_noise, beta_0_with_noise = linear_lasso(X, y, lam=lam, eta=eta)
        train_error_with_noise[i] = compute_training_error(X, y, beta_est_with_noise, beta_0_with_noise)
        pred_error_with_noise[i] = compute_prediction_error(beta0_true, beta_est_with_noise)
    
    return train_error_no_noise, train_error_with_noise, pred_error_no_noise, pred_error_with_noise

# ===================================================================
# 5. 複数回実験の平均計算
# ===================================================================
def run_averaged_experiments(n, p, rho, lambda_seq, noise_variance_value, n_experiments=50):
    """
    n_experiments回の実験を実行し、結果を平均する
    
    Returns:
    --------
    平均された誤差配列のタプル
    """
    r = len(lambda_seq)
    
    # 累積用の配列
    sum_train_error_no_noise = np.zeros(r)
    sum_train_error_with_noise = np.zeros(r)
    sum_pred_error_no_noise = np.zeros(r)
    sum_pred_error_with_noise = np.zeros(r)
    
    print(f"\n{n_experiments}回の実験を開始します...")
    start_time = time.time()
    
    for exp_num in range(n_experiments):
        # 進捗表示
        if (exp_num + 1) % 10 == 0:
            print(f"  実験 {exp_num + 1}/{n_experiments} 完了...")
        
        # 単一実験を実行
        train_no, train_with, pred_no, pred_with = run_single_experiment(
            n, p, rho, lambda_seq, noise_variance_value
        )
        
        # 累積
        sum_train_error_no_noise += train_no
        sum_train_error_with_noise += train_with
        sum_pred_error_no_noise += pred_no
        sum_pred_error_with_noise += pred_with
    
    end_time = time.time()
    print(f"全実験が完了しました。(経過時間: {end_time - start_time:.2f}秒)")
    
    # 平均を計算
    avg_train_error_no_noise = sum_train_error_no_noise / n_experiments
    avg_train_error_with_noise = sum_train_error_with_noise / n_experiments
    avg_pred_error_no_noise = sum_pred_error_no_noise / n_experiments
    avg_pred_error_with_noise = sum_pred_error_with_noise / n_experiments
    
    return avg_train_error_no_noise, avg_train_error_with_noise, avg_pred_error_no_noise, avg_pred_error_with_noise

# ===================================================================
# 6. メイン処理
# ===================================================================
if __name__ == "__main__":
    # --- 6.1. パラメータ設定 ---
    n = 100              # データ数
    p = 200              # 特徴量の次元
    rho = 0.1            # スパース率
    n_experiments = 50   # 実験回数
    noise_variance_value = 0.1  # プライバシーノイズの分散
    
    lambda_seq = np.logspace(-2, 1, 50)
    
    print(f"実験設定:")
    print(f"  データサイズ: n={n}, p={p}")
    print(f"  スパース率: rho={rho}")
    print(f"  実験回数: {n_experiments}")
    print(f"  ノイズ分散: Σ={noise_variance_value}")
    
    # --- 6.2. 平均実験の実行 ---
    avg_train_no, avg_train_with, avg_pred_no, avg_pred_with = run_averaged_experiments(
        n, p, rho, lambda_seq, noise_variance_value, n_experiments
    )
    
    # --- 6.3. 最適λでの統計情報を表示 ---
    print("\n" + "="*70)
    print("【平均結果: ノイズなしの場合】")
    idx_opt_no_noise = np.argmin(avg_pred_no)
    print(f"  最適λ = {lambda_seq[idx_opt_no_noise]:.4f}")
    print(f"  平均訓練誤差 = {avg_train_no[idx_opt_no_noise]:.4f}")
    print(f"  平均予測誤差 = {avg_pred_no[idx_opt_no_noise]:.4f}")
    print(f"  誤差比 (予測/訓練) = {avg_pred_no[idx_opt_no_noise]/avg_train_no[idx_opt_no_noise]:.4f}")
    
    print("\n【平均結果: ノイズありの場合】")
    idx_opt_with_noise = np.argmin(avg_pred_with)
    print(f"  最適λ = {lambda_seq[idx_opt_with_noise]:.4f}")
    print(f"  平均訓練誤差 = {avg_train_with[idx_opt_with_noise]:.4f}")
    print(f"  平均予測誤差 = {avg_pred_with[idx_opt_with_noise]:.4f}")
    print(f"  誤差比 (予測/訓練) = {avg_pred_with[idx_opt_with_noise]/avg_train_with[idx_opt_with_noise]:.4f}")
    print("="*70)
    
    # --- 6.4. 結果のグラフ描画 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (1) 訓練誤差の平均比較
    axes[0, 0].plot(np.log(lambda_seq), avg_train_no, 'r-o', markersize=4, label='ノイズなし')
    axes[0, 0].plot(np.log(lambda_seq), avg_train_with, 'b--^', markersize=4, label=f'ノイズあり (Σ={noise_variance_value})')
    axes[0, 0].axvline(x=np.log(lambda_seq[idx_opt_no_noise]), color='r', linestyle=':', alpha=0.5)
    axes[0, 0].axvline(x=np.log(lambda_seq[idx_opt_with_noise]), color='b', linestyle=':', alpha=0.5)
    axes[0, 0].set_xlabel(r"$\log(\lambda)$")
    axes[0, 0].set_ylabel(r"平均訓練誤差: $\mathbb{E}[||y - X\hat{\beta}||^2]$")
    axes[0, 0].set_title("平均訓練誤差の比較")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # (2) 予測誤差の平均比較
    axes[0, 1].plot(np.log(lambda_seq), avg_pred_no, 'r-o', markersize=4, label='ノイズなし')
    axes[0, 1].plot(np.log(lambda_seq), avg_pred_with, 'b--^', markersize=4, label=f'ノイズあり (Σ={noise_variance_value})')
    axes[0, 1].axvline(x=np.log(lambda_seq[idx_opt_no_noise]), color='r', linestyle=':', alpha=0.5)
    axes[0, 1].axvline(x=np.log(lambda_seq[idx_opt_with_noise]), color='b', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel(r"$\log(\lambda)$")
    axes[0, 1].set_ylabel(r"平均予測誤差: $\mathbb{E}[||\beta_0 - \hat{\beta}||^2]$")
    axes[0, 1].set_title("平均予測誤差の比較")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # (3) 訓練誤差 vs 予測誤差（ノイズなし）
    axes[1, 0].plot(np.log(lambda_seq), avg_train_no, 'g-o', markersize=4, label='平均訓練誤差')
    axes[1, 0].plot(np.log(lambda_seq), avg_pred_no, 'r--s', markersize=4, label='平均予測誤差')
    axes[1, 0].axvline(x=np.log(lambda_seq[idx_opt_no_noise]), color='k', linestyle=':', alpha=0.5, label='最適λ')
    axes[1, 0].set_xlabel(r"$\log(\lambda)$")
    axes[1, 0].set_ylabel("平均誤差")
    axes[1, 0].set_title("ノイズなし: 平均訓練誤差 vs 平均予測誤差")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # (4) 訓練誤差 vs 予測誤差（ノイズあり）
    axes[1, 1].plot(np.log(lambda_seq), avg_train_with, 'g-o', markersize=4, label='平均訓練誤差')
    axes[1, 1].plot(np.log(lambda_seq), avg_pred_with, 'b--s', markersize=4, label='平均予測誤差')
    axes[1, 1].axvline(x=np.log(lambda_seq[idx_opt_with_noise]), color='k', linestyle=':', alpha=0.5, label='最適λ')
    axes[1, 1].set_xlabel(r"$\log(\lambda)$")
    axes[1, 1].set_ylabel("平均誤差")
    axes[1, 1].set_title(f"ノイズあり (Σ={noise_variance_value}): 平均訓練誤差 vs 平均予測誤差")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.suptitle(f"訓練誤差と予測誤差の平均比較 ({n_experiments}回の実験)", fontsize=14, y=0.995)
    plt.tight_layout()
    
    # --- 6.5. 結果を保存 ---
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # グラフを保存
    plt.savefig(f'results/averaged_error_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nグラフを保存しました: results/averaged_error_comparison_{timestamp}.png")
    
    # 結果データをCSVで保存
    results_df = pd.DataFrame({
        'log_lambda': np.log(lambda_seq),
        'lambda': lambda_seq,
        'avg_train_error_no_noise': avg_train_no,
        'avg_pred_error_no_noise': avg_pred_no,
        'avg_train_error_with_noise': avg_train_with,
        'avg_pred_error_with_noise': avg_pred_with
    })
    results_df.to_csv(f'results/averaged_errors_{timestamp}.csv', index=False)
    print(f"平均結果データを保存しました: results/averaged_errors_{timestamp}.csv")