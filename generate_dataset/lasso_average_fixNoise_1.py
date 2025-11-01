import numpy as np
import time
import os
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['axes.unicode_minus'] = False

# ===================================================================
# データ生成
# ===================================================================

def generate_synthetic_data(n, p, beta0, sigma_y=1.0, seed=None):
    """
    MATLAB: y = F*x0/sqrt(N) + noise
    """
    if seed is not None:
        np.random.seed(seed)
    
    F = np.random.normal(0, 1.0, size=(n, p))
    sqN = np.sqrt(p)
    y_true = (F @ beta0) / sqN
    noise = np.random.normal(0, sigma_y, size=n)
    y = y_true + noise
    
    return F, y


def generate_true_beta(p, rho, seed=None):
    """
    K = ceil(N*rho);
    x0 = randn(N,1);
    zero_ind = randperm(N,N-K);
    x0(zero_ind) = 0;
    """
    if seed is not None:
        np.random.seed(seed)
    
    K = int(np.ceil(p * rho))
    x0 = np.random.normal(0, 1.0, size=p)
    
    # N-K個をランダムに選んでゼロに → K個が非ゼロのまま
    zero_indices = np.random.choice(p, p - K, replace=False)
    x0[zero_indices] = 0.0
    
    return x0


# ===================================================================
# Coordinate Descent LASSO
# ===================================================================

def lasso_cd(F, y, lambda1=0, lambda2=0, beta_ini=None, eta=None,
             max_iter=1000, tol=1e-8):
    """
    C実装準拠のCoordinate Descent
    入力: F (n, p) 正規化前の特徴量行列
    処理: F/sqrt(p) を使って計算
    """
    n, p = F.shape
    sqN = np.sqrt(p)
    
    # 初期化
    if beta_ini is None:
        beta = np.zeros(p)
    else:
        beta = beta_ini.copy()
    
    if eta is None:
        eta = np.zeros(p)
    
    # norm_x[i] = lambda_2 + Σ(F[:,i]^2) / N
    norm_x = lambda2 + np.sum(F**2, axis=0) / p
    
    # 初期残差: r = y - F*beta/sqrt(N)
    y_hat = (F @ beta) / sqN
    r = y - y_hat
    
    converged = False
    
    for t in range(max_iter):
        n_converged = 0
        indices = np.random.permutation(p)
        
        for j in indices:
            # Cavity residual
            r_cavity = r + (F[:, j] * beta[j]) / sqN
            
            # b = (F[:,j]/sqrt(N))^T * r_cavity
            b = np.dot(F[:, j], r_cavity) / sqN
            
            beta_old_j = beta[j]
            
            # Soft thresholding
            z = b - eta[j]
            if np.abs(z) > lambda1:
                beta[j] = np.sign(z) * (np.abs(z) - lambda1) / norm_x[j]
            else:
                beta[j] = 0.0
            
            if np.abs(beta[j] - beta_old_j) < tol:
                n_converged += 1
            
            # 残差更新
            r = r - (F[:, j] * (beta[j] - beta_old_j)) / sqN
        
        if n_converged == p:
            converged = True
            break
    
    return beta, converged, t+1


# ===================================================================
# 実験実行
# ===================================================================

def run_single_experiment(n, p, beta0_true, lambda_seq, 
                         fixed_eta, lambda2, n_test, sigma_y):
    """単一実験の実行"""
    # データ生成
    F_train, y_train = generate_synthetic_data(n, p, beta0_true, sigma_y)
    F_test, y_test = generate_synthetic_data(n_test, p, beta0_true, sigma_y)
    
    sqN = np.sqrt(p)
    r = len(lambda_seq)
    
    results = {
        'train_no': np.zeros(r),
        'train_with': np.zeros(r),
        'test_no': np.zeros(r),
        'test_with': np.zeros(r),
        'sparse_no': np.zeros(r),
        'sparse_with': np.zeros(r)
    }
    
    for i, lam in enumerate(lambda_seq):
        # ノイズなし
        beta_no, _, _ = lasso_cd(F_train, y_train, lambda1=lam, 
                                 lambda2=lambda2, eta=None)
        
        y_pred_train = (F_train @ beta_no) / sqN
        y_pred_test = (F_test @ beta_no) / sqN
        
        results['train_no'][i] = np.mean((y_train - y_pred_train)**2)
        results['test_no'][i] = np.mean((y_test - y_pred_test)**2)
        results['sparse_no'][i] = np.sum(np.abs(beta_no) > 1e-10) / p
        
        # ノイズあり
        beta_with, _, _ = lasso_cd(F_train, y_train, lambda1=lam, 
                                   lambda2=lambda2, eta=fixed_eta)
        
        y_pred_train = (F_train @ beta_with) / sqN
        y_pred_test = (F_test @ beta_with) / sqN
        
        results['train_with'][i] = np.mean((y_train - y_pred_train)**2)
        results['test_with'][i] = np.mean((y_test - y_pred_test)**2)
        results['sparse_with'][i] = np.sum(np.abs(beta_with) > 1e-10) / p
    
    return results


def run_averaged_experiments(n, p, rho, lambda_seq, sigma_eta, 
                            lambda2, n_experiments, n_test, sigma_y):
    """複数実験の平均"""
    # 固定ノイズ生成（標準偏差 sigma_eta）
    fixed_eta = np.random.normal(0, sigma_eta, size=p)
    
    print(f"\n{'='*70}")
    print(f"sigma_eta = {sigma_eta} での実験")
    print(f"{'='*70}")
    print(f"固定ノイズ: eta ~ N(0, {sigma_eta}^2), shape={fixed_eta.shape}")
    
    # 累積用
    r = len(lambda_seq)
    avg_results = {
        'train_no': np.zeros(r),
        'train_with': np.zeros(r),
        'test_no': np.zeros(r),
        'test_with': np.zeros(r),
        'sparse_no': np.zeros(r),
        'sparse_with': np.zeros(r)
    }
    
    start = time.time()
    
    for exp_num in range(n_experiments):
        if (exp_num + 1) % 10 == 0:
            print(f"  {exp_num + 1}/{n_experiments} 完了")
        
        beta0 = generate_true_beta(p, rho)
        results = run_single_experiment(n, p, beta0, lambda_seq, 
                                       fixed_eta, lambda2, n_test, sigma_y)
        
        for key in avg_results:
            avg_results[key] += results[key]
    
    # 平均化
    for key in avg_results:
        avg_results[key] /= n_experiments
    
    elapsed = time.time() - start
    print(f"完了: {elapsed:.2f}秒\n")
    
    return avg_results


# ===================================================================
# グラフ描画関数
# ===================================================================

def plot_results(all_results, M, N, rho, n_experiments, timestamp):
    """
    2つのsigma_etaの結果を比較するグラフを作成
    """
    fig = plt.figure(figsize=(16, 10))
    
    # sigma_etaごとに色を設定
    colors = {0.10: 'blue', 1.0: 'red'}
    labels = {0.10: r'$\sigma_\eta = 0.10$', 1.0: r'$\sigma_\eta = 1.0$'}
    
    # --- (1) Training Error比較 ---
    ax1 = plt.subplot(2, 3, 1)
    for sigma_eta, data in all_results.items():
        lam_seq = data['lambda_seq']
        res = data['results']
        color = colors[sigma_eta]
        
        ax1.plot(lam_seq, res['train_no'], '-o', color=color, alpha=0.5,
                markersize=4, label=f'{labels[sigma_eta]} (no noise)')
        ax1.plot(lam_seq, res['train_with'], '--^', color=color,
                markersize=4, label=f'{labels[sigma_eta]} (with noise)')
    
    ax1.set_xlabel(r'$\lambda$', fontsize=11)
    ax1.set_ylabel('Training Error', fontsize=11)
    ax1.set_title('Training Error vs Lambda', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    
    # --- (2) Test Error比較 ---
    ax2 = plt.subplot(2, 3, 2)
    for sigma_eta, data in all_results.items():
        lam_seq = data['lambda_seq']
        res = data['results']
        color = colors[sigma_eta]
        
        ax2.plot(lam_seq, res['test_no'], '-o', color=color, alpha=0.5,
                markersize=4, label=f'{labels[sigma_eta]} (no noise)')
        ax2.plot(lam_seq, res['test_with'], '--^', color=color,
                markersize=4, label=f'{labels[sigma_eta]} (with noise)')
    
    ax2.set_xlabel(r'$\lambda$', fontsize=11)
    ax2.set_ylabel('Test Error', fontsize=11)
    ax2.set_title('Test Error vs Lambda', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # --- (3) Sparsity比較 ---
    ax3 = plt.subplot(2, 3, 3)
    for sigma_eta, data in all_results.items():
        lam_seq = data['lambda_seq']
        res = data['results']
        color = colors[sigma_eta]
        
        ax3.plot(lam_seq, res['sparse_no'], '-o', color=color, alpha=0.5,
                markersize=4, label=f'{labels[sigma_eta]} (no noise)')
        ax3.plot(lam_seq, res['sparse_with'], '--^', color=color,
                markersize=4, label=f'{labels[sigma_eta]} (with noise)')
    
    ax3.set_xlabel(r'$\lambda$', fontsize=11)
    ax3.set_ylabel('Sparsity Ratio', fontsize=11)
    ax3.set_title('Sparsity vs Lambda', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    
    # --- (4) Error Ratio vs Lambda ---
    ax4 = plt.subplot(2, 3, 4)
    for sigma_eta, data in all_results.items():
        lam_seq = data['lambda_seq']
        res = data['results']
        color = colors[sigma_eta]
        
        # ゼロ除算回避
        ratio_no = np.where(res['train_no'] > 1e-10, 
                           res['test_no'] / res['train_no'], np.nan)
        ratio_with = np.where(res['train_with'] > 1e-10,
                             res['test_with'] / res['train_with'], np.nan)
        
        ax4.plot(lam_seq, ratio_no, '-o', color=color, alpha=0.5,
                markersize=4, label=f'{labels[sigma_eta]} (no noise)')
        ax4.plot(lam_seq, ratio_with, '--^', color=color,
                markersize=4, label=f'{labels[sigma_eta]} (with noise)')
    
    ax4.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel(r'$\lambda$', fontsize=11)
    ax4.set_ylabel('Test Error / Training Error', fontsize=11)
    ax4.set_title('Error Ratio vs Lambda', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    
    # --- (5) Train vs Test (sigma_eta=0.10) ---
    ax5 = plt.subplot(2, 3, 5)
    if 0.10 in all_results:
        data = all_results[0.10]
        lam_seq = data['lambda_seq']
        res = data['results']
        
        ax5.plot(lam_seq, res['train_no'], 'g-o', markersize=4, 
                label='Train (no noise)')
        ax5.plot(lam_seq, res['test_no'], 'r--s', markersize=4,
                label='Test (no noise)')
        ax5.plot(lam_seq, res['train_with'], 'b-^', markersize=4,
                label='Train (with noise)')
        ax5.plot(lam_seq, res['test_with'], 'm--d', markersize=4,
                label='Test (with noise)')
    
    ax5.set_xlabel(r'$\lambda$', fontsize=11)
    ax5.set_ylabel('Error', fontsize=11)
    ax5.set_title(r'Train vs Test ($\sigma_\eta = 0.10$)', fontsize=12)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8)
    
    # --- (6) Train vs Test (sigma_eta=1.0) ---
    ax6 = plt.subplot(2, 3, 6)
    if 1.0 in all_results:
        data = all_results[1.0]
        lam_seq = data['lambda_seq']
        res = data['results']
        
        ax6.plot(lam_seq, res['train_no'], 'g-o', markersize=4,
                label='Train (no noise)')
        ax6.plot(lam_seq, res['test_no'], 'r--s', markersize=4,
                label='Test (no noise)')
        ax6.plot(lam_seq, res['train_with'], 'b-^', markersize=4,
                label='Train (with noise)')
        ax6.plot(lam_seq, res['test_with'], 'm--d', markersize=4,
                label='Test (with noise)')
    
    ax6.set_xlabel(r'$\lambda$', fontsize=11)
    ax6.set_ylabel('Error', fontsize=11)
    ax6.set_title(r'Train vs Test ($\sigma_\eta = 1.0$)', fontsize=12)
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=8)
    
    # 全体のタイトル
    fig.suptitle(f'LASSO (M={M}, N={N}, rho={rho}, experiments={n_experiments})',
                fontsize=14, y=0.995)
    
    plt.tight_layout()
    
    # 保存
    filename = f'results/LASSO_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nグラフ保存: {filename}")
    plt.close()


# ===================================================================
# メイン処理
# ===================================================================

if __name__ == "__main__":
    print("="*70)
    print("Coordinate Descent LASSO")
    print("="*70)
    
    # パラメータ設定
    M = 500
    N = 1000
    n = M
    p = N
    rho = 0.1
    K = int(np.ceil(N * rho))
    sigma_y = 1.0
    lambda2 = 0.0
    
    n_experiments = 1
    n_test = 1000
    
    print(f"  n (データ数) = {M}")
    print(f"  p (特徴数) = {N}")
    print(f"  K (非ゼロ要素数) = {K}")
    print(f"  rho = {rho}")
    print(f"  sigma_y = {sigma_y}")
    print(f"  n_experiments = {n_experiments}")
    print("="*70)
    
    # lambda範囲（実験データに合わせる）
    lambda_configs = {
        0.10: np.array([3.0, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7, 
                        0.6, 0.5, 0.4, 0.3, 0.2, 0.18, 0.16, 0.14, 0.12, 0.11]),
        1.0: np.array([3.0, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 
                       1.3, 1.26, 1.22, 1.18, 1.16, 1.15, 1.14, 1.13, 1.1, 1.09])
    }
    
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for sigma_eta in [0.10, 1.0]:
        lambda_seq = lambda_configs[sigma_eta]
        
        results = run_averaged_experiments(
            n, p, rho, lambda_seq, sigma_eta, 
            lambda2, n_experiments, n_test, sigma_y
        )
        
        all_results[sigma_eta] = {
            'lambda_seq': lambda_seq,
            'results': results
        }
        
        # 結果表示
        print(f"【sigma_eta = {sigma_eta}】")
        
        # lambda=1.0付近の結果
        idx = np.argmin(np.abs(lambda_seq - 1.0))
        print(f"\n  lambda = {lambda_seq[idx]:.2f} 付近:")
        print(f"    [ノイズなし]")
        print(f"      Training Error = {results['train_no'][idx]:.4f}")
        print(f"      Test Error = {results['test_no'][idx]:.4f}")
        print(f"      Sparsity = {results['sparse_no'][idx]:.4f}")
        print(f"    [ノイズあり]")
        print(f"      Training Error = {results['train_with'][idx]:.4f}")
        print(f"      Test Error = {results['test_with'][idx]:.4f}")
        print(f"      Sparsity = {results['sparse_with'][idx]:.4f}")
        
        # 期待値（実験データから）
        if sigma_eta == 0.10:
            print(f"\n  【期待値】")
            print(f"    Training Error ~ 0.716, Test Error ~ 1.149")
            print(f"    Sparsity ~ 0.217")
        elif sigma_eta == 1.0:
            print(f"\n  【期待値 (lambda=1.4付近)】")
            idx_14 = np.argmin(np.abs(lambda_seq - 1.4))
            print(f"    lambda = {lambda_seq[idx_14]:.2f}:")
            print(f"      Training Error = {results['train_with'][idx_14]:.4f} (期待: ~0.88)")
            print(f"      Test Error = {results['test_with'][idx_14]:.4f} (期待: ~3.4)")
            print(f"      Sparsity = {results['sparse_with'][idx_14]:.4f} (期待: ~0.49)")
        
        print("\n" + "-"*70)
    
    # グラフ描画
    os.makedirs("results", exist_ok=True)
    plot_results(all_results, M, N, rho, n_experiments, timestamp)
    
    # CSV保存
    for sigma_eta, data in all_results.items():
        lambda_seq = data['lambda_seq']
        res = data['results']
        
        df = pd.DataFrame({
            'lambda': lambda_seq,
            'train_no': res['train_no'],
            'test_no': res['test_no'],
            'train_with': res['train_with'],
            'test_with': res['test_with'],
            'sparse_no': res['sparse_no'],
            'sparse_with': res['sparse_with']
        })
        
        filename = f'results/LASSO_sigma{sigma_eta}_{timestamp}.csv'
        df.to_csv(filename, index=False)
        print(f"CSV保存: {filename}")
    
    print("\n" + "="*70)
    print("実験完了")
    print("="*70)