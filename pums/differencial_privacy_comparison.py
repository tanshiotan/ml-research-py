import pandas as pd
import copy
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import time

# --- CSVファイルを読み込む設定 ---
INPUT_CSV_PATH = "pums_wa_cleaned_full.csv"
TARGET_COLUMN = 'WAGP'

# --- 差分プライバシーパラメータ設定 ---
DELTA = 1e-5   # ガウシアンメカニズム用の失敗確率

# --- 関数定義 ---
def soft_th(lam, x):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def linear_lasso(X, y, lam=0, beta=None, eta=None):
    n, p = X.shape
    if beta is None:
        beta = np.zeros(p)
    
    X_std, y_std, X_bar, X_sd, y_bar = centralize(X, y)
    
    if eta is None:
        eta = np.zeros((p, n))

    max_iter = 500
    for i in range(max_iter):
        beta_old = copy.copy(beta)
        
        for j in range(p):
            r_j = y_std - (np.dot(X_std, beta) - X_std[:, j] * beta[j])
            z = np.dot(X_std[:, j], r_j - eta[j]) / n
            beta[j] = soft_th(lam, z)
            
        eps = np.linalg.norm(beta - beta_old, 2)
        if eps < 0.0001:
            break
            
    beta = beta / X_sd
    beta_0 = y_bar - np.dot(X_bar, beta)
    return beta, beta_0

def centralize(X0, y0, standardize=True):
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

def estimate_sensitivity(X_train, y_train, lam, num_samples=10):
    """
    感度の推定：データを1件除去した時の係数変化の最大値
    """
    n = len(y_train)
    beta_full, beta_0_full = linear_lasso(X_train, y_train, lam=lam)
    
    max_change = 0
    indices = np.random.choice(n, min(num_samples, n), replace=False)
    
    for idx in indices:
        X_loo = np.delete(X_train, idx, axis=0)
        y_loo = np.delete(y_train, idx)
        
        beta_loo, beta_0_loo = linear_lasso(X_loo, y_loo, lam=lam)
        
        change = np.linalg.norm(np.append(beta_full, beta_0_full) - 
                               np.append(beta_loo, beta_0_loo), ord=2)
        max_change = max(max_change, change)
    
    return max_change

# --- ラプラスノイズ（出力摂動法） ---
def add_laplace_noise_to_coefficients(beta, beta_0, sensitivity, epsilon):
    """
    係数にラプラスノイズを追加（出力摂動法）
    """
    scale = sensitivity / epsilon
    noisy_beta = beta + np.random.laplace(0, scale, size=beta.shape)
    noisy_beta_0 = beta_0 + np.random.laplace(0, scale)
    return noisy_beta, noisy_beta_0, scale

# --- ガウシアンノイズ（目的摂動法） ---
def generate_gaussian_noise_eta(n, p, sensitivity, epsilon, delta):
    """
    目的摂動法のためのガウシアンノイズを生成
    """
    sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
    eta = np.random.normal(loc=0, scale=sigma, size=(p, n))
    return eta, sigma

# --- メイン処理 ---

# 1. データ準備
try:
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"'{INPUT_CSV_PATH}' を読み込みました。({len(df)}行, {len(df.columns)}列)")
except FileNotFoundError:
    print(f"エラー: ファイル '{INPUT_CSV_PATH}' が見つかりません。")
    exit()

y_pd = df[TARGET_COLUMN]
X_pd = df.drop(columns=[TARGET_COLUMN])
X = X_pd.values
y = y_pd.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"データを訓練用({len(X_train)}件)とテスト用({len(X_test)}件)に分割しました。")

n_train, p = X_train.shape

# プライバシー予算の設定
epsilon_values = [0.1, 1.0, 10.0]
lambda_seq = np.logspace(0, 4, 50)
r = len(lambda_seq)

# 結果格納用
rmse_no_noise = np.zeros(r)
rmse_laplace = {eps: np.zeros(r) for eps in epsilon_values}
rmse_gaussian = {eps: np.zeros(r) for eps in epsilon_values}
sensitivity_values = np.zeros(r)
noise_scale_laplace = {eps: np.zeros(r) for eps in epsilon_values}
noise_scale_gaussian = {eps: np.zeros(r) for eps in epsilon_values}

print("\n" + "="*70)
print("ラプラスノイズ vs ガウシアンノイズ 比較評価")
print("="*70)
print(f"プライバシーパラメータ:")
print(f"  - ε (epsilon) 候補: {epsilon_values}")
print(f"  - δ (delta): {DELTA} (ガウシアンのみ)")
print(f"  - 感度: データから推定")
print(f"比較手法:")
print(f"  - ラプラスノイズ: 出力摂動法（係数に直接ノイズ）")
print(f"  - ガウシアンノイズ: 目的摂動法（最適化にノイズ）")
print("="*70)

start_time = time.time()
print("\n計算を開始します...")

# 2. 各λに対して評価
for i, lam in enumerate(lambda_seq):
    if (i + 1) % 10 == 0:
        print(f"Progress: {i+1}/{r} (λ={lam:.2f})")
    
    # ノイズなしのベースライン
    beta, beta_0 = linear_lasso(X_train, y_train, lam=lam, eta=None)
    y_pred = np.dot(X_test, beta) + beta_0
    rmse_no_noise[i] = np.sqrt(np.mean((y_test - y_pred)**2))
    
    # 感度の推定
    sensitivity = estimate_sensitivity(X_train, y_train, lam, num_samples=5)
    sensitivity_values[i] = sensitivity
    
    # 各εに対して両方の手法を評価
    for epsilon in epsilon_values:
        # --- ラプラスノイズ版 ---
        noisy_beta_lap, noisy_beta_0_lap, scale_lap = add_laplace_noise_to_coefficients(
            beta, beta_0, sensitivity, epsilon
        )
        y_pred_lap = np.dot(X_test, noisy_beta_lap) + noisy_beta_0_lap
        rmse_laplace[epsilon][i] = np.sqrt(np.mean((y_test - y_pred_lap)**2))
        noise_scale_laplace[epsilon][i] = scale_lap
        
        # --- ガウシアンノイズ版 ---
        eta, sigma_gauss = generate_gaussian_noise_eta(
            n_train, p, sensitivity, epsilon, DELTA
        )
        beta_gauss, beta_0_gauss = linear_lasso(X_train, y_train, lam=lam, eta=eta)
        y_pred_gauss = np.dot(X_test, beta_gauss) + beta_0_gauss
        rmse_gaussian[epsilon][i] = np.sqrt(np.mean((y_test - y_pred_gauss)**2))
        noise_scale_gaussian[epsilon][i] = sigma_gauss

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n計算完了: {int(elapsed_time//60)}分{elapsed_time%60:.2f}秒")

# 3. グラフ描画
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

colors_eps = {0.1: '#e74c3c', 1.0: '#2ecc71', 10.0: '#f39c12'}

# (a) RMSE比較 - ラプラスノイズ
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(np.log(lambda_seq), rmse_no_noise, 'k-', linewidth=2.5, label='ノイズなし')
for eps in epsilon_values:
    ax1.plot(np.log(lambda_seq), rmse_laplace[eps], '--', 
             label=f'ε={eps}', linewidth=2, color=colors_eps[eps])
ax1.set_xlabel(r"$\log(\lambda)$", fontsize=11)
ax1.set_ylabel("RMSE", fontsize=11)
ax1.set_title("(a) ラプラスノイズ: 予測精度", fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (b) RMSE比較 - ガウシアンノイズ
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(np.log(lambda_seq), rmse_no_noise, 'k-', linewidth=2.5, label='ノイズなし')
for eps in epsilon_values:
    ax2.plot(np.log(lambda_seq), rmse_gaussian[eps], '--', 
             label=f'ε={eps}', linewidth=2, color=colors_eps[eps])
ax2.set_xlabel(r"$\log(\lambda)$", fontsize=11)
ax2.set_ylabel("RMSE", fontsize=11)
ax2.set_title("(b) ガウシアンノイズ: 予測精度", fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# (c) 両者の直接比較（ε=1.0の場合）
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(np.log(lambda_seq), rmse_no_noise, 'k-', linewidth=2.5, label='ノイズなし')
ax3.plot(np.log(lambda_seq), rmse_laplace[1.0], '--', 
         label='ラプラス (ε=1.0)', linewidth=2, color='#3498db')
ax3.plot(np.log(lambda_seq), rmse_gaussian[1.0], '-.', 
         label='ガウシアン (ε=1.0)', linewidth=2, color='#9b59b6')
ax3.set_xlabel(r"$\log(\lambda)$", fontsize=11)
ax3.set_ylabel("RMSE", fontsize=11)
ax3.set_title("(c) 直接比較 (ε=1.0)", fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# (d) 精度低下率 - ラプラスノイズ
ax4 = fig.add_subplot(gs[1, 0])
for eps in epsilon_values:
    loss = (rmse_laplace[eps] - rmse_no_noise) / rmse_no_noise * 100
    ax4.plot(np.log(lambda_seq), loss, label=f'ε={eps}', 
             linewidth=2, color=colors_eps[eps])
ax4.set_xlabel(r"$\log(\lambda)$", fontsize=11)
ax4.set_ylabel("精度低下率 (%)", fontsize=11)
ax4.set_title("(d) ラプラス: 精度低下", fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# (e) 精度低下率 - ガウシアンノイズ
ax5 = fig.add_subplot(gs[1, 1])
for eps in epsilon_values:
    loss = (rmse_gaussian[eps] - rmse_no_noise) / rmse_no_noise * 100
    ax5.plot(np.log(lambda_seq), loss, label=f'ε={eps}', 
             linewidth=2, color=colors_eps[eps])
ax5.set_xlabel(r"$\log(\lambda)$", fontsize=11)
ax5.set_ylabel("精度低下率 (%)", fontsize=11)
ax5.set_title("(e) ガウシアン: 精度低下", fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# (f) 精度低下率の差分
ax6 = fig.add_subplot(gs[1, 2])
for eps in epsilon_values:
    loss_lap = (rmse_laplace[eps] - rmse_no_noise) / rmse_no_noise * 100
    loss_gauss = (rmse_gaussian[eps] - rmse_no_noise) / rmse_no_noise * 100
    diff = loss_lap - loss_gauss
    ax6.plot(np.log(lambda_seq), diff, label=f'ε={eps}', 
             linewidth=2, color=colors_eps[eps])
ax6.set_xlabel(r"$\log(\lambda)$", fontsize=11)
ax6.set_ylabel("精度低下率の差 (%)", fontsize=11)
ax6.set_title("(f) ラプラス - ガウシアン", fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)
ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# (g) 感度の変化
ax7 = fig.add_subplot(gs[2, 0])
ax7.plot(np.log(lambda_seq), sensitivity_values, 'r-', linewidth=2.5)
ax7.set_xlabel(r"$\log(\lambda)$", fontsize=11)
ax7.set_ylabel("感度 (Sensitivity)", fontsize=11)
ax7.set_title("(g) 推定感度の変化", fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

# (h) ノイズスケール比較（ε=1.0）
ax8 = fig.add_subplot(gs[2, 1])
ax8.plot(np.log(lambda_seq), noise_scale_laplace[1.0], 
         label='ラプラス scale', linewidth=2, color='#3498db')
ax8.plot(np.log(lambda_seq), noise_scale_gaussian[1.0], 
         label='ガウシアン σ', linewidth=2, color='#9b59b6')
ax8.set_xlabel(r"$\log(\lambda)$", fontsize=11)
ax8.set_ylabel("ノイズパラメータ", fontsize=11)
ax8.set_title("(h) ノイズの大きさ比較 (ε=1.0)", fontsize=12, fontweight='bold')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)

# (i) 相対的な性能
ax9 = fig.add_subplot(gs[2, 2])
best_idx = np.argmin(rmse_no_noise)
x_pos = np.arange(len(epsilon_values))
width = 0.35

rmse_lap_at_best = [rmse_laplace[eps][best_idx] for eps in epsilon_values]
rmse_gauss_at_best = [rmse_gaussian[eps][best_idx] for eps in epsilon_values]
baseline = rmse_no_noise[best_idx]

loss_lap = [(r - baseline) / baseline * 100 for r in rmse_lap_at_best]
loss_gauss = [(r - baseline) / baseline * 100 for r in rmse_gauss_at_best]

ax9.bar(x_pos - width/2, loss_lap, width, label='ラプラス', color='#3498db', alpha=0.8)
ax9.bar(x_pos + width/2, loss_gauss, width, label='ガウシアン', color='#9b59b6', alpha=0.8)
ax9.set_xlabel('ε (プライバシー予算)', fontsize=11)
ax9.set_ylabel('精度低下率 (%)', fontsize=11)
ax9.set_title(f'(i) 最適λでの比較 (λ={lambda_seq[best_idx]:.2f})', 
              fontsize=12, fontweight='bold')
ax9.set_xticks(x_pos)
ax9.set_xticklabels([str(eps) for eps in epsilon_values])
ax9.legend(fontsize=9)
ax9.grid(True, alpha=0.3, axis='y')

plt.savefig('laplace_vs_gaussian_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 4. 詳細な評価サマリー
print("\n" + "="*70)
print("評価サマリー: ラプラスノイズ vs ガウシアンノイズ")
print("="*70)

best_lambda_idx = np.argmin(rmse_no_noise)
best_lambda = lambda_seq[best_lambda_idx]

print(f"\n最適な正則化パラメータ:")
print(f"  λ = {best_lambda:.4f} (log λ = {np.log(best_lambda):.4f})")
print(f"  RMSE (ノイズなし) = {rmse_no_noise[best_lambda_idx]:.2f}")
print(f"  推定感度 = {sensitivity_values[best_lambda_idx]:.4f}")

print(f"\n【ラプラスノイズ】出力摂動法:")
print(f"{'ε':>8} | {'RMSE':>10} | {'精度低下':>10} | {'scale':>10}")
print("-" * 50)
for epsilon in epsilon_values:
    rmse_at_best = rmse_laplace[epsilon][best_lambda_idx]
    degradation = (rmse_at_best - rmse_no_noise[best_lambda_idx]) / rmse_no_noise[best_lambda_idx] * 100
    scale = noise_scale_laplace[epsilon][best_lambda_idx]
    print(f"{epsilon:8.2f} | {rmse_at_best:10.2f} | {degradation:9.2f}% | {scale:10.4f}")

print(f"\n【ガウシアンノイズ】目的摂動法:")
print(f"{'ε':>8} | {'RMSE':>10} | {'精度低下':>10} | {'σ':>10}")
print("-" * 50)
for epsilon in epsilon_values:
    rmse_at_best = rmse_gaussian[epsilon][best_lambda_idx]
    degradation = (rmse_at_best - rmse_no_noise[best_lambda_idx]) / rmse_no_noise[best_lambda_idx] * 100
    sigma = noise_scale_gaussian[epsilon][best_lambda_idx]
    print(f"{epsilon:8.2f} | {rmse_at_best:10.2f} | {degradation:9.2f}% | {sigma:10.4f}")

print(f"\n【比較】どちらが優れているか:")
print(f"{'ε':>8} | {'優れた手法':>15} | {'RMSEの差':>12}")
print("-" * 40)
for epsilon in epsilon_values:
    rmse_lap = rmse_laplace[epsilon][best_lambda_idx]
    rmse_gauss = rmse_gaussian[epsilon][best_lambda_idx]
    diff = rmse_lap - rmse_gauss
    better = "ガウシアン" if rmse_gauss < rmse_lap else "ラプラス"
    print(f"{epsilon:8.2f} | {better:>15} | {abs(diff):11.2f}")

print(f"\n解釈:")
print(f"  - ラプラスノイズ: 係数に直接ノイズを加える（シンプル）")
print(f"  - ガウシアンノイズ: 最適化過程にノイズを加える（理論的）")
print(f"  - 一般的にガウシアンの方がε-δ差分プライバシーで有利")
print(f"  - ラプラスはε差分プライバシー（δ=0）で厳密な保証")
print("="*70)