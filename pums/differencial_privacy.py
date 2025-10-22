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

# --- 新規追加：差分プライバシーを考慮した関数 ---
def add_laplace_noise_to_coefficients(beta, beta_0, sensitivity, epsilon):
    """
    係数にラプラスノイズを追加（出力摂動法）
    
    Parameters:
    - beta: 回帰係数
    - beta_0: 切片
    - sensitivity: 感度（データ1件の変更が係数に与える最大影響）
    - epsilon: プライバシー予算
    """
    scale = sensitivity / epsilon
    noisy_beta = beta + np.random.laplace(0, scale, size=beta.shape)
    noisy_beta_0 = beta_0 + np.random.laplace(0, scale)
    return noisy_beta, noisy_beta_0

def estimate_sensitivity(X_train, y_train, lam, num_samples=10):
    """
    感度の推定：データを1件除去した時の係数変化の最大値
    
    Parameters:
    - X_train, y_train: 訓練データ
    - lam: 正則化パラメータ
    - num_samples: サンプリング数（計算時間削減のため全データではなくサンプリング）
    """
    n = len(y_train)
    beta_full, beta_0_full = linear_lasso(X_train, y_train, lam=lam)
    
    max_change = 0
    indices = np.random.choice(n, min(num_samples, n), replace=False)
    
    for idx in indices:
        # idx番目のデータを除去
        X_loo = np.delete(X_train, idx, axis=0)
        y_loo = np.delete(y_train, idx)
        
        beta_loo, beta_0_loo = linear_lasso(X_loo, y_loo, lam=lam)
        
        # 係数の変化量を計算
        change = np.linalg.norm(np.append(beta_full, beta_0_full) - 
                               np.append(beta_loo, beta_0_loo), ord=2)
        max_change = max(max_change, change)
    
    return max_change

def evaluate_privacy_empirically(X_train, y_train, X_test, y_test, lam, epsilon, num_runs=50):
    """
    経験的プライバシー評価：同じデータで複数回実行し、出力の分布を確認
    """
    predictions = []
    
    for _ in range(num_runs):
        beta, beta_0 = linear_lasso(X_train, y_train, lam=lam)
        sensitivity = estimate_sensitivity(X_train, y_train, lam, num_samples=5)
        noisy_beta, noisy_beta_0 = add_laplace_noise_to_coefficients(
            beta, beta_0, sensitivity, epsilon
        )
        y_pred = np.dot(X_test, noisy_beta) + noisy_beta_0
        predictions.append(y_pred)
    
    predictions = np.array(predictions)
    pred_std = np.std(predictions, axis=0).mean()
    
    return pred_std

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

# プライバシー予算の設定
epsilon_values = [0.1, 1.0, 10.0]  # 小さいほど強いプライバシー保護
lambda_seq = np.logspace(0, 4, 50)  # 計算時間削減のため100→50に削減
r = len(lambda_seq)

rmse_no_noise = np.zeros(r)
rmse_dp = {eps: np.zeros(r) for eps in epsilon_values}
sensitivity_values = np.zeros(r)
privacy_variance = {eps: np.zeros(r) for eps in epsilon_values}

print("\n差分プライバシー評価を開始します...")
start_time = time.time()

# 2. 各λに対して評価
for i, lam in enumerate(lambda_seq):
    print(f"Progress: {i+1}/{r} (λ={lam:.2f})")
    
    # ノイズなしのベースライン
    beta, beta_0 = linear_lasso(X_train, y_train, lam=lam)
    y_pred = np.dot(X_test, beta) + beta_0
    rmse_no_noise[i] = np.sqrt(np.mean((y_test - y_pred)**2))
    
    # 感度の推定（サンプリング数を減らして高速化）
    sensitivity = estimate_sensitivity(X_train, y_train, lam, num_samples=5)
    sensitivity_values[i] = sensitivity
    
    # 各εに対して差分プライバシー版を評価
    for epsilon in epsilon_values:
        # 差分プライバシーを適用した予測
        noisy_beta, noisy_beta_0 = add_laplace_noise_to_coefficients(
            beta, beta_0, sensitivity, epsilon
        )
        y_pred_dp = np.dot(X_test, noisy_beta) + noisy_beta_0
        rmse_dp[epsilon][i] = np.sqrt(np.mean((y_test - y_pred_dp)**2))
        
        # プライバシーの経験的評価（一部のλでのみ実施）
        if i % 10 == 0:  # 計算時間削減
            privacy_variance[epsilon][i] = evaluate_privacy_empirically(
                X_train, y_train, X_test[:100], y_test[:100], 
                lam, epsilon, num_runs=20
            )

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n計算完了: {int(elapsed_time//60)}分{elapsed_time%60:.2f}秒")

# 3. グラフ描画
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# (a) RMSE比較
ax1 = axes[0, 0]
ax1.plot(np.log(lambda_seq), rmse_no_noise, 'k-', linewidth=2, label='ノイズなし')
for epsilon in epsilon_values:
    ax1.plot(np.log(lambda_seq), rmse_dp[epsilon], '--', 
             label=f'DP (ε={epsilon})', linewidth=2)
ax1.set_xlabel(r"$\log(\lambda)$")
ax1.set_ylabel("RMSE")
ax1.set_title("(a) 予測精度の比較")
ax1.legend()
ax1.grid(True)

# (b) 精度低下率
ax2 = axes[0, 1]
for epsilon in epsilon_values:
    accuracy_loss = (rmse_dp[epsilon] - rmse_no_noise) / rmse_no_noise * 100
    ax2.plot(np.log(lambda_seq), accuracy_loss, label=f'ε={epsilon}', linewidth=2)
ax2.set_xlabel(r"$\log(\lambda)$")
ax2.set_ylabel("精度低下率 (%)")
ax2.set_title("(b) プライバシー保護による精度低下")
ax2.legend()
ax2.grid(True)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# (c) 感度の変化
ax3 = axes[1, 0]
ax3.plot(np.log(lambda_seq), sensitivity_values, 'r-', linewidth=2)
ax3.set_xlabel(r"$\log(\lambda)$")
ax3.set_ylabel("感度 (Sensitivity)")
ax3.set_title("(c) モデルの感度（データ1件の影響度）")
ax3.grid(True)

# (d) プライバシー分散
ax4 = axes[1, 1]
for epsilon in epsilon_values:
    mask = privacy_variance[epsilon] > 0
    ax4.plot(np.log(lambda_seq)[mask], privacy_variance[epsilon][mask], 
             'o-', label=f'ε={epsilon}', markersize=8)
ax4.set_xlabel(r"$\log(\lambda)$")
ax4.set_ylabel("予測の標準偏差")
ax4.set_title("(d) プライバシーノイズによる出力の不確実性")
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()

# 4. 評価サマリーの出力
print("\n" + "="*60)
print("差分プライバシー評価サマリー")
print("="*60)
best_lambda_idx = np.argmin(rmse_no_noise)
best_lambda = lambda_seq[best_lambda_idx]
print(f"\n最適なλ: {best_lambda:.4f} (log λ = {np.log(best_lambda):.4f})")
print(f"ノイズなしのRMSE: {rmse_no_noise[best_lambda_idx]:.2f}")
print(f"このλでの感度: {sensitivity_values[best_lambda_idx]:.4f}")

print("\n各プライバシーレベルでの性能:")
for epsilon in epsilon_values:
    rmse_at_best = rmse_dp[epsilon][best_lambda_idx]
    degradation = (rmse_at_best - rmse_no_noise[best_lambda_idx]) / rmse_no_noise[best_lambda_idx] * 100
    print(f"  ε={epsilon:5.1f}: RMSE={rmse_at_best:7.2f} (精度低下: {degradation:5.2f}%)")

print("\n解釈:")
print("- εが小さいほど強いプライバシー保護（但し精度は低下）")
print("- 感度が小さいほど、少ないノイズで高いプライバシー保護が可能")
print("- λ（正則化）を大きくすると感度が下がり、プライバシー保護が容易に")
print("="*60)