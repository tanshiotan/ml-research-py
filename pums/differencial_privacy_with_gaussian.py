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
EPSILON = 1.0  # プライバシー予算（小さいほど強い保護）
DELTA = 1e-5   # 失敗確率
SENSITIVITY = 1.0  # 感度（データの正規化範囲に基づいて設定）

# --- 関数定義 ---
def soft_th(lam, x):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def linear_lasso(X, y, lam=0, beta=None, eta=None):
    """
    Lasso回帰の座標降下法
    
    Parameters:
    - X: 特徴量行列
    - y: 目的変数
    - lam: 正則化パラメータ
    - beta: 初期係数（Noneの場合はゼロ初期化）
    - eta: 目的摂動ノイズ（差分プライバシー用）
    """
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

def generate_objective_perturbation_noise(n, p, sensitivity, epsilon, delta):
    """
    目的摂動法のためのガウシアンノイズを生成
    
    差分プライバシーを満たすノイズの標準偏差:
    σ = (sensitivity * sqrt(2 * ln(1.25/δ))) / ε
    
    Parameters:
    - n: サンプル数
    - p: 特徴量の次元数
    - sensitivity: 感度（データ1件の最大影響）
    - epsilon: プライバシー予算
    - delta: 失敗確率
    
    Returns:
    - eta: (p, n)形状のガウシアンノイズ行列
    """
    # ガウシアンメカニズムの標準偏差計算
    sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
    
    # ノイズ生成
    eta = np.random.normal(loc=0, scale=sigma, size=(p, n))
    
    return eta, sigma

def estimate_sensitivity_simple(X_train, y_train, lam, num_samples=10):
    """
    感度の簡易推定：データを1件除去した時の係数変化
    
    Parameters:
    - X_train, y_train: 訓練データ
    - lam: 正則化パラメータ
    - num_samples: サンプリング数
    
    Returns:
    - 推定感度（L2ノルム）
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

def evaluate_privacy_variance(X_train, y_train, X_test, y_test, lam, 
                              sensitivity, epsilon, delta, num_runs=30):
    """
    プライバシーノイズによる出力の分散を評価
    
    同じデータで複数回実行し、予測値のばらつきを測定
    """
    predictions = []
    
    n, p = X_train.shape
    for _ in range(num_runs):
        eta, _ = generate_objective_perturbation_noise(n, p, sensitivity, epsilon, delta)
        beta, beta_0 = linear_lasso(X_train, y_train, lam=lam, eta=eta)
        y_pred = np.dot(X_test, beta) + beta_0
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

# プライバシー予算の複数設定
epsilon_values = [0.1, 0.5, 1.0, 5.0]
lambda_seq = np.logspace(0, 4, 100)
r = len(lambda_seq)

# 結果格納用
rmse_no_noise = np.zeros(r)
rmse_dp = {eps: np.zeros(r) for eps in epsilon_values}
sensitivity_values = np.zeros(r)
noise_sigma_values = {eps: np.zeros(r) for eps in epsilon_values}
privacy_variance = {eps: np.zeros(r) for eps in epsilon_values}

n_train, p = X_train.shape

print("\n" + "="*70)
print("差分プライバシー付きLasso回帰の評価")
print("="*70)
print(f"プライバシーパラメータ:")
print(f"  - ε (epsilon) 候補: {epsilon_values}")
print(f"  - δ (delta): {DELTA}")
print(f"  - 初期感度設定: {SENSITIVITY}")
print(f"  - ノイズタイプ: ガウシアンノイズ（目的摂動法）")
print("="*70)

start_time = time.time()
print("\nStarting RMSE calculation...")

# 2. 各λに対して評価
for i, lam in enumerate(lambda_seq):
    if (i + 1) % 10 == 0:
        print(f"Progress: {i+1}/{r} (λ={lam:.2f})")
    
    # ノイズなしのベースライン
    beta, beta_0 = linear_lasso(X_train, y_train, lam=lam, eta=None)
    y_pred = np.dot(X_test, beta) + beta_0
    rmse_no_noise[i] = np.sqrt(np.mean((y_test - y_pred)**2))
    
    # 感度の推定（10回に1回実施して計算時間削減）
    if i % 10 == 0:
        estimated_sensitivity = estimate_sensitivity_simple(X_train, y_train, lam, num_samples=5)
        current_sensitivity = max(estimated_sensitivity, SENSITIVITY)  # 最小値を保証
    else:
        current_sensitivity = SENSITIVITY
    
    sensitivity_values[i] = current_sensitivity
    
    # 各εに対して差分プライバシー版を評価
    for epsilon in epsilon_values:
        # ガウシアンノイズ生成
        eta, sigma = generate_objective_perturbation_noise(
            n_train, p, current_sensitivity, epsilon, DELTA
        )
        noise_sigma_values[epsilon][i] = sigma
        
        # 差分プライバシー版のLasso回帰
        beta_dp, beta_0_dp = linear_lasso(X_train, y_train, lam=lam, eta=eta)
        y_pred_dp = np.dot(X_test, beta_dp) + beta_0_dp
        rmse_dp[epsilon][i] = np.sqrt(np.mean((y_test - y_pred_dp)**2))
        
        # プライバシー分散の評価（20回に1回実施）
        if i % 20 == 0:
            privacy_variance[epsilon][i] = evaluate_privacy_variance(
                X_train, y_train, X_test[:100], y_test[:100],
                lam, current_sensitivity, epsilon, DELTA, num_runs=20
            )

end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = elapsed_time % 60

print("\nCalculation finished.")
print(f"Total Elapsed Time: {minutes} minutes and {seconds:.2f} seconds.")

# 3. グラフ描画
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# (a) RMSE比較
ax1 = axes[0, 0]
ax1.plot(np.log(lambda_seq), rmse_no_noise, 'k-', linewidth=2.5, label='ノイズなし')
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for eps, color in zip(epsilon_values, colors):
    ax1.plot(np.log(lambda_seq), rmse_dp[eps], '--', 
             label=f'DP (ε={eps})', linewidth=2, color=color)
ax1.set_xlabel(r"$\log(\lambda)$", fontsize=12)
ax1.set_ylabel("RMSE", fontsize=12)
ax1.set_title("(a) 予測精度の比較（差分プライバシーあり/なし）", fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# (b) 精度低下率
ax2 = axes[0, 1]
for eps, color in zip(epsilon_values, colors):
    accuracy_loss = (rmse_dp[eps] - rmse_no_noise) / rmse_no_noise * 100
    ax2.plot(np.log(lambda_seq), accuracy_loss, label=f'ε={eps}', 
             linewidth=2, color=color)
ax2.set_xlabel(r"$\log(\lambda)$", fontsize=12)
ax2.set_ylabel("精度低下率 (%)", fontsize=12)
ax2.set_title("(b) プライバシー保護による精度低下", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# (c) ノイズの標準偏差
ax3 = axes[1, 0]
for eps, color in zip(epsilon_values, colors):
    ax3.plot(np.log(lambda_seq), noise_sigma_values[eps], 
             label=f'ε={eps}', linewidth=2, color=color)
ax3.set_xlabel(r"$\log(\lambda)$", fontsize=12)
ax3.set_ylabel("ノイズ標準偏差 σ", fontsize=12)
ax3.set_title("(c) 付加されるガウシアンノイズの大きさ", fontsize=13)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# (d) プライバシー分散
ax4 = axes[1, 1]
for eps, color in zip(epsilon_values, colors):
    mask = privacy_variance[eps] > 0
    ax4.plot(np.log(lambda_seq)[mask], privacy_variance[eps][mask], 
             'o-', label=f'ε={eps}', markersize=6, linewidth=2, color=color)
ax4.set_xlabel(r"$\log(\lambda)$", fontsize=12)
ax4.set_ylabel("予測の標準偏差", fontsize=12)
ax4.set_title("(d) プライバシーノイズによる出力の不確実性", fontsize=13)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4. 評価サマリーの出力
print("\n" + "="*70)
print("差分プライバシー評価サマリー")
print("="*70)

best_lambda_idx = np.argmin(rmse_no_noise)
best_lambda = lambda_seq[best_lambda_idx]

print(f"\n最適な正則化パラメータ:")
print(f"  λ = {best_lambda:.4f} (log λ = {np.log(best_lambda):.4f})")
print(f"\nノイズなしの性能:")
print(f"  RMSE = {rmse_no_noise[best_lambda_idx]:.2f}")
print(f"  推定感度 = {sensitivity_values[best_lambda_idx]:.4f}")

print(f"\n各プライバシーレベルでの性能比較:")
print(f"{'ε':>8} | {'RMSE':>10} | {'精度低下':>10} | {'ノイズσ':>10}")
print("-" * 50)
for epsilon in epsilon_values:
    rmse_at_best = rmse_dp[epsilon][best_lambda_idx]
    degradation = (rmse_at_best - rmse_no_noise[best_lambda_idx]) / rmse_no_noise[best_lambda_idx] * 100
    sigma = noise_sigma_values[epsilon][best_lambda_idx]
    print(f"{epsilon:8.2f} | {rmse_at_best:10.2f} | {degradation:9.2f}% | {sigma:10.4f}")

print(f"\n解釈ガイド:")
print(f"  - ε (epsilon): 小さいほど強いプライバシー保護（推奨: 0.1~1.0）")
print(f"  - δ (delta): 失敗確率 = {DELTA}")
print(f"  - 感度: データ1件の変更がモデルに与える最大影響")
print(f"  - λ (正則化): 大きいほど感度が小さくなり、プライバシー保護が容易")
print(f"  - トレードオフ: プライバシー保護を強めると予測精度は低下")
print("="*70)