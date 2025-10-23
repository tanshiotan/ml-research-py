import numpy as np
import matplotlib.pyplot as plt
import os

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

def visualize_data(X, y, beta0, filename=None):
    """
    生成されたデータを可視化し、指定された場合はファイルに保存
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 真のパラメータβ0の可視化
    axes[0, 0].stem(beta0, basefmt=' ')
    axes[0, 0].set_title('True Parameter β0 (Sparse)')
    axes[0, 0].set_xlabel('Parameter Index')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 真のパラメータのヒストグラム(非ゼロ要素のみ)
    non_zero_beta = beta0[beta0 != 0]
    axes[0, 1].hist(non_zero_beta, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title(f'Distribution of Non-zero β0 (n={len(non_zero_beta)})')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 説明変数Xの最初の2次元の散布図
    axes[1, 0].scatter(X[:, 0], X[:, 1], alpha=0.6)
    axes[1, 0].set_title('Scatter Plot of X (first 2 dimensions)')
    axes[1, 0].set_xlabel('X[:, 0]')
    axes[1, 0].set_ylabel('X[:, 1]')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 目的変数yのヒストグラム
    axes[1, 1].hist(y, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Distribution of y')
    axes[1, 1].set_xlabel('y')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        output_dir = os.path.dirname(filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(filename)
        print(f"\nプロットを '{filename}' に保存しました。")
        plt.close(fig) 
    else:
        plt.show()

def print_data_statistics(X, y, beta0):
    """
    生成されたデータの統計情報を表示
    """
    print("=" * 60)
    print("データ生成結果の統計情報")
    print("=" * 60)
    print(f"\n【データサイズ】")
    print(f"  データ数 n: {X.shape[0]}")
    print(f"  パラメータ数 p: {X.shape[1]}")
    print(f"  n < p: {X.shape[0] < X.shape[1]}")
    
    print(f"\n【説明変数 X】")
    print(f"  形状: {X.shape}")
    print(f"  平均: {np.mean(X):.6f}")
    print(f"  分散: {np.var(X):.6f} (期待値: {1/X.shape[1]:.6f})")
    print(f"  標準偏差: {np.std(X):.6f}")
    
    print(f"\n【真のパラメータ β0】")
    print(f"  形状: {beta0.shape}")
    print(f"  非ゼロ要素数: {np.sum(beta0 != 0)}")
    print(f"  スパース率: {np.sum(beta0 == 0) / len(beta0):.2%}")
    if np.sum(beta0 != 0) > 0:
        print(f"  非ゼロ要素の平均: {np.mean(beta0[beta0 != 0]):.6f}")
        print(f"  非ゼロ要素の標準偏差: {np.std(beta0[beta0 != 0]):.6f}")
    
    print(f"\n【目的変数 y】")
    print(f"  形状: {y.shape}")
    print(f"  平均: {np.mean(y):.6f}")
    print(f"  分散: {np.var(y):.6f}")
    print(f"  標準偏差: {np.std(y):.6f}")
    print(f"  最小値: {np.min(y):.6f}")
    print(f"  最大値: {np.max(y):.6f}")
    print("=" * 60)

def save_data_to_csv(X, y, beta0, prefix='output/data', number=1):
    """
    生成されたデータをCSVファイルに保存する。
    Xとyは1つのファイルに結合する。
    """
    output_dir = os.path.dirname(prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Xとyを結合 (yを最後の列にする)
    # yを (n,) から (n, 1) に変形して結合
    combined_data = np.c_[X, y]
    
    # 2. 結合したデータと、真の係数beta0をそれぞれ保存
    dataset_filename = f"{prefix}_dataset_{number}.csv"
    beta0_filename = f"{prefix}_beta0_{number}.csv"
    
    np.savetxt(dataset_filename, combined_data, delimiter=",", fmt='%.18e')
    np.savetxt(beta0_filename, beta0, delimiter=",", fmt='%.18e')
    
    print(f"\nデータを以下のCSVファイルに保存しました:")
    print(f"  - データセット (X, y): {dataset_filename}")
    print(f"  - 真の係数 (beta0): {beta0_filename}")

# === 使用例 ===
if __name__ == "__main__":
    n = 100
    p = 200
    rho = 0.1
    
    X, y, beta0 = generate_synthetic_data(n, p, rho, seed=42)
    print_data_statistics(X, y, beta0)
    
    file_prefix = "output/synthetic_data" 
    file_number = 1
    
    plot_filename = f"{file_prefix}_plot_{file_number}.png"
    visualize_data(X, y, beta0, filename=plot_filename)
    
    save_data_to_csv(X, y, beta0, prefix=file_prefix, number=file_number)
