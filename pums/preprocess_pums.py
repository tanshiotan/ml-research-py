import pandas as pd
import numpy as np

# --- 設定項目 ---
INPUT_CSV_PATH = "psam_p53.csv" 
OUTPUT_CSV_PATH = "pums_wa_cleaned_full.csv" # 出力ファイル名を変更
TARGET_COLUMN = 'WAGP'

FEATURE_COLUMNS = [
    'AGEP', 'SEX', 'MAR', 'RAC1P', 'HISP', 'CIT', 'SCHL', 'ESR', 
    'COW', 'WKHP', 'OCCP', 'INDP', 'PUMA', 'DIS', 'ENG'
]
# ... 他にも試したい基本変数があればここに追加 ...

# ワンホットエンコーディングするカテゴリ変数
CATEGORICAL_COLS = [
    'SEX', 'MAR', 'RAC1P', 'HISP', 'CIT', 'SCHL', 'COW', 
    'OCCP', 'INDP', 'PUMA', 'DIS', 'ENG'
]

# ★★★ 特徴量削減のキーポイント ★★★
# カテゴリ数が多すぎる変数と、残したい上位カテゴリの数を指定
HIGH_CARDINALITY_COLS = {
    'OCCP': 30,  # 職業コードは上位30個に絞る
    'INDP': 30,  # 産業コードは上位30個に絞る
    'PUMA': 20   # 地域コードは上位20個に絞る
}

# --- ここからプログラム本体 ---
def preprocess_pums():
    print(f"--- データ前処理（特徴量削減版）を開始します ---")
    
    # 1. データの読み込み
    try:
        df = pd.read_csv(INPUT_CSV_PATH, usecols=[TARGET_COLUMN] + FEATURE_COLUMNS)
        print(f"読み込み成功: {INPUT_CSV_PATH} ({len(df)}行)")
    except Exception as e:
        print(f"エラー: ファイルの読み込みに失敗しました。詳細: {e}")
        return

    # 2. 分析対象の絞り込み
    initial_rows = len(df)
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    df = df[df[TARGET_COLUMN] > 0]
    df = df[df['ESR'].isin([1, 2])]
    df = df.drop(columns=['ESR'])
    print(f"分析対象を絞り込みました: {initial_rows}行 → {len(df)}行")

    # 3. 高カーディナリティ変数の集約
    for col, top_n in HIGH_CARDINALITY_COLS.items():
        if col in df.columns:
            # 上位Nカテゴリを取得
            top_categories = df[col].value_counts().nlargest(top_n).index
            # 上位N以外を 'Other' に置き換え
            df[col] = df[col].where(df[col].isin(top_categories), 'Other')
            print(f"'{col}'列を上位{top_n}カテゴリと'Other'に集約しました。")

    # 4. 欠損値を含む行を削除
    df.dropna(inplace=True)
    
    # 5. ワンホットエンコーディング
    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns], drop_first=True, dtype=float)

    # 6. 保存
    df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n--- 前処理が完了しました ---")
    print(f"クリーンなデータが'{OUTPUT_CSV_PATH}'として保存されました。")
    print(f"最終的なデータサイズ: {len(df)}行, {len(df.columns)}列")

if __name__ == '__main__':
    preprocess_pums()