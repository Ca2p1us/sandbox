import json
import os
import glob
import matplotlib.pyplot as plt

# ==========================================
# 設定: 日本語フォントとラベル
# ==========================================

# Windows標準の日本語フォントを設定
plt.rcParams['font.family'] = 'MS Gothic'

# ベンチマーク関数リスト
BENCHMARK_FUNCTIONS = [
    "Gaussian",
    "Sphere",
    "Gaussian_cos",
    "Ackley",
    "Gaussian_peaks",
    "Mixed"
]

# Y軸の範囲設定 (最小値, 最大値)
# 必要に応じて数値を調整してください
Y_AXIS_LIMITS = {
    "Gaussian": (0, 6.2),
    "Sphere": (0, 1.5),         # 状況に合わせて調整してください
    "Gaussian_cos": (0, 7.2),
    "Ackley": (0, 6.0),
    "Gaussian_peaks": (0, 6.2),
    "Mixed": (0.0, 6.2)
}

# 手法の設定 (表示順序)
# key: グラフのX軸ラベル
# value: パス生成関数
METHODS_CONFIG = {
    "提案手法": lambda func: os.path.join(
        "result", "proposal", "best", func, "Hybrid", "200inds_9eval"
    ),
    "SAF-IEDA\n(200個体)": lambda func: os.path.join(
        "result", "saf_ieda", "best", func, "200inds"
    ),
    "従来法\n(9個体)": lambda func: os.path.join(
        "result", "conventional", "best", func, "9inds"
    ),
    "従来法\n(50個体)": lambda func: os.path.join(
        "result", "conventional", "best", func, "50inds"
    ),
    "従来法\n(200個体)": lambda func: os.path.join(
        "result", "conventional", "best", func, "200inds"
    ),
}

# 手法ごとの色設定 (ラベルの一部でマッチングさせます)
# これによりSAF-IEDAを抜いても提案手法の色(オレンジ)が維持されます
COLOR_MAP = {
    "従来法": "#d3d3a0",   # 黄緑系
    "SAF-IEDA": "#a0cbe8", # 青系
    "提案手法": "#f0c0a0"   # オレンジ系
}

# 軸ラベルの設定
Y_LABEL = "最終世代における最大適応度"
X_LABEL = "手法"

# ==========================================
# 処理ロジック
# ==========================================

def get_final_fitness_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 最初のキーを取得 (例: "1_results")
        first_key = list(data.keys())[0]
        generations_list = data[first_key]
        
        # 最終世代の結果を取得
        final_gen = generations_list[-1]
        return final_gen['fitness']
        
    except Exception as e:
        return None

def collect_data(func_name):
    """
    データ収集を行う関数
    戻り値: (data_list, label_list, color_list)
    """
    plot_data = []
    labels = []
    colors = []

    print(f"--- ベンチマーク関数: {func_name} の処理中 ---")

    for label, path_builder in METHODS_CONFIG.items():
        dir_path = path_builder(func_name)
        dir_path = os.path.normpath(dir_path)

        json_files = glob.glob(os.path.join(dir_path, "*noiseFalse*.json"))
        
        fitness_values = []
        for json_file in json_files:
            val = get_final_fitness_from_file(json_file)
            if val is not None:
                fitness_values.append(val)
        
        # データが存在する場合のみリストに追加
        if fitness_values:
            plot_data.append(fitness_values)
            labels.append(label)
            
            # 色の決定
            color = "#cccccc" # デフォルトグレー
            for key_word, c_code in COLOR_MAP.items():
                if key_word in label:
                    color = c_code
                    break
            colors.append(color)

            avg = sum(fitness_values) / len(fitness_values)
            print(f"  {label.replace(chr(10), ' ')}: {len(fitness_values)}個ロード (平均: {avg:.4f})")
    
    return plot_data, labels, colors

def draw_boxplot(data, labels, colors, func_name, suffix=""):
    """
    箱ひげ図を描画して保存する
    suffix: ファイル名の末尾につける識別子 ("_all" や "_no_saf" など)
    """
    if not data:
        print(f"  データがないため {func_name}{suffix} のグラフ作成をスキップします。")
        return

    plt.figure(figsize=(8, 6))
    
    # 箱ひげ図の描画
    bp = plt.boxplot(data, labels=labels, patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.5))

    # 色の設定 (渡されたcolorsリストを使用)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.8)

    plt.title(f"評価関数: {func_name}", fontsize=14)
    plt.ylabel(Y_LABEL, fontsize=12)
    plt.xlabel(X_LABEL, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Y軸の範囲設定
    if func_name in Y_AXIS_LIMITS:
        y_min, y_max = Y_AXIS_LIMITS[func_name]
        plt.ylim(y_min, y_max)

    plt.tight_layout()
    output_file = f"boxplot_{func_name}{suffix}.png"
    plt.savefig(output_file, dpi=300)
    print(f"  グラフを保存しました: {output_file}")
    plt.close()

if __name__ == "__main__":
    if not os.path.exists("result"):
        print("警告: 'result' ディレクトリが見つかりません。")

    for func in BENCHMARK_FUNCTIONS:
        # 1. 全データの収集
        all_data, all_labels, all_colors = collect_data(func)
        
        if not all_data:
            continue

        # 2. 全手法入りのグラフ作成 (_all)
        draw_boxplot(all_data, all_labels, all_colors, func, suffix="_all")

        # 3. SAF-IEDAを除外したデータの作成
        no_saf_data = []
        no_saf_labels = []
        no_saf_colors = []

        for d, l, c in zip(all_data, all_labels, all_colors):
            if "SAF-IEDA" not in l:
                no_saf_data.append(d)
                no_saf_labels.append(l)
                no_saf_colors.append(c)

        # 4. SAF-IEDA抜きのグラフ作成 (_no_saf)
        if no_saf_data:
            draw_boxplot(no_saf_data, no_saf_labels, no_saf_colors, func, suffix="_no_saf")
        
        print("") # 空行