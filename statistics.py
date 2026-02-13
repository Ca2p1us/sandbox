import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import numpy as np
from scipy import stats

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
    "Gaussian": (0, 7.0),
    "Sphere": (0, 1.5),         # 状況に合わせて調整してください
    "Gaussian_cos": (0, 7.0),
    "Ackley": (0, 7.0),
    "Gaussian_peaks": (0, 7.0),
    "Mixed": (0.0, 7.0)
}

# 距離グラフ用のY軸範囲設定 (必要に応じて調整)
DISTANCE_Y_LIMITS = (0, 0.7)

# 手法の設定 (表示順序)
# key: グラフのX軸ラベル
# value: パス生成関数
METHODS_CONFIG = {
    "提案手法": lambda func: os.path.join(
        "result", "proposal", "best", func, "Hybrid", "200inds_9eval"
    ),
    "距離項なしサロゲート": lambda func: os.path.join(
        "result", "proposal", "best", func, "TPS", "200inds_9eval"
    ),
    "GA\n(9個体)": lambda func: os.path.join(
        "result", "conventional", "best", func, "9inds"
    ),
    "GA\n(200個体)": lambda func: os.path.join(
        "result", "conventional", "best", func, "200inds"
    ),
    "SAF-IEDA\n(200個体)": lambda func: os.path.join(
        "result", "saf_ieda", "best", func, "200inds"
    ),
}

# 手法ごとの色設定 (ラベルの一部でマッチングさせます)
# これによりSAF-IEDAを抜いても提案手法の色(オレンジ)が維持されます
COLOR_MAP = {
    "GA": "#d3d3a0",   # 黄緑系
    "SAF-IEDA": "#a0cbe8", # 青系
    "提案手法": "#f0c0a0"   # オレンジ系
}

# 軸ラベルの設定
Y_LABEL = "最終世代における最大適応度"
X_LABEL = "手法"

# ==========================================
# 処理ロジック
# ==========================================

def perform_statistical_test(data, labels, func_name):
    """
    1. シャピロ・ウィルク検定で正規性を確認
    2. 正規性がある場合はウェルチのt検定、ない場合はマン・ホイットニーのU検定を実行
    """
    target1 = "提案手法"
    target2 = "距離項なしサロゲート"

    idx1 = next((i for i, l in enumerate(labels) if target1 in l), None)
    idx2 = next((i for i, l in enumerate(labels) if target2 in l), None)

    if idx1 is None or idx2 is None:
        return

    group1 = data[idx1]
    group2 = data[idx2]

    print(f"\n  --- 統計的検定結果 ({func_name}) ---")

    # --- 1. 正規性の検定 (Shapiro-Wilk) ---
    # p > 0.05 なら正規分布とみなせる
    _, p_norm1 = stats.shapiro(group1) if len(group1) >= 3 else (0, 0)
    _, p_norm2 = stats.shapiro(group2) if len(group2) >= 3 else (0, 0)

    is_normal1 = p_norm1 > 0.05
    is_normal2 = p_norm2 > 0.05

    print(f"    正規性確認 (Shapiro-Wilk):")
    print(f"      {target1}: p={p_norm1:.4e} ({'正規分布' if is_normal1 else '非正規分布'})")
    print(f"      {target2}: p={p_norm2:.4e} ({'正規分布' if is_normal2 else '非正規分布'})")

    # --- 2. 検定の選択と実行 ---
    if is_normal1 and is_normal2:
        # 両方正規分布ならウェルチのt検定
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        test_name = "ウェルチのt検定 (パラメトリック)"
    else:
        # どちらかが非正規ならマン・ホイットニーのU検定
        t_stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        test_name = "マン・ホイットニーのU検定 (ノンパラメトリック)"

    print(f"    採用された検定: {test_name}")
    print(f"    p値: {p_val:.4e}")

    if p_val < 0.05:
        print(f"    => 有意差あり (p < 0.05)")
    else:
        print(f"    => 有意差なし (p >= 0.05)")
    print("  ----------------------------------------------")

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

            # --- 統計量の計算と出力 ---
            avg = np.mean(fitness_values)    # 平均値
            std = np.std(fitness_values)     # 標準偏差
            med = np.median(fitness_values)  # 中央値
            
            clean_label = label.replace(chr(10), ' ')
            print(f"  {clean_label}:")
            print(f"    データ数: {len(fitness_values)}")
            print(f"    平均値  : {avg:.4f}")
            print(f"    標準偏差: {std:.4f}")
            print(f"    中央値  : {med:.4f}")
    
    return plot_data, labels, colors

def draw_boxplot(data, labels, colors, func_name, suffix=""):
    if not data:
        return

    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(data, tick_labels=labels, patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.5))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.8)

    plt.ylabel(Y_LABEL, fontsize=18)
    plt.xlabel(X_LABEL, fontsize=18)
    plt.tick_params(axis='x', labelsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    if func_name in Y_AXIS_LIMITS and suffix != "_no_saf":
        plt.ylim(Y_AXIS_LIMITS[func_name])

    plt.tight_layout()
    
    # --- 保存先の変更 ---
    output_dir = os.path.join("result", "analysis", "boxplot")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"boxplot_{func_name}{suffix}.pdf")
    # ------------------
    
    plt.savefig(output_file, dpi=300)
    print(f"  保存完了: {output_file}")
    plt.close()

def get_distance_history_from_file(filepath):
    """
    _distance_history.json から [(gen, dist), ...] のリストを取得
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # キー (例: "1_distance_history") を動的に取得
        first_key = list(data.keys())[0]
        history = data[first_key]
        return history
    except Exception as e:
        # print(f"Error reading {filepath}: {e}")
        return None

def collect_distance_data(func_name):
    """
    各手法ごとの距離推移データの平均を取得する
    戻り値: 辞書 { label: { "gens": [], "means": [], "stds": [], "color": ... } }
    """
    results = {}

    print(f"--- ベンチマーク関数: {func_name} (距離推移) の処理中 ---")

    for label, path_builder in METHODS_CONFIG.items():
        # 通常のパス生成
        best_dir_path = path_builder(func_name)
        
        # パスの置換: 'best' -> 'distance_histories'
        # 注: ディレクトリ構造が 'best' と並列であることを前提としています
        dist_dir_path = best_dir_path.replace("best", "distance_histories")
        dist_dir_path = os.path.normpath(dist_dir_path)

        # ファイル検索 (_distance_history.json を対象)
        json_files = glob.glob(os.path.join(dist_dir_path, "*_distance_history.json"))
        
        if not json_files:
            # GAなどは距離ログがない場合があるのでスキップ
            continue

        # 全試行のデータを収集
        # all_trials[generation_index] = [val_trial1, val_trial2, ...]
        all_trials_data = {} 
        generations = []

        for json_file in json_files:
            history = get_distance_history_from_file(json_file)
            if history:
                for gen, dist in history:
                    # distance_historyが3要素(gen, sel_dist, pop_dist)の場合と2要素の場合に対応
                    # ここでは sel_dist (2番目の要素) を採用する
                    val = dist
                    
                    if gen not in all_trials_data:
                        all_trials_data[gen] = []
                    all_trials_data[gen].append(val)
        
        if not all_trials_data:
            continue

        # 世代順にソート
        generations = sorted(all_trials_data.keys())
        means = []
        stds = []

        for gen in generations:
            vals = all_trials_data[gen]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        
        # 色の決定
        color = "#000000"
        for key_word, c_code in COLOR_MAP.items():
            if key_word in label:
                color = c_code
                break
        
        # ラベルの改行削除
        clean_label = label.replace('\n', ' ')

        results[clean_label] = {
            "gens": generations,
            "means": means,
            "stds": stds,
            "color": color
        }
        print(f"  {clean_label}: {len(json_files)} 試行のデータを集計")

    return results

def draw_distance_transition_graph(distance_data, func_name):
    if not distance_data: return

    plt.figure(figsize=(8, 6))
    for label, data in distance_data.items():
        plt.plot(data["gens"], data["means"], label=label, color=data["color"], linewidth=2, marker='o', markersize=4)
        lower = np.maximum(np.array(data["means"]) - np.array(data["stds"]), 0)
        upper = np.array(data["means"]) + np.array(data["stds"])
        plt.fill_between(data["gens"], lower, upper, color=data["color"], alpha=0.15)

    # plt.title(f"Average Nearest Neighbor Distance: {func_name}", fontsize=14)
    plt.xlabel("Generation", fontsize=16)
    plt.ylabel("Avg NN Distance (Normalized)", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=18, loc='upper right')
    plt.ylim(*DISTANCE_Y_LIMITS)
    plt.tight_layout()
    
    # --- 保存先の変更 ---
    output_dir = os.path.join("result", "analysis", "distance_history")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"distance_history_{func_name}.pdf")
    # ------------------

    plt.savefig(output_file, dpi=300)
    print(f"  保存完了: {output_file}")
    plt.close()


def collect_weight_distance_data(func_name):
    results = {}
    target_dir = os.path.normpath(os.path.join("result", "benchmark", func_name, "Hybrid"))
    
    if not os.path.exists(target_dir):
        # フォルダがない場合はユーザーに通知してスキップ（エラーにはしない）
        # print(f"  [Skip] Weight dir not found: {target_dir}")
        return {}

    json_files = glob.glob(os.path.join(target_dir, "*weight_distance_history.json"))
    if not json_files: return {}

    print(f"--- データ収集: 重み別距離推移 ({func_name}) ---")
    weight_map = {}

    for json_file in json_files:
        match = re.search(r"_(\d+(\.\d+)?)weight_", os.path.basename(json_file))
        if not match: continue
        weight = float(match.group(1))
        
        history = get_distance_history_from_file(json_file)
        if history:
            if weight not in weight_map: weight_map[weight] = {}
            for gen, dist in history:
                if gen not in weight_map[weight]: weight_map[weight][gen] = []
                weight_map[weight][gen].append(dist)
    
    for w in sorted(weight_map.keys()):
        gens = sorted(weight_map[w].keys())
        results[w] = {
            "gens": gens,
            "means": [np.mean(weight_map[w][g]) for g in gens],
            "stds": [np.std(weight_map[w][g]) for g in gens]
        }
        print(f"  w={w}: {len(weight_map[w][gens[0]])} trials")
        
    return results

def draw_weight_distance_transition_graph(weight_data, func_name):
    if not weight_data: return

    plt.figure(figsize=(10, 7))
    weights = sorted(weight_data.keys())
    colors = cm.viridis(np.linspace(0, 1, len(weights)))

    data = None # 変数初期化

    for idx, w in enumerate(weights):
        data = weight_data[w]
        plt.plot(data["gens"], data["means"], label=f"w={w}", color=colors[idx], linewidth=2, marker='o', markersize=4)
        lower = np.maximum(np.array(data["means"]) - np.array(data["stds"]), 0)
        upper = np.array(data["means"]) + np.array(data["stds"])
        plt.fill_between(data["gens"], lower, upper, color=colors[idx], alpha=0.1)

    # plt.title(f"Distance History by Weight: {func_name}", fontsize=14)]
    plt.title("")
    plt.xlabel("世代", fontsize=18)
    
    # --- 修正箇所: 1刻み設定 ---
    if data is not None:
        plt.xticks(np.arange(min(data["gens"]), max(data["gens"]) + 1, 1))
    # -------------------------

    plt.ylabel("平均最近傍距離", fontsize=18)

    # --- 追加箇所: 軸目盛りのフォントサイズを変更 ---
    plt.tick_params(labelsize=15) 
    # ---------------------------------------------

    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.legend(fontsize=20, loc='upper right')
    plt.ylim(*DISTANCE_Y_LIMITS)
    plt.tight_layout()
    
    # --- 保存先の変更 ---
    output_dir = os.path.join("result", "analysis", "weight_distance_history")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"weight_distance_history_{func_name}.pdf")
    # ------------------

    plt.savefig(output_file, dpi=300)
    print(f"  保存完了: {output_file}")
    plt.close()


def draw_weight_win_counts(weight_data, func_name):
    """
    重みごとの勝利数（最小距離記録回数）グラフを描画し、カウントデータを返す
    """
    if not weight_data:
        return {}

    weights = list(weight_data.keys())
    
    # 最終世代での最小距離を持つ重みを特定して表示
    best_final_w = None
    min_final_dist = float('inf')

    for w in weights:
        means = weight_data[w]["means"]
        if means:
            final_val = means[-1] 
            if final_val < min_final_dist:
                min_final_dist = final_val
                best_final_w = w
    
    if best_final_w is not None:
        print(f"  ★ [{func_name}] 最終世代で最も距離が小さい重み: w={best_final_w} (Distance: {min_final_dist:.4f})")

    # 共通する世代を取得
    common_gens = set(weight_data[weights[0]]["gens"])
    for w in weights[1:]:
        common_gens &= set(weight_data[w]["gens"])
    
    sorted_gens = sorted(list(common_gens))
    if not sorted_gens:
        return {}

    win_counts = {w: 0 for w in weights}

    for gen in sorted_gens:
        min_val = float('inf')
        best_w = None
        for w in weights:
            try:
                idx = weight_data[w]["gens"].index(gen)
                val = weight_data[w]["means"][idx]
                if val < min_val:
                    min_val = val
                    best_w = w
            except ValueError:
                pass
        
        if best_w is not None:
            win_counts[best_w] += 1

    plt.figure(figsize=(10, 6))
    sorted_weights = sorted(weights)
    counts = [win_counts[w] for w in sorted_weights]
    labels = [str(w) for w in sorted_weights]
    
    bars = plt.bar(labels, counts, color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.xlabel("Weight (w)", fontsize=14)
    plt.ylabel("Win Count (Generations)", fontsize=14)
    plt.title(f"Number of Generations with Lowest Avg NN Distance: {func_name}", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    
    output_dir = os.path.join("result", "analysis", "weight_distance_history")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"weight_win_counts_{func_name}.pdf")

    plt.savefig(output_file, dpi=300)
    print(f"  保存完了: {output_file}")
    plt.close()

    return win_counts # 戻り値としてカウントデータを返す

def draw_total_weight_win_counts(aggregated_counts):
    """
    全関数の勝利数を合計して棒グラフを描画
    """
    if not aggregated_counts:
        return

    plt.figure(figsize=(10, 6))
    sorted_weights = sorted(aggregated_counts.keys())
    counts = [aggregated_counts[w] for w in sorted_weights]
    labels = [str(w) for w in sorted_weights]
    
    bars = plt.bar(labels, counts, color='lightgreen', edgecolor='black', alpha=0.7)
    
    plt.xlabel("Weight (w)", fontsize=14)
    plt.ylabel("Total Win Count (Generations)", fontsize=14)
    plt.title("Total Number of Generations with Lowest Avg NN Distance (All Functions)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    
    output_dir = os.path.join("result", "analysis", "weight_distance_history")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "total_weight_win_counts.pdf")

    plt.savefig(output_file, dpi=300)
    print(f"  全体勝利数グラフを保存しました: {output_file}")
    plt.close()

def run_analysis(func_name, plot_types):
    """
    指定された関数とプロットタイプに基づいて解析を実行する
    plot_types: list of str ("boxplot", "distance", "weight")
    """
    print(f"\n[{func_name}] の解析を開始します...")
    win_counts = {}

    # 1. 適応度比較 (Boxplot)
    if "boxplot" in plot_types:
        all_data, all_labels, all_colors = collect_data(func_name)
        if all_data:
            draw_boxplot(all_data, all_labels, all_colors, func_name, suffix="_all")
            
            # --- ここでt検定を実行 ---
            perform_statistical_test(all_data, all_labels, func_name)
            
            no_saf_data = [d for d, l in zip(all_data, all_labels) if "SAF-IEDA" not in l]
            no_saf_labels = [l for l in all_labels if "SAF-IEDA" not in l]
            no_saf_colors = [c for c, l in zip(all_colors, all_labels) if "SAF-IEDA" not in l]
            if no_saf_data:
                draw_boxplot(no_saf_data, no_saf_labels, no_saf_colors, func_name, suffix="_no_saf")
        else:
            print("  適応度データが見つかりませんでした。")

    # 2. 手法別距離推移
    if "distance" in plot_types:
        dist_data = collect_distance_data(func_name)
        if dist_data:
            draw_distance_transition_graph(dist_data, func_name)
        else:
            print("  手法別距離データが見つかりませんでした。")

    # 3. 重み別距離推移
    if "weight" in plot_types:
        weight_dist_data = collect_weight_distance_data(func_name)
        if weight_dist_data:
            draw_weight_distance_transition_graph(weight_dist_data, func_name)
            win_counts = draw_weight_win_counts(weight_dist_data, func_name) # 変更: 戻り値を受け取る
        else:
            print("  重み比較データなし")

    return win_counts # 追加: データを返す

def draw_combined_boxplot():
    """
    4つの評価関数を2x2のグリッドで箱ひげ図として描画する
    """
    # 対象とする4つの評価関数
    target_funcs = ["Gaussian", "Gaussian_cos", "Gaussian_peaks", "Ackley"]
    
    # 図の作成 (サイズは log.py の設定を参考に調整)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    output_dir = os.path.join("result", "analysis", "boxplot")
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- 4関数結合箱ひげ図の作成を開始します ---")

    for i, func_name in enumerate(target_funcs):
        ax = axes[i]
        
        # データの収集
        data, labels, colors = collect_data(func_name)
        
        if data:
            # 箱ひげ図の描画
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                            medianprops=dict(color="black", linewidth=1.5))
            
            # 色塗り
            for j, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[j])
                patch.set_alpha(0.8)
            
            # 軸設定
            ax.set_title(func_name, fontsize=18)
            ax.set_ylabel("最終世代における最大適応度", fontsize=14)
            ax.tick_params(axis='x', labelsize=11)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
            # Y軸範囲の適用
            if func_name in Y_AXIS_LIMITS:
                ax.set_ylim(Y_AXIS_LIMITS[func_name])
        else:
            # データがない場合の表示
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=16)
            ax.set_title(func_name, fontsize=18)
            print(f"  警告: {func_name} のデータが見つかりませんでした。")

    # レイアウト調整
    plt.tight_layout()
    
    # 保存
    output_file = os.path.join(output_dir, "combined_boxplot_4metrics.pdf")
    plt.savefig(output_file, dpi=300)
    print(f"保存完了: {output_file}")
    plt.close()


if __name__ == "__main__":
    if not os.path.exists("result"):
        print("警告: 'result' ディレクトリが見つかりません。実行場所を確認してください。")

    print("=========================================")
    print("      統計グラフ作成・解析ツール")
    print("=========================================")
    print("解析モードを選択してください:")
    print("1: 全ベンチマーク関数を一括出力 (従来モード)")
    print("2: 個別のベンチマーク関数を選択して出力")
    
    mode = input("モードを選択 (1/2): ")

    if mode == "1":
        # 全関数一括実行
        print("\n--- 全関数一括モードを実行します ---")
        
        aggregated_win_counts = {} # 追加: 集計用辞書
        
        for func in BENCHMARK_FUNCTIONS:
            counts = run_analysis(func, plot_types=["boxplot", "distance", "weight"])
            
            # 追加: 勝利数を集計
            if counts:
                for w, c in counts.items():
                    aggregated_win_counts[w] = aggregated_win_counts.get(w, 0) + c
            print("")
            
        # 追加: 最後に合計グラフを描画
        if aggregated_win_counts:
            print("\n--- 全関数合計の重み比較グラフを作成します ---")
            draw_total_weight_win_counts(aggregated_win_counts)

    elif mode == "2":
        # 関数選択
        print("\n解析するベンチマーク関数を選択してください:")
        for i, func in enumerate(BENCHMARK_FUNCTIONS):
            print(f"{i+1}: {func}")
        
        try:
            func_idx = int(input(f"番号を入力 (1-{len(BENCHMARK_FUNCTIONS)}): ")) - 1
            if 0 <= func_idx < len(BENCHMARK_FUNCTIONS):
                target_func = BENCHMARK_FUNCTIONS[func_idx]
                
                # グラフタイプ選択
                print(f"\n[{target_func}] に対して作成するグラフを選択してください:")
                print("1: 適応度箱ひげ図 (Boxplot)")
                print("2: 手法別距離推移 (Distance History)")
                print("3: 重み別距離推移 (Weight Distance Comparison)")
                print("4: すべて作成")
                
                plot_choice = input("番号を入力 (1/2/3/4): ")
                
                selected_plots = []
                if plot_choice == "1":
                    selected_plots = ["boxplot"]
                elif plot_choice == "2":
                    selected_plots = ["distance"]
                elif plot_choice == "3":
                    selected_plots = ["weight"]
                elif plot_choice == "4":
                    selected_plots = ["boxplot", "distance", "weight"]
                else:
                    print("無効な入力です。すべて作成します。")
                    selected_plots = ["boxplot", "distance", "weight"]

                run_analysis(target_func, selected_plots)
            
            else:
                print("無効な番号です。終了します。")
        except ValueError:
            print("入力エラーです。数値を入力してください。")
    elif mode == "3":
        # 結合箱ひげ図モード
        draw_combined_boxplot()

    else:
        print("無効なモードです。終了します。")