import numpy as np
import uuid
import copy
from .config import PARAM_CONSTRAINTS

# --- ヘルパー関数: ネストされた辞書へのアクセス ---
def get_nested_value(ind, dot_key):
    """ドット区切りのキーを使ってネストされた辞書から値を取得する"""
    val = ind
    for k in dot_key.split('.'):
        val = val[k]
    return val

def set_nested_value(ind, dot_key, value):
    """ドット区切りのキーを使ってネストされた辞書に値を設定する"""
    keys = dot_key.split('.')
    current = ind
    # 最後のキーの手前まで掘り下げる
    for k in keys[:-1]:
        current = current[k]
    # 最後のキーに値をセット
    current[keys[-1]] = value

# ---------------------------------------------------

RNG = np.random.default_rng(252)

def sample_new_population_from_probability_model(best_individuals, num_samples, params_info, generation):
    next_generation = []
    
    # --- 1. 統計情報の計算 ---
    
    # 適応度と重みの計算
    fitnesses = np.array([ind['fitness'] for ind in best_individuals])
    if np.sum(fitnesses) == 0:
        weights = np.ones(len(best_individuals)) / len(best_individuals)
    else:
        weights = fitnesses / np.sum(fitnesses)

    current_std_devs = {}
    
    # 標準偏差の計算（これは全個体の分散を使う：カーネルの幅に相当）
    for param_key in params_info:
        vals = np.array([get_nested_value(ind, param_key) for ind in best_individuals])
        std = np.std(vals)
        
        # ★追加: カーネル幅の拡大 (Bandwidth Scaling)
        # Top-Ncの分布よりも「少し広め」に探索することで、
        # 確率モデルの裾野を広げ、早期収束を防ぎます。
        # 1.5 〜 2.0 程度の値を推奨します。
        std *= 2.0 

        # 最小探索幅 (5%ルール) は維持
        param_range = PARAM_CONSTRAINTS.get(param_key, (0, 100))
        val_range = param_range[1] - param_range[0]
        min_std = val_range * 0.05
        
        if std < min_std:
            std = min_std
        current_std_devs[param_key] = std

    # --- 2. 混合分布からのサンプリング ---
    
    for _ in range(num_samples):
        new_ind = copy.deepcopy(best_individuals[0])
        
        # ★変更点: ここで「どの山（個体）周辺を探索するか」を確率的に決める
        # これにより、複数の有望な領域を同時に探索できる（多峰性の維持）
        selected_parent = RNG.choice(best_individuals, p=weights)
        
        for param_key in params_info:
            param_range = PARAM_CONSTRAINTS.get(param_key, (0, 100))
            min_val, max_val = param_range
            
            # ★変更点: 全体の平均ではなく、「選ばれた個体の値」を中心にする
            center_value = get_nested_value(selected_parent, param_key)
            
            # 標準偏差は「集団全体の広がり」を使う（または少し縮小しても良い）
            # 論文のカーネル幅の概念に従い、集団の標準偏差をそのまま使うのが安全
            std_dev = current_std_devs[param_key]
            
            # リサンプリング (Rejection Sampling)
            max_retries = 100
            new_value = center_value 
            
            for _ in range(max_retries):
                candidate = np.random.normal(center_value, std_dev)
                if min_val <= candidate <= max_val:
                    new_value = candidate
                    break
            else:
                candidate = np.random.normal(center_value, std_dev)
                new_value = max(min_val, min(max_val, candidate))
            
            set_nested_value(new_ind, param_key, new_value)
            
        new_ind["chromosomeId"] = str(uuid.uuid4())
        new_ind["generation"] = generation
        new_ind["fitness"] = 0.0
        new_ind["pre_evaluation"] = 0.0
        new_ind["true_fitness"] = 0.0
        
        next_generation.append(new_ind)
        
    return next_generation