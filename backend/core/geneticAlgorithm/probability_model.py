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
    """
    論文の SAF-IEDA に基づくサンプリング (確率モデルからの次世代生成)
    """
    next_generation = []
    
    # 全個体の適応度(fitness)の合計
    fitness_sum = sum(ind['fitness'] for ind in best_individuals)
    if fitness_sum == 0: 
        fitness_sum = 1.0

    for _ in range(num_samples):
        # 1. ベースとなる親を選択（ルーレット選択）
        rand_val = RNG.uniform() * fitness_sum
        current_sum = 0
        selected_parent_ind = best_individuals[0]
        
        for ind in best_individuals:
            current_sum += ind['fitness']
            if current_sum >= rand_val:
                selected_parent_ind = ind
                break
        
        # 2. 親個体をディープコピーして新しい個体の雛形にする
        # これによりネストされた構造(fmParamsListなど)を維持できます
        new_ind = copy.deepcopy(selected_parent_ind)
        
        # 3. 各パラメータを確率モデルに従って更新
        for param_key in params_info:
            # 定義域情報の取得
            param_range = PARAM_CONSTRAINTS.get(param_key, (0, 100))
            min_val, max_val = param_range
            val_range = max_val - min_val
            
            # --- 修正箇所: ヘルパー関数を使って値を取得 ---
            base_value = get_nested_value(selected_parent_ind, param_key)
            # ---------------------------------------------
            
            # サンプリング (摂動を加える)
            std_dev = val_range * 0.2 
            if std_dev == 0: std_dev = 1.0
            
            new_value = np.random.normal(base_value, std_dev)
            
            # 定義域内にクリッピング
            new_value = max(min_val, min(max_val, new_value))
            
            # --- 修正箇所: ヘルパー関数を使って値を設定 ---
            set_nested_value(new_ind, param_key, new_value)
            # ---------------------------------------------
            
        # 4. 個体情報の更新
        new_ind["chromosomeId"] = str(uuid.uuid4())
        new_ind["generation"] = generation
        new_ind["fitness"] = 0.0
        new_ind["pre_evaluation"] = 0.0
        new_ind["true_fitness"] = 0.0
        
        next_generation.append(new_ind)
        
    return next_generation