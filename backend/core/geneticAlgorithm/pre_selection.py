#評価する個体を選ぶプログラム
from ..geneticAlgorithm.config import NUM_GENERATIONS
from typing import List
import numpy as np

def select_top_individuals_by_pre_evaluation(population: List[dict], total_n: int = 10, gen:int = 1) -> List[str]:
    """
    pre_evaluation（事前評価済み）が整数・小数どちらでも対応し、降順ソートして上位top_n個体のIDリストを返す
    """
    def is_number(s):
        try:
            float(s)
            return True
        except Exception:
            return False

    
    # pre_evaluationが未設定または変換できない個体は除外
    valid_population = [
        ind for ind in population
        if "pre_evaluation" in ind and is_number(ind["pre_evaluation"]) and "chromosomeId" in ind
    ]
    if not valid_population:
        print("有効な個体が存在しません。")
        return []

    # 降順ソート
    sorted_population = sorted(valid_population, key=lambda x: float(x["pre_evaluation"]), reverse=True)

    if gen < int(NUM_GENERATIONS/2):
        # 上位と下位をそれぞれ選択
        half_top = total_n // 2 + (total_n % 2)  # 奇数なら上位を1つ多く
        half_bottom = total_n // 2

        top_inds = sorted_population[:half_top]
        bottom_inds = sorted_population[-half_bottom:] if half_bottom > 0 else []

        # 合体（順序は上位→下位）
        selected_inds = top_inds + bottom_inds
    else:
        selected_inds = sorted_population[:total_n]


    # 重複を防止（極端に少ないpopulationでも安全）
    selected_ids = []
    for ind in selected_inds:
        cid = ind["chromosomeId"]
        if cid not in selected_ids:
            selected_ids.append(cid)
    # 個体IDリストで返す
    return selected_ids

def select_active_samples(population, evaluated_inds, n_select, param_keys):
    """
    補間モデルの不確実性（＝評価済み個体からの距離）に基づき、未評価個体から選択
    """
    def get_param_vec(ind):
        vals = []
        for k in param_keys:
            val = ind
            for kk in k.split('.'):
                val = val.get(kk, None)
                if val is None:
                    break
            vals.append(val)
        return np.array(vals, dtype=float)
    
    evaluated_vecs = [get_param_vec(ind) for ind in evaluated_inds]
    
    # 各未評価個体について、最も近い評価済み個体までの距離を算出
    candidates = []
    for ind in population:
        if ind in evaluated_inds:
            continue
        v = get_param_vec(ind)
        dist = min(np.linalg.norm(v - e) for e in evaluated_vecs)
        candidates.append((dist, ind))
    
    # 距離が大きい＝不確実性が高い
    candidates.sort(reverse=True, key=lambda x: x[0])
    return [ind for _, ind in candidates[:n_select]]
