#事前評価・適応度評価を行う

from typing import List, Dict, Tuple
import math
from scipy.interpolate import RBFInterpolator
import numpy as np
from ..geneticAlgorithm.config import PARAMS,NUM_GENERATIONS
from ...engine import evaluate
from .make_chromosome_params import make_chromosome_params
from ...engine.evaluate import evaluate_fitness, get_best_and_worst_individuals
from ..log import log
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=10)
EPS_RATIO = 0.02  # ガウス補間のeps計算用割合

def interpolation(
  population: List[dict] = None,
  evaluated_population: List[dict] = None,
  best: dict = None,
  worst: dict = None,
  method_num: int = 0,
  gen:int = 0,
  param_keys: List[str] = PARAMS,
  target_key: str = "pre_evaluation",
  refernce_key = "fitness",      
):
    # print(f"補間を開始します。")
    if not best or not worst:
        print(f"bestまたはworstがNoneです。ランダムな{target_key}を付与します。")
        for ind in population:
            ind[target_key] = RNG.uniform(0.0, 6.0)
        return
    if not evaluated_population:
        print(f"評価済み個体群がNoneです。ランダムな{target_key}を付与します。")
        for ind in population:
            ind[target_key] = RNG.uniform(0.0, 6.0)
        return
    
    if param_keys is None:
        # デフォルトはoperator1のfrequencyのみ
        param_keys = ["fmParamsList.operator1.frequency"]

    # 全個体（未評価 + 評価済み）からMin/Maxを取得
    all_inds = population
    min_max_dict = get_min_max_dict(all_inds, param_keys)

    # 評価済み個体を「正規化ベクトル」と「正解値」のペアリストに変換しておく
    # 構造: [(normalized_vec, fitness_value), ...]
    norm_eval_data = []
    for ind in evaluated_population:
        vec = to_normalized_vec(ind, param_keys, min_max_dict)
        val = float(ind.get(refernce_key, 0.0))
        norm_eval_data.append((vec, val))

    best_params = to_normalized_vec(best, param_keys, min_max_dict)
    worst_params = to_normalized_vec(worst, param_keys, min_max_dict)

    if any(p is None for p in best_params) or any(p is None for p in worst_params):
        raise ValueError("best/worst に param_keys が存在しません。")
    best_val = float(best.get(refernce_key, 1.0))
    worst_val = float(worst.get(refernce_key, 1.0))
    #パラメータの事前計算
    if method_num == 0:
        #距離補間用のパラメータ計算
        max_dist = euclidean(best_params, worst_params)
    elif method_num ==1 or (method_num == 4 and gen > int(NUM_GENERATIONS/2)):
        #ガウス補間用のパラメータ計算
        eps = (best_val - worst_val) * EPS_RATIO
        C = worst_val - eps
        A = best_val - C
        # ratio for sigma estimation（数値安定化）
        denom = (best_val - C)
        if denom == 0:
            raise ValueError(f"best の target が C と等しい。sigma を推定できません。\nbest: {best_val}, C: {C}")
        ratio = (worst_val - C) / denom
        if not (0 < ratio < 1):
            raise ValueError(f"ratio が (0,1) の範囲にない。best/worst の target を確認してください。\nratio: {ratio}")
        sigma = get_sigma(best_params=best_params, worst_params=worst_params,ratio=ratio)
    elif method_num ==2:
        # RBF補間用の学習データの計算
        train_X = []
        train_Y = []
        for individual in evaluated_population:
            train_X.append(to_normalized_vec(individual, param_keys=param_keys, min_max_dict=min_max_dict))
            train_Y.append(float(individual.get(refernce_key, 0.0)))
        # print(f"学習データの次元数: {np.shape(np.array(train_X))}, ラベル数: {len(train_Y)}")
        interpolator = RBFInterpolator(np.array(train_X), np.array(train_Y), kernel='linear',smoothing=0.1)
        

    # print(f"best_val: {best_val}, worst_val: {worst_val}")
    for ind in population:
        target_vec = to_normalized_vec(ind, param_keys, min_max_dict)
        if method_num == 0:
            ind[target_key] = calculate_by_distance(
                target_vec=target_vec,
                target_key=target_key,
                max_dist=max_dist,
                best_val=best_val,
                worst_val=worst_val,
                best_params=best_params,
            )
        elif method_num == 1:
            ind[target_key] = calculate_by_Gaussian(
                target_vec=target_vec,
                best_params=best_params,
                C=C,
                A=A,
                # sigma=[200,200,200,200,200,200],
                sigma=sigma,
            )
        elif method_num == 2:
            ind[target_key] = calculate_by_RBF(
                target_vec=target_vec,
                interpolater=interpolator,
            )
        elif method_num == 3:
            ind[target_key] = calculate_by_IDW(
                target_vec=target_vec,
                norm_eval_data=norm_eval_data
            )
        elif method_num == 4:
            if gen <= int(NUM_GENERATIONS/2):
                ind[target_key] = calculate_by_IDW(
                    target_vec=target_vec,
                    norm_eval_data=norm_eval_data
                )
            else:
                ind[target_key] = calculate_by_Gaussian(
                    individual=ind,
                    best_params=best_params,
                    C=C,
                    A=A,
                    # sigma=[200,200,200,200,200,200],
                    sigma=sigma,
                )
    return

def calculate_by_distance(
    target_vec: List[float] = None,
    max_dist: float = 0.0,
    best_val: float = 1.0,
    worst_val: float = 1.0,
    best_params: List[float] = None,
) -> float:
    if max_dist == 0:
        # 全員同じ場合
        print(f"距離が同一です。\nbest {best_val}, worst {worst_val}")        
        return best_val
    dist_best = euclidean(target_vec, best_params)
    ratio = dist_best / max_dist
    value = best_val * (1 - ratio) + worst_val * ratio
    return value
    
 
def get_sigma(
    best_params: List[float] = None,
    worst_params: List[float] = None,
    ratio: float = 0.5,
    eps_floor_ratio: float = 1e-6,
 ) -> float:
    sigma = []
    for b, w in zip(best_params, worst_params):
        raw = abs(b - w)
        # if raw == 0, we still need a meaningful scale; use relative floor
        floor = max(abs(b), abs(w), 1.0) * eps_floor_ratio
        raw = max(raw, floor)
        s = raw / math.sqrt(-2.0 * math.log(ratio))
        s = max(s, floor)  # avoid exact zero
        sigma.append(s)
    return sigma


def calculate_by_Gaussian(
    target_vec:List[float] = None,
    best_params: List[float] = None,
    C: float = 0.0,
    A: float = 1.0,
    sigma: List[float] = None,
) -> float:
    mu = best_params
    # compute normalized squared distance
    dist_sq = 0.0
    for xj, muj, sj in zip(target_vec, mu, sigma):
        z = (xj - muj) / sj
        dist_sq += z * z
    # note: DO NOT divide by N; the scale is already handled by sigma
    value = A * math.exp(-0.5 * dist_sq) + C
    return value


def calculate_by_RBF(
    target_vec: List[float] = None,
    interpolater = None,
) -> float:
    x_vec = np.array(target_vec)
    est_value = interpolater(x_vec)[0]
    return float(est_value)


def calculate_by_IDW(
    target_vec: List[float],
    norm_eval_data: List[Tuple[List[float], float]],
    p: float = 2.0
) -> float:
    """
    正規化されたベクトルを用いてIDW（逆距離加重）補間を行う。
    target_vec: ターゲット個体の正規化ベクトル
    norm_eval_data: [(正規化ベクトル, fitness), ...] のリスト
    p: 距離の重み係数 (通常 2.0)
    """
    weights_sum = 0.0
    weighted_val_sum = 0.0
    
    for ref_vec, ref_val in norm_eval_data:
        # ユークリッド距離計算
        dist = math.sqrt(sum((t - r) ** 2 for t, r in zip(target_vec, ref_vec)))
        
        # 距離が極めて近い（同じ個体）場合は、その値をそのまま採用
        if dist < 1e-6:
            return ref_val
        
        # 重み計算
        weight = 1.0 / (dist ** p)
        
        weights_sum += weight
        weighted_val_sum += weight * ref_val
    
    if weights_sum == 0:
        return 0.0
        
    return weighted_val_sum / weights_sum


def get_evaluated_individuals(
        population: List[dict],
        id_list: List[str]
) -> List[dict]:
    """
    UUIDリストで管理された評価済み個体を抽出する。
    """
    evaluated_individuals = []
    if id_list is None or len(id_list) == 0:
        return evaluated_individuals
    for individual in population:
        if "chromosomeId" not in individual:
            continue
        # UUID型にも対応
        try:
            if str(individual["chromosomeId"]) in [str(i) for i in id_list]:
                evaluated_individuals.append(individual)
        except Exception:
            continue
    return evaluated_individuals

def get_param(ind, key):
        val = ind
        for k in key.split('.'):
            val = val.get(k, None)
            if val is None:
                break
        return val

def to_vec(ind,param_keys):
    vec = []
    for k in param_keys:
        val = get_param(ind, k)
        try:
            # None または非数値の場合は 0.0 を使用
            vec.append(float(val) if val is not None else 0.0)
        except ValueError:
            vec.append(0.0)
    return vec

# 距離計算
def euclidean(vec1, vec2):
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(vec1, vec2) if a is not None and b is not None))

def get_min_max_dict(all_inds: List[dict], param_keys: List[str]) -> Dict[str, Tuple[float, float]]:
    """全個体から各パラメータの最小・最大値を取得して辞書で返す"""
    min_max_dict = {}
    for key in param_keys:
        values = []
        for ind in all_inds:
            val = get_param(ind, key)
            if val is not None:
                values.append(float(val))
        
        if not values:
            min_max_dict[key] = (0.0, 1.0) # ダミー
        else:
            min_v, max_v = min(values), max(values)
            if max_v == min_v:
                max_v += 1.0 # ゼロ除算防止
            min_max_dict[key] = (min_v, max_v)
    return min_max_dict

def to_normalized_vec(
    ind: dict, 
    param_keys: List[str], 
    min_max_dict: Dict[str, Tuple[float, float]]
) -> List[float]:
    """個体のパラメータを0.0-1.0に正規化したベクトルを返す"""
    vec = []
    for key in param_keys:
        val = get_param(ind, key)
        if val is None:
            vec.append(0.0)
            continue
        
        val = float(val)
        min_v, max_v = min_max_dict[key]
        
        # 正規化: (x - min) / (max - min)
        norm_val = (val - min_v) / (max_v - min_v)
        vec.append(norm_val)
    return vec

def get_total_error(
    population: List[dict],
    target_param: str = "fitness",
    param_keys: List[str] = PARAMS,
    evaluate_num: int = None,
):
    total_error = 0.0
    if evaluate_num is None:
        raise ValueError("evaluate_num が指定されていません。")
    elif evaluate_num == 1:
        for ind in population:
            true_val = float(evaluate.calculate_Gaussian(ind, param_keys))
            error = np.abs(true_val - ind[target_param])
            total_error += error**2
    elif evaluate_num == 2:
        for ind in population:
            true_val = float(evaluate.calculate_Sphere(ind, param_keys))
            error = np.abs(true_val - ind[target_param])
            total_error += error**2
    elif evaluate_num == 3:
        for ind in population:
            true_val = float(evaluate.calculate_Gaussian_cos(ind, param_keys))
            error = np.abs(true_val - ind[target_param])
            total_error += error**2
    elif evaluate_num == 4:
        for ind in population:
            true_val = float(evaluate.calculate_Ackley(ind, param_keys))
            error = np.abs(true_val - ind[target_param])
            total_error += error**2
    elif evaluate_num == 5:
        for ind in population:
            true_val = float(evaluate.calculate_Gaussian_two_peak(ind, param_keys))
            error = np.abs(true_val - ind[target_param])
            total_error += error**2
    return total_error

if __name__ == "__main__":
    experimental_selected_population = [make_chromosome_params() for _ in range(9)]
    evaluate_fitness(experimental_selected_population)
    best, worst = get_best_and_worst_individuals(experimental_selected_population)
    experimental_population = [make_chromosome_params() for _ in range(200)]
    interpolation(
        population=experimental_population,
        evaluated_population=experimental_selected_population,
        method_num=3,
        best=best,
        worst=worst,
        param_keys=PARAMS,
        target_key="pre_evaluation",
    )
    evaluate_fitness(experimental_population)
    log("tests/interpolation_test.json", experimental_population)
    fig, ax = plt.subplots()
    pre_eval = [ind["pre_evaluation"] for ind in experimental_population]
    fitness = [ind["fitness"] for ind in experimental_population]
    ax.scatter(pre_eval, fitness)
    ax.set_xlabel("Pre-evaluation")
    ax.set_ylabel("Fitness")
    ax.set_title("Pre-evaluation vs Fitness after Gaussian Interpolation")
    plt.savefig("tests/interpolation_result.png")
    plt.show()
