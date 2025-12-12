#事前評価・適応度評価を行う

from typing import List, Dict
import math
from scipy.interpolate import RBFInterpolator
import numpy as np
from ..geneticAlgorithm.config import FITNESS_KEY, PARAMS
from ...engine import evaluate

RNG = np.random.default_rng(seed=10)
EPS_RATIO = 0.02  # ガウス補間のeps計算用割合

def interpolation(
  population: List[dict] = None,
  evaluated_population: List[dict] = None,
  best: dict = None,
  worst: dict = None,
  method_num: int = 0,
  param_keys: List[str] = PARAMS,
  fitness_key: List[str] = FITNESS_KEY,
  target_key: str = "pre_evaluation",      
):
    print(f"補間を開始します。")
    if not best or not worst:
        print(f"bestまたはworstがNoneです。ランダムな{target_key}を付与します。")
        for ind in population:
            ind[target_key] = RNG.uniform(1.0, 10.0)
        return
    
    if param_keys is None:
        # デフォルトはoperator1のfrequencyのみ
        param_keys = ["fmParamsList.operator1.frequency"]

    best_params = [get_param(best, k) for k in param_keys]
    worst_params = [get_param(worst, k) for k in param_keys]

    if any(p is None for p in best_params) or any(p is None for p in worst_params):
        raise ValueError("best/worst に param_keys が存在しません。")
    best_val = float(best.get(fitness_key[0], 1.0))
    worst_val = float(worst.get(fitness_key[0], 1.0))
    #パラメータの事前計算
    if method_num == 0:
        #距離補間用のパラメータ計算
        max_dist = euclidean(best_params, worst_params)
    elif method_num ==1:
        #ガウス補間用のパラメータ計算
        eps = (best_val - worst_val) * EPS_RATIO
        C = worst_val - eps
        A = best_val - C
        # ratio for sigma estimation（数値安定化）
        denom = (best_val - C)
        if denom == 0:
            raise ValueError("best の target が C と等しい。sigma を推定できません。")
        ratio = (worst_val - C) / denom
        if not (0 < ratio < 1):
            raise ValueError("ratio が (0,1) の範囲にない。best/worst の target を確認してください。")
        sigma = get_sigma(best_params=best_params, worst_params=worst_params,ratio=ratio)
    if method_num ==2:
        # RBF補間用の学習データの計算
        train_X = []
        train_Y = []
        for individual in evaluated_population:
            train_X.append(to_vec(individual, param_keys=param_keys))
            train_Y.append(float(individual.get(target_key, 0.0)))
        print(f"学習データの次元数: {np.shape(np.array(train_X))}, ラベル数: {len(train_Y)}")
        interpolator = RBFInterpolator(np.array(train_X), np.array(train_Y), kernel='gaussian', epsilon=1.5)
        
    for ind in population:
        if method_num == 0:
            ind[target_key] = calculate_by_distance(
                individual=ind,
                param_keys=param_keys,
                target_key=target_key,
                max_dist=max_dist,
                best_val=best_val,
                worst_val=worst_val,
                best_params=best_params,
                sigma=sigma,
            )
        elif method_num == 1:
            ind[target_key] = calculate_by_Gaussian(
            individual=ind,
            param_keys=param_keys,
            target_key=target_key,
            best_params=best_params,
            best_val=best_val,
            worst_val=worst_val,
            C=C,
            A=A,
            sigma=sigma,
            )
        elif method_num == 2:
            ind[target_key] = calculate_by_RBF(
                individual = ind,
                evaluated_population=evaluated_population,
                param_keys=param_keys,
                interpolater=interpolator,
            )
    
    return

def calculate_by_distance(
    individual:dict= None,
    param_keys: List[str] = None,
    target_key: str = "pre_evaluation",
    max_dist: float = 0.0,
    best_val: float = 1.0,
    worst_val: float = 1.0,
    best_params: List[float] = None,
):
    if max_dist == 0:
        # 全員同じ場合
        value = best_val
        individual[target_key] = float(value)
        print(f"距離が同一です。\nbest {best_val}, worst {worst_val}")        
        return
    ind_params = to_vec(individual, param_keys=param_keys)
    dist_best = euclidean(ind_params, best_params)
    ratio = dist_best / max_dist
    value = best_val * (1 - ratio) + worst_val * ratio
    return value
    
 
def get_sigma(
    best_params: List[float] = None,
    worst_params: List[float] = None,
    ratio: float = 0.5,
    eps_floor_ratio: float = 1e-6,
 ):
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
    individual:dict= None,
    param_keys: List[str] = None,
    target_key: str = "pre_evaluation",
    best_params: List[float] = None,
    best_val: float = 1.0,
    worst_val: float = 1.0,
    C: float = 0.0,
    A: float = 1.0,
    sigma: List[float] = None,
):
    print(f"best_val: {best_val}, worst_val: {worst_val}")
    mu = best_params
    ind_params = [get_param(individual, k) for k in param_keys]
    if any(p is None for p in ind_params):
        individual[target_key] = RNG.uniform(1.0, 10.0)
        return
    # compute normalized squared distance
    dist_sq = 0.0
    for xj, muj, sj in zip(ind_params, mu, sigma):
        z = (xj - muj) / sj
        dist_sq += z * z
    # note: DO NOT divide by N; the scale is already handled by sigma
    value = A * math.exp(-0.5 * dist_sq) + C
    return value


def calculate_by_RBF(
    individual:dict= None,
    evaluated_population: List[dict] = None,
    param_keys: List[str] = None,
    interpolater = None,
):
    if individual in evaluated_population:
        return
    x_vec = np.array([to_vec(individual, param_keys=param_keys)])
    est_value = interpolater(x_vec)[0]
    return float(est_value)


def interpolate_by_distance(
    population: List[dict],
    best: dict = None,
    worst: dict = None,
    param_keys: List[str] = None,
    target_key: str = "pre_evaluation"
):
    """
    best, worstを基準に、個体群のtarget_key（pre_evaluationやfitnessなど）を距離に応じて線形補間で再評価する。
    best/worstがNoneの場合はランダムな値を付与する。
    param_keys: 距離計算に使うパラメータ名リスト（Noneならoperator1のfrequencyのみ）
    target_key: 補間して格納するキー名（"pre_evaluation"または"fitness"など）
    """
    print(f"補間を開始します。")
    if not best or not worst:
        print(f"bestまたはworstがNoneです。ランダムな{target_key}を付与します。")
        for ind in population:
            ind[target_key] = RNG.uniform(1.0, 10.0)
        return

    if param_keys is None:
        # デフォルトはoperator1のfrequencyのみ
        param_keys = ["fmParamsList.operator1.frequency"]

    best_params = to_vec(best, param_keys=param_keys)
    worst_params = to_vec(worst, param_keys=param_keys)

    max_dist = euclidean(best_params, worst_params)
    if max_dist == 0:
        # 全員同じ場合
        for ind in population:
            value = best.get(target_key, best.get("fitness", 1))
            ind[target_key] = float(value)
        print(f"距離が同一です。\nbest {best[target_key]}, worst {worst[target_key]}")
        return

    best_val = float(best.get("fitness", 1))
    worst_val = float(worst.get("fitness", 1))

    for ind in population:
        ind_vec = to_vec(ind, param_keys=param_keys)
        dist_best = euclidean(ind_vec, best_params)
        # 距離が近いほどbest_valに近づく
        ratio = dist_best / max_dist
        value = best_val * (1 - ratio) + worst_val * ratio
        ind[target_key] = value
    print(f"{target_key}を補間しました。\nbest {best_val}, worst {worst_val}")
    return


def interpolate_by_Gaussian(
    population: List[dict],
    best: dict = None,
    worst: dict = None,
    param_keys: List[str] = None,
    target_key: str = "pre_evaluation",
    eps_ratio: float = 0.02,
    eps_floor_ratio: float = 1e-6
):
    """
    ガウス関数 f(x) = A exp(-(x-μ)^2/(2σ^2)) + C のパラメータ推定
    C ≈ y2 として、わずかに ε を下げた近似で計算。
    
    eps_ratio: 全振幅に対する補正割合（例: 0.02 = 2%）
    """
    print(f"ガウス補間を開始します。")
    if not best or not worst:
        print(f"bestまたはworstがNoneです。ランダムな{target_key}を付与します。")
        for ind in population:
            ind[target_key] = RNG.uniform(1.0, 10.0)
        return
    if param_keys is None:
        # デフォルトはoperator1のfrequencyのみ
        param_keys = ["fmParamsList.operator1.frequency"]

    best_params = [get_param(best, k) for k in param_keys]
    worst_params = [get_param(worst, k) for k in param_keys]

    if any(p is None for p in best_params) or any(p is None for p in worst_params):
        raise ValueError("best/worst に param_keys が存在しません。")
    
    best_val = float(best.get("fitness", 1.0))
    worst_val = float(worst.get("fitness", 1.0))
    eps = (best_val - worst_val) * eps_ratio
    C = worst_val - eps
    A = best_val - C
    print(f"best_val: {best_val}, worst_val: {worst_val}")
    print(best[target_key] - C)

    # ratio for sigma estimation（数値安定化）
    denom = (best.get("fitness", 1e-12) - C)
    if denom == 0:
        raise ValueError("best の target が C と等しい。sigma を推定できません。")
    ratio = (worst.get("fitness", 1e-12) - C) / denom
    if not (0 < ratio < 1):
        raise ValueError("ratio が (0,1) の範囲にない。best/worst の target を確認してください。")

    sigma = []
    for b, w in zip(best_params, worst_params):
        raw = abs(b - w)
        # if raw == 0, we still need a meaningful scale; use relative floor
        floor = max(abs(b), abs(w), 1.0) * eps_floor_ratio
        raw = max(raw, floor)
        s = raw / math.sqrt(-2.0 * math.log(ratio))
        s = max(s, floor)  # avoid exact zero
        sigma.append(s)
    mu = best_params
    print(f"ガウス補間のパラメータ: \n\tA={A}, \n\tmu={mu}, \n\tsigma={sigma}, \n\tC={C}")

    # 個体群のガウス補間
    for ind in population:
        ind_params = [get_param(ind, k) for k in param_keys]
        if any(p is None for p in ind_params):
            ind[target_key] = RNG.uniform(1.0, 10.0)
            continue
        # compute normalized squared distance
        dist_sq = 0.0
        for xj, muj, sj in zip(ind_params, mu, sigma):
            z = (xj - muj) / sj
            dist_sq += z * z
        # note: DO NOT divide by N; the scale is already handled by sigma
        value = A * math.exp(-0.5 * dist_sq) + C
        ind[target_key] = value
    print(f"{target_key}をガウス補間しました。\nbest {best_val}, worst {worst_val}")
    return


def interpolate_by_RBF(
    population: List[dict],        # {UUID: 個体オブジェクト} の辞書
    evaluated_ind: List[dict] = None,      # 評価済み個体のリスト
    param_keys: List[str] = None,
    target_key: str = "pre_evaluation",
):
    """
    UUIDリストで管理された評価済み個体の fitness を用いて、
    pre_evaluation をRBF補間する。
    カーネルはGaussianを使用する。
    """
    train_X = []
    train_Y = []
    print(f"RBF補間を開始します。")
    if evaluated_ind is None or len(evaluated_ind) == 0:
        print(f"事前評価がNoneです。ランダムな{target_key}を付与します。")
        for ind in population:
            ind[target_key] = RNG.uniform(1.0, 10.0)
        return
    if param_keys is None:
        # デフォルトはoperator1のfrequencyのみ
        param_keys = ["fmParamsList.operator1.frequency"]
        
    for individual in evaluated_ind:
        train_X.append(to_vec(individual, param_keys=param_keys))
        train_Y.append(float(individual.get(target_key, 0.0)))
    print(f"学習データの次元数: {np.shape(np.array(train_X))}, ラベル数: {len(train_Y)}")
    interpolator = RBFInterpolator(np.array(train_X), np.array(train_Y), kernel='gaussian', epsilon=1.5)

    for individual in population:
        if individual in evaluated_ind:
            continue
        x_vec = np.array([to_vec(individual)])
        est_value = interpolator(x_vec)[0]
        individual[target_key] = float(est_value)

    return

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