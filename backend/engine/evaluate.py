import random
import math
import uuid
import numpy as np
from scipy.stats import norm
from typing import List
from ..core.geneticAlgorithm.config import PARAMS, TARGET_PARAMS, TARGET_PARAMS_1, TARGET_PARAMS_2, SIGMA, RATE

def add_noise(value: float, noise_sigma: float = 1.0, noise_mean = 0, scale = 1.0) -> float:
    # 平均0、標準偏差noise_sigmaの正規分布ノイズを加算
    rng = np.random.default_rng(seed=50)
    val = rng.normal(loc=noise_mean, scale=noise_sigma)
    val = norm.cdf(val,loc=noise_mean,scale=noise_sigma) * scale  # 0〜1に正規化してscaleをかける
    return value + val

# 完全ランダムな評価（1〜10）
def evaluate_fitness_random(population: List[dict], noise_is_added: bool = False):
    for individual in population:
        if not isinstance(individual, dict):
            print("警告: individualがdict型ではありません:", individual)
            continue  # またはraise Exceptionで止めてもOK
        fitness = random.randint(1, 10)
        if noise_is_added:
            fitness = add_noise(value=fitness,scale=1.0)
        individual["fitness"] = fitness

def evaluate_fitness(
    population: List[dict],
    evaluate_num: int = 1,
    param_keys: List[str] = PARAMS,
    noise_is_added: bool = False,
    target_key: str = "fitness"
    ):
    """"
    評価関数の振り分け
    """
    if param_keys is None:
        param_keys = ["fmParamsList.operator1.frequency"]
    for ind in population:
        if not isinstance(ind, dict):
            print("警告: individualがdict型ではありません:", ind)
            continue
        if evaluate_num == 1:
            ind[target_key] = calculate_Gaussian(individual=ind, param_keys=param_keys, target_params=TARGET_PARAMS, noise_is_added=noise_is_added)
        elif evaluate_num == 2:
            ind[target_key] = calculate_Sphere(individual=ind, param_keys=param_keys, target_params=TARGET_PARAMS, noise_is_added=noise_is_added)
        elif evaluate_num == 3:
            ind[target_key] = calculate_Gaussian_cos(individual=ind, param_keys=param_keys, target_params=TARGET_PARAMS, noise_is_added=noise_is_added)
        elif evaluate_num == 4:
            ind[target_key] = calculate_Ackley(individual=ind, param_keys=param_keys, target_params=TARGET_PARAMS, noise_is_added=noise_is_added)
        elif evaluate_num == 5:
            ind[target_key] = calculate_Gaussian_peaks(individual=ind, param_keys=param_keys, target_params=[TARGET_PARAMS,TARGET_PARAMS_1,TARGET_PARAMS_2], noise_is_added=noise_is_added)
        elif evaluate_num == 6:
            ind[target_key] = calculate_mixed(individual=ind, param_keys=param_keys, target_params=TARGET_PARAMS, noise_is_added=noise_is_added)
    return None

def calculate_Gaussian(
        individual: dict = None,
        param_keys: List[str] = None,
        target_params: List[float] = TARGET_PARAMS,
        noise_is_added: bool = False,
        sigma: float = 75.0
    ):
    scores = []
    for key, target in zip(param_keys, target_params):
        # ドット区切りでアクセス
        val = individual
        for k in key.split('.'):
            val = val.get(k, None)
            if val is None:
                break
        if val is None:
            scores.append(0)
        else:
            # 正規分布の確率密度関数（最大値1）
            score = compute_Gaussian(val=val, target=target, sigma=sigma)
            scores.append(score)

    # 統合
    total_score = sum(scores)  if scores else 0
    if noise_is_added:
        total_score = add_noise(value=total_score, scale=1.0)  # ノイズを加えて

    # total_score = total_score * 10  # 0～10にスケール
    # total_score = int(round(total_score))  # 0～10の整数に丸める
    # total_score = (max(0, min(10, total_score)))  # 範囲外は補正
    return total_score
def compute_Gaussian(
    val: float,
    target: float,
    sigma: float = 75.0
):
    return np.exp(-((float(val) - target) ** 2) / (2 * sigma ** 2))

def calculate_Sphere(
    individual: dict = None,
    param_keys: List[str] = None,
    target_params: List[float] = TARGET_PARAMS,
    noise_is_added: bool = False
):
    scores = []
    for key, target in zip(param_keys, target_params):
        # ドット区切りでアクセス
        val = individual
        for k in key.split('.'):
            val = val.get(k, None)
            if val is None:
                break
        if val is None:
            scores.append(0)
        else:
            # 正規分布の確率密度関数（最大値1）
            score = compute_Sphere(val=val, target=target)
            scores.append(score)

    # 統合
    total_score = sum(scores)  if scores else 0
    if noise_is_added:
        total_score = add_noise(value=total_score, scale=1.0)  # ノイズを加えて

    # total_score = total_score * 10  # 0～10にスケール
    # total_score = int(round(total_score))  # 0～10の整数に丸める
    # individual["fitness"] = (max(0, min(10, total_score)))  # 範囲外は補正
    return total_score

def compute_Sphere(
    val: float,
    target: float,
):
    return -1 * (float(val) - target) ** 2
    
def calculate_Gaussian_cos(
    individual: dict,
    param_keys: List[str] = None,
    target_params: List[float] = TARGET_PARAMS,
    noise_is_added: bool = False,
    sigma: float = 75.0,
    frequency: float = 0.02,
):
    """
    param_keysで指定した各パラメータがtarget_paramsの値に近いほど高評価（正規分布に基づく）
    id_listが指定された場合は、そのID（chromosomeId）を持つ個体のみfitnessを付与
    統合手法は平均値
    """
    scores = []
    for key, target in zip(param_keys, target_params):
        # ドット区切りでアクセス
        val = individual
        for k in key.split('.'):
            val = val.get(k, None)
            if val is None:
                break
        if val is None:
            scores.append(0)
        else:
            # 正規分布の確率密度関数（最大値1）
            score = compute_Gaussian_cos(val=val, target=target, sigma=sigma, frequency=frequency)
            scores.append(score)

    # 統合
    total_score = sum(scores)  if scores else 0
    if noise_is_added:
        total_score = add_noise(value=total_score, scale=1.0)  # ノイズを加えて

    # total_score = total_score * 10  # 0～10にスケール
    # total_score = int(round(total_score))  # 0～10の整数に丸める
    # individual["fitness"] = (max(0, min(10, total_score)))  # 範囲外は補正
    return total_score + 0.1

def compute_Gaussian_cos(
    val: float,
    target: float,
    sigma: float = 75.0,
    frequency: float = 0.02,
):
    return np.exp(-((float(val) - target) ** 2) / (2 * sigma ** 2)) + 0.1 * np.cos(2 * np.pi * frequency * float(val)) +0.1
    
def calculate_Ackley(
        individual: dict,
        param_keys: List[str] = None,
        target_params: List[float] = TARGET_PARAMS,
        noise_is_added: bool = False,
        A = 20,
        B = 0.04,
        C = 0.04
):
    values = []
    for key in param_keys:
        val = individual
        for k in key.split('.'):
            val = val.get(k, None)
            if val is None:
                break
        if val is None:
            values.append(0)
        else:
            values.append(float(val))

    # 統合
    fitness = compute_Ackley(values=values, target_params=target_params, A=A, B=B, C=C, n = 6.0)
    # 必要に応じてスケーリングやノイズ付与も可能
    # max = A + np.e
    # fitness = 6.0 * (1.0 - fitness / max)
    if noise_is_added:
        fitness = add_noise(value=fitness, scale=1.0)
    return fitness

# def compute_Ackley(
#     values: List[float],
#     target_params: List[float],
#     A = 20,
#     B = 0.04,
#     C = 0.04
# ):
#     return -A * np.exp(-B * np.sqrt(sum((values[i] - target_params[i])**2 for i in range(len(target_params)))/len(values))) - np.exp(sum(np.cos(C*(values[i] - target_params[i])) for i in range(len(target_params)))/len(values)) + A + np.e

def compute_Ackley(
    values: List[float],
    target_params: List[float],
    A = 20,
    B = 0.04,
    C = 0.04,
    n = 1.0
):
    # Ackley関数の生の値を計算 (0に近いほど最適)
    raw_val = -A * np.exp(-B * np.sqrt(sum((values[i] - target_params[i])**2 for i in range(len(target_params)))/len(values))) - np.exp(sum(np.cos(C*(values[i] - target_params[i])) for i in range(len(target_params)))/len(values)) + A + np.e

    # calculate_Ackleyと同様の計算方法で適応度に変換
    # (最大値(A+e)で正規化し反転させ、6.0倍する)
    max_val = A + np.e
    fitness = n * (1.0 - raw_val / max_val)

    return fitness

def calculate_Gaussian_peaks(
    individual: dict,
    param_keys: List[str] = None,
    target_params: list[list] = [TARGET_PARAMS,TARGET_PARAMS_1,TARGET_PARAMS_2],
    sigmas:list = SIGMA,
    rates:list = RATE,
    noise_is_added: bool = False,
):
    scores = []
    for i,key in enumerate(param_keys):
        # ドット区切りでアクセス
        val = individual
        for k in key.split('.'):
            val = val.get(k, None)
            if val is None:
                break
        if val is None:
            scores.append(0)
        else:
            # 正規分布の確率密度関数（最大値1）
            dim_score = 0.0
            for j in range(len(rates)):
                mu = target_params[j][i]
                sigma = sigmas[j]
                rate = rates[j]
                dim_score += rate * np.exp(-((val - mu) ** 2) / (2 * sigma **2))
            scores.append(dim_score)

    # 統合
    total_score = sum(scores)  if scores else 0
    if noise_is_added:
        total_score = add_noise(value=total_score, scale=1.0)  # ノイズを加えて

    # total_score = total_score * 10  # 0～10にスケール
    # total_score = int(round(total_score))  # 0～10の整数に丸める
    # individual["fitness"] = (max(0, min(10, total_score)))  # 範囲外は補正
    return total_score

def compute_Gaussian_peaks(
    val:float,
    target_params:list[list] = [TARGET_PARAMS,TARGET_PARAMS_1,TARGET_PARAMS_2],
    sigmas:list = SIGMA,
    rates:list = RATE,
):
    score = 0.0
    for j in range(len(rates)):
        mu = target_params[j][0]
        sigma = sigmas[j]
        rate = rates[j]
        score += rate * np.exp(-((val - mu) ** 2) / (2 * sigma **2))
    return score

def calculate_mixed(
    individual: dict,
    param_keys: List[str] = None,
    target_params: list[list] = TARGET_PARAMS,
    sigmas:list = SIGMA,
    rates:list = RATE,
    noise_is_added: bool = False,
):
    
    scores = []
    for i,key in enumerate(param_keys):
        # ドット区切りでアクセス
        val = individual
        for k in key.split('.'):
            val = val.get(k, None)
            if val is None:
                break
        if val is None:
            scores.append(0)
        else:
            # 正規分布の確率密度関数（最大値1）
            dim_score = 0.0
            if i == 0:
                dim_score += compute_Gaussian(val=val, target=float(target_params[0]))
            elif i == 1:
                dim_score += compute_Gaussian(val=val, target=float(target_params[0]))
            elif i == 2:
                dim_score += compute_Gaussian_cos(val=val, target=float(target_params[0]))
            elif i == 3:
                dim_score += compute_Gaussian_peaks(val=val)
            elif i == 4:
                dim_score += compute_Ackley(values=[val], target_params=[target_params[0]])
            elif i == 5:
                dim_score += compute_Ackley(values=[val], target_params=[target_params[0]])
            scores.append(dim_score)

    # 統合
    total_score = sum(scores)  if scores else 0
    if noise_is_added:
        total_score = add_noise(value=total_score, scale=1.0)  # ノイズを加えて

    # total_score = total_score * 10  # 0～10にスケール
    # total_score = int(round(total_score))  # 0～10の整数に丸める
    # individual["fitness"] = (max(0, min(10, total_score)))  # 範囲外は補正
    return total_score

# 最も適応度の高い個体と最も低い個体を取得
def get_best_and_worst_individuals(population: List[dict]):
    # fitnessが未設定の場合は除外
    valid_population = [ind for ind in population if "fitness" in ind]
    if not valid_population:
        return None, None
    # fitnessはstrなのでfloatに変換して比較
    best = max(valid_population, key=lambda x: float(x["fitness"]))
    worst = min(valid_population, key=lambda x: float(x["fitness"]))
    return best, worst

def proposal_evaluate_random(id_list: List[str], population: List[dict]):
    """
    id_listに含まれるID（chromosomeId）を持つ個体のみ、fitnessにランダムな値（1〜10）を与える
    """
    for individual in population:
        if not isinstance(individual, dict):
            continue
        if "chromosomeId" in individual and individual["chromosomeId"] in id_list:
            individual["fitness"] = (round(add_noise(random.randint(1, 10))))

def get_best_and_worst_individuals_by_id(evaluated_population: List[dict]):
    """
    id_listに含まれるchromosomeIdを持つ個体群から、最もfitnessが高い個体と低い個体を返す
    fitnessが小数値でも取得できるようにfloat変換で判定
    """
    if not evaluated_population:
        return None, None
    best = max(evaluated_population, key=lambda x: float(x["fitness"]))
    worst = min(evaluated_population, key=lambda x: float(x["fitness"]))
    return best, worst

def get_average_fitness(population: List[dict], evaluate_population: List[dict] = None) -> float:
    """
    population内のfitnessの平均値を返す（fitnessが未設定・不正な個体は除外）
    id_listが指定された場合は、そのID（chromosomeId）を持つ個体のみ対象
    """
    target_ids = None
    if evaluate_population is not None:
        target_ids = {str(ind["chromosomeId"]) for ind in evaluate_population if "chromosomeId" in ind}
   
    valid_fitness_values = []
   
    for ind in population:
        if "fitness" not in ind or ind["fitness"] is None:
            continue

        if target_ids is not None:
            if "chromosomeId" not in ind or str(ind["chromosomeId"]) not in target_ids:
                continue

        try:
            val = float(ind["fitness"])
            valid_fitness_values.append(val)
        except (ValueError, TypeError):
            continue

    if not valid_fitness_values:
        return 0.0
    return sum(valid_fitness_values) / len(valid_fitness_values)