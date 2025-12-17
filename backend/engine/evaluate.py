import random
import math
import uuid
import numpy as np
from scipy.stats import norm
from typing import List
from ..core.geneticAlgorithm.config import PARAMS, TARGET_PARAMS, TARGET_PARAMS_1, TARGET_PARAMS_2

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
            ind[target_key] = calculate_Gaussian_two_peak(individual=ind, param_keys=param_keys, target_params=TARGET_PARAMS_1, target_params_2=TARGET_PARAMS_2, noise_is_added=noise_is_added)
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
            score = np.exp(-((float(val) - target) ** 2) / (2 * sigma ** 2))
            # score = np.exp(-(float(val) ** 2) / (2 * (sigma ** 2)))
            scores.append(score)

    # 統合
    total_score = sum(scores)  if scores else 0
    if noise_is_added:
        total_score = add_noise(value=total_score, scale=1.0)  # ノイズを加えて

    # total_score = total_score * 10  # 0～10にスケール
    # total_score = int(round(total_score))  # 0～10の整数に丸める
    # total_score = (max(0, min(10, total_score)))  # 範囲外は補正
    return total_score

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
            score = -1 * (float(val) - target) ** 2
            # score = np.exp(-(float(val) ** 2) / (2 * (sigma ** 2)))
            scores.append(score)

    # 統合
    total_score = sum(scores)  if scores else 0
    if noise_is_added:
        total_score = add_noise(value=total_score, scale=1.0)  # ノイズを加えて

    # total_score = total_score * 10  # 0～10にスケール
    # total_score = int(round(total_score))  # 0～10の整数に丸める
    # individual["fitness"] = (max(0, min(10, total_score)))  # 範囲外は補正
    return total_score
    
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
            score = np.exp(-((float(val) - target) ** 2) / (2 * sigma ** 2)) + 0.5 * np.cos(2 * np.pi * frequency * float(val))
            # score = np.exp(-(float(val) ** 2) / (2 * (sigma ** 2)))
            scores.append(score)

    # 統合
    total_score = sum(scores)  if scores else 0
    if noise_is_added:
        total_score = add_noise(value=total_score, scale=1.0)  # ノイズを加えて

    # total_score = total_score * 10  # 0～10にスケール
    # total_score = int(round(total_score))  # 0～10の整数に丸める
    # individual["fitness"] = (max(0, min(10, total_score)))  # 範囲外は補正
    return total_score
    
def calculate_Ackley(
        individual: dict,
        param_keys: List[str] = None,
        target_params: List[float] = TARGET_PARAMS,
        noise_is_added: bool = False,
        A = 600,
        B = 0.00005,
        C = 0.0625 * np.pi
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
    fitness = 0
    fitness -= -A * np.exp(-B * np.sqrt(sum((values[i] - target_params[i])**2 for i in range(len(target_params)))/len(values))) - np.exp(sum(np.cos(C*(values[i] - target_params[i])) for i in range(len(target_params)))/len(values)) + A + np.e
    # 必要に応じてスケーリングやノイズ付与も可能
    # fitness = -fitness
    fitness = fitness / 50.0
    if noise_is_added:
        fitness = add_noise(value=fitness, scale=1.0)
    return fitness

def calculate_Gaussian_two_peak(
    individual: dict,
    param_keys: List[str] = None,
    target_params: List[float] = TARGET_PARAMS_1,
    target_params_2: List[float] = TARGET_PARAMS_2,
    noise_is_added: bool = False,
    sigma: float = 30.0,
):
    scores = []
    for key, target, target_2 in zip(param_keys, target_params, target_params_2):
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
            score = np.exp(-((float(val) - target) ** 2) / (2 * sigma ** 2)) + np.exp(-((float(val) - target_2) ** 2) / (2 * sigma ** 2))
            # score = np.exp(-(float(val) ** 2) / (2 * (sigma ** 2)))
            scores.append(score)

    # 統合
    total_score = sum(scores)  if scores else 0
    if noise_is_added:
        total_score = add_noise(value=total_score, scale=1.0)  # ノイズを加えて

    # total_score = total_score * 10  # 0～10にスケール
    # total_score = int(round(total_score))  # 0～10の整数に丸める
    # individual["fitness"] = (max(0, min(10, total_score)))  # 範囲外は補正
    return total_score

# FMパラメータに基づいた評価（例：operator1のfrequencyに基づく）
def evaluate_fitness_by_param(
    population: List[dict],
    target_params: List[float] = TARGET_PARAMS,
    sigma: float = 75.0,
    param_keys: List[str] = None,
    evaluate_population: List[dict] = None,
    noise_is_added: bool = False
):
    """
    param_keysで指定した各パラメータがtarget_paramsの値に近いほど高評価（正規分布に基づく）
    id_listが指定された場合は、そのID（chromosomeId）を持つ個体のみfitnessを付与
    統合手法は平均値
    """
    if param_keys is None:
        param_keys = ["fmParamsList.operator1.frequency"]
    if evaluate_population is None:
        evaluate_population = population

    for individual in population:
        if not isinstance(individual, dict):
            print("警告: individualがdict型ではありません:", individual)
            continue
        if not individual in evaluate_population:
            continue

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
                score = np.exp(-((float(val) - target) ** 2) / (2 * sigma ** 2))
                # score = np.exp(-(float(val) ** 2) / (2 * (sigma ** 2)))
                scores.append(score)

        # 統合
        total_score = sum(scores)  if scores else 0
        if noise_is_added:
            total_score = add_noise(value=total_score, scale=1.0)  # ノイズを加えて

        # total_score = total_score * 10  # 0～10にスケール
        # total_score = int(round(total_score))  # 0～10の整数に丸める
        # individual["fitness"] = (max(0, min(10, total_score)))  # 範囲外は補正
        individual['fitness'] = total_score


def evaluate_fitness_sphere(
    population: List[dict],
    target_params: List[float] = TARGET_PARAMS,
    param_keys: List[str] = None,
    evaluate_population: List[dict] = None,
    noise_is_added: bool = False
):
    """
    各個体の二乗和を取ることで評価
    param_keys: 評価対象パラメータ名リスト（例: ["fmParamsList.operator1.frequency", ...]）
    id_list: 評価対象のchromosomeIdリスト（Noneなら全個体）
    """
    if evaluate_population is None:
        evaluate_population = population

    for individual in population:
        if not isinstance(individual, dict):
            print("警告: individualがdict型ではありません:", individual)
            continue
        if not individual in evaluate_population:
            continue

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

        # 線形結合
        fitness = 0
        for i in range(len(values)):
            fitness += -1 * (values[i] - target_params[i]) ** 2
            # fitness += -1 * values[i] ** 2
        # fitness = 10 + 3*fitness
        # 必要に応じてスケーリングやノイズ付与も可能
        if noise_is_added:
            fitness = add_noise(value=fitness, scale=1.0)
        individual['fitness'] = float(fitness)


def evaluate_fitness_cos(
        population: List[dict],
        id_list: List[str] = None,
        param_keys: List[str] = None,
        A = 10,
        noise_is_added: bool = False
):
    """
    Rastrigin関数で評価
    param_keys: 評価対象パラメータ名リスト（例: ["fmParamsList.operator1.frequency", ...]）
    id_list: 評価対象のchromosomeIdリスト（Noneなら全個体）
    """
    def should_evaluate(ind):
        if id_list is None:
            return True
        if "chromosomeId" not in ind:
            return False
        try:
            return str(ind["chromosomeId"]) in [str(i) for i in id_list]
        except Exception:
            return False

    for individual in population:
        if not isinstance(individual, dict):
            print("警告: individualがdict型ではありません:", individual)
            continue
        if not should_evaluate(individual):
            continue

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
        fitness = (8 * A - sum(0.005 * v**2 - 100 *np.cos(np.pi*v / 16) for v in values))
        # 必要に応じてスケーリングやノイズ付与も可能
        if noise_is_added:
            fitness = add_noise(value=fitness, scale=1.0)
        individual["fitness"] = fitness


def evaluate_fitness_Ackley(
        population: List[dict],
        param_keys: List[str] = None,
        target_params: List[float] = TARGET_PARAMS,
        evaluate_population: List[dict] = None,
        A = 300,
        B = 0.00005,
        C = 0.0625 * np.pi,
        noise_is_added: bool = False
):
    """
    Ackley関数で評価
    param_keys: 評価対象パラメータ名リスト（例: ["fmParamsList.operator1.frequency", ...]）
    id_list: 評価対象のchromosomeIdリスト（Noneなら全個体）
    """
    if evaluate_population is None:
        evaluate_population = population
    for individual in population:
        if not isinstance(individual, dict):
            print("警告: individualがdict型ではありません:", individual)
            continue
        if not individual in evaluate_population:
            continue

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
        fitness = 0
        fitness -= -A * np.exp(-B * np.sqrt(sum((values[i] - target_params[i])**2 for i in range(len(target_params)))/len(values))) - np.exp(sum(np.cos(C*(values[i] - target_params[i])) for i in range(len(target_params)))/len(values)) + A + np.e
        # 必要に応じてスケーリングやノイズ付与も可能
        fitness = -fitness
        fitness = fitness / 50.0
        if noise_is_added:
            fitness = add_noise(value=fitness, scale=1.0)
        individual["fitness"] = fitness


def evaluate_fitness_Schwefel(
        population: List[dict],
        param_keys: List[str] = None,
        id_list: List[str] = None,
        noise_is_added: bool = False
):
    """
    Schwefel関数で評価
    param_keys: 評価対象パラメータ名リスト（例: ["fmParamsList.operator1.frequency", ...]）
    id_list: 評価対象のchromosomeIdリスト（Noneなら全個体）
    """
    def should_evaluate(ind):
        if id_list is None:
            return True
        if "chromosomeId" not in ind:
            return False
        try:
            return str(ind["chromosomeId"]) in [str(i) for i in id_list]
        except Exception:
            return False

    for individual in population:
        if not isinstance(individual, dict):
            print("警告: individualがdict型ではありません:", individual)
            continue
        if not should_evaluate(individual):
            continue

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

        # 統合,(418.9829,)
        fitness = 0
        # fitness += 418.9829 * len(values) - sum(v * np.sin(np.sqrt(abs(v))) for v in values)
        fitness += sum(v * np.sin(np.sqrt(abs(v))) for v in values)
        # 必要に応じてスケーリングやノイズ付与も可能
        if noise_is_added:
            fitness = add_noise(value=fitness, scale=1.0)
        individual["fitness"] = fitness


def evaluate_fitness_gaussian_two_peak(
    population: List[dict],
    target_params: List[float] = TARGET_PARAMS_1,
    target_params_2: List[float] = TARGET_PARAMS_2,
    sigma: float = 30.0,
    param_keys: List[str] = None,
    evaluate_population: List[dict] = None,
    noise_is_added: bool = False
):
    """
    param_keysで指定した各パラメータがtarget_paramsの値に近いほど高評価（正規分布に基づく）
    id_listが指定された場合は、そのID（chromosomeId）を持つ個体のみfitnessを付与
    統合手法は平均値
    """
    if param_keys is None:
        param_keys = ["fmParamsList.operator1.frequency"]
    if evaluate_population is None:
        evaluate_population = population

    for individual in population:
        if not isinstance(individual, dict):
            print("警告: individualがdict型ではありません:", individual)
            continue
        if not individual in evaluate_population:
            continue

        scores = []
        for key, target, target_2 in zip(param_keys, target_params, target_params_2):
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
                score = np.exp(-((float(val) - target) ** 2) / (2 * sigma ** 2)) + 0.5 * np.exp(-((float(val) - target_2) ** 2) / (2 * sigma ** 2))
                # score = np.exp(-(float(val) ** 2) / (2 * (sigma ** 2)))
                scores.append(score)

        # 統合
        total_score = sum(scores)  if scores else 0
        if noise_is_added:
            total_score = add_noise(value=total_score, scale=1.0)  # ノイズを加えて

        # total_score = total_score * 10  # 0～10にスケール
        # total_score = int(round(total_score))  # 0～10の整数に丸める
        # individual["fitness"] = (max(0, min(10, total_score)))  # 範囲外は補正
        individual['fitness'] = total_score


def evaluate_fitness_gaussian_cos(
    population: List[dict],
    target_params: List[float] = TARGET_PARAMS,
    sigma: float = 100.0,
    frequency: float = 0.02,
    param_keys: List[str] = None,
    evaluate_population: List[dict] = None,
    noise_is_added: bool = False
):
    """
    param_keysで指定した各パラメータがtarget_paramsの値に近いほど高評価（正規分布に基づく）
    id_listが指定された場合は、そのID（chromosomeId）を持つ個体のみfitnessを付与
    統合手法は平均値
    """
    if param_keys is None:
        param_keys = ["fmParamsList.operator1.frequency"]
    if evaluate_population is None:
        evaluate_population = population

    for individual in population:
        if not isinstance(individual, dict):
            print("警告: individualがdict型ではありません:", individual)
            continue
        if not individual in evaluate_population:
            continue

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
                score = np.exp(-((float(val) - target) ** 2) / (2 * sigma ** 2)) + 0.5 * np.cos(2 * np.pi * frequency * float(val))
                # score = np.exp(-(float(val) ** 2) / (2 * (sigma ** 2)))
                scores.append(score)

        # 統合
        total_score = sum(scores)  if scores else 0
        if noise_is_added:
            total_score = add_noise(value=total_score, scale=1.0)  # ノイズを加えて

        # total_score = total_score * 10  # 0～10にスケール
        # total_score = int(round(total_score))  # 0～10の整数に丸める
        # individual["fitness"] = (max(0, min(10, total_score)))  # 範囲外は補正
        individual['fitness'] = total_score


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