import random
import math
import uuid
from typing import List

# 完全ランダムな評価（1〜10）
def evaluate_fitness_random(population: List[dict]):
    for individual in population:
        if not isinstance(individual, dict):
            print("警告: individualがdict型ではありません:", individual)
            continue  # またはraise Exceptionで止めてもOK
        individual["fitness"] = str(random.randint(1, 10))

# FMパラメータに基づいた評価（例：operator1のfrequencyに基づく）
def evaluate_fitness_by_param(
    population: List[dict],
    target_params: List[float],
    sigma: float = 100.0,
    param_keys: List[str] = None,
    method: str = "product"  # "product", "mean", "max", "min", "median"
):
    """
    param_keysで指定した各パラメータがtarget_paramsの値に近いほど高評価（正規分布に基づく）
    methodで統合手法を選択可能
    """
    if param_keys is None:
        param_keys = ["fmParamsList.operator1.frequency"]

    for individual in population:
        if not isinstance(individual, dict):
            print("警告: individualがdict型ではありません:", individual)
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
                score = math.exp(-((float(val) - target) ** 2) / (2 * sigma ** 2))
                scores.append(score)

        # 統合手法の選択
        if method == "product":
            total_score = 1.0
            for s in scores:
                total_score *= s
        elif method == "mean":
            total_score = sum(scores) / len(scores) if scores else 0
        elif method == "max":
            total_score = max(scores) if scores else 0
        elif method == "min":
            total_score = min(scores) if scores else 0
        elif method == "median":
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            if n == 0:
                total_score = 0
            elif n % 2 == 1:
                total_score = sorted_scores[n // 2]
            else:
                total_score = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
        else:
            print(f"未知のmethod: {method}。productを使用します。")
            total_score = 1.0
            for s in scores:
                total_score *= s

        normalized = int(round(total_score * 10))  # 0～10に丸める
        individual["fitness"] = str(normalized)

# 最も適応度の高い個体と最も低い個体を取得
def get_best_and_worst_individuals(population: List[dict]):
    # fitnessが未設定の場合は除外
    valid_population = [ind for ind in population if "fitness" in ind]
    if not valid_population:
        return None, None
    # fitnessはstrなのでintに変換して比較
    best = max(valid_population, key=lambda x: int(x["fitness"]))
    worst = min(valid_population, key=lambda x: int(x["fitness"]))
    return best, worst

def proposal_evaluate_random(id_list: List[str], population: List[dict]):
    """
    id_listに含まれるID（chromosomeId）を持つ個体のみ、fitnessにランダムな値（1〜10）を与える
    """
    for individual in population:
        if not isinstance(individual, dict):
            continue
        if "chromosomeId" in individual and individual["chromosomeId"] in id_list:
            individual["fitness"] = str(random.randint(1, 10))

def get_best_and_worst_individuals_by_id(id_list: List[str], population: List[dict]):
    """
    id_listに含まれるchromosomeIdを持つ個体群から、最もfitnessが高い個体と低い個体を返す
    """
    # id_listをUUID型に変換
    id_set = set()
    for cid in id_list:
        try:
            id_set.add(uuid.UUID(str(cid)))
        except Exception:
            continue

    valid_population = [
        ind for ind in population
        if "fitness" in ind and "chromosomeId" in ind
        and (
            (isinstance(ind["chromosomeId"], uuid.UUID) and ind["chromosomeId"] in id_set)
            or
            (isinstance(ind["chromosomeId"], str) and uuid.UUID(ind["chromosomeId"]) in id_set)
        )
        and str(ind["fitness"]).strip() not in ("", "None")
        and str(ind["fitness"]).replace('.', '', 1).isdigit()
    ]
    if not valid_population:
        return None, None
    best = max(valid_population, key=lambda x: float(x["fitness"]))
    worst = min(valid_population, key=lambda x: float(x["fitness"]))
    return best, worst

def get_average_fitness(population: List[dict]) -> float:
    """
    population内のfitnessの平均値を返す（fitnessが未設定・不正な個体は除外）
    """
    fitness_values = [
        float(ind["fitness"])
        for ind in population
        if "fitness" in ind
        and str(ind["fitness"]).strip() not in ("", "None")
        and str(ind["fitness"]).replace('.', '', 1).isdigit()
    ]
    if not fitness_values:
        return 0.0
    return sum(fitness_values) / len(fitness_values)

def evaluate_fitness_by_distribution(population: List[dict], target_freq: float = 440, sigma: float = 100):
    """
    operator1のfrequencyがtarget_freqに近いほど高評価（正規分布に基づく）
    sigmaは分布の幅（標準偏差）
    """
    for individual in population:
        if not isinstance(individual, dict):
            print("警告: individualがdict型ではありません:", individual)
            continue
        op1 = individual["fmParamsList"]["operator1"]
        frequency = op1.get("frequency", 0)
        # 正規分布の確率密度関数（最大値を10にスケール）
        score = math.exp(-((frequency - target_freq) ** 2) / (2 * sigma ** 2))
        normalized = int(round(score * 10))  # 0～10に丸める
        individual["fitness"] = str(normalized")