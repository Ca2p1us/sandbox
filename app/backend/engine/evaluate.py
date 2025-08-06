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
def evaluate_fitness_by_param(population: List[dict]):
    for individual in population:
        if not isinstance(individual, dict):
            print("警告: individualがdict型ではありません:", individual)
            continue
        op1 = individual["fmParamsList"]["operator1"]
        frequency = op1.get("frequency", 0)
        # 例えば frequency を使って評価（220Hz ～ 880Hz を1～10にマッピング）
        normalized = max(1, min(10, int((frequency - 220) / (880 - 220) * 9 + 1)))
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
        individual["fitness"] = str(normalized)