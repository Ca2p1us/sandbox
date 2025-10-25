import random
import math
import uuid
import numpy as np
from typing import List

def add_noise(value: float, noise_sigma: float = 1.0, noise_mean = 0, scale = 1.0) -> float:
    # 平均0、標準偏差noise_sigmaの正規分布ノイズを加算
    return value + random.gauss(noise_mean, noise_sigma) * scale

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

# FMパラメータに基づいた評価（例：operator1のfrequencyに基づく）
def evaluate_fitness_by_param(
    population: List[dict],
    target_params: List[float],
    sigma: float = 100.0,
    param_keys: List[str] = None,
    id_list: List[str] = None,
    noise_is_added: bool = False
):
    """
    param_keysで指定した各パラメータがtarget_paramsの値に近いほど高評価（正規分布に基づく）
    id_listが指定された場合は、そのID（chromosomeId）を持つ個体のみfitnessを付与
    統合手法は平均値
    """
    if param_keys is None:
        param_keys = ["fmParamsList.operator1.frequency"]

    # id_listが指定されていれば、そのIDのみ評価
    def should_evaluate(ind):
        if id_list is None:
            return True
        if "chromosomeId" not in ind:
            return False
        # UUID型にも対応
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
                # score = math.exp(-((float(val) - target) ** 2) / (2 * sigma ** 2))
                score = math.exp(-(float(val) ** 2) / (2 * (sigma ** 2)))
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
    target_params: List[float],
    param_keys: List[str],
    id_list: List[str] = None,
    noise_is_added: bool = False
):
    """
    各個体の二乗和を取ることで評価
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

        # 線形結合
        fitness = 0
        for i in range(len(values)):
            # fitness += -1 * (values[i] - target_params[i]) ** 2
            fitness += -1 * values[i] ** 2
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
        id_list: List[str] = None,
        A = 300,
        B = 0.005,
        C = 2*np.pi,
        noise_is_added: bool = False
):
    """
    Ackley関数で評価
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
        fitness = 0
        fitness -= -A * math.exp(-B * math.sqrt(sum(v**2 for v in values)/len(values))) - math.exp(sum(np.cos(C*v) for v in values)/len(values)) + A + math.e
        # 必要に応じてスケーリングやノイズ付与も可能
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
        # fitness += 418.9829 * len(values) - sum(v * math.sin(math.sqrt(abs(v))) for v in values)
        fitness += sum(v * np.sin(math.sqrt(abs(v))) for v in values)
        # 必要に応じてスケーリングやノイズ付与も可能
        if noise_is_added:
            fitness = add_noise(value=fitness, scale=1.0)
        individual["fitness"] = fitness


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
            individual["fitness"] = (round(add_noise(random.randint(1, 10))))

def get_best_and_worst_individuals_by_id(id_list: List[str], population: List[dict]):
    """
    id_listに含まれるchromosomeIdを持つ個体群から、最もfitnessが高い個体と低い個体を返す
    fitnessが小数値でも取得できるようにfloat変換で判定
    """
    # id_listをUUID型に変換
    id_set = set()
    for cid in id_list:
        try:
            id_set.add(uuid.UUID(str(cid)))
        except Exception:
            continue

    def is_number(s):
        try:
            float(s)
            return True
        except Exception:
            return False

    valid_population = [
        ind for ind in population
        if "fitness" in ind and "chromosomeId" in ind
        and (
            (isinstance(ind["chromosomeId"], uuid.UUID) and ind["chromosomeId"] in id_set)
            or
            (isinstance(ind["chromosomeId"], str) and uuid.UUID(ind["chromosomeId"]) in id_set)
        )
        and is_number(ind["fitness"])
    ]
    if not valid_population:
        return None, None
    best = max(valid_population, key=lambda x: float(x["fitness"]))
    worst = min(valid_population, key=lambda x: float(x["fitness"]))
    return best, worst

def get_average_fitness(population: List[dict], id_list: List[str] = None) -> float:
    """
    population内のfitnessの平均値を返す（fitnessが未設定・不正な個体は除外）
    id_listが指定された場合は、そのID（chromosomeId）を持つ個体のみ対象
    """
    if id_list is not None:
        # UUID型も考慮してセット化
        id_set = set()
        for cid in id_list:
            try:
                id_set.add(str(uuid.UUID(str(cid))))
            except Exception:
                id_set.add(str(cid))
        fitness_values = [
            float(ind["fitness"])
            for ind in population
            if "fitness" in ind
            and "chromosomeId" in ind
            and str(ind["chromosomeId"]) in id_set
            and str(ind["fitness"]).strip() not in ("", "None")
            and str(ind["fitness"]).replace('.', '', 1).isdigit()
        ]
    else:
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