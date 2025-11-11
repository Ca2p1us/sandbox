import random
import numpy as np
from collections.abc import Callable

RNG = np.random.default_rng(seed=30)


def exec_blx_alpha(parents_chromosomes: list, func_repair_gene, mutate) -> dict:
    offspring = {"fmParamsList": {}}
    parents1 = parents_chromosomes[0]
    parents2 = parents_chromosomes[1]
    for chromosome1_operator, chromosome2_operator in zip(parents1["fmParamsList"].items(), parents2["fmParamsList"].items()):
        # return operator1,operator2 ...
        operator_num: str = chromosome1_operator[0]
        offspring_operators_params = {}
        offspring_operators_params.setdefault(operator_num, {})
        for chromosome1_operator_params, chromosome2_operator_params in zip(chromosome1_operator[1].items(), chromosome2_operator[1].items()):
            # attack,decayなどのパラメータ
            param_name = chromosome1_operator_params[0]
            chromosome1_operator_param_value = chromosome1_operator_params[1]
            chromosome2_operator_param_value = chromosome2_operator_params[1]
            offspring_param_range: tuple = culc_offspring_param_range(
                chromosome1_operator_param_value, chromosome2_operator_param_value)
            # 子孫のパラメータを範囲内からランダムに決定
            # # offspring_param: float = random.uniform(
            # #     offspring_param_range[0], offspring_param_range[1])
            offspring_param: float = RNG.uniform(
                offspring_param_range[0], offspring_param_range[1]
            )
            offspring_param: float = RNG.uniform(
                offspring_param_range[0], offspring_param_range[1]
            )
            # 一定確率で突然変異 確率は通常[1/遺伝子型の数]
            # mutete_probability = random.random()
            mutete_probability = RNG.uniform()
            if mutete_probability <= 1/100:
                offspring_param = mutate(param_name)
            # パラメータの修正
            offspring_param = func_repair_gene(param_name, offspring_param)
            operator_param = {param_name: offspring_param}
            offspring_operators_params[operator_num].update(operator_param)
        offspring["fmParamsList"].update(offspring_operators_params)
    return offspring


def culc_offspring_param_range(param1, param2, expand_rate=0.5) -> tuple:
    param1 = float(param1)
    param2 = float(param2)
    param_range = abs(param1-param2)
    expanded_range = param_range * expand_rate
    if param1 >= param2:
        return (param2-expanded_range, param1+expanded_range)
    else:
        return (param1-expanded_range, param2+expanded_range)
