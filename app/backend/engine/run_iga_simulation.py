from typing import List
from ..core.geneticAlgorithm.selection import tournament
from ..core.geneticAlgorithm.make_chromosome_params import make_chromosome_params
from ..core.geneticAlgorithm import BLX_alpha
from ..core.geneticAlgorithm.repair import repair_fm_params as repair_gene
from ..core.geneticAlgorithm.mutate import mutate 
from ..core.geneticAlgorithm import make_chromosome_params
from ..core.geneticAlgorithm.interpolation import interpolate_by_distance
from ..core.geneticAlgorithm.pre_selection import select_top_individuals_by_pre_evaluation
from ..core.log import log, log_fitness, sound_check, load_params_from_json
from ..engine import evaluate
import uuid


Chromosomes = List[dict]

NUM_GENERATIONS = 10
POPULATION_SIZE = 10

def make_initial_population(num_individuals=10):
    return [make_chromosome_params.make_chromosome_params() for _ in range(num_individuals)]

def run_simulation_normal_IGA():
    best_fitness_history = []
    # 1. 初期個体生成
    population = make_initial_population(POPULATION_SIZE)
    
    for generation in range(NUM_GENERATIONS):
        # 2. 評価
        evaluate.evaluate_fitness_by_param(
            population = population,
            target_params=[0.03, 0.16, 0.89, 0.29, 0.06, 0.31690]
            )
        best, worst = evaluate.get_best_and_worst_individuals(population)
        print(f"Generation {generation + 1}\n \t Best fitness = {best['fitness']}\n \tWorst fitness = {worst['fitness']}")
        # --- ここで履歴に追加 ---
        if best is not None and "fitness" in best:
            best_fitness_history.append((generation + 1, float(best["fitness"])))
        # 評価の平均値を表示
        print(f"average fitness:", evaluate.get_average_fitness(population))
        next_generation:List[Chromosomes]  = []
        for _ in range(POPULATION_SIZE):
            # 3. 選択
            selected = tournament.exec_tournament_selection(population)

            # 4. 交叉&突然変異
            offspring = BLX_alpha.exec_blx_alpha(
                parents_chromosomes=selected,
                func_repair_gene=repair_gene,
                mutate=mutate
            )
            # offspringがリストの場合
            if isinstance(offspring, list):
                for ind in offspring:
                    if isinstance(ind, dict):
                        # algorithmNumを親からコピー（selected[0]を例とする）
                        ind["algorithmNum"] = selected[0].get("algorithmNum", None)
                        # 新しいchromosomeIdを付与
                        ind["chromosomeId"] = str(uuid.uuid4())
                        next_generation.append(ind)
                    else:
                        print("警告: offspring内にdict以外が含まれています:", ind)
            elif isinstance(offspring, dict):
                offspring["algorithmNum"] = selected[0].get("algorithmNum", None)
                offspring["chromosomeId"] = str(uuid.uuid4())
                next_generation.append(offspring)
            else:
                print("警告: offspringがdictまたはlist[dict]ではありません:", offspring)

        # 5. 次世代への更新
        population = next_generation
        #評価
        evaluate.evaluate_fitness_by_param(
            population = population,
            target_params=[0.03, 0.16, 0.89, 0.29, 0.06, 0.31690]
            )

    # 6. 最終結果の出力
    log("result/simulation_random.json", population)
    log_fitness(best_fitness_history,"random")
    return population

PROPOSAL_POPULATION_SIZE = 100
EVALUATE_SIZE = 9
PARAMS = ["fmParamsList.operator1.attack", "fmParamsList.operator1.decay", "fmParamsList.operator1.sustain", "fmParamsList.operator1.sustain_time", "fmParamsList.operator1.release", "fmParamsList.operator1.frequency"]
def run_simulation_proposal_IGA(evaluate_num=0):
    best_fitness_history = []
    # 1. 初期個体生成
    population = make_initial_population(PROPOSAL_POPULATION_SIZE)
    # 初期個体の事前評価(補間)
    interpolate_by_distance(
        population,
        param_keys=PARAMS,
        target_key="pre_evaluation"
        )
    # 評価個体の選択
    evaluate_population = select_top_individuals_by_pre_evaluation(population, top_n=EVALUATE_SIZE)

    for generation in range(NUM_GENERATIONS - 1):
        # 2. 評価
        print(f"\n--- Generation {generation + 1} の評価を行います ---")
        if evaluate_num == 0:
        # 2-1. ガウス関数
            evaluate_method = "gaussian"
            evaluate.evaluate_fitness_by_param(
                #評価対象集団
                population,
                #目標値
                target_params=[0.03, 0.16, 0.89, 0.29, 0.06, 0.31690],
                #標準偏差
                sigma=500.0,
                #評価対象パラメータ
                param_keys=PARAMS,
                #評価方法"product", "mean", "max", "min", "median"
                #評価個体のIDリスト
                id_list=evaluate_population
            )
        elif evaluate_num == 1:
        # 2-2. スフィア関数
            evaluate_method = "sphere"
            evaluate.evaluate_fitness_sphere(
                population=population,
                target_params=[0.03, 0.16, 0.89, 0.29, 0.06, 0.31690],
                param_keys=PARAMS,
                id_list=evaluate_population
            )
        elif evaluate_num == 2:
        # 2-3. ノイズ関数
            evaluate_method = "noise"
            evaluate.evaluate_fitness_noise(
                population=population,
                param_keys=PARAMS,
                id_list=evaluate_population
            )
        elif evaluate_num == 3:
        # 2-4. コサイン関数
            evaluate_method = "cos"
            evaluate.evaluate_fitness_cos(
                population=population,
                param_keys=PARAMS,
                id_list=evaluate_population
            )
        # ベスト・ワースト個体の取得
        best, worst = evaluate.get_best_and_worst_individuals_by_id(evaluate_population, population)
        print(f"best {best}\nworst {worst}")
        # ほかの個体の評価を補間
        interpolate_by_distance(population, best, worst,param_keys=PARAMS, target_key="fitness")
        # 評価の平均値を表示
        print(f"average fitness:", evaluate.get_average_fitness(population))
        # 上位9個体の平均評価値を表示
        print(f"average fitness of top {EVALUATE_SIZE}:", evaluate.get_average_fitness(population,evaluate_population))
        print(f"\t Best fitness = {best['fitness']}\n \tWorst fitness = {worst['fitness']}")
        # --- ここで履歴に追加 ---
        if best is not None and "fitness" in best:
            best_fitness_history.append((generation + 1, float(best["fitness"])))

        next_generation:List[Chromosomes]  = []
        for _ in range(PROPOSAL_POPULATION_SIZE):
            # 3. 選択
            selected = tournament.exec_tournament_selection(population)

            # 4. 交叉&突然変異
            offspring = BLX_alpha.exec_blx_alpha(
                parents_chromosomes=selected,
                func_repair_gene=repair_gene,
                mutate=mutate
            )
            # offspringがリストの場合
            if isinstance(offspring, list):
                for ind in offspring:
                    if isinstance(ind, dict):
                        # algorithmNumを親からコピー（selected[0]を例とする）
                        ind["algorithmNum"] = selected[0].get("algorithmNum", None)
                        # 新しいchromosomeIdを付与
                        ind["chromosomeId"] = str(uuid.uuid4())
                        ind["fitness"] = 0.0
                        ind["pre_evaluation"] = 0
                        next_generation.append(ind)
                    else:
                        print("警告: offspring内にdict以外が含まれています:", ind)
            elif isinstance(offspring, dict):
                offspring["algorithmNum"] = selected[0].get("algorithmNum", None)
                offspring["chromosomeId"] = str(uuid.uuid4())
                offspring["fitness"] = 0.0
                offspring["pre_evaluation"] = 0
                next_generation.append(offspring)
            else:
                print("警告: offspringがdictまたはlist[dict]ではありません:", offspring)

        # 5. 次世代への更新
        print(f"--- Generation {generation + 1} の次世代を生成しました ---")
        population = next_generation
        #事前評価(補間)
        interpolate_by_distance(
            population,
            best,
            worst,
            param_keys=PARAMS,
            target_key="pre_evaluation"
            )
        # 評価個体の選択
        evaluate_population = select_top_individuals_by_pre_evaluation(population, top_n=EVALUATE_SIZE)

    # --- ここで最終世代の評価値を再計算 ---
    if evaluate_num == 0:
        evaluate.evaluate_fitness_by_param(
            population,
            target_params=[0.03, 0.16, 0.89, 0.29, 0.06, 0.31690],
            sigma=500.0,
            param_keys=PARAMS,
            id_list=evaluate_population
        )
    elif evaluate_num == 1:
        evaluate.evaluate_fitness_sphere(
            population =population,
            target_params=[0.03, 0.16, 0.89, 0.29, 0.06, 0.31690],
            param_keys=PARAMS,
            id_list=evaluate_population
        )
    elif evaluate_num == 2:
        evaluate.evaluate_fitness_noise(
            population=population,
            param_keys=PARAMS,
            id_list=evaluate_population
        )
    elif evaluate_num == 3:
        evaluate.evaluate_fitness_cos(
            population=population,
            param_keys=PARAMS,
            id_list=evaluate_population
        )
    # ベスト・ワースト個体の取得
    best, worst = evaluate.get_best_and_worst_individuals_by_id(evaluate_population, population)
    interpolate_by_distance(population, best, worst, target_key='fitness')
    # 評価の平均値を表示
    print(f"Generation {NUM_GENERATIONS}\n average fitness:", evaluate.get_average_fitness(population))
    # 上位9個体の平均評価値を表示
    print(f" average fitness of top {EVALUATE_SIZE}:", evaluate.get_average_fitness(population,evaluate_population))
    print(f"\t Best fitness = {best['fitness']}\n \tWorst fitness = {worst['fitness']}")
    # 6. 最終結果の出力
    log("result/simulation_"+evaluate_method+".json", population)
    log_fitness(best_fitness_history,evaluate_method)
    return population
