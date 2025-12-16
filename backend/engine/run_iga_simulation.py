from typing import List
from ..core.geneticAlgorithm.selection import tournament
from ..core.geneticAlgorithm.make_chromosome_params import make_chromosome_params
from ..core.geneticAlgorithm import BLX_alpha
from ..core.geneticAlgorithm.repair import repair_fm_params as repair_gene
from ..core.geneticAlgorithm.mutate import mutate 
from ..core.geneticAlgorithm import make_chromosome_params
from ..core.geneticAlgorithm.interpolation import interpolation, get_evaluated_individuals, get_total_error
from ..core.geneticAlgorithm.pre_selection import select_top_individuals_by_pre_evaluation
from ..core.geneticAlgorithm.config import TARGET_PARAMS, PARAMS, TARGET_PARAMS_1, TARGET_PARAMS_2
from ..core.log import log, log_fitness, plot_individual_params, log_average_fitness
from ..engine.evaluate import evaluate_fitness, get_best_and_worst_individuals, get_best_and_worst_individuals_by_id, get_average_fitness
import uuid


Chromosomes = List[dict]


def make_initial_population(num_individuals=10):
    return [make_chromosome_params.make_chromosome_params() for _ in range(num_individuals)]

def run_simulation_normal_IGA(NUM_GENERATIONS=9, POPULATION_SIZE=10, evaluate_num=0, times:int=1, noise_is_added: bool = False, look: bool = False, tournament_size=3):
    best_fitness_history = []
    average_fitness_history = []
    bests = []
    if evaluate_num == 1:
    # 2-1. ガウス関数
        evaluate_method = "Gaussian"
    elif evaluate_num == 2:
    # 2-2. スフィア関数
        evaluate_method = "Sphere"
    elif evaluate_num == 3:
    # 2-3. Gauss関数+cos関数
        evaluate_method = "Gaussian_cos"
    elif evaluate_num == 4:
    # 2-4. Ackley関数
        evaluate_method = "Ackley"
    elif evaluate_num == 5:
    # 2-5. Gaussian_two_peak関数
            evaluate_method = "Gaussian_two_peak"
    # 1. 初期個体生成
    population = make_initial_population(POPULATION_SIZE)
    
    for generation in range(NUM_GENERATIONS - 1):
        # 2. 評価
        evaluate_fitness(
            population=population,
            evaluate_num=evaluate_num,
            param_keys=PARAMS,
            noise_is_added=noise_is_added
        )
        best, worst = get_best_and_worst_individuals(population)
        # --- ここで履歴に追加 ---
        if best is not None and "fitness" in best:
            best_fitness_history.append((generation + 1, float(best["fitness"])))
            bests.append(best)
        # 評価の平均値を表示
        average = get_average_fitness(population)
        if average is not None:
            average_fitness_history.append((generation + 1, float(average)))
        next_generation:List[Chromosomes]  = []
        if look and times == 1:
            plot_individual_params(population=population,best=best,worst=worst, param_keys=PARAMS, generation=generation + 1, file_path=f'./result/conventional/graph/{evaluate_method}/scatter/{evaluate_method}_noise{str(noise_is_added)}_{str(POPULATION_SIZE)}_individuals_{str(generation + 1)}gens')
        for _ in range(POPULATION_SIZE):
            # 3. 選択
            selected = tournament.exec_tournament_selection(chromosomes_params=population, participants_num=tournament_size)
            # selected = tournament.exec_tournament_selection(chromosomes_params=population, participants_num=6)
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
                        # ind["algorithmNum"] = selected[0].get("algorithmNum", None)
                        ind["generation"] = generation + 2
                        # 新しいchromosomeIdを付与
                        ind["chromosomeId"] = str(uuid.uuid4())
                        next_generation.append(ind)
                    else:
                        print("警告: offspring内にdict以外が含まれています:", ind)
            elif isinstance(offspring, dict):
                # offspring["algorithmNum"] = selected[0].get("algorithmNum", None)
                offspring["generation"] = generation + 2
                offspring["chromosomeId"] = str(uuid.uuid4())
                next_generation.append(offspring)
            else:
                print("警告: offspringがdictまたはlist[dict]ではありません:", offspring)

        # 5. 次世代への更新
        population = next_generation
    # --- ここで最終世代の評価値を再計算 ---
    evaluate_fitness(
            population=population,
            evaluate_num=evaluate_num,
            param_keys=PARAMS,
            noise_is_added=noise_is_added
        )

    # ベスト・ワースト個体の取得
    best, worst = get_best_and_worst_individuals(population)
    # 評価の平均値を表示
    average = get_average_fitness(population)
    average = get_average_fitness(population)
    best_fitness_history.append((NUM_GENERATIONS, float(best["fitness"])))
    average_fitness_history.append((NUM_GENERATIONS, float(average)))
    bests.append(best)
    if look and times == 1:
        plot_individual_params(population=population,best=best,worst=worst, param_keys=PARAMS, generation=NUM_GENERATIONS, file_path=f'./result/conventional/graph/{evaluate_method}/scatter/{evaluate_method}_noise{str(noise_is_added)}_{str(POPULATION_SIZE)}_individuals_{str(NUM_GENERATIONS)}gens')
    # 6. 最終結果の出力
    log(f"result/conventional/last_gen_individuals/{evaluate_method}/{str(POPULATION_SIZE)}inds/simulation_{evaluate_method}_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(times)}.json", population,times = times)
    log(f"result/conventional/best/{evaluate_method}/{str(POPULATION_SIZE)}inds/best_individual_{evaluate_method}_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(times)}.json", bests,times = times)
    log_fitness(method=evaluate_method, file_path=f"result/conventional/graph/{evaluate_method}/best_fitnesses/{evaluate_method}_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(POPULATION_SIZE)}_{str(times)}_best_fitness_history.png", best_fitness_history=best_fitness_history, average_fitness_history=average_fitness_history)
    log_average_fitness(method=evaluate_method, file_path=f"_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(POPULATION_SIZE)}_{str(times)}_average_fitness_history.json", average_fitness_history=average_fitness_history, times=times)
    return best_fitness_history,average_fitness_history



def run_simulation_proposal_IGA(NUM_GENERATIONS=9, PROPOSAL_POPULATION_SIZE=200, EVALUATE_SIZE=9, evaluate_num=0, interpolate_num=0, times:int=1, noise_is_added:bool=False, look: bool = False, tournament_size=3):
    best_fitness_history = []
    average_fitness_history = []
    error_history = []
    bests = []
    evaluate_method = ""
    interpolate = "linear"
    if evaluate_num == 1:
    # 2-1. ガウス関数
        evaluate_method = "Gaussian"
    elif evaluate_num == 2:
    # 2-2. スフィア関数
        evaluate_method = "Sphere"
    elif evaluate_num == 3:
    # 2-3. Gauss関数+cos関数
        evaluate_method = "Gaussian_cos"
    elif evaluate_num == 4:
    # 2-4. Ackley関数
        evaluate_method = "Ackley"
    elif evaluate_num == 5:
    # 2-5. Gaussian_two_peak関数
        evaluate_method = "Gaussian_two_peak"
    # 1. 初期個体生成
    population = make_initial_population(PROPOSAL_POPULATION_SIZE)
    # 初期個体の事前評価(補間)
    if interpolate_num == 0:
        interpolate = "linear"
    elif interpolate_num == 1:
        interpolate = "Gauss"
    elif interpolate_num == 2:
        interpolate = "RBF"
    elif interpolate_num == 3:
        interpolate = "IDW"
    interpolation(
            population,
            method_num=interpolate_num,
            param_keys=PARAMS,
            target_key="pre_evaluation",
            )
    # 評価個体の選択
    evaluate_id = select_top_individuals_by_pre_evaluation(population, total_n=EVALUATE_SIZE)
    evaluate_population = get_evaluated_individuals(population, evaluate_id)

    for generation in range(NUM_GENERATIONS - 1):
        # 2. 評価
        evaluate_fitness(
            population=evaluate_population,
            evaluate_num=evaluate_num,
            param_keys=PARAMS,
            noise_is_added=noise_is_added
        )
        # ベスト・ワースト個体の取得
        best, worst = get_best_and_worst_individuals_by_id(evaluate_population)
        # ほかの個体の評価を補間
        interpolation(population=population, evaluated_population=evaluate_population , best=best, worst=worst, method_num=interpolate_num, target_key="fitness")
        # 真値の取得
        evaluate_fitness(
            population=population,
            evaluate_num=evaluate_num,
            param_keys=PARAMS,
            noise_is_added=noise_is_added,
            target_key="true_fitness"
        )

        # 評価の平均値を表示
        average = get_average_fitness(population)
        print(best["fitness"])
        # --- ここで履歴に追加 ---
        if best is not None and "fitness" in best:
            best_fitness_history.append((generation + 1, float(best["fitness"])))
            bests.append(best)
        if average is not None:
            average_fitness_history.append((generation + 1, float(average)))
        error_history.append((generation + 1, float(get_total_error(population=population, evaluate_num=evaluate_num))))
        if look and times == 1:
            plot_individual_params(population=population,best=best,worst=worst, param_keys=PARAMS, generation=generation + 1, file_path=f'./result/proposal/graph/{evaluate_method}/{interpolate}/scatter/{evaluate_method}_noise{str(noise_is_added)}_{str(PROPOSAL_POPULATION_SIZE)}_{str(EVALUATE_SIZE)}_individuals_{str(generation + 1)}gens')
        next_generation:List[Chromosomes]  = []
        # for _ in range(PROPOSAL_POPULATION_SIZE):
        while len(next_generation) < PROPOSAL_POPULATION_SIZE:
            # 3. 選択
            # if generation > NUM_GENERATIONS // 2:
            #     selected = tournament.exec_tournament_selection(chromosomes_params=population, participants_num=30)
            # else:
            #     selected = tournament.exec_tournament_selection(chromosomes_params=population, participants_num=tournament_size)
            selected = tournament.exec_tournament_selection(chromosomes_params=population, participants_num=tournament_size)
            # 4. 交叉&突然変異
            offspring = BLX_alpha.exec_blx_alpha(
                parents_chromosomes=selected,
                func_repair_gene=repair_gene,
                mutate=mutate
            )
            # offspringがリストの場合
            if isinstance(offspring, list):
                for ind in offspring:
                    if len(next_generation) >= PROPOSAL_POPULATION_SIZE:
                        break
                    if isinstance(ind, dict):
                        # algorithmNumを親からコピー（selected[0]を例とする）
                        # ind["algorithmNum"] = selected[0].get("algorithmNum", None)
                        # 新しいchromosomeIdを付与
                        ind["fitness"] = 0.0
                        ind["pre_evaluation"] = 0
                        ind["true_fitness"] = 0.0
                        ind["generation"] = generation + 2
                        ind["chromosomeId"] = str(uuid.uuid4())
                        next_generation.append(ind)
                    else:
                        print("警告: offspring内にdict以外が含まれています:", ind)
            elif isinstance(offspring, dict):
                # offspring["algorithmNum"] = selected[0].get("algorithmNum", None)
                offspring["fitness"] = 0.0
                offspring["pre_evaluation"] = 0
                offspring["true_fitness"] = 0.0
                offspring["generation"] = generation + 2
                offspring["chromosomeId"] = str(uuid.uuid4())
                next_generation.append(offspring)
            else:
                print("警告: offspringがdictまたはlist[dict]ではありません:", offspring)

        # 5. 次世代への更新

        population = next_generation
        #事前評価(補間)
        interpolation(
            population = population,
            evaluated_population = evaluate_population,
            best = best, 
            worst = worst,
            method_num=interpolate_num,
            param_keys=PARAMS,
            target_key="pre_evaluation",
        )
        # 評価個体の選択
        evaluate_id = select_top_individuals_by_pre_evaluation(population, total_n=EVALUATE_SIZE)
        evaluate_population = get_evaluated_individuals(population, evaluate_id)

    # --- ここで最終世代の評価値を再計算 ---
    evaluate_fitness(
            population=evaluate_population,
            evaluate_num=evaluate_num,
            param_keys=PARAMS,
            noise_is_added=noise_is_added
        )
    # ベスト・ワースト個体の取得
    best, worst = get_best_and_worst_individuals_by_id(evaluate_population)
    # ほかの個体の評価を補間
    interpolation(population=population,evaluated_population=evaluate_population , best=best, worst=worst, method_num=interpolate_num, target_key="fitness")
    evaluate_fitness(
            population=population,
            evaluate_num=evaluate_num,
            param_keys=PARAMS,
            noise_is_added=noise_is_added,
            target_key="true_fitness"
        )
    # 評価の平均値を表示
    average = get_average_fitness(population)
    best_fitness_history.append((NUM_GENERATIONS, float(best["fitness"])))
    average_fitness_history.append((NUM_GENERATIONS, float(average)))
    bests.append(best)
    error_history.append((NUM_GENERATIONS, float(get_total_error(population=population, evaluate_num=evaluate_num))))
    if look and times == 1:
        plot_individual_params(population=population,best=best,worst=worst, param_keys=PARAMS, generation=NUM_GENERATIONS, file_path=f'./result/proposal/graph/{evaluate_method}/{interpolate}/scatter/{evaluate_method}_noise{str(noise_is_added)}_{str(PROPOSAL_POPULATION_SIZE)}_{str(EVALUATE_SIZE)}_individuals_{str(NUM_GENERATIONS)}gens')
    # 6. 最終結果の出力
    log(f"result/proposal/last_gen_individuals/{evaluate_method}/{interpolate}/{str(PROPOSAL_POPULATION_SIZE)}inds_{str(EVALUATE_SIZE)}eval/simulation_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(times)}.json", population,times = times)
    log(f"result/proposal/best/{evaluate_method}/{interpolate}/{str(PROPOSAL_POPULATION_SIZE)}inds_{str(EVALUATE_SIZE)}eval/best_individual_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(times)}.json", bests,times = times)
    log_fitness(method=evaluate_method, file_path=f"result/proposal/graph/{evaluate_method}/{interpolate}/best_fitnesses/{evaluate_method}_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(PROPOSAL_POPULATION_SIZE)}_{str(EVALUATE_SIZE)}eval_{str(times)}_best_fitness_history.png", best_fitness_history=best_fitness_history, average_fitness_history=average_fitness_history)
    log_average_fitness(method=evaluate_method, interpolate_method=interpolate, file_path=f"_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(PROPOSAL_POPULATION_SIZE)}_{str(EVALUATE_SIZE)}eval_{str(times)}_average_fitness_history.json", average_fitness_history=average_fitness_history, times=times)
    return best_fitness_history, average_fitness_history, error_history
