from typing import List
from ..core.geneticAlgorithm.selection import tournament
from ..core.geneticAlgorithm.make_chromosome_params import make_chromosome_params
from ..core.geneticAlgorithm import BLX_alpha
from ..core.geneticAlgorithm.repair import repair_fm_params as repair_gene
from ..core.geneticAlgorithm.mutate import mutate 
from ..core.geneticAlgorithm import make_chromosome_params
from ..core.geneticAlgorithm.interpolation import interpolate_by_distance
from ..core.geneticAlgorithm.pre_selection import select_top_individuals_by_pre_evaluation
from ..core.geneticAlgorithm.config import TARGET_PARAMS, PARAMS
from ..core.log import log, log_fitness, plot_individual_params
from ..engine import evaluate
import uuid


Chromosomes = List[dict]


def make_initial_population(num_individuals=10):
    return [make_chromosome_params.make_chromosome_params() for _ in range(num_individuals)]

def run_simulation_normal_IGA(NUM_GENERATIONS=9, POPULATION_SIZE=10, evaluate_num=0, times:int=1, noise_is_added: bool = False, look: bool = False):
    best_fitness_history = []
    average_fitness_history = []
    bests = []
    # 1. 初期個体生成
    population = make_initial_population(POPULATION_SIZE)
    
    for generation in range(NUM_GENERATIONS - 1):
        # 2. 評価
        print(f"\n--- Generation {generation + 1} の評価を行います ---")
        if evaluate_num == 1:
        # 2-1. ガウス関数
            evaluate_method = "Gaussian"
            evaluate.evaluate_fitness_by_param(
                #評価対象集団
                population,
                #目標値
                target_params=TARGET_PARAMS,
                #評価対象パラメータ
                param_keys=PARAMS,
                noise_is_added=noise_is_added
            )
        elif evaluate_num == 2:
        # 2-2. スフィア関数
            evaluate_method = "Sphere"
            evaluate.evaluate_fitness_sphere(
                population=population,
                target_params=TARGET_PARAMS,
                param_keys=PARAMS,
                noise_is_added=noise_is_added
            )
        elif evaluate_num == 3:
        # 2-3. Rastrigin関数
            evaluate_method = "Rastrigin"
            evaluate.evaluate_fitness_cos(
                population=population,
                param_keys=PARAMS,
                noise_is_added=noise_is_added
            )
        elif evaluate_num == 4:
        # 2-4. Ackley関数
            evaluate_method = "Ackley"
            evaluate.evaluate_fitness_Ackley(
                population=population,
                param_keys=PARAMS,
                noise_is_added=noise_is_added
            )
        elif evaluate_num == 5:
        # 2-5. Schwefel関数
            evaluate_method = "Schwefel"
            evaluate.evaluate_fitness_Schwefel(
                population=population,
                param_keys=PARAMS,
                noise_is_added=noise_is_added
            )
        best, worst = evaluate.get_best_and_worst_individuals(population)
        print(f"Generation {generation + 1}\n \t Best fitness = {best['fitness']}\n \tWorst fitness = {worst['fitness']}")
        # --- ここで履歴に追加 ---
        if best is not None and "fitness" in best:
            best_fitness_history.append((generation + 1, float(best["fitness"])))
            bests.append(best)
        # 評価の平均値を表示
        average = evaluate.get_average_fitness(population)
        print(f"average fitness:", average)
        if average is not None:
            average_fitness_history.append((generation + 1, float(average)))
        next_generation:List[Chromosomes]  = []
        if look:
            plot_individual_params(population, PARAMS, generation+1)
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
                        # ind["algorithmNum"] = selected[0].get("algorithmNum", None)
                        ind["generation"] = generation + 1
                        # 新しいchromosomeIdを付与
                        ind["chromosomeId"] = str(uuid.uuid4())
                        next_generation.append(ind)
                    else:
                        print("警告: offspring内にdict以外が含まれています:", ind)
            elif isinstance(offspring, dict):
                # offspring["algorithmNum"] = selected[0].get("algorithmNum", None)
                offspring["generation"] = generation + 1
                offspring["chromosomeId"] = str(uuid.uuid4())
                next_generation.append(offspring)
            else:
                print("警告: offspringがdictまたはlist[dict]ではありません:", offspring)

        # 5. 次世代への更新
        population = next_generation
        print(f"------------------------------------")
    # --- ここで最終世代の評価値を再計算 ---
    if evaluate_num == 1:
        evaluate.evaluate_fitness_by_param(
            population,
            target_params=TARGET_PARAMS,
            param_keys=PARAMS,
            noise_is_added=noise_is_added
        )
    elif evaluate_num == 2:
        evaluate.evaluate_fitness_sphere(
            population =population,
            target_params=TARGET_PARAMS,
            param_keys=PARAMS,
            noise_is_added=noise_is_added
        )
    elif evaluate_num == 3:
        evaluate.evaluate_fitness_cos(
            population=population,
            param_keys=PARAMS,
            noise_is_added=noise_is_added
        )
    elif evaluate_num == 4:
        evaluate.evaluate_fitness_Ackley(
            population=population,
            param_keys=PARAMS,
            noise_is_added=noise_is_added
        )
    elif evaluate_num == 5:
        evaluate.evaluate_fitness_Schwefel(
            population=population,
            param_keys=PARAMS,
            noise_is_added=noise_is_added
        )

    # ベスト・ワースト個体の取得
    best, worst = evaluate.get_best_and_worst_individuals(population)
    interpolate_by_distance(population, best, worst, target_key='fitness')
    # 評価の平均値を表示
    average = evaluate.get_average_fitness(population)
    print(f"Generation {NUM_GENERATIONS}\n average fitness:", average)
    print(f"\t Best fitness = {best['fitness']}\n \tWorst fitness = {worst['fitness']}")
    best_fitness_history.append((NUM_GENERATIONS, float(best["fitness"])))
    bests.append(best)
    average_fitness_history.append((NUM_GENERATIONS, float(average)))
    if look:
        plot_individual_params(population, PARAMS, NUM_GENERATIONS)
    # 6. 最終結果の出力
    log("result/conventional/last_gen_individuals/"+evaluate_method+"/simulation_"+evaluate_method+"_"+str(NUM_GENERATIONS)+"gens_"+str(POPULATION_SIZE)+"_"+str(times)+".json", population)
    log("result/conventional/best/"+evaluate_method+"/best_individual_"+evaluate_method+"_"+str(NUM_GENERATIONS)+"gens_"+str(POPULATION_SIZE)+"_"+str(times)+".json", bests)
    log_fitness(evaluate_method, "result/conventional/graph/"+evaluate_method+"/"+evaluate_method+"_"+str(NUM_GENERATIONS)+"gens_"+str(POPULATION_SIZE)+"_"+str(times)+"_best_fitness_history.png", best_fitness_history,  average_fitness_history=average_fitness_history)
    return best_fitness_history




def run_simulation_proposal_IGA(NUM_GENERATIONS=9, PROPOSAL_POPULATION_SIZE=200, EVALUATE_SIZE=9, evaluate_num=0, times:int=1, noise_is_added:bool=False, look: bool = False):
    best_fitness_history = []
    average_fitness_history = []
    bests = []
    # 1. 初期個体生成
    population = make_initial_population(PROPOSAL_POPULATION_SIZE)
    # 初期個体の事前評価(補間)
    interpolate_by_distance(
        population,
        param_keys=PARAMS,
        target_key="pre_evaluation"
        )
    # 評価個体の選択
    evaluate_population = select_top_individuals_by_pre_evaluation(population, total_n=EVALUATE_SIZE)

    for generation in range(NUM_GENERATIONS - 1):
        # 2. 評価
        print(f"\n--- Generation {generation + 1} の評価を行います ---")
        if evaluate_num == 1:
        # 2-1. ガウス関数
            evaluate_method = "Gaussian"
            evaluate.evaluate_fitness_by_param(
                #評価対象集団
                population,
                #目標値
                target_params=TARGET_PARAMS,
                #評価対象パラメータ
                param_keys=PARAMS,
                #評価方法"product", "mean", "max", "min", "median"
                #評価個体のIDリスト
                id_list=evaluate_population,
                noise_is_added=noise_is_added
            )
        elif evaluate_num == 2:
        # 2-2. スフィア関数
            evaluate_method = "Sphere"
            evaluate.evaluate_fitness_sphere(
                population=population,
                target_params=TARGET_PARAMS,
                param_keys=PARAMS,
                id_list=evaluate_population,
                noise_is_added=noise_is_added
            )
        elif evaluate_num == 3:
        # 2-4. Rastrigin関数
            evaluate_method = "Rastrigin"
            evaluate.evaluate_fitness_cos(
                population=population,
                param_keys=PARAMS,
                id_list=evaluate_population,
                noise_is_added=noise_is_added
            )
        elif evaluate_num == 4:
        # 2-5. Ackley関数
            evaluate_method = "Ackley"
            evaluate.evaluate_fitness_Ackley(
                population=population,
                param_keys=PARAMS,
                id_list=evaluate_population,
                noise_is_added=noise_is_added
            )
        elif evaluate_num == 5:
        # 2-6. Schwefel関数
            evaluate_method = "Schwefel"
            evaluate.evaluate_fitness_Schwefel(
                population=population,
                param_keys=PARAMS,
                id_list=evaluate_population,
                noise_is_added=noise_is_added
            )
        # ベスト・ワースト個体の取得
        best, worst = evaluate.get_best_and_worst_individuals_by_id(evaluate_population, population)
        # ほかの個体の評価を補間
        interpolate_by_distance(population, best, worst,param_keys=PARAMS, target_key="fitness")
        # 評価の平均値を表示
        average = evaluate.get_average_fitness(population)
        print(f"average fitness:", average)
        # 上位9個体の平均評価値を表示
        print(f"\t Best fitness = {best['fitness']}\n \tWorst fitness = {worst['fitness']}")

        # --- ここで履歴に追加 ---
        if best is not None and "fitness" in best:
            best_fitness_history.append((generation + 1, float(best["fitness"])))
            bests.append(best)
        if average is not None:
            average_fitness_history.append((generation + 1, float(average)))
        if look:
            plot_individual_params(population, PARAMS, generation+1)
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
                        # ind["algorithmNum"] = selected[0].get("algorithmNum", None)
                        # 新しいchromosomeIdを付与
                        ind["fitness"] = 0.0
                        ind["pre_evaluation"] = 0
                        ind["generation"] = generation + 2
                        ind["chromosomeId"] = str(uuid.uuid4())
                        next_generation.append(ind)
                    else:
                        print("警告: offspring内にdict以外が含まれています:", ind)
            elif isinstance(offspring, dict):
                # offspring["algorithmNum"] = selected[0].get("algorithmNum", None)
                offspring["fitness"] = 0.0
                offspring["pre_evaluation"] = 0
                offspring["generation"] = generation + 2
                offspring["chromosomeId"] = str(uuid.uuid4())
                next_generation.append(offspring)
            else:
                print("警告: offspringがdictまたはlist[dict]ではありません:", offspring)

        # 5. 次世代への更新
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
        evaluate_population = select_top_individuals_by_pre_evaluation(population, total_n=EVALUATE_SIZE)
        print(f"------------------------------------")

    # --- ここで最終世代の評価値を再計算 ---
    if evaluate_num == 1:
        evaluate.evaluate_fitness_by_param(
            population,
            target_params=TARGET_PARAMS,
            param_keys=PARAMS,
            id_list=evaluate_population,
            noise_is_added=noise_is_added
        )
    elif evaluate_num == 2:
        evaluate.evaluate_fitness_sphere(
            population =population,
            target_params=TARGET_PARAMS,
            param_keys=PARAMS,
            id_list=evaluate_population,
            noise_is_added=noise_is_added
        )
    elif evaluate_num == 3:
        evaluate.evaluate_fitness_cos(
            population=population,
            param_keys=PARAMS,
            id_list=evaluate_population,
            noise_is_added=noise_is_added
        )
    elif evaluate_num == 4:
        evaluate.evaluate_fitness_Ackley(
            population=population,
            param_keys=PARAMS,
            id_list=evaluate_population,
            noise_is_added=noise_is_added
        )
    elif evaluate_num == 5:
        evaluate.evaluate_fitness_Schwefel(
            population=population,
            param_keys=PARAMS,
            id_list=evaluate_population,
            noise_is_added=noise_is_added
        )
    # ベスト・ワースト個体の取得
    best, worst = evaluate.get_best_and_worst_individuals_by_id(evaluate_population, population)
    interpolate_by_distance(population, best, worst, target_key='fitness')
    # 評価の平均値を表示
    average = evaluate.get_average_fitness(population)
    print(f"Generation {NUM_GENERATIONS}\n average fitness:", average)
    # 上位9個体の平均評価値を表示
    print(f" average fitness of top {EVALUATE_SIZE}:", evaluate.get_average_fitness(population,evaluate_population))
    print(f"\t Best fitness = {best['fitness']}\n \tWorst fitness = {worst['fitness']}")
    best_fitness_history.append((NUM_GENERATIONS, float(best["fitness"])))
    bests.append(best)
    average_fitness_history.append((NUM_GENERATIONS, float(average)))
    if look:
        plot_individual_params(population, PARAMS, NUM_GENERATIONS)
    # 6. 最終結果の出力
    log("result/proposal/last_gen_individuals/"+evaluate_method+"/simulation_"+evaluate_method+"_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(PROPOSAL_POPULATION_SIZE)+"_"+str(times)+".json", population)
    log("result/proposal/best/"+evaluate_method+"/best_individual_"+evaluate_method+"_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(PROPOSAL_POPULATION_SIZE)+"_"+str(times)+".json", bests)
    log_fitness(evaluate_method, "result/proposal/graph/"+evaluate_method+"/"+evaluate_method+"_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(PROPOSAL_POPULATION_SIZE)+"_"+str(times)+"_best_fitness_history.png", best_fitness_history, average_fitness_history=average_fitness_history)
    return best_fitness_history
