from typing import List
from ..core.geneticAlgorithm.selection import tournament
from ..core.geneticAlgorithm.make_chromosome_params import make_chromosome_params
from ..core.geneticAlgorithm import BLX_alpha
from ..core.geneticAlgorithm.repair import repair_fm_params as repair_gene
from ..core.geneticAlgorithm.mutate import mutate 
from ..core.geneticAlgorithm import make_chromosome_params
from ..core.geneticAlgorithm.interpolation import interpolate_fitness_by_distance
from ..core.geneticAlgorithm.pre_selection import select_top_individuals_by_fitness
from ..core.log import log
from ..engine import evaluate
import uuid


Chromosomes = List[dict]

NUM_GENERATIONS = 10
POPULATION_SIZE = 10

def make_initial_population(num_individuals=10):
    return [make_chromosome_params.make_chromosome_params() for _ in range(num_individuals)]

def run_simulation_normal_IGA():
    # 1. 初期個体生成
    population = make_initial_population(POPULATION_SIZE)
    
    for generation in range(NUM_GENERATIONS):
        # 2. 評価
        evaluate.evaluate_fitness_random(population)
        best, worst = evaluate.get_best_and_worst_individuals(population)
        print(f"Generation {generation + 1}\n \t Best fitness = {best['fitness']}\n \tWorst fitness = {worst['fitness']}")

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
        evaluate.evaluate_fitness_by_param(population)

    # 6. 最終結果の出力
    log("result/random/simulation_random_results.json", population)
    return population

PROPOSAL_POPULATION_SIZE = 100
EVALUATE_SIZE = 10
def run_simulation_proposal_IGA():
    # 1. 初期個体生成
    population = make_initial_population(PROPOSAL_POPULATION_SIZE)
    # 初期個体の事前評価
    interpolate_fitness_by_distance(population)
    # 評価個体の選択
    evaluate_population = select_top_individuals_by_fitness(population, top_n=EVALUATE_SIZE)

    for generation in range(NUM_GENERATIONS):
        # 2. 評価
        fitnesses = evaluate.evaluate_fitness_random(population)
        best, worst = evaluate.get_best_and_worst_individuals(population)
        print(f"Generation {generation + 1}\n \t Best fitness = {best['fitness']}\n \tWorst fitness = {worst['fitness']}")

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
        evaluate.evaluate_fitness_by_param(population)
        # 事前評価の補間
        interpolate_fitness_by_distance(population, best, worst)
        # 評価個体の選択
        evaluate_population = select_top_individuals_by_fitness(population, top_n=EVALUATE_SIZE)

    # 6. 最終結果の出力
    log("result/random/simulation_proposal_results.json", population)
    return population

print(f"IGAシミュレーション\n1: 普通のIGAシミュレーション\n2: 提案型IGAシミュレーション")
choice = input("実行するシミュレーションを選択 (1/2): ")
if choice == "2":
    print("提案型IGAシミュレーションを実行")
    run_simulation_proposal_IGA()
    print("提案型IGAシミュレーションが完了")
else:
    print("普通のIGAシミュレーションを実行")
    run_simulation_normal_IGA()
    print("普通のIGAシミュレーションが完了")