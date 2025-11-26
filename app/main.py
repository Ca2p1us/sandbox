from backend.engine import run_iga_simulation as iga
from backend.core.log import sound_check, log_fitness, log_fitness_histories
import numpy as np


NUM_GENERATIONS = 9
POPULATION_SIZE = 200
PROPOSAL_POPULATION_SIZE = 200
EVALUATE_SIZE = 9
EXPERIMENT_TIMES = 5


best_fitness_histories = []
best_fitness_histories_few = []
best_fitness_histories_many = []
average_fitness_histories = []
average_fitness_histories_few = []
average_fitness_histories_many = []
noise_is_added = False
look = False
interpolate_num = 100
print(f"IGAシミュレーション\n1: 普通のIGAシミュレーション\n2: 提案型IGAシミュレーション\n3: 3種比較")
choice = input("実行するシミュレーションを選択 (1/3): ")
if choice == "2":
    print(f"IGAシミュレーションの評価関数を選択\n1: ガウス関数\n2: スフィア関数\n3: Rastrigin関数\n4: Ackley関数\n5: Schwefel関数")
    evaluate_num = input("評価関数の番号を入力してください: ")
    print(f"補間方法を選択してください。\n0: 距離に基づく線形補間\n1: ガウス関数に基づく補間\n2: RBF補間")
    interpolate_num = input("補間方法の番号を入力してください: ")
    print(f"ノイズを追加しますか？\n0: 追加しない\n1: 追加する")
    TF = input("ノイズを追加しますか？ (0/1): ")
    if TF == "1":
        noise_is_added = True
    print(f"途中経過を見ますか？\n0: 見ない\n1: 見る")
    TF2 = input("途中経過を見ますか？ (0/1): ")
    if TF2 == "1":
        look = True
    for i in range(EXPERIMENT_TIMES):
        print(f"評価関数 {i}: {evaluate_num}")
        print("提案型IGAシミュレーション"+str(i+1)+"回目を実行")
        best_fitness, average_fitness = iga.run_simulation_proposal_IGA(NUM_GENERATIONS=NUM_GENERATIONS, PROPOSAL_POPULATION_SIZE=PROPOSAL_POPULATION_SIZE, EVALUATE_SIZE=EVALUATE_SIZE, evaluate_num = int(evaluate_num), interpolate_num = int(interpolate_num), times = i+1, noise_is_added=noise_is_added, look=look)
        best_fitness_histories.append(best_fitness)
        average_fitness_histories.append(average_fitness)
        print("提案型IGAシミュレーション"+str(i+1)+"回目が完了")
    log_fitness_histories(int(evaluate_num), int(interpolate_num), "_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(PROPOSAL_POPULATION_SIZE)+"_"+str(EVALUATE_SIZE)+"eval_best_fitness_histories.png", best_fitness_histories, ver="proposal")

elif choice == "1":
    print(f"IGAシミュレーションの評価関数を選択\n1: ガウス関数\n2: スフィア関数\n3: Rastrigin関数\n4: Ackley関数\n5: Schwefel関数")
    evaluate_num = input("評価関数の番号を入力してください: ")
    print(f"ノイズを追加しますか？\n0: 追加しない\n1: 追加する")
    TF = input("ノイズを追加しますか？ (0/1): ")
    if TF == "1":
        noise_is_added = True
    print(f"途中経過を見ますか？\n0: 見ない\n1: 見る")
    TF2 = input("途中経過を見ますか？ (0/1): ")
    if TF2 == "1":
        look = True
    for i in range(EXPERIMENT_TIMES):
        print("普通のIGAシミュレーション"+str(i+1)+"回目を実行")
        best_fitness, average_fitness = iga.run_simulation_normal_IGA(NUM_GENERATIONS=NUM_GENERATIONS, POPULATION_SIZE=POPULATION_SIZE, evaluate_num = int(evaluate_num), times = i+1, noise_is_added=noise_is_added, look=look)
        best_fitness_histories.append(best_fitness)
        average_fitness_histories.append(average_fitness)
        print("普通のIGAシミュレーション"+str(i+1)+"回目が完了")
    log_fitness_histories(int(evaluate_num), int(interpolate_num), "_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(POPULATION_SIZE)+"_best_fitness_histories.png", best_fitness_histories, ver="conventional")

elif choice == "3":
    population_size = int(input(f"個体群サイズを入力してください\n個体群サイズ: "))
    if population_size <= 0:
        population_size = POPULATION_SIZE
    evaluate_size = int(input(f"評価個体数を入力してください\n評価個体数: "))
    if evaluate_size <= 0:
        evaluate_size = EVALUATE_SIZE
    print(f"IGAシミュレーションの評価関数を選択\n1: ガウス関数\n2: スフィア関数\n3: Rastrigin関数\n4: Ackley関数\n5: Schwefel関数")
    evaluate_num = input("評価関数の番号を入力してください: ")
    print(f"補間方法を選択してください。\n0: 距離に基づく線形補間\n1: ガウス関数に基づく補間\n2: RBF補間")
    interpolate_num = input("補間方法の番号を入力してください: ")
    print(f"ノイズを追加しますか？\n0: 追加しない\n1: 追加する")
    TF = input("ノイズを追加しますか？ (0/1): ")
    if TF == "1":
        noise_is_added = True
    print(f"途中経過を見ますか？\n0: 見ない\n1: 見る")
    TF2 = input("途中経過を見ますか？ (0/1): ")
    if TF2 == "1":
        look = True
    for i in range(EXPERIMENT_TIMES):
        print(f"{evaluate_num}個体のIGAシミュレーション"+str(i+1)+"回目を実行")
        best_fitness, average_fitness = iga.run_simulation_normal_IGA(NUM_GENERATIONS=NUM_GENERATIONS, POPULATION_SIZE=evaluate_size, evaluate_num = int(evaluate_num), times = i+1, noise_is_added=noise_is_added, look=look)
        best_fitness_histories_few.append(best_fitness)
        average_fitness_histories_few.append(average_fitness)
        print(f"{evaluate_num}個体のIGAシミュレーション"+str(i+1)+"回目が完了")
    log_fitness_histories(int(evaluate_num), 100, "_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(evaluate_size)+"_best_fitness_histories.png", best_fitness_histories_few, ver="conventional")
    log_fitness("_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(evaluate_size)+"_average_fitness_histories.png", best_fitness_history= best_fitness_histories_few, average_fitness_history=average_fitness_histories_few,evaluate_num=int(evaluate_num),ver="conventional")
    for i in range(EXPERIMENT_TIMES):
        print(f"{population_size}個体のIGAシミュレーション"+str(i+1)+"回目を実行")
        best_fitness, average_fitness = iga.run_simulation_normal_IGA(NUM_GENERATIONS=NUM_GENERATIONS, POPULATION_SIZE=population_size, evaluate_num = int(evaluate_num), times = i+1, noise_is_added=noise_is_added, look=look)
        best_fitness_histories_many.append(best_fitness)
        average_fitness_histories_many.append(average_fitness)
        print(f"{population_size}個体のIGAシミュレーション"+str(i+1)+"回目が完了")
    log_fitness_histories(int(evaluate_num), 100, "_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_best_fitness_histories.png", best_fitness_histories_many, ver="conventional")
    log_fitness("_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_average_fitness_histories.png", best_fitness_history= best_fitness_histories_many, average_fitness_history=average_fitness_histories_many,evaluate_num=int(evaluate_num),ver="conventional")
    for  i in range(EXPERIMENT_TIMES):
        print(f"提案型IGAシミュレーション"+str(i+1)+"回目を実行")
        best_fitness, average_fitness = iga.run_simulation_proposal_IGA(NUM_GENERATIONS=NUM_GENERATIONS, PROPOSAL_POPULATION_SIZE=population_size, EVALUATE_SIZE=evaluate_size, evaluate_num = int(evaluate_num), interpolate_num = int(interpolate_num), times = i+1, noise_is_added=noise_is_added, look=look)
        best_fitness_histories.append(best_fitness)
        average_fitness_histories.append(average_fitness)
        print(f"提案型IGAシミュレーション"+str(i+1)+"回目が完了")
    best_fitness_histories = np.mean(best_fitness_histories, axis=0)
    best_fitness_histories = [tuple(row) for row in best_fitness_histories]
    average_fitness_histories = np.mean(average_fitness_histories, axis=0)
    average_fitness_histories = [tuple(row) for row in average_fitness_histories]
    print(best_fitness_histories)
    print(average_fitness_histories)
    log_fitness_histories(int(evaluate_num), int(interpolate_num), "_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_"+str(evaluate_size)+"eval_best_fitness_histories.png", best_fitness_histories, ver="proposal")
    log_fitness("_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_"+str(evaluate_size)+"eval_average_fitness_histories.png", best_fitness_history= best_fitness_histories, average_fitness_history=average_fitness_histories,evaluate_num=int(evaluate_num), interpolate_num=int(interpolate_num), ver="proposal")