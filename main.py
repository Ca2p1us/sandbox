from backend.engine import run_iga_simulation as iga
from backend.core.log import log_fitness, log_fitness_histories, log_comparison, log_error_history, log_compare, log_fitness_variance, log_distance_history, log_distance_history_json, log_four_metrics, log_four_metrics_comparison
import numpy as np
from backend.core.geneticAlgorithm.config import NUM_GENERATIONS, POPULATION_SIZE, PROPOSAL_POPULATION_SIZE, EVALUATE_SIZE, EXPERIMENT_TIMES



best_fitness_histories = []
best_fitness_histories_few = []
best_fitness_histories_many = []
best_fitness_histories_benchmark = []
average_fitness_histories = []
average_fitness_histories_few = []
average_fitness_histories_many = []
average_fitness_histories_benchmark = []
best_fitness_histories_saf = []
average_fitness_histories_saf = []
noise_is_added = False
look = False
interpolate_num = 100
print(f"IGAシミュレーション\n1: 普通のIGAシミュレーション\n2: 提案型IGAシミュレーション\n3: 比較\n4: トーナメントサイズの比較\n5: 個体数の比較\n6: SAF-IEDAシミュレーション\n7: 重み付きサロゲートモデル比較シミュレーション\n8: 4つの評価関数シミュレーション")
choice = input("実行するシミュレーションを選択 (1/7): ")
if choice == "2":
    print(f"IGAシミュレーションの評価関数を選択\n1: ガウス関数\n2: スフィア関数\n3: Gauss関数+cos関数\n4: Ackley関数\n5: Gaussian_peaks関数\n6: Mixed関数")
    evaluate_num = input("評価関数の番号を入力してください: ")
    print(f"補間方法を選択してください。\n0: 距離に基づく線形補間\n1: ガウス関数補間\n2: TPS補間(距離項なし)\n3: IDW補間\n4: TPS補間\n5: Gaussian_RBF補間\n6: IMQ補間")
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
        print("提案型IGAシミュレーション"+str(i+1)+"回目を実行")
        best_fitness, average_fitness, error_history, distance_history = iga.run_simulation_proposal_IGA(NUM_GENERATIONS=NUM_GENERATIONS, PROPOSAL_POPULATION_SIZE=PROPOSAL_POPULATION_SIZE, EVALUATE_SIZE=EVALUATE_SIZE, evaluate_num = int(evaluate_num), interpolate_num = int(interpolate_num), times = i+1, noise_is_added=noise_is_added, look=look, tournament_size=4)
        best_fitness_histories.append(best_fitness)
        average_fitness_histories.append(average_fitness)
        print("提案型IGAシミュレーション"+str(i+1)+"回目が完了")
    log_fitness_histories(
        evaluate_num=int(evaluate_num),
        interpolate_num=int(interpolate_num),
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(PROPOSAL_POPULATION_SIZE)+"_"+str(EVALUATE_SIZE)+"eval_best_fitness_histories.pdf",
        best_fitness_histories=best_fitness_histories,
        ver="proposal"
    )
    log_fitness_variance(
        evaluate_num=int(evaluate_num),
        interpolate_num=int(interpolate_num),
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(PROPOSAL_POPULATION_SIZE)+"_"+str(EVALUATE_SIZE)+"eval_fitness_variance.pdf",
        best_fitness_histories=best_fitness_histories,
        ver="proposal"
    )

elif choice == "1":
    print(f"IGAシミュレーションの評価関数を選択\n1: ガウス関数\n2: スフィア関数\n3: Gauss関数+cos関数\n4: Ackley関数\n5: Gaussian_peaks関数\n6: Mixed関数")
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
        best_fitness, average_fitness = iga.run_simulation_normal_IGA(NUM_GENERATIONS=NUM_GENERATIONS, POPULATION_SIZE=POPULATION_SIZE, evaluate_num = int(evaluate_num), times = i+1, noise_is_added=noise_is_added, look=look, tournament_size=4)
        best_fitness_histories.append(best_fitness)
        average_fitness_histories.append(average_fitness)
        print("普通のIGAシミュレーション"+str(i+1)+"回目が完了")
    log_fitness_histories(
        evaluate_num=int(evaluate_num),
        interpolate_num=100,
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(POPULATION_SIZE)+"_best_fitness_histories.pdf",
        best_fitness_histories=best_fitness_histories,
        ver="conventional"
    )
    log_fitness_variance(
        evaluate_num=int(evaluate_num),
        interpolate_num=int(interpolate_num),
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(PROPOSAL_POPULATION_SIZE)+"_"+str(EVALUATE_SIZE)+"eval_fitness_variance.pdf",
        best_fitness_histories=best_fitness_histories,
        ver="proposal"
    )

elif choice == "3":
    population_size = int(input(f"個体群サイズを入力してください\n個体群サイズ: "))
    if population_size <= 0 or population_size == None:
        population_size = PROPOSAL_POPULATION_SIZE
    evaluate_size = int(input(f"評価個体数を入力してください\n評価個体数: "))
    if evaluate_size <= 0 or evaluate_size == None:
        evaluate_size = EVALUATE_SIZE
    print(f"IGAシミュレーションの評価関数を選択\n1: ガウス関数\n2: スフィア関数\n3: Gauss関数+cos関数\n4: Ackley関数\n5: Gaussian_peaks関数\n6: Mixed関数")
    evaluate_num = input("評価関数の番号を入力してください: ")
    print(f"補間方法を選択してください。\n0: 距離に基づく線形補間\n1: ガウス関数に基づく補間\n2: TPS補間(距離項なし)\n3: IDW補間\n4: TPS補間\n5: Gaussian_RBF補間\n6: IMQ補間")
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
        print(f"{evaluate_size}個体のIGAシミュレーション"+str(i+1)+"回目を実行")
        best_fitness, average_fitness = iga.run_simulation_normal_IGA(NUM_GENERATIONS=NUM_GENERATIONS, POPULATION_SIZE=evaluate_size, evaluate_num = int(evaluate_num), times = i+1, noise_is_added=noise_is_added, look=look, tournament_size=4)
        best_fitness_histories_few.append(best_fitness)
        average_fitness_histories_few.append(average_fitness)
        print(f"{evaluate_size}個体のIGAシミュレーション"+str(i+1)+"回目が完了")
    best_fitness_histories_few_ave = np.mean(best_fitness_histories_few, axis=0)
    best_fitness_histories_few_ave = [tuple(row) for row in best_fitness_histories_few_ave]
    average_fitness_histories_few_ave = np.mean(average_fitness_histories_few, axis=0)
    average_fitness_histories_few_ave = [tuple(row) for row in average_fitness_histories_few_ave]
    log_fitness_histories(
        evaluate_num=int(evaluate_num),
        interpolate_num=100,
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(evaluate_size)+"_best_fitness_histories.pdf",
        best_fitness_histories=best_fitness_histories_few,
        ver="conventional")
    log_fitness(file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(evaluate_size)+"_average_fitness_histories.pdf", best_fitness_history= best_fitness_histories_few_ave, average_fitness_history=average_fitness_histories_few_ave,evaluate_num=int(evaluate_num),ver="conventional")
    for i in range(EXPERIMENT_TIMES):
        print(f"{population_size}個体のIGAシミュレーション"+str(i+1)+"回目を実行")
        best_fitness, average_fitness = iga.run_simulation_normal_IGA(NUM_GENERATIONS=NUM_GENERATIONS, POPULATION_SIZE=population_size, evaluate_num = int(evaluate_num), times = i+1, noise_is_added=noise_is_added, look=look,tournament_size=4)
        best_fitness_histories_many.append(best_fitness)
        average_fitness_histories_many.append(average_fitness)
        print(f"{population_size}個体のIGAシミュレーション"+str(i+1)+"回目が完了")
    best_fitness_histories_many_ave = np.mean(best_fitness_histories_many, axis=0)
    best_fitness_histories_many_ave = [tuple(row) for row in best_fitness_histories_many_ave]
    average_fitness_histories_many_ave = np.mean(average_fitness_histories_many, axis=0)
    average_fitness_histories_many_ave = [tuple(row) for row in average_fitness_histories_many_ave]
    log_fitness_histories(
        evaluate_num=int(evaluate_num),
        interpolate_num=100,
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_best_fitness_histories.pdf",
        best_fitness_histories=best_fitness_histories_many,
        ver="conventional")
    log_fitness(file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_average_fitness_histories.pdf", best_fitness_history= best_fitness_histories_many_ave, average_fitness_history=average_fitness_histories_many_ave,evaluate_num=int(evaluate_num),ver="conventional")
    for  i in range(EXPERIMENT_TIMES):
        print(f"提案型IGAシミュレーション"+str(i+1)+"回目を実行")
        best_fitness, average_fitness, error_history, distance_history = iga.run_simulation_proposal_IGA(NUM_GENERATIONS=NUM_GENERATIONS, PROPOSAL_POPULATION_SIZE=population_size, EVALUATE_SIZE=evaluate_size, evaluate_num = int(evaluate_num), interpolate_num = int(interpolate_num), times = i+1, noise_is_added=noise_is_added, look=look, tournament_size=4)
        best_fitness_histories.append(best_fitness)
        average_fitness_histories.append(average_fitness)
        log_distance_history_json(
            evaluate_num=int(evaluate_num),
            interpolate_num=int(interpolate_num),
            file_path=f"_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(population_size)}_{str(evaluate_size)}eval_{str(i+1)}_distance_history.json",
            distance_history=distance_history,
            times=i+1,
            ver="proposal"
        )
        print(f"提案型IGAシミュレーション"+str(i+1)+"回目が完了")
    best_fitness_histories_ave = np.mean(best_fitness_histories, axis=0)
    best_fitness_histories_ave = [tuple(row) for row in best_fitness_histories_ave]
    average_fitness_histories_ave = np.mean(average_fitness_histories, axis=0)
    average_fitness_histories_ave = [tuple(row) for row in average_fitness_histories_ave]
    log_fitness_histories(
        evaluate_num=int(evaluate_num),
        interpolate_num=int(interpolate_num),
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_"+str(evaluate_size)+"eval_best_fitness_histories.pdf",
        best_fitness_histories=best_fitness_histories,
        ver="proposal"
    )
    benchmark_population_size = 50
    for i in range(EXPERIMENT_TIMES):
        print(f"距離項なしサロゲート"+str(i+1)+"回目を実行")
        best_fitness, average_fitness, error_history, distance_history = iga.run_simulation_proposal_IGA(NUM_GENERATIONS=NUM_GENERATIONS, PROPOSAL_POPULATION_SIZE=population_size, EVALUATE_SIZE=evaluate_size, evaluate_num = int(evaluate_num), interpolate_num = 2, times = i+1, noise_is_added=noise_is_added, look=look, tournament_size=4)
        best_fitness_histories_benchmark.append(best_fitness)
        average_fitness_histories_benchmark.append(average_fitness)
        log_distance_history_json(
            evaluate_num=int(evaluate_num),
            interpolate_num=2,
            file_path=f"_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(population_size)}_{str(evaluate_size)}eval_{str(i+1)}_distance_history.json",
            distance_history=distance_history,
            times=i+1,
            ver="proposal"
        )
        print(f"距離項なしサロゲート"+str(i+1)+"回目が完了")
    best_fitness_histories_benchmark = np.mean(best_fitness_histories_benchmark, axis=0)
    best_fitness_histories_benchmark = [tuple(row) for row in best_fitness_histories_benchmark]
    average_fitness_histories_benchmark = np.mean(average_fitness_histories_benchmark,axis=0)
    average_fitness_histories_benchmark = [tuple(row) for row in average_fitness_histories_benchmark]

    # 4. SAF-IEDA (追加部分)
    for i in range(EXPERIMENT_TIMES):
        print(f"SAF-IEDAシミュレーション"+str(i+1)+"回目を実行")
        # 比較のため、Top-Nc(真の評価数)には evaluate_size を使用
        best_fitness, average_fitness = iga.run_simulation_SAF_IEDA(
            NUM_GENERATIONS=NUM_GENERATIONS,
            POPULATION_SIZE=population_size,
            TOP_NC=evaluate_size, # 評価コストを揃えるため
            evaluate_num=int(evaluate_num),
            times=i+1,
            noise_is_added=noise_is_added,
            look=look,
            tournament_size=4
        )
        best_fitness_histories_saf.append(best_fitness)
        average_fitness_histories_saf.append(average_fitness)
        print(f"SAF-IEDAシミュレーション"+str(i+1)+"回目が完了")
    
    # SAF-IEDAの平均計算
    best_fitness_histories_saf_ave = np.mean(best_fitness_histories_saf, axis=0)
    best_fitness_histories_saf_ave = [tuple(row) for row in best_fitness_histories_saf_ave]
    average_fitness_histories_saf_ave = np.mean(average_fitness_histories_saf, axis=0)
    average_fitness_histories_saf_ave = [tuple(row) for row in average_fitness_histories_saf_ave]

    # 個別の履歴保存 (SAF)
    log_fitness_histories(
        evaluate_num=int(evaluate_num),
        interpolate_num=100, 
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_SAF_best_fitness_histories.pdf",
        best_fitness_histories=best_fitness_histories_saf,
        ver="saf_ieda"
    )

    best_fitness_list = [
    {
        'label': '提案手法', 
        'data': best_fitness_histories_ave, 
        'marker': 'o', 
        'linestyle': '-'
    },
    {
        'label': '距離項なしサロゲート',
        'data': best_fitness_histories_benchmark,
        'marker': 'x',
        'linestyle': '-.'
    },
    {
        'label': 'GA(9個体)', 
        'data': best_fitness_histories_few_ave, 
        'marker': 'o', 
        'linestyle': '--'
    },
    {
        'label': 'GA(200個体)', 
        'data': best_fitness_histories_many_ave, 
        'marker': '^', 
        'linestyle': ':'
    },
    {
        'label': 'SAF-IEDA', 
        'data': best_fitness_histories_saf_ave, 
        'marker': 's', 
        'linestyle': '-.'
    },
    # # 追加のデータも辞書を足すだけ
    # {
    #     'label': '参考データ',
    #     'data': data_new,
    #     'marker': 'x',
    #     'linestyle': '-.'
    # }
    ]
    ave_fitness_list = [
    {
        'label': '提案手法', 
        'data': average_fitness_histories_ave, 
        'marker': 'o', 
        'linestyle': '-'
    },
    {
        'label': '距離項なしサロゲート',
        'data': average_fitness_histories_benchmark,
        'marker': 'x',
        'linestyle': '-.'
    },
    {
        'label': 'GA(9個体)', 
        'data': average_fitness_histories_few_ave, 
        'marker': 'o', 
        'linestyle': '--'
    },
    {
        'label': 'GA(200個体)', 
        'data': average_fitness_histories_many_ave, 
        'marker': '^', 
        'linestyle': ':'
    },
    {
        'label': 'SAF-IEDA', 
        'data': average_fitness_histories_saf_ave, 
        'marker': 's', 
        'linestyle': '-.'
    },
    # # 追加のデータも辞書を足すだけ
    # {
    #     'label': '参考データ',
    #     'data': data_new,
    #     'marker': 'x',
    #     'linestyle': '-.'
    # }
]

    log_fitness(
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_"+str(evaluate_size)+"eval_average_fitness_histories.pdf",
        best_fitness_history= best_fitness_histories_ave,
        average_fitness_history=average_fitness_histories_ave,
        evaluate_num=int(evaluate_num),
        interpolate_num=int(interpolate_num),
        ver="proposal"
    )
    log_error_history(
        evaluate_num=int(evaluate_num),
        interpolate_num=int(interpolate_num),
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_"+str(evaluate_size)+"eval_error_history.pdf",
        error_history= error_history,
        ver="proposal"
    )
    log_comparison(
        evaluate_num=int(evaluate_num),
        interpolate_num=int(interpolate_num),
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_"+str(evaluate_size)+"eval_comparison.pdf",
        plot_series_list=best_fitness_list,
        indicator="最大",
        ver="comparison"
    )
    log_comparison(
        evaluate_num=int(evaluate_num),
        interpolate_num=int(interpolate_num),
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_"+str(evaluate_size)+"eval_comparison_average.pdf",
        plot_series_list=ave_fitness_list,
        indicator="平均",
        ver="comparison"
    )
    log_fitness_variance(
        evaluate_num=int(evaluate_num),
        interpolate_num=int(interpolate_num),
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(PROPOSAL_POPULATION_SIZE)+"_"+str(EVALUATE_SIZE)+"eval_fitness_variance.pdf",
        best_fitness_histories=best_fitness_histories,
        ver="proposal"
    )

elif choice == "4":
    proposal_or_conventional = input("提案型IGAを実行する場合は1、普通のIGAを実行する場合は0を入力してください (1/0): ")
    population_size = int(input(f"個体群サイズを入力してください\n個体群サイズ: "))
    if population_size <= 0 or population_size == None:
        population_size = PROPOSAL_POPULATION_SIZE
    evaluate_size = int(input(f"評価個体数を入力してください\n評価個体数: "))
    if evaluate_size <= 0 or evaluate_size == None:
        evaluate_size = EVALUATE_SIZE
    print(f"IGAシミュレーションの評価関数を選択\n1: ガウス関数\n2: スフィア関数\n3: Gauss関数+cos関数\n4: Ackley関数\n5: Gaussian_peaks関数\n6: Mixed関数")
    evaluate_num = input("評価関数の番号を入力してください: ")
    print(f"補間方法を選択してください。\n0: 距離に基づく線形補間\n1: ガウス関数に基づく補間\n2: TPS補間(距離項なし)\n3: IDW補間\n4: TPS補間\n5: Gaussian_RBF補間\n6: IMQ補間")
    interpolate_num = input("補間方法の番号を入力してください: ")
    print(f"ノイズを追加しますか？\n0: 追加しない\n1: 追加する")
    TF = input("ノイズを追加しますか？ (0/1): ")
    if TF == "1":
        noise_is_added = True
    print(f"途中経過を見ますか？\n0: 見ない\n1: 見る")
    TF2 = input("途中経過を見ますか？ (0/1): ")
    if TF2 == "1":
        look = True
    # tornament_sizes = [2,4,6,8,10,20,30,40]
    tornament_sizes = [2,3,4,5,6]
    best_fitness_histories_all = []
    for ts in tornament_sizes:
        best_fitness_history = []
        for i in range(EXPERIMENT_TIMES):
            if proposal_or_conventional == "0":
                print(f"普通のIGAシミュレーション トーナメントサイズ {ts} "+str(i+1)+"回目を実行")
                best_fitness, average_fitness = iga.run_simulation_normal_IGA(NUM_GENERATIONS=NUM_GENERATIONS, POPULATION_SIZE=population_size, evaluate_num = int(evaluate_num), times = i+1, noise_is_added=noise_is_added, look=look, tournament_size=ts)
            elif proposal_or_conventional == "1":
                print(f"提案型IGAシミュレーション トーナメントサイズ {ts} "+str(i+1)+"回目を実行")
                best_fitness, average_fitness, error_history, _ = iga.run_simulation_proposal_IGA(NUM_GENERATIONS=NUM_GENERATIONS, PROPOSAL_POPULATION_SIZE=population_size, EVALUATE_SIZE=evaluate_size, evaluate_num = int(evaluate_num), interpolate_num = int(interpolate_num), times = i+1, noise_is_added=noise_is_added, look=look, tournament_size=ts)
            best_fitness_history.append(best_fitness)
        best_fitness_histories_all.append(np.mean(best_fitness_history, axis=0))
    log_compare(
        evaluate_num=int(evaluate_num),
        interpolate_num=int(interpolate_num),
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_"+str(evaluate_size)+"eval_tornament_size_comparison.pdf",
        fitness_histories=best_fitness_histories_all,
        tornament_sizes=tornament_sizes,
        population_size=population_size,
    )
elif choice == "5":
    proposal_or_conventional = input("提案型IGAを実行する場合は1、普通のIGAを実行する場合は0を入力してください (1/0): ")
    print(f"IGAシミュレーションの評価関数を選択\n1: ガウス関数\n2: スフィア関数\n3: Gauss関数+cos関数\n4: Ackley関数\n5: Gaussian_peaks関数\n6: Mixed関数")
    evaluate_num = input("評価関数の番号を入力してください: ")
    
    if proposal_or_conventional == "1":
        print(f"補間方法を選択してください。\n0: 距離に基づく線形補間\n1: ガウス関数に基づく補間\n2: TPS補間(距離項なし)\n3: IDW補間\n4: TPS補間\n5: Gaussian_RBF補間\n6: IMQ補間")
        interpolate_num = input("補間方法の番号を入力してください: ")
        evaluate_size = int(input(f"評価個体数を入力してください\n評価個体数: "))
        if evaluate_size <= 0 or evaluate_size == None:
            evaluate_size = EVALUATE_SIZE
    else:
        interpolate_num = 100 # ダミー値
        evaluate_size = 0 # 使用しない

    print(f"ノイズを追加しますか？\n0: 追加しない\n1: 追加する")
    TF = input("ノイズを追加しますか？ (0/1): ")
    if TF == "1":
        noise_is_added = True
    print(f"途中経過を見ますか？\n0: 見ない\n1: 見る")
    TF2 = input("途中経過を見ますか？ (0/1): ")
    if TF2 == "1":
        look = True

    # 比較する個体数のリスト（適宜変更可能）
    test_population_sizes = [9, 20, 50, 100, 200]
    
    plot_series_list = []
    markers = ['o', '^', 's', 'D', 'x', '*', 'v', '<']
    
    for idx, pop_size in enumerate(test_population_sizes):
        best_fitness_history_for_size = []
        average_fitness_history_for_size = []
        
        for i in range(EXPERIMENT_TIMES):
            if proposal_or_conventional == "0":
                print(f"普通のIGAシミュレーション 個体数 {pop_size} "+str(i+1)+"回目を実行")
                best_fitness, average_fitness = iga.run_simulation_normal_IGA(
                    NUM_GENERATIONS=NUM_GENERATIONS, 
                    POPULATION_SIZE=pop_size, 
                    evaluate_num=int(evaluate_num), 
                    times=i+1, 
                    noise_is_added=noise_is_added, 
                    look=look, 
                    tournament_size=4 # デフォルトまたは適当な値
                )
            elif proposal_or_conventional == "1":
                print(f"提案型IGAシミュレーション 総個体数 {pop_size} "+str(i+1)+"回目を実行")
                # 提案型の場合、総個体数をpop_sizeに変更
                best_fitness, average_fitness, error_history, _ = iga.run_simulation_proposal_IGA(
                    NUM_GENERATIONS=NUM_GENERATIONS, 
                    PROPOSAL_POPULATION_SIZE=pop_size, 
                    EVALUATE_SIZE=evaluate_size, 
                    evaluate_num=int(evaluate_num), 
                    interpolate_num=int(interpolate_num), 
                    times=i+1, 
                    noise_is_added=noise_is_added, 
                    look=look, 
                    tournament_size=4
                )
            best_fitness_history_for_size.append(best_fitness)
            average_fitness_history_for_size.append(average_fitness)

        # 平均を計算
        avg_best_fitness = np.mean(best_fitness_history_for_size, axis=0)
        avg_best_fitness = [tuple(row) for row in avg_best_fitness]

        # プロット用データに追加
        plot_series_list.append({
            'label': f'Pop Size {pop_size}',
            'data': avg_best_fitness,
            'marker': markers[idx % len(markers)],
            'linestyle': '-'
        })

    # 結果の保存・描画
    log_comparison(
        evaluate_num=int(evaluate_num),
        interpolate_num=int(interpolate_num),
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_pop_size_comparison.pdf",
        plot_series_list=plot_series_list,
        indicator="Best Fitness by Pop Size ",
        ver="benchmark"
    )
    print("個体数比較シミュレーションが完了しました。")
elif choice == "6":
    print(f"SAF-IEDAシミュレーションを開始します")
    
    # パラメータ入力
    population_size_input = input(f"個体群サイズを入力してください (default: {POPULATION_SIZE}): ")
    population_size = int(population_size_input) if population_size_input else POPULATION_SIZE
    
    # Top-Nc (SAFにおける真評価個体数)
    default_top_nc = 5
    top_nc_input = input(f"Top-Nc (評価個体数) を入力してください (default: {default_top_nc}): ")
    top_nc = int(top_nc_input) if top_nc_input else default_top_nc

    print(f"IGAシミュレーションの評価関数を選択\n1: ガウス関数\n2: スフィア関数\n3: Gauss関数+cos関数\n4: Ackley関数\n5: Gaussian_peaks関数\n6: Mixed関数")
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
        print("SAF-IEDAシミュレーション"+str(i+1)+"回目を実行")
        
        best_fitness, average_fitness = iga.run_simulation_SAF_IEDA(
            NUM_GENERATIONS=NUM_GENERATIONS,
            POPULATION_SIZE=population_size,
            TOP_NC=top_nc,
            evaluate_num=int(evaluate_num),
            times=i+1,
            noise_is_added=noise_is_added,
            look=look,
            tournament_size=4
        )
        best_fitness_histories.append(best_fitness)
        average_fitness_histories.append(average_fitness)
        print("SAF-IEDAシミュレーション"+str(i+1)+"回目が完了")
        
    # 結果の出力
    log_fitness_histories(
        evaluate_num=int(evaluate_num),
        interpolate_num=100, # SAFでは補間番号はダミーで100などにしておくか、適宜調整
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_SAF_IEDA_best_fitness_histories.pdf",
        best_fitness_histories=best_fitness_histories,
        ver="saf_ieda"
    )
    # 分散などの出力が必要であれば以下も追加
    log_fitness_variance(
        evaluate_num=int(evaluate_num),
        interpolate_num=100,
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_SAF_IEDA_fitness_variance.pdf",
        best_fitness_histories=best_fitness_histories,
        ver="saf_ieda"
    )
elif choice == "7":
    print("重み付きサロゲートモデル比較シミュレーション")
    population_size_input = input(f"個体群サイズを入力してください (default: {PROPOSAL_POPULATION_SIZE}): ")
    population_size = int(population_size_input) if population_size_input else PROPOSAL_POPULATION_SIZE

    evaluate_size_input = input(f"評価個体数を入力してください (default: {EVALUATE_SIZE}): ")
    evaluate_size = int(evaluate_size_input) if evaluate_size_input else EVALUATE_SIZE

    print(f"IGAシミュレーションの評価関数を選択\n1: ガウス関数\n2: スフィア関数\n3: Gauss関数+cos関数\n4: Ackley関数\n5: Gaussian_peaks関数\n6: Mixed関数")
    evaluate_num = input("評価関数の番号を入力してください: ")
    print(f"補間方法を選択してください。\n0: 距離に基づく線形補間\n1: ガウス関数に基づく補間\n2: TPS補間(距離項なし)\n3: IDW補間\n4: TPS補間\n5: Gaussian_RBF補間\n6: IMQ補間")
    interpolate_num = input("補間方法の番号を入力してください: ")
    
    print(f"ノイズを追加しますか？\n0: 追加しない\n1: 追加する")
    TF = input("ノイズを追加しますか？ (0/1): ")
    if TF == "1":
        noise_is_added = True
    print(f"途中経過を見ますか？\n0: 見ない\n1: 見る")
    TF2 = input("途中経過を見ますか？ (0/1): ")
    if TF2 == "1":
        look = True

    # 比較する重み w のリスト
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plot_series_list = []
    markers = ['o', '^', 's', 'D', 'x', '*', 'v', '<', '>', 'p', 'h']

    for idx, w in enumerate(weights):
        best_fitness_histories_w = []
        print(f"重み w={w} のシミュレーションを実行中...")
        for i in range(EXPERIMENT_TIMES):
            print(f"  w={w} : {i+1}回目")
            best_fitness, average_fitness, error_history, distance_history = iga.run_simulation_proposal_IGA(
                NUM_GENERATIONS=NUM_GENERATIONS, 
                PROPOSAL_POPULATION_SIZE=population_size, 
                EVALUATE_SIZE=evaluate_size, 
                evaluate_num = int(evaluate_num), 
                interpolate_num = int(interpolate_num), 
                times = i+1, 
                noise_is_added=noise_is_added, 
                look=look, 
                tournament_size=4, 
                w=w
            )
            best_fitness_histories_w.append(best_fitness)
            log_distance_history_json(
                evaluate_num=int(evaluate_num),
                interpolate_num=int(interpolate_num),
                file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_"+str(evaluate_size)+"_"+str(w)+"weight_distance_history.json",
                distance_history=distance_history,
                ver="benchmark"
            )
        
        # 平均を計算
        best_fitness_histories_w_ave = np.mean(best_fitness_histories_w, axis=0)
        best_fitness_histories_w_ave = [tuple(row) for row in best_fitness_histories_w_ave]
        
        plot_series_list.append({
            'label': f'w={w}',
            'data': best_fitness_histories_w_ave,
            'marker': markers[idx % len(markers)],
            'linestyle': '-'
        })

    # グラフの保存
    log_comparison(
        evaluate_num=int(evaluate_num),
        interpolate_num=int(interpolate_num),
        file_path="_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(population_size)+"_"+str(evaluate_size)+"eval_weight_comparison.pdf",
        plot_series_list=plot_series_list,
        indicator="Best Fitness by Weight ",
        ver="benchmark"
    )

elif choice == "8":
    print("4つの評価関数シミュレーション (比較モード)")
    
    # 共通パラメータ入力
    print(f"補間方法を選択してください。\n0: 距離に基づく線形補間\n1: ガウス関数に基づく補間\n2: TPS補間(距離項なし)\n3: IDW補間\n4: TPS補間\n5: Gaussian_RBF補間\n6: IMQ補間")
    interpolate_num_val = int(input("補間方法の番号を入力してください: "))
    
    print(f"ノイズを追加しますか？\n0: 追加しない\n1: 追加する")
    TF = input("ノイズを追加しますか？ (0/1): ")
    noise_is_added = (TF == "1")
    
    print(f"途中経過を見ますか？\n0: 見ない\n1: 見る")
    TF2 = input("途中経過を見ますか？ (0/1): ")
    look = (TF2 == "1")

    # 評価関数のリスト (IDと名前の対応)
    target_evaluations = [
        (1, "Gaussian"),
        (3, "Gaussian_cos"),
        (5, "Gaussian_peaks"), 
        (4, "Ackley")
    ]
    
    # 全関数の比較データを格納する辞書
    # key: FunctionName, value: list of series dicts
    all_comparison_data = {}

    for eval_id, eval_name in target_evaluations:
        print(f"\n==========================================")
        print(f"--- {eval_name} (ID: {eval_id}) シミュレーション開始 ---")
        print(f"==========================================")
        
        # 各手法のデータ格納用リスト
        best_fitness_histories_proposal = []
        best_fitness_histories_benchmark = []
        best_fitness_histories_few = []
        best_fitness_histories_many = []
        best_fitness_histories_saf = []
        
        # 1. Proposal Method
        print(f"[{eval_name}] 提案手法実行中...")
        for i in range(EXPERIMENT_TIMES):
            print(f"  Proposal {i+1}th run")
            best_fitness, _, _, _ = iga.run_simulation_proposal_IGA(
                NUM_GENERATIONS=NUM_GENERATIONS, 
                PROPOSAL_POPULATION_SIZE=PROPOSAL_POPULATION_SIZE, 
                EVALUATE_SIZE=EVALUATE_SIZE, 
                evaluate_num=eval_id, 
                interpolate_num=interpolate_num_val, 
                times=i+1, 
                noise_is_added=noise_is_added, 
                look=look, 
                tournament_size=4
            )
            best_fitness_histories_proposal.append(best_fitness)

        # 2. Benchmark (TPS distance-free, fixed interpolate_num=2)
        print(f"[{eval_name}] 距離項なしサロゲート実行中...")
        for i in range(EXPERIMENT_TIMES):
            print(f"  Benchmark {i+1}th run")
            best_fitness, _, _, _ = iga.run_simulation_proposal_IGA(
                NUM_GENERATIONS=NUM_GENERATIONS, 
                PROPOSAL_POPULATION_SIZE=PROPOSAL_POPULATION_SIZE, 
                EVALUATE_SIZE=EVALUATE_SIZE, 
                evaluate_num=eval_id, 
                interpolate_num=2, 
                times=i+1, 
                noise_is_added=noise_is_added, 
                look=look, 
                tournament_size=4
            )
            best_fitness_histories_benchmark.append(best_fitness)

        # 3. GA Small Population (evaluate_size)
        print(f"[{eval_name}] GA(少数個体)実行中...")
        for i in range(EXPERIMENT_TIMES):
            print(f"  GA(Few) {i+1}th run")
            best_fitness, _ = iga.run_simulation_normal_IGA(
                NUM_GENERATIONS=NUM_GENERATIONS, 
                POPULATION_SIZE=EVALUATE_SIZE, 
                evaluate_num=eval_id, 
                times=i+1, 
                noise_is_added=noise_is_added, 
                look=look, 
                tournament_size=4
            )
            best_fitness_histories_few.append(best_fitness)

        # 4. GA Large Population (proposal_population_size)
        print(f"[{eval_name}] GA(多数個体)実行中...")
        for i in range(EXPERIMENT_TIMES):
            print(f"  GA(Many) {i+1}th run")
            best_fitness, _ = iga.run_simulation_normal_IGA(
                NUM_GENERATIONS=NUM_GENERATIONS, 
                POPULATION_SIZE=PROPOSAL_POPULATION_SIZE, 
                evaluate_num=eval_id, 
                times=i+1, 
                noise_is_added=noise_is_added, 
                look=look, 
                tournament_size=4
            )
            best_fitness_histories_many.append(best_fitness)

        # # 5. SAF-IEDA (Added part)
        # print(f"[{eval_name}] SAF-IEDA実行中...")
        # for i in range(EXPERIMENT_TIMES):
        #     print(f"  SAF-IEDA {i+1}th run")
        #     best_fitness, _ = iga.run_simulation_SAF_IEDA(
        #         NUM_GENERATIONS=NUM_GENERATIONS,
        #         POPULATION_SIZE=PROPOSAL_POPULATION_SIZE,
        #         TOP_NC=EVALUATE_SIZE,
        #         evaluate_num=eval_id,
        #         times=i+1,
        #         noise_is_added=noise_is_added,
        #         look=look,
        #         tournament_size=4
        #     )
        #     best_fitness_histories_saf.append(best_fitness)

        # --- 平均値の計算 ---
        def calc_average_history(histories):
            if not histories: return []
            num_gens = len(histories[0])
            avg_hist = []
            for g in range(num_gens):
                vals = [h[g][1] for h in histories]
                avg = sum(vals) / len(vals)
                avg_hist.append((histories[0][g][0], avg))
            return avg_hist

        avg_proposal = calc_average_history(best_fitness_histories_proposal)
        avg_benchmark = calc_average_history(best_fitness_histories_benchmark)
        avg_few = calc_average_history(best_fitness_histories_few)
        avg_many = calc_average_history(best_fitness_histories_many)
        avg_saf = calc_average_history(best_fitness_histories_saf)

        # --- プロット用データ作成 ---
        series_list = [
            {
                'label': '提案手法', 
                'data': avg_proposal, 
                'marker': 'o', 
                'linestyle': '-'
            },
            {
                'label': '距離項なし',
                'data': avg_benchmark,
                'marker': 'x',
                'linestyle': '-.'
            },
            {
                'label': f'GA ({EVALUATE_SIZE}個体)', 
                'data': avg_few, 
                'marker': 'o', 
                'linestyle': '--'
            },
            {
                'label': f'GA ({PROPOSAL_POPULATION_SIZE}個体)', 
                'data': avg_many, 
                'marker': '^', 
                'linestyle': ':'
            },
            {
                'label': 'SAF-IEDA', 
                'data': avg_saf, 
                'marker': 's', 
                'linestyle': '-.'
            }
        ]
        
        all_comparison_data[eval_name] = series_list

    # プロット実行
    log_four_metrics_comparison(
        interpolate_num=interpolate_num_val,
        file_path=f"_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_4metrics_FULL_comparison.pdf",
        comparison_data=all_comparison_data,
        ver="comparison"
    )
    print("\n全シミュレーション完了。比較画像を確認してください。")
    
