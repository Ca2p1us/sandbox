from backend.engine import run_iga_simulation as iga
from backend.core.log import sound_check, log_fitness, log_fitness_histories


NUM_GENERATIONS = 9
POPULATION_SIZE = 10
PROPOSAL_POPULATION_SIZE = 200
EVALUATE_SIZE = 9


best_fitness_histories = []
noise_is_added = False
print(f"IGAシミュレーション\n1: 普通のIGAシミュレーション\n2: 提案型IGAシミュレーション\n3: 音の確認")
choice = input("実行するシミュレーションを選択 (1/3): ")
if choice == "2":
    print(f"IGAシミュレーションの評価関数を選択\n1: ガウス関数\n2: スフィア関数\n3: Rastrigin関数\n4: Ackley関数\n5: Schwefel関数")
    evaluate_num = input("評価関数の番号を入力してください: ")
    print(f"ノイズを追加しますか？\n0: 追加しない\n1: 追加する")
    TF = input("ノイズを追加しますか？ (0/1): ")
    if TF == "1":
        noise_is_added = True
    for i in range(5):
        print(f"評価関数 {i}: {evaluate_num}")
        print("提案型IGAシミュレーション"+str(i+1)+"回目を実行")
        best_fitness_histories.append(iga.run_simulation_proposal_IGA(NUM_GENERATIONS=NUM_GENERATIONS, PROPOSAL_POPULATION_SIZE=PROPOSAL_POPULATION_SIZE, EVALUATE_SIZE=EVALUATE_SIZE, evaluate_num = int(evaluate_num), times = i+1, noise_is_added=noise_is_added))
        print("提案型IGAシミュレーション"+str(i+1)+"回目が完了")
    log_fitness_histories(int(evaluate_num), "_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(PROPOSAL_POPULATION_SIZE)+"_best_fitness_histories.png", best_fitness_histories, ver="proposal")
elif choice == "1":
    print(f"IGAシミュレーションの評価関数を選択\n1: ガウス関数\n2: スフィア関数\n3: Rastrigin関数\n4: Ackley関数\n5: Schwefel関数")
    evaluate_num = input("評価関数の番号を入力してください: ")
    print(f"ノイズを追加しますか？\n0: 追加しない\n1: 追加する")
    TF = input("ノイズを追加しますか？ (0/1): ")
    if TF == "1":
        noise_is_added = True
    for i in range(5):
        print("普通のIGAシミュレーション"+str(i+1)+"回目を実行")
        best_fitness_histories.append(iga.run_simulation_normal_IGA(NUM_GENERATIONS=NUM_GENERATIONS, POPULATION_SIZE=POPULATION_SIZE, evaluate_num = int(evaluate_num), times = i+1, noise_is_added=noise_is_added))
        print("普通のIGAシミュレーション"+str(i+1)+"回目が完了")
    log_fitness_histories(int(evaluate_num), "_noise"+str(noise_is_added)+"_"+str(NUM_GENERATIONS)+"gens_"+str(POPULATION_SIZE)+"_best_fitness_histories.png", best_fitness_histories, ver="conventional")
elif choice == "3":
    print("音の確認を実行")
    # json_file_path = input("音の確認を行う評価関数を選択\n0: ガウス関数\n1: スフィア関数\n2: ノイズ関数\n3: Rastrigin関数\n4: Ackley関数\n5: Schweful関数\n6: ランダム評価\n7: テスト音声\n番号を入力してください: ")
    # if json_file_path == "0":
    #     json_file_path = "result/simulation_gaussian.json"
    # elif json_file_path == "1":
    #     json_file_path = "result/simulation_sphere.json"
    # elif json_file_path == "2":
    #     json_file_path = "result/simulation_noise.json"
    # elif json_file_path == "3":
    #     json_file_path = "result/simulation_Rastrigin.json"
    # elif json_file_path == "4":
    #     json_file_path = "result/simulation_Ackley.json"
    # elif json_file_path == "5":
    #     json_file_path = "result/simulation_Schwefel.json"
    # elif json_file_path == "6":
    #     json_file_path = "result/simulation_random.json"
    # elif json_file_path == "7":
    #     json_file_path = "result/simulation_test.json"
    # sound_check(file_path=json_file_path)
    print("音の確認が完了")