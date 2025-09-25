from ..engine import run_iga_simulation as iga
from ..core import sound_check

print(f"IGAシミュレーション\n1: 普通のIGAシミュレーション\n2: 提案型IGAシミュレーション\n3: 音の確認")
choice = input("実行するシミュレーションを選択 (1/3): ")
if choice == "2":
    print(f"提案型IGAシミュレーションの評価関数を選択\n0: ガウス関数\n1: スフィア関数\n2: ノイズ関数\n3: コサイン関数")
    evaluate_num = input("評価関数の番号を入力してください: ")
    print("提案型IGAシミュレーションを実行")
    iga.run_simulation_proposal_IGA(evaluate_num = int(evaluate_num))
    print("提案型IGAシミュレーションが完了")
elif choice == "1":
    print("普通のIGAシミュレーションを実行")
    iga.run_simulation_normal_IGA()
    print("普通のIGAシミュレーションが完了")
elif choice == "3":
    print("音の確認を実行")
    json_file_path = input("音の確認を行う評価関数を選択\n0: ガウス関数\n1: スフィア関数\n2: ノイズ関数\n3: コサイン関数\n番号を入力してください: ")
    if json_file_path == "0":
        json_file_path = "result/simulation_gaussian.json"
    elif json_file_path == "1":
        json_file_path = "result/simulation_sphere.json"
    elif json_file_path == "2":
        json_file_path = "result/simulation_noise.json"
    elif json_file_path == "3":
        json_file_path = "result/simulation_cos.json"
    sound_check(file_path=json_file_path)
    print("音の確認が完了")