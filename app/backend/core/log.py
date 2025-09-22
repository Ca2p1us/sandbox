import matplotlib.pyplot as plt
import os
import json

def log(file_path: str, answer):
    save_data = [answer]
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # ファイルが存在しない場合は新規作成
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump({"results": [answer]}, f, indent=2)
        return

    # with open(file_path, 'r') as f:
    #     content = f.read()
    #     if not content.strip():
    #         read_data = []
    #     else:
    #         read_data = json.loads(content).get("results", [])
    #     save_data = read_data + [answer]

    with open(file_path, 'w') as f:
        json.dump({"results": save_data}, f, indent=2)

    return


def log_fitness(best_fitness_history):
    """
    世代ごとのbest個体のfitness履歴をグラフ表示する
    best_fitness_history: [(世代番号, fitness値), ...] のリスト
    """
    if not best_fitness_history:
        print("履歴データがありません。")
        return

    generations = [item[0] for item in best_fitness_history]
    fitness_values = [item[1] for item in best_fitness_history]

    plt.figure(figsize=(8, 5))
    plt.plot(generations, fitness_values, marker='o', linestyle='-', color='blue')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Best Fitness History')
    plt.grid(True)
    plt.tight_layout()
    plt.show()