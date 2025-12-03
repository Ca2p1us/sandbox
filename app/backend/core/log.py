import matplotlib.pyplot as plt
import matplotlib.animation as ani
import os
import json
import numpy as np
import pyaudio
import wave
from scipy.signal import sawtooth
import uuid
from .geneticAlgorithm.config import ATTACK_RANGE, NUM_GENERATIONS


def log(file_path: str, answer,times: int):
    # UUID型をstr型に変換するヘルパー関数
    def convert_uuid_to_str(obj):
        if isinstance(obj, dict):
            return {k: convert_uuid_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_uuid_to_str(v) for v in obj]
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        else:
            return obj

    save_data = convert_uuid_to_str(answer)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({f"{times}_results": save_data}, f, indent=2)

    return


def log_fitness(method: str = None, file_path: str = None, best_fitness_history=None, average_fitness_history=None, evaluate_num: int = None, interpolate_num: int = None, ver: str = None, title: str = None):
    """
    世代ごとのbest個体のfitness履歴をグラフ表示する
    best_fitness_history: [(世代番号, fitness値), ...] のリスト
    """
    if evaluate_num == 1:
        method = "Gaussian"
    elif evaluate_num == 2:
        method = "Sphere"
    elif evaluate_num == 3:
        method = "Rastrigin"
    elif evaluate_num == 4:
        method = "Ackley"
    elif evaluate_num == 5:
        method = "Schwefel"
    if interpolate_num == 0:
        interpolate = "linear"
    elif interpolate_num == 1:
        interpolate = "Gauss"
    elif interpolate_num == 2:
        interpolate = "RBF"
    if isinstance(evaluate_num, int):
        if isinstance(interpolate_num, int):
            file_path = f'./result/{ver}/graph/{method}/{interpolate}/best_fitnesses/{method}{file_path}'
        else:
            file_path = f'./result/{ver}/graph/{method}/best_fitnesses/{method}{file_path}'
    
    fig, ax = plt.subplots()
    if not best_fitness_history:
        print("履歴データがありません。")
        return

    generations = [item[0] for item in best_fitness_history]
    fitness_values = [item[1] for item in best_fitness_history]
    average_values = [item[1] for item in average_fitness_history] if average_fitness_history else None

    ax.set_xlabel('Generation')  # x軸ラベル
    ax.set_ylabel('Fitness')  # y軸ラベル
    ax.set_title(method+' Fitness History'+ (f" - {title}" if title else ""))  # グラフタイトル
    ax.set_xlim(0.5,NUM_GENERATIONS+0.5)
    if method == "Gaussian":
        ax.set_ylim(0,6)
    if method == "Ackley":
        ax.set_ylim(3.0,4.5)
    ax.grid(True)
    ax.plot(generations, fitness_values,
             marker='o', linestyle='-', color='blue', label='Best Fitness')
    if average_values:
        ax.plot(generations, average_values,
                 marker='o', linestyle='--', color='orange', label='Average Fitness')
    ax.legend(loc=0)
    # fig.tight_layout()
    # plt.figure(figsize=(8, 5))
    # plt.plot(generations, fitness_values,
    #          marker='o', linestyle='-', color='blue')
    # plt.xlabel('Generation')
    # plt.ylabel('Best Fitness')
    # plt.title(method+' Best Fitness History')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f'./result/graph/{method}_fitness_history.png')
    plt.savefig(file_path)
    # plt.show()
    plt.close()

def log_fitness_histories(method_num: int, interpolate_num: int, file_path: str, best_fitness_histories, ver: str):
    """
    複数回のシミュレーションの世代ごとのbest個体のfitness履歴をグラフ表示する
    best_fitness_histories: [[(世代番号, fitness値), ...], ...] のリスト
    """

    if not best_fitness_histories:
        print("履歴データがありません。")
        return
    if method_num == 1:
        method = "Gaussian"
    elif method_num == 2:
        method = "Sphere"
    elif method_num == 3:
        method = "Rastrigin"
    elif method_num == 4:
        method = "Ackley"
    elif method_num == 5:
        method = "Schwefel"
    if interpolate_num == 0:
        interpolate = "linear"
    elif interpolate_num == 1:
        interpolate = "Gauss"
    elif interpolate_num == 2:
        interpolate = "RBF"

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, best_fitness_history in enumerate(best_fitness_histories):
        generations = [item[0] for item in best_fitness_history]
        fitness_values = [item[1] for item in best_fitness_history]
        ax.plot(generations, fitness_values,
                 marker='o', linestyle='-', label=f'Run {idx+1}')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.set_title(method+' Best Fitness Histories')
    ax.set_xlim(0.5,NUM_GENERATIONS+0.5)
    if method == "Gaussian":
        ax.set_ylim(0,6)
    if method == "Ackley":
        ax.set_ylim(3.0,4.5)
    ax.grid(True)
    # fig.tight_layout()
    # plt.savefig(f'./result/graph/{method}_fitness_histories.png')
    if interpolate_num == 100:
        plt.savefig(f'./result/{ver}/graph/{method}/best_fitnesses/{method}{file_path}')
    else:
        plt.savefig(f'./result/{ver}/graph/{method}/{interpolate}/best_fitnesses/{method}{file_path}')
    plt.show()
    plt.close()
    return
def log_comparison(evaluate_num: int, interpolate_num: int, file_path: str, best_fitness_histories_few_ave, best_fitness_histories_many_ave, best_fitness_histories_ave,indicator : str):
    """
    複数回のシミュレーションの世代ごとのbest個体のfitness履歴をグラフ表示する
    best_fitness_histories: [[(世代番号, fitness値), ...], ...] のリスト
    """

    if not best_fitness_histories_ave:
        print("履歴データがありません。")
        return
    if evaluate_num == 1:
        method = "Gaussian"
    elif evaluate_num == 2:
        method = "Sphere"
    elif evaluate_num == 3:
        method = "Rastrigin"
    elif evaluate_num == 4:
        method = "Ackley"
    elif evaluate_num == 5:
        method = "Schwefel"
    if interpolate_num == 0:
        interpolate = "linear"
    elif interpolate_num == 1:
        interpolate = "Gauss"
    elif interpolate_num == 2:
        interpolate = "RBF"

    fig, ax = plt.subplots(figsize=(8, 5))

    generations = [item[0] for item in best_fitness_histories_ave]
    fitness_values_few = [item[1] for item in best_fitness_histories_few_ave]
    fitness_values_many = [item[1] for item in best_fitness_histories_many_ave]
    fitness_values_proposal = [item[1] for item in best_fitness_histories_ave]

    ax.plot(generations, fitness_values_few,
             marker='o', linestyle='-', label='Normal IGA (few)')
    ax.plot(generations, fitness_values_many,
             marker='o', linestyle='-', label='Normal IGA (many)')
    ax.plot(generations, fitness_values_proposal,
             marker='o', linestyle='-', label='Proposed IGA')

    ax.set_xlabel('Generation')
    ax.set_ylabel(indicator+'Fitness')
    ax.set_title(method+' '+indicator+' Fitness Comparison')
    ax.set_xlim(0.5,NUM_GENERATIONS+0.5)
    if method == "Gaussian":
        ax.set_ylim(0,6)
    if method == "Ackley":
        ax.set_ylim(3.0,4.5)
    ax.grid(True)
    ax.legend(["補間なし9個体","補間なし200個体","補間あり"],prop={"family":"MS Gothic"},loc=0)
    # fig.tight_layout()
    # plt.savefig(f'./result/graph/{method}_fitness_histories.png')
    if interpolate_num == 100:
        plt.savefig(f'./result/comparison/{method}{file_path}')
    else:
        plt.savefig(f'./result/comparison/{method}/{interpolate}/{method}{file_path}')
    plt.show()
    plt.close()
    return


# サンプリング周波数と再生時間の設定
SAMPLE_RATE = 44100  # CD品質のサンプリングレート
CARRIER_FREQ_BASE = 440.0  # 基本となるキャリア周波数（A4音）


def load_params_from_json(file_path):
    """
    JSONファイルからパラメータを読み込む関数
    """
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません -> {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def generate_adsr_envelope(params, duration, sample_rate):
    """
    ADSRパラメータに基づいてエンベロープ波形を生成する関数 (前の回答と同じ)
    """
    attack_time = params['attack']
    decay_time = params['decay']
    sustain_level = params['sustain']
    sustain_time = params['sustainTime']
    release_time = params['release']

    attack_samples = int(attack_time * sample_rate)
    decay_samples = int(decay_time * sample_rate)
    sustain_samples = int(sustain_time * sample_rate)
    release_samples = int(release_time * sample_rate)
    total_samples = int(duration * sample_rate)

    envelope = np.zeros(total_samples)

    # Attack, Decay, Sustain, Releaseの各セクションの処理
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if decay_samples > 0:
        envelope[attack_samples:attack_samples +
                 decay_samples] = np.linspace(1, sustain_level, decay_samples)
    envelope[attack_samples + decay_samples:attack_samples +
             decay_samples + sustain_samples] = sustain_level
    release_start_idx = attack_samples + decay_samples + sustain_samples
    if release_samples > 0:
        envelope[release_start_idx:release_start_idx +
                 release_samples] = np.linspace(sustain_level, 0, release_samples)

    return envelope[:total_samples]


def play_sound(waveform, sample_rate):
    """
    指定された波形をPyAudioで再生する関数 (前の回答と同じ)
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True)

    audio_data = (waveform * 0.5).astype(np.float32).tobytes()
    stream.write(audio_data)

    stream.stop_stream()
    stream.close()
    p.terminate()

# WAVファイルに書き出す関数


def save_sound_as_wave(waveform, sample_rate, filename):
    """
    指定された波形をWAVファイルに保存する関数
    """
    # 波形データを16ビットPCMに変換
    waveform_integers = np.int16(waveform * 32767)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # モノラル
        wf.setsampwidth(2)  # 16ビット
        wf.setframerate(sample_rate)
        wf.writeframes(waveform_integers.tobytes())

    print(f"WAVファイルを保存しました: {filename}")

# --- メイン処理 ---


def sound_check(file_path=None):
    method = file_path[18:]
    method = method[:-5]
    # 1. JSONファイルからデータを読み込む
    fm_data = load_params_from_json(file_path)

    if fm_data:
        # 最も評価値の高い個体を取得、複数いた場合は最初の個体を選択
        best_individual = max(
            fm_data["results"][-1],
            key=lambda ind: float(ind.get("fitness", 0))
        )
        params = best_individual['fmParamsList']['operator1']

        # 2. ADSRエンベロープと波形を生成
        NOTE_DURATION = params['attack'] + params['decay'] + \
            params['sustainTime'] + params['release']
        envelope = generate_adsr_envelope(params, NOTE_DURATION, SAMPLE_RATE)
        t = np.linspace(0, NOTE_DURATION, len(envelope), endpoint=False)
        frequency = params['frequency'] * 1000  # 周波数をHzに変換
        # キャリア波形の生成（ここではサイン波を使用）
        carrier_wave = np.sin(2 * np.pi * frequency * t)

        # 3. エンベロープを適用
        modulated_wave = carrier_wave * envelope

        # 4. 音声の再生
        print("JSONファイルからパラメータを読み込み、音声を再生します...")
        play_sound(modulated_wave, SAMPLE_RATE)
        print("再生が終了しました。")
        # 5. WAVファイルに保存
        save_sound_as_wave(modulated_wave, SAMPLE_RATE,
                           f"./result/sound/{method}.wav")
        print(f"WAVファイルに保存しました: app/result/sound/{method}.wav")
    else:
        print("音声の再生をスキップします。")

# def plot_individual_params(population: list[dict], param_keys: list[str], generation: int, file_path: str = None):
#     """
#     個体群の指定されたパラメータをプロットする関数
#     param_key: "fmParamsList.operator1.frequency" のようなドット区切りで指定
#     """
#     def get_param(ind, key):
#         # "fmParamsList.operator1.frequency" のようなドット区切りでアクセス
#         val = ind
#         for k in key.split('.'):
#             val = val.get(k, None)
#             if val is None:
#                 break
#         return val

#     param_values1 = [get_param(ind, param_keys[0]) for ind in population]
#     param_values2 = [get_param(ind, param_keys[1]) for ind in population]

#     plt.xlim(-50,ATTACK_RANGE[1]+50)
#     plt.ylim(-50,ATTACK_RANGE[1]+50)
#     plt.title(f"Generation {generation} - individuals'")
#     plt.xlabel("1st parameter")
#     plt.ylabel("2nd parameter")
#     plt.grid(True)
#     plt.scatter(param_values1, param_values2)
#     plt.savefig(file_path)
#     plt.close()
#     return

def plot_individual_params(population: list[dict], param_keys: list[str], generation: int, file_path: str):
    """
    個体群の指定されたパラメータを3つのペアでプロットし、1枚の画像にまとめる関数
    param_keys: 少なくとも6つのキーが必要
    """
    def get_param(ind, key):
        # "fmParamsList.operator1.frequency" のようなドット区切りでアクセス
        val = ind
        for k in key.split('.'):
            val = val.get(k, None)
            if val is None:
                break
        return val

    # 1行3列のサブプロットを作成 (figsizeは横長に設定)
    
    # プロットするペアのインデックスリスト
    pair_indices = [(0, 1), (2, 3), (4, 5)]


    # 3つのグラフをループで描画
    for i, (idx1, idx2) in enumerate(pair_indices):
        fig, ax = plt.subplots(figsize=(6, 6))
        key1 = param_keys[idx1]
        key2 = param_keys[idx2]

        # データの抽出
        param_values1 = [get_param(ind, key1) for ind in population]
        param_values2 = [get_param(ind, key2) for ind in population]

        # プロット設定
        ax.scatter(param_values1, param_values2, alpha=1.0) # 重なりが見やすいように透過度を設定
        
        # 軸範囲の設定 (元のコードに準拠)
        ax.set_xlim(-50, ATTACK_RANGE[1] + 50)
        ax.set_ylim(-50, ATTACK_RANGE[1] + 50)
        
        # ラベルとグリッド
        ax.set_xlabel(f"param{idx1+1}", fontsize=9) # パラメータ番号をラベルにする
        ax.set_ylabel(f"param{idx2+1}", fontsize=9)
        ax.set_title(f"Gen {generation} - Pair {i+1} (Param {idx1+1} vs Param {idx2+1})")
        ax.grid(True)

        # レイアウトを自動調整して重なりを防ぐ
        plt.tight_layout()
        
        # ファイル保存
        plt.savefig(f"{file_path}_pair{i+1}.png")
        plt.close() # メモリ解放
    return
def log_average_fitness(method: str = None, interpolate_method:str = None,file_path: str = None, average_fitness_history=None, times: int = None):
    """
    世代ごとのaverage個体のfitness履歴をjsonファイルに保存する
    average_fitness_history: [(世代番号, fitness値), ...] のリスト
    """
    method_num = None
    interpolate_num = None
    if method == "Gaussian":
        method_num = 1
    elif method == "Sphere":
        method_num = 2
    elif method == "Rastrigin":
        method_num = 3
    elif method == "Ackley":
        method_num = 4
    elif method == "Schwefel":
        method_num = 5
    if interpolate_method == "linear":
        interpolate_num = 0
    elif interpolate_method == "Gauss":
        interpolate_num = 1
    elif interpolate_method == "RBF":
        interpolate_num = 2
    if isinstance(method_num, int):
        if isinstance(interpolate_num, int):
            file_path = f'./result/proposal/average/{method}/{interpolate_method}/{method}{file_path}'
        else:
            file_path = f'./result/conventional/average/{method}/{method}{file_path}'
    
    # JSONファイルに保存
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump({f"{times}_average_fitness":average_fitness_history}, f, indent=2)

    return

def compute_interpolation_error(population, true_eval_func, param_keys):
    """
    全個体について補間値と真値のMSEを計算
    """
    errors = []
    for ind in population:
        # 真値計算
        params = [get_nested_value(ind, k) for k in param_keys]
        true_val = true_eval_func(params)
        interp_val = ind.get("pre_evaluation", None)
        if interp_val is None:
            continue
        errors.append((true_val - interp_val) ** 2)
    return np.mean(errors) if errors else None
