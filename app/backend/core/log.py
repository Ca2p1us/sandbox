import matplotlib.pyplot as plt
import os
import json
import numpy as np
import pyaudio
import wave
from scipy.signal import sawtooth
import uuid


def log(file_path: str, answer):
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
        json.dump({"results": save_data}, f, indent=2)

    return


def log_fitness(method: str, file_path: str, best_fitness_history, average_fitness_history=None):
    """
    世代ごとのbest個体のfitness履歴をグラフ表示する
    best_fitness_history: [(世代番号, fitness値), ...] のリスト
    """

    if not best_fitness_history:
        print("履歴データがありません。")
        return
    fig, ax = plt.subplots()

    generations = [item[0] for item in best_fitness_history]
    fitness_values = [item[1] for item in best_fitness_history]
    average_values = [item[1] for item in average_fitness_history] if average_fitness_history else None

    ax.set_xlabel('Generation')  # x軸ラベル
    ax.set_ylabel('Fitness')  # y軸ラベル
    ax.set_title(method+' Fitness History')  # グラフタイトル
    ax.grid(True)
    ax.plot(generations, fitness_values,
             marker='o', linestyle='-', color='blue', label='Best Fitness')
    if average_values:
        ax.plot(generations, average_values,
                 marker='o', linestyle='--', color='orange', label='Average Fitness')
        ax.legend()
    ax.legend(loc=0)
    fig.tight_layout()
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

def log_fitness_histories(method_num: int, file_path: str, best_fitness_histories, ver: str):
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

    plt.figure(figsize=(8, 5))

    for idx, best_fitness_history in enumerate(best_fitness_histories):
        generations = [item[0] for item in best_fitness_history]
        fitness_values = [item[1] for item in best_fitness_history]
        plt.plot(generations, fitness_values,
                 marker='o', linestyle='-', label=f'Run {idx+1}')

    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title(method+' Best Fitness Histories')
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(f'./result/graph/{method}_fitness_histories.png')
    plt.savefig(f'./result/{ver}/graph/{method}/{method}{file_path}')
    plt.show()


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
