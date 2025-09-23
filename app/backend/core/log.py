import matplotlib.pyplot as plt
import os
import json
import numpy as np
import pyaudio
from scipy.signal import sawtooth

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
        envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain_level, decay_samples)
    envelope[attack_samples + decay_samples:attack_samples + decay_samples + sustain_samples] = sustain_level
    release_start_idx = attack_samples + decay_samples + sustain_samples
    if release_samples > 0:
        envelope[release_start_idx:release_start_idx + release_samples] = np.linspace(sustain_level, 0, release_samples)

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

# --- メイン処理 ---
def sound_check(file_path=None):
    # 1. JSONファイルからデータを読み込む
    fm_data = load_params_from_json(file_path)

    if fm_data:
        best_individual = max(
            fm_data["results"][-1],
            key=lambda ind: float(ind.get("fitness", 0))
        )
        params = best_individual['fmParamsList']['operator1']

        # 2. ADSRエンベロープと波形を生成
        NOTE_DURATION = params['attack'] + params['decay'] + params['sustainTime'] + params['release']
        envelope = generate_adsr_envelope(params, NOTE_DURATION, SAMPLE_RATE)
        t = np.linspace(0, NOTE_DURATION, len(envelope), endpoint=False)
        frequency = CARRIER_FREQ_BASE * (1 + params['frequency'])
        carrier_wave = np.sin(2 * np.pi * frequency * t)
        
        # 3. エンベロープを適用
        modulated_wave = carrier_wave * envelope

        # 4. 音声の再生
        print("JSONファイルからパラメータを読み込み、音声を再生します...")
        play_sound(modulated_wave, SAMPLE_RATE)
        print("再生が終了しました。")
    else:
        print("音声の再生をスキップします。")

