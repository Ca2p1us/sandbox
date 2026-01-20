from typing import List
from ..core.geneticAlgorithm.selection import tournament
from ..core.geneticAlgorithm.make_chromosome_params import make_chromosome_params
from ..core.geneticAlgorithm import BLX_alpha
from ..core.geneticAlgorithm.repair import repair_fm_params as repair_gene
from ..core.geneticAlgorithm.mutate import mutate 
from ..core.geneticAlgorithm import make_chromosome_params
from ..core.geneticAlgorithm.interpolation import interpolation, get_evaluated_individuals, get_total_error, build_interpolator
from ..core.geneticAlgorithm.pre_selection import select_top_individuals_by_pre_evaluation
from ..core.geneticAlgorithm.config import TARGET_PARAMS, PARAMS, TARGET_PARAMS_1, TARGET_PARAMS_2
from ..core.log import log, log_fitness, plot_individual_params, log_average_fitness, plot_interpolated_heatmap
from ..engine.evaluate import evaluate_fitness, get_best_and_worst_individuals, get_best_and_worst_individuals_by_id, get_average_fitness
from ..engine.SAF_IEDA import SAF_SurrogateModel
import numpy as np
import uuid
import copy


Chromosomes = List[dict]


def make_initial_population(num_individuals: int) -> Chromosomes:
    return [make_chromosome_params.make_chromosome_params() for _ in range(num_individuals)]

def run_simulation_normal_IGA(NUM_GENERATIONS=9, POPULATION_SIZE=10, evaluate_num=0, times:int=1, noise_is_added: bool = False, look: bool = False, tournament_size=3):
    best_fitness_history = []
    average_fitness_history = []
    bests = []
    if evaluate_num == 1:
    # 2-1. ガウス関数
        evaluate_method = "Gaussian"
    elif evaluate_num == 2:
    # 2-2. スフィア関数
        evaluate_method = "Sphere"
    elif evaluate_num == 3:
    # 2-3. Gauss関数+cos関数
        evaluate_method = "Gaussian_cos"
    elif evaluate_num == 4:
    # 2-4. Ackley関数
        evaluate_method = "Ackley"
    elif evaluate_num == 5:
    # 2-5. Gaussian_peaks関数
        evaluate_method = "Gaussian_peaks"
    elif evaluate_num == 6:
    # 2-6. Mixed関数
        evaluate_method = "Mixed"
    # 1. 初期個体生成
    population = make_initial_population(POPULATION_SIZE)
    
    for generation in range(NUM_GENERATIONS - 1):
        # 2. 評価
        evaluate_fitness(
            population=population,
            evaluate_num=evaluate_num,
            param_keys=PARAMS,
            noise_is_added=noise_is_added
        )
        best, worst = get_best_and_worst_individuals(population)
        # --- ここで履歴に追加 ---
        if best is not None and "fitness" in best:
            best_fitness_history.append((generation + 1, float(best["fitness"])))
            bests.append(best)
        # 評価の平均値を表示
        average = get_average_fitness(population)
        if average is not None:
            average_fitness_history.append((generation + 1, float(average)))
        next_generation:List[Chromosomes]  = []
        if look and times == 1:
            plot_individual_params(population=population,best=best,worst=worst, param_keys=PARAMS, generation=generation + 1, file_path=f'./result/conventional/graph/{evaluate_method}/scatter/{evaluate_method}_noise{str(noise_is_added)}_{str(POPULATION_SIZE)}_individuals_{str(generation + 1)}gens')
        for _ in range(POPULATION_SIZE):
            # 3. 選択
            selected = tournament.exec_tournament_selection(chromosomes_params=population, participants_num=tournament_size)
            # selected = tournament.exec_tournament_selection(chromosomes_params=population, participants_num=6)
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
                        ind["generation"] = generation + 2
                        # 新しいchromosomeIdを付与
                        ind["chromosomeId"] = str(uuid.uuid4())
                        next_generation.append(ind)
                    else:
                        print("警告: offspring内にdict以外が含まれています:", ind)
            elif isinstance(offspring, dict):
                # offspring["algorithmNum"] = selected[0].get("algorithmNum", None)
                offspring["generation"] = generation + 2
                offspring["chromosomeId"] = str(uuid.uuid4())
                next_generation.append(offspring)
            else:
                print("警告: offspringがdictまたはlist[dict]ではありません:", offspring)

        # 5. 次世代への更新
        population = next_generation
    # --- ここで最終世代の評価値を再計算 ---
    evaluate_fitness(
            population=population,
            evaluate_num=evaluate_num,
            param_keys=PARAMS,
            noise_is_added=noise_is_added
        )

    # ベスト・ワースト個体の取得
    best, worst = get_best_and_worst_individuals(population)
    # 評価の平均値を表示
    average = get_average_fitness(population)
    best_fitness_history.append((NUM_GENERATIONS, float(best["fitness"])))
    average_fitness_history.append((NUM_GENERATIONS, float(average)))
    bests.append(best)
    if look and times == 1:
        plot_individual_params(population=population,best=best,worst=worst, param_keys=PARAMS, generation=NUM_GENERATIONS, file_path=f'./result/conventional/graph/{evaluate_method}/scatter/{evaluate_method}_noise{str(noise_is_added)}_{str(POPULATION_SIZE)}_individuals_{str(NUM_GENERATIONS)}gens')
    # 6. 最終結果の出力
    log(f"result/conventional/last_gen_individuals/{evaluate_method}/{str(POPULATION_SIZE)}inds/simulation_{evaluate_method}_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(times)}.json", population,times = times)
    log(f"result/conventional/best/{evaluate_method}/{str(POPULATION_SIZE)}inds/best_individual_{evaluate_method}_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(times)}.json", bests,times = times)
    log_fitness(
        evaluate_num=evaluate_num,
        file_path=f"_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(POPULATION_SIZE)}_{str(times)}_best_fitness_history.png",
        best_fitness_history=best_fitness_history,
        average_fitness_history=average_fitness_history,
        ver="conventional"
    )
    log_average_fitness(
        evaluate_num=evaluate_num,
        file_path=f"_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(POPULATION_SIZE)}_{str(times)}_average_fitness_history.json",
        average_fitness_history=average_fitness_history,
        times=times
        )
    return best_fitness_history,average_fitness_history


def run_simulation_proposal_IGA(NUM_GENERATIONS=9, PROPOSAL_POPULATION_SIZE=200, EVALUATE_SIZE=9, evaluate_num=0, interpolate_num=0, times:int=1, noise_is_added:bool=False, look: bool = False, tournament_size=3):
    best_fitness_history = []
    average_fitness_history = []
    error_history = []
    bests = []
    archive = []
    evaluate_method = ""
    interpolate = "linear"
    if evaluate_num == 1:
    # 2-1. ガウス関数
        evaluate_method = "Gaussian"
    elif evaluate_num == 2:
    # 2-2. スフィア関数
        evaluate_method = "Sphere"
    elif evaluate_num == 3:
    # 2-3. Gauss関数+cos関数
        evaluate_method = "Gaussian_cos"
    elif evaluate_num == 4:
    # 2-4. Ackley関数
        evaluate_method = "Ackley"
    elif evaluate_num == 5:
    # 2-5. Gaussian_peaks関数
        evaluate_method = "Gaussian_peaks"
    elif evaluate_num == 6:
    # 2-6. Mixed関数
        evaluate_method = "Mixed"
    # 1. 初期個体生成
    population = make_initial_population(PROPOSAL_POPULATION_SIZE)
    # 初期個体の事前評価(補間)
    if interpolate_num == 0:
        interpolate = "linear"
    elif interpolate_num == 1:
        interpolate = "Gauss"
    elif interpolate_num == 2:
        interpolate = "TPS"
    elif interpolate_num == 3:
        interpolate = "IDW"
    elif interpolate_num == 4:
        interpolate = "Hybrid_RBF"
    elif interpolate_num == 5:
        interpolate = "Gaussian_RBF"
    elif interpolate_num == 6:
        interpolate = "IMQ_RBF"
    interpolation(
            population=population,
            method_num=interpolate_num,
            param_keys=PARAMS,
            target_key="pre_evaluation",
            )
    # 評価個体の選択
    evaluate_id = select_top_individuals_by_pre_evaluation(population, total_n=EVALUATE_SIZE,gen = 1)
    evaluate_population = get_evaluated_individuals(population, evaluate_id)

    for generation in range(NUM_GENERATIONS - 1):
        # 2. 評価
        evaluate_fitness(
            population=evaluate_population,
            evaluate_num=evaluate_num,
            param_keys=PARAMS,
            noise_is_added=noise_is_added
        )
        archive.extend([copy.deepcopy(ind) for ind in evaluate_population])
        interpolator = build_interpolator(
            evaluated_population=archive,
            method_num=interpolate_num,
            param_keys=PARAMS,
            refernce_key="fitness",
            generation=generation
        )
        # ベスト・ワースト個体の取得
        best, worst = get_best_and_worst_individuals_by_id(archive)
        # ほかの個体の評価を補間
        interpolation(
            population=population,
            evaluated_population=archive,
            best=best,
            worst=worst,
            method_num=interpolate_num,
            gen=generation+1,
            target_key="fitness",
            interpolator=interpolator
        )
        # 真値の取得
        evaluate_fitness(
            population=population,
            evaluate_num=evaluate_num,
            param_keys=PARAMS,
            noise_is_added=noise_is_added,
            target_key="true_fitness"
        )

        # 評価の平均値を表示
        average = get_average_fitness(population)
        # print(best["fitness"])
        # --- ここで履歴に追加 ---
        if best is not None and "fitness" in best:
            best_fitness_history.append((generation + 1, float(best["fitness"])))
            bests.append(best)
        if average is not None:
            average_fitness_history.append((generation + 1, float(average)))
        error_history.append((generation + 1, float(get_total_error(population=population, evaluate_num=evaluate_num))))
        if look and times == 1:
            plot_individual_params(
                population=population,
                best=best,worst=worst,
                param_keys=PARAMS,
                generation=generation + 1,
                file_path=f'./result/proposal/graph/{evaluate_method}/{interpolate}/scatter/{evaluate_method}_noise{str(noise_is_added)}_{str(PROPOSAL_POPULATION_SIZE)}_{str(EVALUATE_SIZE)}_individuals_{str(generation + 1)}gens'
            )
            plot_interpolated_heatmap(
                interpolator=interpolator,
                evaluated_population=archive,
                best=best,
                param_keys=PARAMS,
                pair_indices=[(0,1),(2,3),(4,5)],
                generation=generation + 1,
                file_path=f'./result/proposal/heatmap/{evaluate_method}/{interpolate}/{evaluate_method}_noise{str(noise_is_added)}_{str(PROPOSAL_POPULATION_SIZE)}_{str(EVALUATE_SIZE)}_individuals_{str(generation + 1)}gens'
            )
        next_generation:List[Chromosomes]  = []
        # for _ in range(PROPOSAL_POPULATION_SIZE):
        while len(next_generation) < PROPOSAL_POPULATION_SIZE:
            # 3. 選択
            # if generation > NUM_GENERATIONS // 2:
            #     selected = tournament.exec_tournament_selection(chromosomes_params=population, participants_num=30)
            # else:
            #     selected = tournament.exec_tournament_selection(chromosomes_params=population, participants_num=tournament_size)
            selected = tournament.exec_tournament_selection(chromosomes_params=population, participants_num=tournament_size)
            # 4. 交叉&突然変異
            offspring = BLX_alpha.exec_blx_alpha(
                parents_chromosomes=selected,
                func_repair_gene=repair_gene,
                mutate=mutate
            )
            # offspringがリストの場合
            if isinstance(offspring, list):
                for ind in offspring:
                    if len(next_generation) >= PROPOSAL_POPULATION_SIZE:
                        break
                    if isinstance(ind, dict):
                        # algorithmNumを親からコピー（selected[0]を例とする）
                        # ind["algorithmNum"] = selected[0].get("algorithmNum", None)
                        # 新しいchromosomeIdを付与
                        ind["fitness"] = 0.0
                        ind["pre_evaluation"] = 0
                        ind["true_fitness"] = 0.0
                        ind["generation"] = generation + 2
                        ind["chromosomeId"] = str(uuid.uuid4())
                        next_generation.append(ind)
                    else:
                        print("警告: offspring内にdict以外が含まれています:", ind)
            elif isinstance(offspring, dict):
                # offspring["algorithmNum"] = selected[0].get("algorithmNum", None)
                offspring["fitness"] = 0.0
                offspring["pre_evaluation"] = 0
                offspring["true_fitness"] = 0.0
                offspring["generation"] = generation + 2
                offspring["chromosomeId"] = str(uuid.uuid4())
                next_generation.append(offspring)
            else:
                print("警告: offspringがdictまたはlist[dict]ではありません:", offspring)

        # 5. 次世代への更新

        population = next_generation
        #事前評価(補間)
        interpolation(
            population = population,
            evaluated_population = archive,
            best = best, 
            worst = worst,
            method_num=interpolate_num,
            gen=generation+2,
            param_keys=PARAMS,
            target_key="pre_evaluation",
            interpolator=interpolator
        )
        # 評価個体の選択
        evaluate_id = select_top_individuals_by_pre_evaluation(population, total_n=EVALUATE_SIZE, gen=generation+1)
        evaluate_population = get_evaluated_individuals(population, evaluate_id)

    # --- ここで最終世代の評価値を再計算 ---
    evaluate_fitness(
            population=evaluate_population,
            evaluate_num=evaluate_num,
            param_keys=PARAMS,
            noise_is_added=noise_is_added
        )
    # ベスト・ワースト個体の取得
    archive.extend(copy.deepcopy(ind) for ind in evaluate_population)
    interpolator = build_interpolator(
            evaluated_population=archive,
            method_num=interpolate_num,
            param_keys=PARAMS,
            refernce_key="fitness",
            generation=NUM_GENERATIONS
        )
    best, worst = get_best_and_worst_individuals_by_id(archive)
    # ほかの個体の評価を補間
    interpolation(
            population=population,
            evaluated_population=archive,
            best=best,
            worst=worst,
            method_num=interpolate_num,
            gen=NUM_GENERATIONS,
            target_key="fitness",
            interpolator=interpolator
        )
    evaluate_fitness(
            population=population,
            evaluate_num=evaluate_num,
            param_keys=PARAMS,
            noise_is_added=noise_is_added,
            target_key="true_fitness"
        )
    # 評価の平均値を表示
    average = get_average_fitness(population)
    best_fitness_history.append((NUM_GENERATIONS, float(best["fitness"])))
    average_fitness_history.append((NUM_GENERATIONS, float(average)))
    bests.append(best)
    error_history.append((NUM_GENERATIONS, float(get_total_error(population=population, evaluate_num=evaluate_num))))
    if look and times == 1:
        plot_individual_params(
            population=population,
            best=best,worst=worst,
            param_keys=PARAMS,
            generation=NUM_GENERATIONS,
            file_path=f'./result/proposal/graph/{evaluate_method}/{interpolate}/scatter/{evaluate_method}_noise{str(noise_is_added)}_{str(PROPOSAL_POPULATION_SIZE)}_{str(EVALUATE_SIZE)}_individuals_{str(NUM_GENERATIONS)}gens'
            )
        plot_interpolated_heatmap(
                interpolator=interpolator,
                evaluated_population=archive,
                best=best,
                param_keys=PARAMS,
                pair_indices=[(0,1),(2,3),(4,5)],
                generation=NUM_GENERATIONS,
                file_path=f'./result/proposal/heatmap/{evaluate_method}/{interpolate}/{evaluate_method}_noise{str(noise_is_added)}_{str(PROPOSAL_POPULATION_SIZE)}_{str(EVALUATE_SIZE)}_individuals_{str(NUM_GENERATIONS)}gens'
            )
    # 6. 最終結果の出力
    log(f"result/proposal/last_gen_individuals/{evaluate_method}/{interpolate}/{str(PROPOSAL_POPULATION_SIZE)}inds_{str(EVALUATE_SIZE)}eval/simulation_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(times)}.json", population,times = times)
    log(f"result/proposal/best/{evaluate_method}/{interpolate}/{str(PROPOSAL_POPULATION_SIZE)}inds_{str(EVALUATE_SIZE)}eval/best_individual_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(times)}.json", bests,times = times)
    log_fitness(
        evaluate_num=evaluate_num,
        interpolate_num=interpolate_num,
        file_path=f"_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(PROPOSAL_POPULATION_SIZE)}_{str(EVALUATE_SIZE)}eval_{str(times)}_best_fitness_history.png",
        best_fitness_history=best_fitness_history,
        average_fitness_history=average_fitness_history,
        ver="proposal"
    )
    log_average_fitness(
        evaluate_num=evaluate_num,
        interpolate_num=interpolate_num,
        file_path=f"_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(PROPOSAL_POPULATION_SIZE)}_{str(EVALUATE_SIZE)}eval_{str(times)}_average_fitness_history.json",
        average_fitness_history=average_fitness_history,
        times=times
    )
    return best_fitness_history, average_fitness_history, error_history

def run_simulation_SAF_IEDA(NUM_GENERATIONS=9, POPULATION_SIZE=9, TOP_NC=5, evaluate_num=0, times:int=1, noise_is_added: bool = False, look: bool = False, tournament_size=3):
    """
    SAF-IEDAアルゴリズムを用いたシミュレーションを実行する関数
    Top-Nc戦略: 全個体の属性頻度(密度)を計算し、上位Nc個体のみを真の評価に使用する
    """
    
    # --- 1. ネストされた辞書から値を取得するヘルパー関数 ---
    def get_nested_value(ind, dot_key):
        val = ind
        for k in dot_key.split('.'):
            val = val[k]
        return val

    # --- 2. Top-Nc選抜用のスコア計算関数 (Top-Nc Strategy) ---
    def calculate_preference_scores(population, param_keys, bins=20):
        """
        各個体の「属性頻度スコア」を計算する。
        連続値のため、ヒストグラムを用いて各パラメータ値の密度(頻度)を推定し、
        個体ごとの総和をスコアとする（スコアが高い＝集団内で一般的な特徴を持つ＝好ましい候補）。
        """
        # 全個体のパラメータを配列化 (N x D)
        all_values = np.array([[get_nested_value(ind, k) for k in param_keys] for ind in population])
        num_individuals, num_params = all_values.shape
        
        # 個体ごとのスコア初期化
        scores = np.zeros(num_individuals)
        
        # 各次元(パラメータ)ごとに密度を計算
        for d in range(num_params):
            # ヒストグラムで密度分布を作成
            hist, bin_edges = np.histogram(all_values[:, d], bins=bins, density=True)
            # 各個体の値がどのビンに含まれるか判定
            # digitizeは1から始まるインデックスを返すため-1する
            indices = np.digitize(all_values[:, d], bin_edges) - 1
            # 範囲外(最大値ジャストなど)の補正
            indices = np.clip(indices, 0, bins - 1)
            # そのビンの密度をスコアに加算
            scores += hist[indices]
            
        return scores
    # -----------------------------------------------------

    best_fitness_history = []
    average_fitness_history = []
    bests = []
    
    # 評価手法の特定
    evaluate_method = ""
    if evaluate_num == 1: evaluate_method = "Gaussian"
    elif evaluate_num == 2: evaluate_method = "Sphere"
    elif evaluate_num == 3: evaluate_method = "Gaussian_cos"
    elif evaluate_num == 4: evaluate_method = "Ackley"
    elif evaluate_num == 5: evaluate_method = "Gaussian_peaks"
    elif evaluate_num == 6: evaluate_method = "Mixed"

    # 1. 初期個体生成
    population = make_initial_population(POPULATION_SIZE)
    
    # SAFモデルの初期化
    saf_model = SAF_SurrogateModel(num_vars=len(PARAMS))

    for generation in range(NUM_GENERATIONS - 1):
        
        # --- Top-Nc 戦略の実装 ---
        
        # A. 属性頻度スコアの計算
        pref_scores = calculate_preference_scores(population, PARAMS)
        
        # B. スコアに基づいて個体群をソート (スコアが高い順)
        # 情報を付与してソートしやすくする
        for i, ind in enumerate(population):
            ind['pre_evaluation'] = pref_scores[i]
        
        # スコア降順でソート
        sorted_population = sorted(population, key=lambda x: x['pre_evaluation'], reverse=True)
        
        # C. 上位Nc個体 (Top-Nc) と 残りの個体 (Estimated) に分割
        top_nc_individuals = sorted_population[:TOP_NC]
        estimated_individuals = sorted_population[TOP_NC:]
        
        # --- 2. 真の評価 (Top-Ncのみ) ---
        # ユーザー評価のシミュレーション
        evaluate_fitness(
            population=top_nc_individuals,
            evaluate_num=evaluate_num,
            param_keys=PARAMS,
            noise_is_added=noise_is_added,
            target_key="fitness"
        )
        # true_fitnessに真値を格納 (Top-Ncは真値で確定)
        for ind in top_nc_individuals:
            ind['true_fitness'] = ind['fitness']

        # (ログ用: 残りの個体の真値も計算しておくが、学習には使わない)
        evaluate_fitness(
            population=estimated_individuals,
            evaluate_num=evaluate_num,
            param_keys=PARAMS,
            noise_is_added=noise_is_added
        )

        # --- ログ記録 (真値ベース) ---
        # 全体の中から真のベストを探す
        best_ind = max(population, key=lambda x: x['fitness'])
        best_fitness_history.append((generation + 1, float(best_ind["true_fitness"])))
        bests.append(copy.deepcopy(best_ind))

        total_fitness = sum(ind['fitness'] for ind in population)
        average_fitness = total_fitness / len(population)
        average_fitness_history.append((generation + 1, float(average_fitness)))

        if look and times == 1:
            # プロット
            plot_individual_params(
                population=population,
                best=best_ind,
                worst=min(population, key=lambda x: x['true_fitness']),
                param_keys=PARAMS,
                generation=generation + 1,
                file_path=f'./result/saf_ieda/graph/{evaluate_method}/scatter/{evaluate_method}_noise{str(noise_is_added)}_{str(POPULATION_SIZE)}_individuals_{str(generation + 1)}gens'
            )
        
        # --- 3. SAFモデルの学習 (Top-Ncの情報のみ使用) ---
        top_vectors = np.array([[get_nested_value(ind, k) for k in PARAMS] for ind in top_nc_individuals])
        top_fitness_vals = np.array([ind['fitness'] for ind in top_nc_individuals])
        
        saf_model.fit(top_vectors, top_fitness_vals)

        # --- 4. 適応度予測 (残りの個体) ---
        if len(estimated_individuals) > 0:
            est_vectors = np.array([[get_nested_value(ind, k) for k in PARAMS] for ind in estimated_individuals])
            predicted_vals = saf_model.predict(est_vectors)
            
            # 予測値でfitnessを上書き (選択に使用するため)
            for i, ind in enumerate(estimated_individuals):
                ind['fitness'] = predicted_vals[i]

        # --- 5. 次世代生成 (EDA/GA) ---
        # 選択のためにpopulationリストを再構成（Top-Ncは真値、残りは予測値が入っている状態）
        # population変数はそのままオブジェクト参照しているので、中身のdictは更新されている
        
        next_generation: List[Chromosomes] = []
        
        while len(next_generation) < POPULATION_SIZE:
            # 選択 (Top-Ncの真値と、他個体の予測値が混在したfitnessを使用)
            selected = tournament.exec_tournament_selection(
                chromosomes_params=population, 
                participants_num=tournament_size
            )
            
            # 交叉 & 突然変異
            offspring = BLX_alpha.exec_blx_alpha(
                parents_chromosomes=selected,
                func_repair_gene=repair_gene,
                mutate=mutate
            )
            
            if isinstance(offspring, list):
                for ind in offspring:
                    if len(next_generation) >= POPULATION_SIZE:
                        break
                    if isinstance(ind, dict):
                        ind["fitness"] = 0.0
                        ind["pre_evaluation"] = 0.0
                        ind["true_fitness"] = 0.0
                        ind["generation"] = generation + 2
                        ind["chromosomeId"] = str(uuid.uuid4())
                        next_generation.append(ind)
            elif isinstance(offspring, dict):
                offspring["fitness"] = 0.0
                offspring["pre_evaluation"] = 0.0
                offspring["true_fitness"] = 0.0
                offspring["generation"] = generation + 2
                offspring["chromosomeId"] = str(uuid.uuid4())
                next_generation.append(offspring)

        population = next_generation

    # --- 最終世代の処理 ---
    # A. 属性頻度スコアの計算
    pref_scores = calculate_preference_scores(population, PARAMS)
    
    # B. スコアに基づいて個体群をソート (スコアが高い順)
    # 情報を付与してソートしやすくする
    for i, ind in enumerate(population):
        ind['pre_evaluation'] = pref_scores[i]
    
    # スコア降順でソート
    sorted_population = sorted(population, key=lambda x: x['pre_evaluation'], reverse=True)
    
    # C. 上位Nc個体 (Top-Nc) と 残りの個体 (Estimated) に分割
    top_nc_individuals = sorted_population[:TOP_NC]
    estimated_individuals = sorted_population[TOP_NC:]

    # --- 2. 真の評価 (Top-Ncのみ) ---
    # ユーザー評価のシミュレーション
    evaluate_fitness(
        population=top_nc_individuals,
        evaluate_num=evaluate_num,
        param_keys=PARAMS,
        noise_is_added=noise_is_added,
        target_key="fitness"
    )
    # true_fitnessに真値を格納 (Top-Ncは真値で確定)
    for ind in top_nc_individuals:
        ind['true_fitness'] = ind['fitness']

    # (ログ用: 残りの個体の真値も計算しておくが、学習には使わない)
    evaluate_fitness(
        population=estimated_individuals,
        evaluate_num=evaluate_num,
        param_keys=PARAMS,
        noise_is_added=noise_is_added
    )

    # --- ログ記録 (真値ベース) ---
    # 全体の中から真のベストを探す
    best_ind = max(population, key=lambda x: x['fitness'])
    best_fitness_history.append((generation + 1, float(best_ind["true_fitness"])))
    bests.append(copy.deepcopy(best_ind))

    total_fitness = sum(ind['fitness'] for ind in population)
    average_fitness = total_fitness / len(population)
    average_fitness_history.append((generation + 1, float(average_fitness)))

    if look and times == 1:
        # プロット
        plot_individual_params(
            population=population,
            best=best_ind,
            worst=min(population, key=lambda x: x['true_fitness']),
            param_keys=PARAMS,
            generation=generation + 1,
            file_path=f'./result/saf_ieda/graph/{evaluate_method}/scatter/{evaluate_method}_noise{str(noise_is_added)}_{str(POPULATION_SIZE)}_individuals_{str(generation + 1)}gens'
        )
    
    # --- 3. SAFモデルの学習 (Top-Ncの情報のみ使用) ---
    top_vectors = np.array([[get_nested_value(ind, k) for k in PARAMS] for ind in top_nc_individuals])
    top_fitness_vals = np.array([ind['fitness'] for ind in top_nc_individuals])
    
    saf_model.fit(top_vectors, top_fitness_vals)

    # --- 4. 適応度予測 (残りの個体) ---
    if len(estimated_individuals) > 0:
        est_vectors = np.array([[get_nested_value(ind, k) for k in PARAMS] for ind in estimated_individuals])
        predicted_vals = saf_model.predict(est_vectors)
        
        # 予測値でfitnessを上書き (選択に使用するため)
        for i, ind in enumerate(estimated_individuals):
            ind['fitness'] = predicted_vals[i]


    # 最終結果の出力
    log(f"result/saf_ieda/last_gen_individuals/{evaluate_method}/{str(POPULATION_SIZE)}inds/simulation_{evaluate_method}_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(times)}.json", population, times=times)
    log(f"result/saf_ieda/best/{evaluate_method}/{str(POPULATION_SIZE)}inds/best_individual_{evaluate_method}_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(times)}.json", bests, times=times)
    
    log_fitness(
        evaluate_num=evaluate_num,
        file_path=f"_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(POPULATION_SIZE)}_{str(times)}_best_fitness_history.png",
        best_fitness_history=best_fitness_history,
        average_fitness_history=average_fitness_history,
        ver="saf_ieda"
    )
    
    log_average_fitness(
        evaluate_num=evaluate_num,
        file_path=f"_noise{str(noise_is_added)}_{str(NUM_GENERATIONS)}gens_{str(POPULATION_SIZE)}_{str(times)}_average_fitness_history.json",
        average_fitness_history=average_fitness_history,
        times=times
    )

    return best_fitness_history, average_fitness_history
