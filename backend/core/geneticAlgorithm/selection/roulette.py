def _exec_roulette_wheel_selection(self) -> List[Chromosome]:
    """
    ルーレット選択を行い、交叉などで利用する2つの個体（染色体）を
    取得する。

    Returns
    -------
    selected_chromosomes : list of Chromosome
        選択された2つの個体（染色体）を格納したリスト。選択処理は評価関数
        （fitnessメソッド）による重みが設定された状態でランダムに抽出される。

    Notes
    -----
    評価関数の結果の値が負になる問題には利用できない。
    """
    weights: List[float] = [
        chromosome.get_fitness() for chromosome in self._population]
    selected_chromosomes: List[Chromosome] = choices(
        self._population, weights=weights, k=2)
    return selected_chromosomes

