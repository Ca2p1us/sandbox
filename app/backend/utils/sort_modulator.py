def sort_modulators(modulators_params:list):
# モジュレータ周波数で昇順にソート,周波数が同じ場合は変調指数でソート（モジュレータの周波数がキャリアの周波数に依存しない実装の場合に有効）
    return sorted(modulators_params, key=lambda params: (float(params['ratioToFundamentalFrequency'])*float(params['frequency']), float(params['modulationIndex'])))

# キャリア周波数に対する比率で昇順にソート,同じ場合は変調指数でソート（モジュレータの周波数がキャリアの周波数に依存しない実装の場合に有効）
    # return sorted(modulators_params, key=lambda params: (float(params['ratioToFundamentalFrequency']),float(params['frequency']), float(params['modulationIndex'])))
# 変調指数でソート
    # return sorted(modulators_params, key=lambda params: (float(params['modulationIndex'])))
