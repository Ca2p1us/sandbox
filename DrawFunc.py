import matplotlib.pyplot as plt
import numpy as np
import os

def plot_gauss_function():
    # 1. パラメータの設定
    mu = 250        # 平均 (μ)
    sigma = 75      # 標準偏差 (σ)
    x_min, x_max = 0, 500  # 定義域 (x軸の範囲)
    y_min, y_max = 0, 6    # 値域 (y軸の範囲)
    
    # ガウス関数のピーク（高さ）の設定
    # 注意: 通常の確率密度関数のピークは約0.008と低いため、
    # 指定された値域[0, 6]で見えるようにピークを6.0に設定します。
    amplitude = 6.0 

    # 2. データの生成
    x = np.linspace(x_min, x_max, 1000)
    
    # ガウス関数の計算: f(x) = A * exp( - (x - μ)^2 / (2σ^2) )
    y = amplitude * np.exp(-((x - mu)**2) / (2 * sigma**2))

    # 4. プロット設定
    fig, ax = plt.subplots(figsize=(8, 6)) # fig, ax を取得するように変更
    
    # 4. 指定された点の計算 (x=250, x=0)
    points_x = [250, 0]
    points_y = [amplitude * np.exp(-((px - mu)**2) / (2 * sigma**2)) for px in points_x]
    
    # ガウス関数の描画
    ax.plot(x, y, color='black', linewidth=1.5, label='Gaussian Curve')
    
    # 点のプロット (赤色、サイズ大きめ)
    ax.scatter(points_x[0], points_y[0], color='blue', s=100, zorder=5, label='Points')
    
    # 点の座標を表示
    
    ax.text(points_x[0], points_y[0] + 0.1, f'Ans', ha='center', fontsize=12)
    # ax.text(points_x[1]+20, points_y[1] + 0.2, f'worst', ha='center', fontsize=12)

    # 軸（枠線）の太さを変える
    axis_width = 1.5  # 太さの設定
    for spine in ax.spines.values():
        spine.set_linewidth(axis_width)

    # メモリ（数字）のフォントサイズと、メモリ自体の太さを変える
    ax.tick_params(axis='both', which='major', labelsize=16, width=axis_width, length=6)

    # x軸: 0, 100, 200, 300, 400, 500 (np.arangeは終了値を含まないので+1します)
    ax.set_xticks(np.arange(0, 501, 50)) 
    
    # y軸: 0, 1, 2, 3, 4, 5, 6
    ax.set_yticks(np.arange(0, 7, 1))
    
    # 軸の範囲設定
    ax.set_xlim(x_min-0.5, x_max+0.5)
    ax.set_ylim(y_min, y_max+0.5)
    
    # グリッドを消す
    ax.grid(False)
    
    # ラベル等の設定
    ax.set_title("")
    ax.set_xlabel('parameter value',fontsize=18)
    ax.set_ylabel('fitness',fontsize=18)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    # 4. グラフの保存と表示
    save_dir = "for_slide"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = os.path.join(save_dir, "gaussian_plot_styled.png")
    plt.savefig(filename,bbox_inches='tight')
    print(f"グラフを {filename} として保存しました。")
    
    plt.show()

def plot_gauss_func_two_peak():
    # 1. パラメータの設定
    mu1 = 150        # 平均1 (μ1)
    mu2 = 350        # 平均2 (μ2)
    sigma = 50      # 標準偏差 (σ)
    x_min, x_max = 0, 500  # 定義域 (x軸の範囲)
    y_min, y_max = 0, 6    # 値域 (y軸の範囲)
    
    # ガウス関数のピーク（高さ）の設定
    amplitude1 = 6.0 
    amplitude2 = 6.0

    # 2. データの生成
    x = np.linspace(x_min, x_max, 1000)
    
    # ガウス関数の計算: f(x) = A * [exp( - (x - μ1)^2 / (2σ^2) ) + exp( - (x - μ2)^2 / (2σ^2) )]
    y = amplitude1 * np.exp(-((x - mu1)**2) / (2 * sigma**2)) + amplitude1 * 0.5 * np.exp(-((x - mu2)**2) / (2 * sigma**2))

    # 4. プロット設定
    fig, ax = plt.subplots(figsize=(8, 6)) # fig, ax を取得するように変更
    point_x = [150]
    point_y = [amplitude1 * np.exp(-((px - mu1)**2) / (2 * sigma**2)) + amplitude1 * 0.5 * np.exp(-((px - mu2)**2) / (2 * sigma**2)) for px in point_x]
    # 点のプロット (赤色、サイズ大きめ)
    ax.scatter(point_x, point_y, color='blue', s=100, zorder=5, label='Point')
    ax.text(point_x[0], point_y[0] + 0.1, f'Ans', ha='center', fontsize=12)
    
    # ガウス関数の描画
    ax.plot(x, y, color='black', linewidth=1.5, label='Gaussian Curve with Two Peaks')
    
    # 軸（枠線）の太さを変える
    axis_width = 1.5  # 太さの設定
    for spine in ax.spines.values():
        spine.set_linewidth(axis_width)

    # メモリ（数字）のフォントサイズと、メモリ自体の太さを変える
    ax.tick_params(axis='both', which='major', labelsize=16, width=axis_width, length=6)

    # x軸: 0, 100, 200, 300, 400, 500 (np.arangeは終了値を含まないので+1します)
    ax.set_xticks(np.arange(0, 501, 50)) 
    
    # y軸: 0, 1, 2, 3, 4, 5, 6
    ax.set_yticks(np.arange(0, 7.5, 1))
    # 軸の範囲設定
    ax.set_xlim(x_min-0.5, x_max+0.5)
    ax.set_ylim(y_min, y_max+0.5)
    # グリッドを消す
    ax.grid(False)
    # ラベル等の設定
    ax.set_title("")
    ax.set_xlabel('parameter value',fontsize=18)
    ax.set_ylabel('fitness',fontsize=18)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # 4. グラフの保存と表示
    save_dir = "for_slide"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, "gaussian_two_peak_plot_styled.png")
    plt.savefig(filename,bbox_inches='tight')
    print(f"グラフを {filename} として保存しました。")
    plt.show()
    return


def plot_gaussian_add_cosine():
    # 1. パラメータの設定
    mu = 250        # 平均 (μ)
    sigma = 100      # 標準偏差 (σ)
    f = 0.02         # 周波数
    x_min, x_max = 0, 500  # 定義域 (x軸の範囲)
    y_min, y_max = 0, 7    # 値域 (y軸の範囲)
    
    # ガウス関数のピーク（高さ）の設定
    # 注意: 通常の確率密度関数のピークは約0.008と低いため、
    # 指定された値域[0, 6]で見えるようにピークを6.0に設定します。
    amplitude = 6.0 

    # 2. データの生成
    x = np.linspace(x_min, x_max, 1000)
    
    # ガウス関数の計算: f(x) = A * exp( - (x - μ)^2 / (2σ^2) )
    y = amplitude * np.exp(-((x - mu)**2) / (2 * sigma**2)) + 0.5 * np.cos(2 * np.pi * f * x)

    # 4. プロット設定
    fig, ax = plt.subplots(figsize=(8, 6)) # fig, ax を取得するように変更
    
    # 4. 指定された点の計算 (x=250, x=0)
    points_x = [250, 0]
    points_y = [amplitude * np.exp(-((px - mu)**2) / (2 * sigma**2))+ 0.5 * np.cos(2 * np.pi * f * px) for px in points_x]
    
    # ガウス関数の描画
    ax.plot(x, y, color='black', linewidth=1.5, label='Gaussian Curve')
    
    # 点のプロット (赤色、サイズ大きめ)
    ax.scatter(points_x[0], points_y[0], color='blue', s=100, zorder=5, label='Points')
    
    # 点の座標を表示
    
    ax.text(points_x[0], points_y[0] + 0.1, f'Ans', ha='center', fontsize=12)
    # ax.text(points_x[1]+20, points_y[1] + 0.2, f'worst', ha='center', fontsize=12)

    # 軸（枠線）の太さを変える
    axis_width = 1.5  # 太さの設定
    for spine in ax.spines.values():
        spine.set_linewidth(axis_width)

    # メモリ（数字）のフォントサイズと、メモリ自体の太さを変える
    ax.tick_params(axis='both', which='major', labelsize=16, width=axis_width, length=6)

    # x軸: 0, 100, 200, 300, 400, 500 (np.arangeは終了値を含まないので+1します)
    ax.set_xticks(np.arange(0, 501, 50)) 
    
    # y軸: 0, 1, 2, 3, 4, 5, 6
    ax.set_yticks(np.arange(0, 7.5, 1))
    
    # 軸の範囲設定
    ax.set_xlim(x_min-0.5, x_max+0.5)
    ax.set_ylim(y_min, y_max+0.5)
    
    # グリッドを消す
    ax.grid(False)
    
    # ラベル等の設定
    ax.set_title("")
    ax.set_xlabel('parameter value',fontsize=18)
    ax.set_ylabel('fitness',fontsize=18)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    # 4. グラフの保存と表示
    save_dir = "for_slide"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = os.path.join(save_dir, "gaussian_cosine_plot_styled.png")
    plt.savefig(filename,bbox_inches='tight')
    print(f"グラフを {filename} として保存しました。")
    
    plt.show()

def plot_Ackley_function():
    A = 20
    B = 0.04
    C = 0.04
    x_min, x_max = 0, 500
    y_min, y_max = 0,6
    mu = 250

    x = np.linspace(x_min, x_max, 1000)
    ackley_val = [(-A * np.exp(-B * np.sqrt(0.5 * ((xi - mu)**2))) - np.exp(0.5 * (np.cos(C * (xi - mu)))) + A + np.exp(1))
         for xi in x ]
    max_ackley_val = A + np.e
    y = [6.0 * (1.0 - val / max_ackley_val) for val in ackley_val]

    fig, ax = plt.subplots(figsize=(8, 6))
    # 4. 指定された点の計算 (x=250, x=0)
    points_x = [250, 0]
    val = [(-A * np.exp(-B * np.sqrt(0.5 * ((px - mu)**2))) - np.exp(0.5 * (np.cos(C * (px - mu)))) + A + np.exp(1)) for px in points_x]

    points_y = [6.0 * (1.0 - val / max_ackley_val) for val in val]
    
    # ガウス関数の描画
    ax.plot(x, y, color='black', linewidth=1.5, label='Gaussian Curve')
    
    # 点のプロット (赤色、サイズ大きめ)
    ax.scatter(points_x[0], points_y[0], color='blue', s=100, zorder=5, label='Points')
    
    # 点の座標を表示
    
    ax.text(points_x[0]+ 20, points_y[0], f'Ans', ha='center', fontsize=12)
    # ax.text(points_x[1]+20, points_y[1] + 0.2, f'worst', ha='center', fontsize=12)
    # 軸（枠線）の太さを変える
    axis_width = 1.5  # 太さの設定
    for spine in ax.spines.values():
        spine.set_linewidth(axis_width)
    ax.plot(x, y, color='black', linewidth=1.5)
    ax.set_title("")
    ax.set_xlabel('parameter value',fontsize=18)
    ax.set_ylabel('fitness',fontsize=18)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # メモリ（数字）のフォントサイズと、メモリ自体の太さを変える
    ax.tick_params(axis='both', which='major', labelsize=16, width=axis_width, length=6)
    ax.set_xticks(np.arange(0, 501, 50)) 
    ax.set_xlim(x_min-0.5, x_max)
    ax.set_ylim(y_min, y_max+0.5)
    # plt.ylim(-10, 10)
    ax.grid(False)
    save_dir = "for_slide"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, "Ackley_function_plot_styled.png")
    plt.savefig(filename,bbox_inches='tight')
    print(f"グラフを {filename} として保存しました。")
    plt.show()


def plot_sphere_function():
    x_min, x_max = 0, 500
    mu = 250
    x = np.linspace(x_min, x_max, 400)
    y = [-(xi - mu)**2 for xi in x ]
    fig, ax = plt.subplots(figsize=(8, 6))
    # 4. 指定された点の計算 (x=250, x=0)
    points_x = [250, 0]
    points_y = [-(px - mu)**2 for px in points_x]
    # ガウス関数の描画
    ax.scatter(points_x[0], points_y[0], color='blue', s=100, zorder=5, label='Points')
    # 点の座標を表示
    ax.text(points_x[0], points_y[0] - 3000, f'Ans', ha='center', fontsize=12)
    # 軸（枠線）の太さを変える
    axis_width = 1.5  # 太さの設定
    for spine in ax.spines.values():
        spine.set_linewidth(axis_width)
    ax.plot(x, y, color='black', linewidth=1.5)
    ax.set_title("")
    ax.set_xlabel('parameter value',fontsize=18)
    ax.set_ylabel('fitness',fontsize=18)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # メモリ（数字）のフォントサイズと、メモリ自体の太さを変える
    ax.tick_params(axis='both', which='major', labelsize=16, width=axis_width, length=6)
    ax.set_xlim(x_min-0.5, x_max)
    # plt.ylim(-10000, 0)
    ax.grid(False)
    save_dir = "for_slide"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, "Sphere_function_plot_styled.png")
    plt.savefig(filename,bbox_inches='tight')
    print(f"グラフを {filename} として保存しました。")
    plt.show()


if __name__ == "__main__":
    plot_gauss_function()
    plot_gauss_func_two_peak()
    plot_gaussian_add_cosine()
    plot_Ackley_function()
    plot_sphere_function()