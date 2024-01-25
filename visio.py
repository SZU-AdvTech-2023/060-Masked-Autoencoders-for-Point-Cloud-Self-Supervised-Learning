import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def pc_show(pc_normal):
    lim = 0.5
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    x_vals = pc_normal[:, 0]
    y_vals = pc_normal[:, 1]
    z_vals = pc_normal[:, 2]
    # z_vals = - pc_normal[:, 2]

    # 根据点的z轴坐标大小，对点进行渐变颜色渲染
    # plasma
    cmap = matplotlib.cm.get_cmap('plasma')
    norm = matplotlib.colors.Normalize(vmin=min(z_vals), vmax=max(z_vals))
    colors = [cmap(norm(value)) for value in z_vals]

    # colors.reverse()
    ax.scatter(x_vals, y_vals, z_vals, c=colors, s=3, depthshade=True)
    ax.grid(False)
    ax.axis(False)
    plt.show()
b=np.loadtxt("/data1/gaoziqi/3DFaceMAE-copy/data/syn_Data/1.txt")
pc_show(b)