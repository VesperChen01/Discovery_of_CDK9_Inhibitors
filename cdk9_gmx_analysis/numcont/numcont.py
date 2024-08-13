# 导入所需的库
import matplotlib.pyplot as plt

def plot_data():
    # 提供的数据
    data = {
        "VAL301": 0.02, "ARG250": 0.34, "ALA381": 0.34, "MET261": 0.98, "THR330": 1.00, "ILE314": 2.10,
        "GLU249": 3.10, "ASN321": 5.88, "MET292": 6.38, "GLU320": 7.94, "ASP382": 16.48, "ASP326": 22.12,
        "SER323": 23.30, "LYS273": 31.79, "LEU371": 38.71, "ALA271": 41.45, "TYR318": 45.89, "LEU251": 50.25,
        "VAL259": 51.39, "MET319": 99.94
    }

    # 将数据分解为两个列表：标签和值，并按照值的大小排序
    labels, values = zip(*sorted(data.items(), key=lambda x: x[1]))

    # 创建一个条形图
    plt.figure(figsize=(10, 8))
    plt.barh(labels, values, color='skyblue')
    plt.xlabel('% of occupancy')
    #plt.title('Percentage Representation of Variables')
    plt.tight_layout()

    # 显示图表
    plt.show()

if __name__ == "__main__":
    plot_data()
