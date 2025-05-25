import matplotlib.pyplot as plt

# 数据
strides = [1, 2, 4, 8, 16, 32]
bandwidth_bitwidth_2 = [4227.10, 4019.72, 2145.04, 831.123, 423.803, 214.139]
bandwidth_bitwidth_4 = [8529.12, 4301.24, 2025.68, 1009.35, 503.37, 249.593]
bandwidth_bitwidth_8 = [8632.03, 4342.10, 2187.54, 1099.67, 554.569, 543.169]

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(strides, bandwidth_bitwidth_2, marker='o', linestyle='-', color='r', label='Bitwidth=2')
plt.plot(strides, bandwidth_bitwidth_4, marker='s', linestyle='-', color='g', label='Bitwidth=4')
plt.plot(strides, bandwidth_bitwidth_8, marker='^', linestyle='-', color='b', label='Bitwidth=8')
plt.xlabel('Stride')
plt.ylabel('Bandwidth (GB/s)')
plt.title('Bandwidth vs Stride for Different Bitwidths')
plt.grid(True)
plt.xscale('log', base=2)  # 使用对数刻度以更好地展示 stride 的变化
plt.xticks(strides, strides)  # 设置 x 轴刻度
plt.legend()  # 显示图例
plt.show()