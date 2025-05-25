import matplotlib.pyplot as plt

# 数据
strides = [1, 2, 4, 8]
bandwidths = [532.121, 183.471, 90.9098, 47.2128]

# 绘制折线图
plt.figure(figsize=(8, 6))
plt.plot(strides, bandwidths, marker='o', linestyle='-', color='b')
plt.xlabel('Stride')
plt.ylabel('Bandwidth (GB/s)')
plt.title('Bandwidth vs Stride')
plt.grid(True)
plt.xscale('log', base=2)  # 使用对数刻度以更好地展示 stride 的变化
plt.xticks(strides, strides)  # 设置 x 轴刻度
plt.show()