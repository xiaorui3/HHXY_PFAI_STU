import numpy as np
import matplotlib.pyplot as plt


i = 0
file_path = 'result/raw_result__netzone22.json'
count = 0
xpoints = np.array([])
ypoints = np.array([])

plt.title("PFAI-tsynbio")
plt.xlabel("x ")
plt.ylabel("y ")
with open(file_path, 'r') as file:
    for line in file:
        i = i + 1

        if 'Right' in line:
            count += 1
        print(f"{i}   {file_path}的准确率为{count / i}")
        xpoints=np.append(xpoints,count / i)
        ypoints = np.append(ypoints, count / i)
# print(ypoints)

plt.hist(xpoints, bins=30, color='skyblue', alpha=0.8)
# 设置图表属性
plt.title('PFAI-tsynbio')
plt.xlabel('Value')
plt.ylabel('Frequency')
# 显示图表
plt.show()


plt.figure(figsize=(10, 8))  # 设置图形尺寸为 10x8 英寸
plt.grid(color = 'r', linestyle = '--', linewidth = 0.5)
plt.plot(xpoints, linestyle = 'dotted', marker = 'o')
plt.show()


