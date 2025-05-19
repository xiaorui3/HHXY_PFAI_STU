i = 0
file_path = 'D:\\赵锐\\蛋白质大模型\\data_temp\\result\\画图\\rst_compar__xiaorui1-498.txt'
count = 0
with open(file_path, 'r') as file:
    for line in file:
        i = i + 1
        if i>=401 and i<=498:
            if 'Right' in line:
                count += 1
        if i>498:
            break
print(f"{file_path}的准确率为{count / 100}  {count}  {i}")
