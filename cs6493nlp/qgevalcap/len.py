def count_lines(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for line in file)

# 替换为你的文件路径
file_path = 'src.txt'
line_count = count_lines(file_path)
print(f"The file has {line_count} lines.")