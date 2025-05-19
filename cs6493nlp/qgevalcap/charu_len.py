def insert_line(file_path, line_number, new_line):
    for i in range(0,line_number):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 插入新行
        lines.insert(line_number - 1, new_line + '\n')

        # 写回文件
        with open(file_path, 'w') as file:
            file.writelines(lines)

# 使用示例
file_path = 'src.txt'  # 替换为你的文件路径
line_number = 10000  # 你想要插入新行的行号
new_line = 'This is a new line.'  # 你想要插入的新行内容
insert_line(file_path, line_number, new_line)