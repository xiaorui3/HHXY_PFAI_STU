import json

# 定义输入和输出文件名
input_file_name = 'PFAI转换tsynbio格式_sft题8月9号新爬取的加之前的所有数据_只有答案.json.json'
output_file_name = 'formatted_pqal.json'

# 打开JSON文件并读取内容
with open(input_file_name, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 初始化一个空列表来存储格式化后的数据
formatted_entries = []

# 获取前10000个条目
entries_to_process = list(data.values())[:10000]

# 处理每个条目
for entry in entries_to_process:
    # 提取问题和详细回答
    question = entry["instruction"]
    long_answer = entry["output"]

    # 创建新的字典，包含问题和详细回答
    new_entry = {
        "instruction": question,
        "input": "",
        "output": long_answer
    }

    # 将新条目添加到列表中
    formatted_entries.append(new_entry)

# 将转换后的数据写入新的JSON文件
with open(output_file_name, 'w', encoding='utf-8') as output_file:
    json.dump(formatted_entries, output_file, indent=4)

print(f"The formatted data has been written to {output_file_name}.")