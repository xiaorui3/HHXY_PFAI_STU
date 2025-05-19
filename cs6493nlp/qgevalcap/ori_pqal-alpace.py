import json

with open('ori_pqal(1).json', 'r') as file:
    data = json.load(file)

formatted_data = []

# 遍历原始数据中的每个条目
for entry in data.values():
    # 提取问题和详细回答
    question = entry["QUESTION"]
    long_answer = entry["LONG_ANSWER"]

    new_entry = {
        "instruction": question,
        "input": "",
        "output": long_answer
    }

    # 将新条目添加到列表中
    formatted_data.append(new_entry)

# 将转换后的数据写入新的JSON文件
with open('formatted_pqal.json', 'w') as file:
    json.dump(formatted_data, file, indent=4)

print("数据转换完成并保存到formatted_pqal.json文件。")