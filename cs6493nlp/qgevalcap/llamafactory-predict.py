import json

# 假设你的JSONL文件名为data.jsonl
file_name = 'generated_predictions.jsonl'
output_file_name = 'data.txt'  # 输出文件名

# 打开文件并逐行读取
with open(file_name, 'r') as file, open(output_file_name, 'w') as output_file:
    for line in file:
        data = json.loads(line)

        prediction = data.get('predict', 'No prediction found')

        output_file.write(prediction + '\n')