import json
from tqdm import tqdm

file_name = 'PFAI转换tsynbio格式_sft题8月9号新爬取的加之前的所有数据_只有答案.json'
output_file_name = 'output.txt'  # 你想要保存output内容的TXT文件名
i=0
# 打开JSON文件并读取内容
with open(file_name, 'r') as file:
    data = json.load(file)

    # 如果data是一个列表，我们可以使用tqdm来遍历它
    if isinstance(data, list):
        for item in tqdm(data, desc='Processing items'):
            i=i+1
            if i==10001:
                break
            output = item.get('output', 'No output found')
            # 这里你可以处理每个item的output
            # 例如，你可以将它们写入到一个文件中
            with open(output_file_name, 'a') as output_file:  # 使用'a'以追加模式打开文件
                output_file.write(output + '\n')  # 假设每个output之间需要换行
    else:
        # 如果data不是一个列表，我们可以直接处理它
        output = data.get('output', 'No output found')
        with open(output_file_name, 'w') as output_file:
            output_file.write(output)

print(f"The output has been written to {output_file_name}.")