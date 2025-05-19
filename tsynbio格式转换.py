import json

input_file_path = 'ProteinLMBench.json'

output_file_path = 'extracted_data(2).json'

with open(input_file_path, 'r') as file:
    data = json.load(file)
i=0
extracted_data = []
for item in data:
    i=i+1
    extracted_item=[]
    if(i>=801 and i<=944):
        extracted_item = {
            'question': item['question'],
            'options': item['options'],
            'answer': item['answer']
        }
        extracted_data.append(extracted_item)
    if i==498:
        print(extracted_item)

with open(output_file_path, 'w') as file:
    json.dump(extracted_data, file, indent=4)
