import subprocess
import os
import threading
import pandas as pd
import json
from tqdm import tqdm
import ijson
__author__ = 'xinya'
import sys
sys.path.append('../')
import yaml
from bleu1.bleu import Bleu
from meteor1.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from collections import defaultdict
from argparse import ArgumentParser
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def multichoice(model_name,index):
    QnA_dir_path = 'PFAI/ProteinLMBench.json'
    with open(QnA_dir_path, 'r') as f:
        file_data = json.load(f)

    model_path = f'{model_name}'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto").eval()

    import re
    answer_list = [f['answer'] for f in file_data]
    answer_list = [re.search(r'\d+', a).group() for a in answer_list]

    prompt = ("""
Answer the multiple-choice question based solely on the provided context. 
If you are still unsure about the answer, output option 7.
Select only ONE correct option by its number. Start your response with 'The correct option is' followed by the option number ONLY. eg: "The correct option is Option X."
Think step by step.
    """)
    question = []

    for f in file_data:
        options = ''
        for o in f['options']:
            options += o + '\n'
        sb = prompt + '\n Question: \n' + f['question'] + '\n Options: \n' + options + '\nThe correct option is:'
        question.append(sb)

    inputs = []
    tokenizer.pad_token = tokenizer.eos_token
    for q in question:
        # for q in question:
        #     a = tokenizer.apply_chat_template([{"role": "user", "content": q}], return_tensors="pt").to("cuda")
        a = tokenizer(q, return_tensors="pt", padding=True)
        input_ids = a.input_ids.to('cuda')
        inputs.append(input_ids)

    print(len(inputs))

    chat_model = ('chat' in model_name) or ('Chat' in model_name)
    if 'Yi' in model_name:
        chat_model = False
    from tqdm import tqdm
    output_list = []

    temp = 0.1
    mnt = 20
    ppp = 0
    import warnings
    from transformers import logging
    hhxyname="hhxy_"+str(index)
    logging.set_verbosity_info()
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    # chat_model=ture
    for q in tqdm(inputs[:]):
        wh = 0
        while True:  # 创建一个无限循环
            arr1 = model.generate(q, max_new_tokens=mnt, do_sample=True, temperature=temp, top_p=0.7)

            qq = tokenizer.decode(q[0], skip_special_tokens=True)
            str1 = tokenizer.decode(arr1[0], skip_special_tokens=True)

            print(qq)

            # 移除用户提问的数据q
            str1 = str1.replace(qq, "")

            # 打印生成的文本
            # print("生成的文本为：", str1)

            # 如果str1不为空且不只包含空格，则退出循环
            if str1.strip():
                break
            else:
                # 如果str1为空或只包含空格，可以选择打印一条消息，然后继续循环
                wh = wh + 1
                print(f"{model_name} 第{wh}次生成的文本为: &{str1}&，正在重新生成...")

                if wh == 3:
                    print(model_name + " 3次均未成功生成")
                    break
                # 可选：在这里添加一些延迟或随机性，以避免潜在的无限循环问题
                # time.sleep(1)  # 例如，这里可以添加1秒的延迟
        output_list.append(arr1)
        lst = [tokenizer.decode(i[0], skip_special_tokens=True) for i in output_list]
        after = []
        lkj = 0
        for i, j in zip(lst, question):
            # print(lst[lkj])
            lkj += 1
            # for i, j in zip(output_list, question):
            after.append(i.replace(j, ''))
            # print(f'模型回答：{i}\n问题：{j}')
            print(f'--------这里: \n模型回答：{i}\n')
        #  print(after)

        # print(after)
        v_ans = []
        non_number = 0
        for o in after:
            # print(o)
            try:
                # v_ans.append(re.search(r'The correct option number is: (\d+)', o).group(1))
                v_ans.append(re.search(r'\d+', o).group())
            except:
                non_number += 1
                v_ans.append("None")

        # print(non_number)
        psd = 0
        # wrong_list = []
        from datetime import datetime
        now = datetime.now()
        formatted_time = now.strftime("%Y%m%d_%H%M%S")
        if "/" in model_name:
            model_name = model_name.split("/")[2]

        with open(f'PFAI/result/raw_result__{hhxyname}.json', 'w') as jj:
            json.dump(after, jj)

        with open(f'PFAI/result/rst_compar__{hhxyname}.txt', 'w') as results:
            for i in range(len(v_ans)):
                print(f'模型给出的答案是：{v_ans[i]}   正确答案是：{answer_list[i]}')
                if v_ans[i] != answer_list[i]:
                    results.write(str(v_ans[i]) + "   " + str(answer_list[i]))
                    results.write("\n")
                    continue
                else:
                    results.write("Right")
                    psd += 1
                    results.write("\n")

        accuracy = psd / len(v_ans)
        print('correct rate: ' + str(psd / len(v_ans)))
        ppp = ppp + 1
        # print(model.generate(q, max_new_tokens=mnt, do_sample=True, temperature=temp))
    lst = [tokenizer.decode(i[0], skip_special_tokens=True) for i in output_list]

    after = []
    for i, j in zip(lst, question):
        # for i, j in zip(output_list, question):
        after.append(i.replace(j, ''))

    # print(after)
    v_ans = []
    non_number = 0
    for o in after:
        try:
            # v_ans.append(re.search(r'The correct option number is: (\d+)', o).group(1))
            v_ans.append(re.search(r'\d+', o).group())
        except:
            non_number += 1
            v_ans.append("None")

    print(non_number)
    psd = 0
    # wrong_list = []
    from datetime import datetime
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")
    if "/" in model_name:
        model_name = model_name.split("/")[2]

    with open(f'PFAI/result/raw_result__{hhxyname}.json', 'w') as jj:
        json.dump(after, jj)

    with open(f'PFAI/result/rst_compar__{hhxyname}.txt', 'w') as results:
        for i in range(len(v_ans)):
            # print(i)
            if v_ans[i] != answer_list[i]:
                results.write(str(v_ans[i]) + "   " + str(answer_list[i]))
                results.write("\n")
                continue
            else:
                results.write("Right")
                psd += 1
                results.write("\n")

    accuracy = psd / len(v_ans)
    print('correct rate: ' + str(psd / len(v_ans)))

    del tokenizer
    del model
    return accuracy


class QGEvalCap:
    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self):
        output = []
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        # print('gts',self.gts)
        # print('res',self.res)

        # =================================================
        # Compute scores
        # =================================================

        # gts = word_tokenize(self.gts['0'][0].decode())
        # res = word_tokenize(self.res['0'][0].decode())
        # print(meteor_score.meteor_score([gts], res))
        for scorer, method in scorers:
            # print 'computing %s score...'%(scorer.method())
            # print(self.gts[0],self.res[0])
            score, scores = scorer.compute_score(self.gts, self.res)
            print('success return')
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.5f" % (m, sc))
                    output.append(sc)
            else:
                print("%s: %0.5f" % (method, score))
                output.append(score)
        # exit(0)
        # 45.22	29.94	22.01	16.76
        # Bleu_1: 0.26487
        # Bleu_2: 0.10851
        # Bleu_3: 0.05112
        # Bleu_4: 0.02147
        return output


def eval(out_file, src_file, tgt_file, isDIn=False, num_pairs=500):
    """
        Given a filename, calculate the metric scores for that prediction file

        isDin: boolean value to check whether input file is DirectIn.txt
    """

    pairs = []
    with open(src_file, 'r') as infile:
        for line in infile:
            pair = {}
            pair['tokenized_sentence'] = line[:-1]
            pairs.append(pair)

    with open(tgt_file, "r") as infile:
        cnt = 0
        for line in infile:
            pairs[cnt]['tokenized_question'] = line[:-1]
            cnt += 1

    output = []
    with open(out_file, 'r') as infile:
        for line in infile:
            line = line[:-1]
            output.append(line)

    for idx, pair in enumerate(pairs):
        pair['prediction'] = output[idx]

    ## eval
    from eval import QGEvalCap
    import json
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.4f')

    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])
    for pair in pairs[:]:
        key = pair['tokenized_sentence']
        res[key] = [pair['prediction'].encode('utf-8')]

        ## gts
        gts[key].append(pair['tokenized_question'].encode('utf-8'))

    QGEval = QGEvalCap(gts, res)
    return QGEval.evaluate()



def print_colorful(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m")

# 函数：实时打印输出
def print_output(pipe, name):
    for line in iter(pipe.readline, ''):  # 逐行读取
        print(f"{name}: {line}", end='')  # 打印每一行
# 函数：执行命令
def execute_command(command, cuda_visible_devices):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    print_colorful("设置CUDA环境变量，使用以下GPU卡ID: "+cuda_visible_devices,91)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    stdout_thread = threading.Thread(target=print_output, args=(process.stdout,"PFAI_RUN"))
    stderr_thread = threading.Thread(target=print_output, args=(process.stderr,"PFAI_ERR"))

    stdout_thread.start()
    stderr_thread.start()

    process.wait()
    stdout_thread.join()
    stderr_thread.join()



def print_colorful(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
num_rows_to_execute = 23
model_name_or_path="/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/qwen27/"
dataset="PFAI转换tsynbio格式_sft题8月9号新爬取的加之前的所有数据_只有答案"
hhxy_max=[]
weitiao=0
wenda=0
pfai_huida=0
print("\033[91m程序将执行"+f'{num_rows_to_execute}次'+"\033[0m")
print("\033[91m================PFAI================ \n共三个问题1.是否微调模型? 2.是为否执行自动化问答? 3.是否运行模型蛋白质答题预测和模型导出? 输入1为执行 0为不执行 \033[0m")
print("\033[91m！！！！！请准确输入数字！否则程序将出现异常及其bug，出现bug不给予解答！！！！！！ 按任意键确认\033[0m")
__=input()
print("\033[91m是否微调模型?\033[0m")
weitiao=int(input())
print("\033[91m是否执行自动化问答?\033[0m")
wenda=int(input())
print("\033[91m是否运行模型蛋白质答题预测和模型导出?\033[0m")
pfai_huida=int(input())

print("\033[91m================PFAI================\033[0m")
print("\033[94m启动自动评测脚本\033[0m")
print("\033[94m执行自动设置环境变量\033[0m")
new_path = "/opt/conda/bin/"
command = ['which', 'python']
result = subprocess.run(['bash', '-c', f'export PATH="{new_path}:$PATH" && {" ".join(command)}'], capture_output=True, text=True)
print("执行结果：", result.stdout)
print("\033[91m================PFAI================\033[0m")

print("\033[94m转换为LLaMA-Factory工作目录：/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/LLaMA-Factory\033[0m")
os.chdir('/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/LLaMA-Factory')
print_colorful("启动LLaMA-Factory训练过程...", "93")
print_colorful("创建/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/LLaMA-Factory/PFAI 文件夹",91)
#  nohup python 自动化脚本.py > output.log 2> error.log &
file_path = '../工作簿2.xlsx'
df = pd.read_excel(file_path)
path = "PFAI"

if not os.path.exists(path):
    os.makedirs(path)
    print("文件夹创建成功")
else:
    print("文件夹已存在，继续执行后续命令")

# 定义命令的基础参数
base_command = [
    'llamafactory-cli', 'train',
    '--stage', 'sft',
    '--do_train', 'True',
    '--model_name_or_path', f'{model_name_or_path}',
    '--preprocessing_num_workers', '16',
    '--finetuning_type', 'lora',
    '--template', 'default',
    '--flash_attn', 'fa2',
    '--dataset_dir', 'data',
    '--dataset', f'{dataset}',
    '--cutoff_len', '102400',
    '--learning_rate', '{learning_rate}',
    '--num_train_epochs', '{num_train_epochs}',
    '--max_samples', '10000',
    '--per_device_train_batch_size', '2',
    '--gradient_accumulation_steps', '8',
    '--lr_scheduler_type', 'cosine',
    '--max_grad_norm', '{max_grad_norm}',
    '--logging_steps', '1',
    '--save_steps', '100',
    '--warmup_steps', '0',
    '--optim', 'adamw_torch',
    '--packing', 'False',
    '--report_to', 'none',
    '--output_dir', '{output_dir}',
    '--fp16', 'True',  # 默认设置为True，如果计算类型是fp32，则需要移除
    '--plot_loss', 'True',
    '--ddp_timeout', '180000000',
    '--include_num_input_tokens_seen', 'True',
    '--quantization_bit', '4',
    '--quantization_method', 'bitsandbytes',
    '--lora_rank', '{lora_rank}',
    '--lora_alpha', '{lora_alpha}',
    '--lora_dropout', '0',
    '--use_rslora', 'True',
    '--lora_target', 'all'
]


dddd=""
lora_z=""
lora_sf=""
for index, row in df.head(num_rows_to_execute).iterrows():

    # 替换命令中的占位符
    command = [
        arg.format(
            learning_rate=row['学习率'],
            num_train_epochs=row['训练轮数'],
            max_grad_norm=row['最大梯度范围'],
            lora_rank=row['Lora秩'],
            lora_alpha=row['LoRA 缩放系数'],
            output_dir=f'PFAI/Qwen2-7B/lora/train_{index+1}_{row["Lora秩"]}_{row["LoRA 缩放系数"]}_{row["学习率"]}'
            # output_dir=f'./test/train_{index + 1}_{row["Lora秩"]}_{row["LoRA 缩放系数"]}'
        ) for arg in base_command
    ]
    lora_z=row["Lora秩"]
    lora_sf=row["LoRA 缩放系数"]
    dddd=f'./PFAI/Qwen2-7B/lora/train_{index + 1}_{row["Lora秩"]}_{row["LoRA 缩放系数"]}_{row["学习率"]}'
    print("\033[91m================PFAI================\033[0m")
    print_colorful("训练开始 第"+str(index+1)+"轮",94)
    print_colorful("学习率："+str(row['学习率']),94)
    print_colorful("训练轮数：" + str(row['训练轮数']), 94)
    print_colorful("最大梯度范围：" + str(row['最大梯度范围']), 94)
    print_colorful("Lora秩：" + str(row['Lora秩']), 94)
    print_colorful("LoRA 缩放系数：" + str(row['LoRA 缩放系数']), 94)
    print_colorful("输出目录：" +dddd , 94)
    print_colorful("index  :  "+f'{index}',94)

    print("将要执行以下命令: ", " ".join(command))

    print("\033[91m================PFAI================\033[0m")

    if weitiao==1:
        execute_command(command, '0')
    else:
        print_colorful("将不执行微调，继续执行后续命令", 91)



    print("\033[91m================PFAI================\033[0m")
    print_colorful("模型已保存到目录" + dddd, 91)

    print_colorful("执行文件提前脚本.....", 91)
    print_colorful("将"+f'{dataset}'+"提前前10000个", 91)
    input_file_name = f'./data/{dataset}.json'
    output_file_name = './PFAI/formatted_pqal.json'

    file_size = os.path.getsize(input_file_name)

    with tqdm(total=file_size, desc="加载数据进度", unit='B') as pbar:
        chunks = []
        chunk_size = 1024 * 1024  # 1MB chunk size

        with open(input_file_name, 'r', encoding='utf-8') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                pbar.update(len(chunk))
                chunks.append(chunk)

    json_string = ''.join(chunks)

    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误：{e}")
        exit(1)

    formatted_entries = []

    entries_to_process = data[:10000]  # 只处理前10000个条目

    for entry in tqdm(entries_to_process, desc="Processing entries"):
        # 提取问题和详细回答
        instruction = entry["instruction"]
        long_answer = entry["output"]

        # 分割instruction为问题和选项
        parts = instruction.split("Options:", 1)
        question = parts[0].strip()
        options = parts[1].strip()

        # 提取选项
        option_lines = options.split("\n")
        options_list = []
        for line in option_lines:
            if line.startswith("option"):
                option = line.split(":", 1)[1].strip()
                options_list.append(option)

        # 创建新的条目
        new_entry = {
            "instruction": question,
            "input": "",
            "output": long_answer
        }

        formatted_entries.append(new_entry)

    with open(output_file_name, 'w', encoding='utf-8') as output_file:
        json.dump(formatted_entries, output_file, indent=4)

    print_colorful(f"文件生成成功： {output_file_name}.", 91)


    output_dir='./PFAI/Qwen2-7B/lora/eval_' + f'{index + 1}_{row["Lora秩"]}_{row["LoRA 缩放系数"]}_{row["学习率"]}'
    command = [
        'llamafactory-cli', 'train',
        '--stage', 'sft',
        '--model_name_or_path', f'{model_name_or_path}',
        '--preprocessing_num_workers', '16',
        '--finetuning_type', 'lora',
        '--quantization_method', 'bitsandbytes',
        '--template', 'default',
        '--flash_attn', 'fa2',
        '--dataset_dir', 'PFAI',
        '--eval_dataset', 'formatted_pqal',
        '--cutoff_len', '102400',
        '--max_samples', '100000',
        '--per_device_eval_batch_size', '2',
        '--predict_with_generate', 'True',
        '--max_new_tokens', '512',
        '--top_p', '0.7',
        '--temperature', '0.95',
        '--output_dir', output_dir,
        '--do_predict', 'True',
        '--adapter_name_or_path', f'{dddd}',
        '--quantization_bit', '4'
    ]
    print("将要执行以下命令: ", " ".join(command))

    if wenda == 1:
        execute_command(command, '0')
    else:
        print_colorful("将不执行问答，程序停止", 91)
        print("\033[91m================PFAI================\033[0m")

    print("\033[91m================PFAI================\033[0m")
    print_colorful("将启动文本提取及其转换工程",91)
    path = "PFAI/res_xiaorui"

    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path}'+" 文件夹创建成功")
    else:
        print(f'{path}'+" 文件夹已存在，继续执行后续命令")

    print("\033[91m================PFAI================\033[0m")
    print_colorful("启动分离alpaca的output工程", 91)
    file_name = f'data/{dataset}'+".json"
    output_file_name = 'PFAI/res_xiaorui/input.txt'  # 你想要保存output内容的TXT文件名
    os.remove(output_file_name)
    i = 0
    # 打开JSON文件并读取内容
    with open(file_name, 'r') as file:
        data = json.load(file)

        # 如果data是一个列表，我们可以使用tqdm来遍历它
        if isinstance(data, list):
            for item in tqdm(data, desc='加载中'):
                i = i + 1
                if i == 10001:
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

    print(f"生成成功： {output_file_name}.")


    def count_lines(file_path):
        with open(file_path, 'r') as file:
            return sum(1 for _ in file)
    line_count = count_lines(output_file_name)
    print(f"共有{line_count} 行.")

    print("\033[91m================PFAI================\033[0m")
    print_colorful("启动分离generated_predictions程序", 91)

    file_name = f'{output_dir}'+'/generated_predictions.jsonl'
    output_file_name = 'PFAI/res_xiaorui/data.txt'  # 输出文件名

    # 打开文件并逐行读取
    with open(file_name, 'r') as file, open(output_file_name, 'w') as output_file:
        for line in file:
            data = json.loads(line)

            prediction = data.get('predict', 'No prediction found')

            output_file.write(prediction + '\n')

    line_count = count_lines(output_file_name)
    print(f"共有{line_count} 行.")
    print("\033[91m================PFAI================\033[0m")
    print_colorful("启动自动填充程序", 91)


    def insert_line(file_path, line_number, new_line):
        for i in range(0, line_number):
            with open(file_path, 'r') as file:
                lines = file.readlines()
            # 插入新行
            lines.insert(line_number - 1, new_line + '\n')
            # 写回文件
            with open(file_path, 'w') as file:
                file.writelines(lines)


    line_number = 10000  # 你想要插入新行的行号
    new_line = 'This is a new line.'  # 你想要插入的新行内容
    tttt="PFAI/res_xiaorui/src.txt"
    if not os.path.exists(tttt):
        f = open(tttt, 'w')
        print(tttt)
        f.close()
    else:
        print(tttt + "文件已存在")
    insert_line("PFAI/res_xiaorui/src.txt", line_number, new_line)
    line_count = count_lines("PFAI/res_xiaorui/src.txt")
    print(f"共有{line_count} 行.")
    if line_count>10000:
        os.remove("PFAI/res_xiaorui/src.txt")
        line_number = 10000  # 你想要插入新行的行号
        new_line = 'This is a new line.'  # 你想要插入的新行内容
        tttt = "PFAI/res_xiaorui/src.txt"
        if not os.path.exists(tttt):
            f = open(tttt, 'w')
            print(tttt)
            f.close()
        else:
            print(tttt + "文件已存在")
        insert_line("PFAI/res_xiaorui/src.txt", line_number, new_line)
        line_count = count_lines("PFAI/res_xiaorui/src.txt")
        print(f"共有{line_count} 行.")

    yaml_file_path = './PFAI/llama3_lora_sft.yaml'  # 替换为你的YAML文件路径
    # 读取YAML文件
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    # 修改指定的数据
    data['model_name_or_path'] = f'{model_name_or_path}'
    data['adapter_name_or_path'] = f'{dddd}'
    data['export_dir'] = './PFAI/models/' + f'{index + 1}_{row["Lora秩"]}_{row["LoRA 缩放系数"]}_{row["学习率"]}'

    # 将修改后的数据写回文件if
    if pfai_huida==1:
        with open(yaml_file_path, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, default_flow_style=False, allow_unicode=True)
        model_path_test01 = data['export_dir']
        print("YAML文件已更新。")
        test01 = data['export_dir']
        print("\033[91m================PFAI================\033[0m")
        print_colorful("启动模型导出程序，模型将要导出到" + f'{test01}目录中', 94)
        command = [
            'llamafactory-cli',
            'export',
            'PFAI/llama3_lora_sft.yaml'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout_thread = threading.Thread(target=print_output, args=(process.stdout, 'PFAI_RUN'))
        stderr_thread = threading.Thread(target=print_output, args=(process.stderr, 'PFAI_ERR'))
        stdout_thread.start()
        stderr_thread.start()
        process.wait()
        stdout_thread.join()
        stderr_thread.join()
        print("\033[91m================PFAI================\033[0m")
        print_colorful("模型导出成功在" + f'{test01}目录中，正在准备执行944道题问答程序.......', 94)
        model_name_list = [
            f'{model_path_test01}'
        ]
        acc=0
        for model in model_name_list:
            try:
                acc = multichoice(model,index)
                print(f"Acc of {model} is: {acc}")
            except Exception as e:
                print(e)
                continue
        zhun = []
        i = 0
        file_path = './PFAI/result/rst_compar__models.txt'
        count = 0
        with open(file_path, 'r') as file:
            for line in file:
                i = i + 1
                if 'Right' in line:
                    count += 1
        print(f"准确率为{count / 100}  {count}  {i}")
        zhun.append(f'{count / 100}')
        df['准确率'] = df['准确率'].astype('object')
        print(f'{count / 100}')
        df.loc[index,'准确率'] = acc
    else:
        df['准确率'] = df['准确率'].astype('object')

        # 现在你可以安全地赋值字符串
        df.loc[index, '准确率'] = "未作预测test"
        print("\033[91m================PFAI================\033[0m")
        print_colorful("未启动模型预测和模型导出功能，将继续执行", 94)
    df.to_excel('工作簿2_modified.xlsx', index=False)

    script_path = '../cs6493nlp/qgevalcap/eval.py'
    print("\033[91m================PFAI================\033[0m")
    print("\033[91mLLama-factory检测的结果为：\033[0m")
    json_file_path = f'{output_dir}'+'/predict_results.json'

    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    predict_bleu4 = float(data["predict_bleu-4"])
    predict_rouge1 = float(data["predict_rouge-1"])
    predict_rouge2 = float(data["predict_rouge-2"])
    predict_rougeL = float(data["predict_rouge-l"])
    df.loc[index, 'llama-factory-predict_bleu-4'] = predict_bleu4
    df.loc[index, 'llama-factory-predict_rouge-1'] = predict_rouge1
    df.loc[index, 'llama-factory-predict_rouge-2'] = predict_rouge2
    df.loc[index, 'llama-factory-predict_rouge-l'] = predict_rougeL
    df.to_excel('工作簿2_modified.xlsx', index=False)





    print(predict_bleu4, predict_rouge1, predict_rouge2, predict_rougeL)
    print(json.dumps(data, indent=4, ensure_ascii=False))
    print("\033[91m================PFAI================\033[0m")
    # process = subprocess.Popen(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # stdout_thread = threading.Thread(target=print_output, args=(process.stdout, 'STDOUT'))
    # stderr_thread = threading.Thread(target=print_output, args=(process.stderr, 'STDERR'))
    #
    # stdout_thread.start()
    # stderr_thread.start()
    #
    # process.wait()
    #
    # stdout_thread.join()
    # stderr_thread.join()
    # os.remove("PFAI/res_xiaorui/src.txt")

    parser = ArgumentParser()
    parser.add_argument("-type", dest="type", default="PFAI", help="squad or nqg")
    # parser.add_argument("-out", "--out_file", dest="out_file", default="/root/autodl-tmp/neural-question-generation/nqg/generated.txt", help="output file to compare")
    # parser.add_argument("-src", "--src_file", dest="src_file", default="/root/autodl-tmp/neural-question-generation/squad_nqg/para-test.txt", help="src file")
    # parser.add_argument("-tgt", "--tgt_file", dest="tgt_file", default="/root/autodl-tmp/neural-question-generation/squad_nqg/tgt-test.txt", help="target file")
    args = parser.parse_args()

    if args.type == 'nqg':
        out_file = '../nqg/generated.txt'
        src_file = '../squad_nqg/para-test.txt'
        tgt_file = '../squad_nqg/tgt-test.txt'
    elif args.type == 'squad':
        out_file = '../ans_squad/generated.txt'
        src_file = '../ans_squad/src.txt'  # ----
        tgt_file = '../ans_squad/golden.txt'
    elif args.type == 'PFAI':
        out_file = 'PFAI/res_xiaorui/data.txt'
        src_file = 'PFAI/res_xiaorui/src.txt'
        tgt_file = 'PFAI/res_xiaorui/input.txt'
    else:
        print('please input again')
    with open('./PFAI/scores.txt', 'w') as f:
        sys.stdout = f

    sys.stdout = sys.__stdout__
    print("scores: \n")
    a = eval(out_file, src_file, tgt_file)
    df.loc[index, 'cs6493nlp-bleu-1'] = float(a[0])
    df.loc[index, 'cs6493nlp-bleu-2'] = float(a[1])
    df.loc[index, 'cs6493nlp-bleu-3'] = float(a[2])
    df.loc[index, 'cs6493nlp-bleu-4'] = float(a[3])
    df.loc[index, 'cs6493nlp-bleu'] = float(a[4])
    df.loc[index, 'cs6493nlp-meteor'] = float(a[5])
    df.loc[index, 'cs6493nlp-rouge_L'] = float(a[6])
    df.to_excel('工作簿2_modified.xlsx', index=False)

    print("\033[91m================PFAI================\033[0m")


def multichoice(model_name,index):
    QnA_dir_path = 'PFAI/ProteinLMBench.json'
    with open(QnA_dir_path, 'r') as f:
        file_data = json.load(f)

    model_path = f'{model_name}'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto").eval()

    import re
    answer_list = [f['answer'] for f in file_data]
    answer_list = [re.search(r'\d+', a).group() for a in answer_list]

    prompt = ("""
Answer the multiple-choice question based solely on the provided context. 
If you are still unsure about the answer, output option 7.
Select only ONE correct option by its number. Start your response with 'The correct option is' followed by the option number ONLY. eg: "The correct option is Option X."
Think step by step.
    """)
    question = []

    for f in file_data:
        options = ''
        for o in f['options']:
            options += o + '\n'
        sb = prompt + '\n Question: \n' + f['question'] + '\n Options: \n' + options + '\nThe correct option is:'
        question.append(sb)

    inputs = []
    tokenizer.pad_token = tokenizer.eos_token
    for q in question:
        # for q in question:
        #     a = tokenizer.apply_chat_template([{"role": "user", "content": q}], return_tensors="pt").to("cuda")
        a = tokenizer(q, return_tensors="pt", padding=True)
        input_ids = a.input_ids.to('cuda')
        inputs.append(input_ids)

    print(len(inputs))

    chat_model = ('chat' in model_name) or ('Chat' in model_name)
    if 'Yi' in model_name:
        chat_model = False
    from tqdm import tqdm
    output_list = []

    temp = 0.1
    mnt = 20
    ppp = 0
    import warnings
    from transformers import logging
    hhxyname="hhxy_"+str(index)
    logging.set_verbosity_info()
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    # chat_model=ture
    for q in tqdm(inputs[:]):
        wh = 0
        while True:  # 创建一个无限循环
            arr1 = model.generate(q, max_new_tokens=mnt, do_sample=True, temperature=temp, top_p=0.7)

            qq = tokenizer.decode(q[0], skip_special_tokens=True)
            str1 = tokenizer.decode(arr1[0], skip_special_tokens=True)

            print(qq)

            # 移除用户提问的数据q
            str1 = str1.replace(qq, "")

            # 打印生成的文本
            # print("生成的文本为：", str1)

            # 如果str1不为空且不只包含空格，则退出循环
            if str1.strip():
                break
            else:
                # 如果str1为空或只包含空格，可以选择打印一条消息，然后继续循环
                wh = wh + 1
                print(f"{model_name} 第{wh}次生成的文本为: &{str1}&，正在重新生成...")

                if wh == 3:
                    print(model_name + " 3次均未成功生成")
                    break
                # 可选：在这里添加一些延迟或随机性，以避免潜在的无限循环问题
                # time.sleep(1)  # 例如，这里可以添加1秒的延迟
        output_list.append(arr1)
        lst = [tokenizer.decode(i[0], skip_special_tokens=True) for i in output_list]
        after = []
        lkj = 0
        for i, j in zip(lst, question):
            # print(lst[lkj])
            lkj += 1
            # for i, j in zip(output_list, question):
            after.append(i.replace(j, ''))
            # print(f'模型回答：{i}\n问题：{j}')
            print(f'--------这里: \n模型回答：{i}\n')
        #  print(after)

        # print(after)
        v_ans = []
        non_number = 0
        for o in after:
            # print(o)
            try:
                # v_ans.append(re.search(r'The correct option number is: (\d+)', o).group(1))
                v_ans.append(re.search(r'\d+', o).group())
            except:
                non_number += 1
                v_ans.append("None")

        # print(non_number)
        psd = 0
        # wrong_list = []
        from datetime import datetime
        now = datetime.now()
        formatted_time = now.strftime("%Y%m%d_%H%M%S")
        if "/" in model_name:
            model_name = model_name.split("/")[2]

        with open(f'PFAI/result/raw_result__{hhxyname}.json', 'w') as jj:
            json.dump(after, jj)

        with open(f'PFAI/result/rst_compar__{hhxyname}.txt', 'w') as results:
            for i in range(len(v_ans)):
                print(f'模型给出的答案是：{v_ans[i]}   正确答案是：{answer_list[i]}')
                if v_ans[i] != answer_list[i]:
                    results.write(str(v_ans[i]) + "   " + str(answer_list[i]))
                    results.write("\n")
                    continue
                else:
                    results.write("Right")
                    psd += 1
                    results.write("\n")

        accuracy = psd / len(v_ans)
        print('correct rate: ' + str(psd / len(v_ans)))
        ppp = ppp + 1
        # print(model.generate(q, max_new_tokens=mnt, do_sample=True, temperature=temp))
    lst = [tokenizer.decode(i[0], skip_special_tokens=True) for i in output_list]

    after = []
    for i, j in zip(lst, question):
        # for i, j in zip(output_list, question):
        after.append(i.replace(j, ''))

    # print(after)
    v_ans = []
    non_number = 0
    for o in after:
        try:
            # v_ans.append(re.search(r'The correct option number is: (\d+)', o).group(1))
            v_ans.append(re.search(r'\d+', o).group())
        except:
            non_number += 1
            v_ans.append("None")

    print(non_number)
    psd = 0
    # wrong_list = []
    from datetime import datetime
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")
    if "/" in model_name:
        model_name = model_name.split("/")[2]

    with open(f'PFAI/result/raw_result__{hhxyname}.json', 'w') as jj:
        json.dump(after, jj)

    with open(f'PFAI/result/rst_compar__{hhxyname}.txt', 'w') as results:
        for i in range(len(v_ans)):
            # print(i)
            if v_ans[i] != answer_list[i]:
                results.write(str(v_ans[i]) + "   " + str(answer_list[i]))
                results.write("\n")
                continue
            else:
                results.write("Right")
                psd += 1
                results.write("\n")

    accuracy = psd / len(v_ans)
    print('correct rate: ' + str(psd / len(v_ans)))

    del tokenizer
    del model
    return accuracy


class QGEvalCap:
    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self):
        output = []
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        # print('gts',self.gts)
        # print('res',self.res)

        # =================================================
        # Compute scores
        # =================================================

        # gts = word_tokenize(self.gts['0'][0].decode())
        # res = word_tokenize(self.res['0'][0].decode())
        # print(meteor_score.meteor_score([gts], res))
        for scorer, method in scorers:
            # print 'computing %s score...'%(scorer.method())
            # print(self.gts[0],self.res[0])
            score, scores = scorer.compute_score(self.gts, self.res)
            print('success return')
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.5f" % (m, sc))
                    output.append(sc)
            else:
                print("%s: %0.5f" % (method, score))
                output.append(score)
        # exit(0)
        # 45.22	29.94	22.01	16.76
        # Bleu_1: 0.26487
        # Bleu_2: 0.10851
        # Bleu_3: 0.05112
        # Bleu_4: 0.02147
        return output


def eval(out_file, src_file, tgt_file, isDIn=False, num_pairs=500):
    """
        Given a filename, calculate the metric scores for that prediction file

        isDin: boolean value to check whether input file is DirectIn.txt
    """

    pairs = []
    with open(src_file, 'r') as infile:
        for line in infile:
            pair = {}
            pair['tokenized_sentence'] = line[:-1]
            pairs.append(pair)

    with open(tgt_file, "r") as infile:
        cnt = 0
        for line in infile:
            pairs[cnt]['tokenized_question'] = line[:-1]
            cnt += 1

    output = []
    with open(out_file, 'r') as infile:
        for line in infile:
            line = line[:-1]
            output.append(line)

    for idx, pair in enumerate(pairs):
        pair['prediction'] = output[idx]

    ## eval
    from eval import QGEvalCap
    import json
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.4f')

    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])
    for pair in pairs[:]:
        key = pair['tokenized_sentence']
        res[key] = [pair['prediction'].encode('utf-8')]

        ## gts
        gts[key].append(pair['tokenized_question'].encode('utf-8'))

    QGEval = QGEvalCap(gts, res)
    return QGEval.evaluate()



def print_colorful(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m")

# 函数：实时打印输出
def print_output(pipe, name):
    for line in iter(pipe.readline, ''):  # 逐行读取
        print(f"{name}: {line}", end='')  # 打印每一行
# 函数：执行命令
def execute_command(command, cuda_visible_devices):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    print_colorful("设置CUDA环境变量，使用以下GPU卡ID: "+cuda_visible_devices,91)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    stdout_thread = threading.Thread(target=print_output, args=(process.stdout,"PFAI_RUN"))
    stderr_thread = threading.Thread(target=print_output, args=(process.stderr,"PFAI_ERR"))

    stdout_thread.start()
    stderr_thread.start()

    process.wait()
    stdout_thread.join()
    stderr_thread.join()






