import shutil
import subprocess
import os
import threading
from json import encoder

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
import config  # 导入配置文件

import shutil
import subprocess
import os
import threading
import json
import yaml
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import config

# 颜色代码
RED = '91'
GREEN = '32'
YELLOW = '33'
BLUE = '34'
PURPLE = '35'
CYAN = '36'
print("\033[?25l")  # 隐藏光标
# 进度跟踪全局变量
current_step = 0
start_time = time.time()
steps = [
    {"name": "环境准备", "duration": 5},
    {"name": "克隆NLTK数据", "duration": 120},
    {"name": "模型训练配置", "duration": 60},
    {"name": "微调模型", "duration": 300},
    {"name": "模型导出", "duration": 180},
    {"name": "问答生成", "duration": 600},
    {"name": "结果评估", "duration": 120}
]

def print_output(pipe, name, color_code):
    """输出处理线程"""
    color = f"\033[{color_code}m"
    reset = "\033[0m"
    for line in iter(pipe.readline, ''):
        print(f"{color}{name}: {line.strip()}{reset}")


def print_colorful(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m")


def clear_screen():
    print("\033[2J\033[H", end="")


def draw_interface():
    global current_step
    now = time.time()
    elapsed = now - start_time
    term_width = shutil.get_terminal_size().columns

    clear_screen()

    # 顶部信息区
    print("\033[1;1H")
    if current_step < len(steps):
        current_step_name = steps[current_step]["name"]
        print_colorful(f"▶ 当前步骤: {current_step_name}", CYAN)
        remaining_time = sum(s["duration"] for s in steps[current_step:])
        print_colorful(f"⏳ 预计剩余时间: {remaining_time} 秒", YELLOW)
    else:
        print_colorful("✅ 所有步骤已完成", GREEN)

    # 步骤列表区
    print("\n\033[K步骤进度:")
    for idx, step in enumerate(steps):
        status_icon = "✓" if idx < current_step else "◻"
        color = GREEN if idx < current_step else RED
        duration = f"{step['duration']}s" if step['duration'] > 0 else ""
        print(f"\033[{color}m {status_icon} {step['name']} {duration}\033[0m")

    # 时间统计区
    print_colorful(f"\n⏱ 总运行时间: {elapsed:.1f} 秒", PURPLE)
    print("-" * term_width)


def update_progress():
    global current_step
    current_step += 1
    draw_interface()


def check_root_and_prepare():
    """权限检查和目录准备"""
    try:
        os.chdir('/')
        print_colorful("✓ 当前工作目录已切换到根目录", GREEN)
        os.makedirs(config.BASE_DIR, exist_ok=True)
        print_colorful(f"✓ 创建项目目录 {config.BASE_DIR}", GREEN)
    except Exception as e:
        print_colorful(f"✗ 初始化失败: {e}", RED)
        exit(1)
    update_progress()


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
        for scorer, method in scorers:
            score, scores = scorer.compute_score(self.gts, self.res)
            print('success return')
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.5f" % (m, sc))
                    output.append(sc)
            else:
                print("%s: %0.5f" % (method, score))
                output.append(score)
        return output

def create_folder(folder_path):
    if os.path.exists(folder_path):
        print_output(f"文件夹 '{folder_path}' 已经存在", "34")
    else:
        os.makedirs(folder_path)
        print_output(f"创建 '{folder_path}' 文件夹成功", "32")
    with open(f'./{folder_path}/src.txt', 'w') as file:
        # 循环sum次
        for i in range(config.flie_len):
            # 每次循环写入一行内容，这里以数字i为例，你可以根据需要修改写入的内容
            file.write(f'这是第{i + 1}行\n')

    print('写入完成')

def clone_nltk_data():
    print_output("正在克隆nltk_data项目......", 91)
    try:
        print_colorful("▶ 开始克隆nltk_data项目", BLUE)
        subprocess.run(['git', 'clone', 'https://github.com/nltk/nltk_data.git'], check=True)
        temp_dir = os.path.join(os.getcwd(), 'nltk_data')
        target_dir = '/root/nltk_data'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        packages_dir = os.path.join(temp_dir, 'packages')
        if os.path.exists(packages_dir):
            for item in os.listdir(packages_dir):
                src_path = os.path.join(packages_dir, item)
                dst_path = os.path.join(target_dir, item)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dst_path)
        else:
            print_output("nltk_data项目中不存在packages目录。", 34)
        shutil.rmtree(temp_dir)
        update_progress()
    except subprocess.CalledProcessError as e:
        print_colorful(f"✗ 克隆失败: {str(e)}", RED)
        print_output(f"克隆nltk_data项目失败：{e}", 34)
        exit(1)
    except Exception as e:
        print_colorful(f"✗ 克隆失败: {str(e)}", RED)
        print_output(f"复制nltk_data项目失败：{e}", 34)
        exit(1)


def execute_command(command, desc, cuda_devices='0'):
    """执行命令并显示进度"""
    draw_interface()
    start = time.time()

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = cuda_devices
    print_colorful(f"▶ 正在执行: {desc}", BLUE)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )

    stdout_thread = threading.Thread(target=print_output, args=(process.stdout, "OUT", GREEN))
    stderr_thread = threading.Thread(target=print_output, args=(process.stderr, "ERR", RED))

    stdout_thread.start()
    stderr_thread.start()

    process.wait()
    actual_duration = time.time() - start
    if current_step < len(steps):
        steps[current_step]["duration"] = max(steps[current_step]["duration"], int(actual_duration))

    stdout_thread.join()
    stderr_thread.join()
    update_progress()


def print_colorful(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m")

draw_interface()
model_name_or_path = config.model_name_or_path
dataset = config.dataset
num_rows_to_execute = config.num_rows_to_execute
learning_rate = config.learning_rate
num_train_epochs = config.num_train_epochs
max_grad_norm = config.max_grad_norm
lora_rank = config.lora_rank
lora_alpha = config.lora_alpha
output_dir_base = config.output_dir_base
cuda_visible_devices = config.cuda_visible_devices

hhxy_max = []
weitiao = 0
wenda = 0
pfai_huida = 0
print("\033[91m程序将执行" + f'{num_rows_to_execute}次' + "\033[0m")
print(
    "\033[91m================PFAI================ \n共三个问题1.是否微调模型? 2.是为否执行自动化问答? 3.是否运行模型蛋白质答题预测和模型导出? 输入1为执行 0为不执行 \033[0m")
print("\033[91m！！！！！请准确输入数字！否则程序将出现异常及其bug，出现bug不给予解答！！！！！！ 按任意键确认\033[0m")
__ = input()
print("\033[91m是否微调模型?\033[0m")
weitiao = int(input())
print("\033[91m是否执行自动化问答?\033[0m")
wenda = int(input())
print("\033[91m是否运行模型蛋白质答题预测和模型导出?\033[0m")
pfai_huida = int(input())

print("\033[91m================PFAI================\033[0m")
print("\033[94m启动自动评测脚本\033[0m")
print("\033[94m执行自动设置环境变量\033[0m")
new_path = "/opt/conda/bin/"
command = ['which', 'python']
result = subprocess.run(['bash', '-c', f'export PATH="{new_path}:$PATH" && {" ".join(command)}'], capture_output=True,
                        text=True)
print("执行结果：", result.stdout)
print("\033[91m================PFAI================\033[0m")

print("\033[94m转换为LLaMA-Factory工作目录：/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/LLaMA-Factory\033[0m")
os.chdir('/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/LLaMA-Factory')
print_colorful("启动LLaMA-Factory训练过程...", "93")
print_colorful("创建/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/LLaMA-Factory/PFAI 文件夹", 91)
#  nohup python 自动化脚本.py > output.log 2> error.log &
path = "PFAI"

if not os.path.exists(path):
    os.makedirs(path)
    print("文件夹创建成功")
else:
    print("文件夹已存在，继续执行后续命令")

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
    '--learning_rate', f'{learning_rate}',
    '--num_train_epochs', f'{num_train_epochs}',
    '--max_samples', '10000',
    '--per_device_train_batch_size', '2',
    '--gradient_accumulation_steps', '8',
    '--lr_scheduler_type', 'cosine',
    '--max_grad_norm', f'{max_grad_norm}',
    '--logging_steps', '1',
    '--save_steps', '100',
    '--warmup_steps', '0',
    '--optim', 'adamw_torch',
    '--packing', 'False',
    '--report_to', 'none',
    '--output_dir', f'{config.output_dir}',
    '--fp16', 'True',
    '--plot_loss', 'True',
    '--ddp_timeout', '180000000',
    '--include_num_input_tokens_seen', 'True',
    '--quantization_bit', '4',
    '--quantization_method', 'bitsandbytes',
    '--lora_rank', f'{lora_rank}',
    '--lora_alpha', f'{lora_alpha}',
    '--lora_dropout', '0',
    '--use_rslora', 'True',
    '--lora_target', 'all'
]



# 函数：执行命令
def execute_command(command, desc, cuda_devices='0'):
    """执行命令并显示进度"""
    draw_interface()
    start = time.time()

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = cuda_devices
    print_colorful(f"▶ 正在执行: {desc}", BLUE)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )

    stdout_thread = threading.Thread(target=print_output, args=(process.stdout, "OUT", GREEN))
    stderr_thread = threading.Thread(target=print_output, args=(process.stderr, "ERR", RED))

    stdout_thread.start()
    stderr_thread.start()

    process.wait()
    actual_duration = time.time() - start
    if current_step < len(steps):
        steps[current_step]["duration"] = max(steps[current_step]["duration"], int(actual_duration))

    stdout_thread.join()
    stderr_thread.join()
    update_progress()

dddd = ""
lora_z = ""
lora_sf = ""
for index in range(0,num_rows_to_execute):
    try:
        command = [
            arg.format(
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_train_epochs,
                max_grad_norm=config.max_grad_norm,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                output_dir=config.output_dir
                # output_dir=f'./test/train_{index + 1}_{row["Lora秩"]}_{row["LoRA 缩放系数"]}'
            ) for arg in base_command
        ]
        lora_z = config.lora_rank
        lora_sf = config.lora_alpha
        dddd = config.output_dir
        print("\033[91m================PFAI================\033[0m")
        print_colorful("训练开始 第" + str(index + 1) + "轮", 94)
        print_colorful("学习率：" + str(config.learning_rate), 94)
        print_colorful("训练轮数：" + str(config.num_rows_to_execute), 94)
        print_colorful("最大梯度范围：" + str(config.max_grad_norm), 94)
        print_colorful("Lora秩：" + str(config.lora_rank), 94)
        print_colorful("LoRA 缩放系数：" + str(config.lora_alpha), 94)
        print_colorful("输出目录：" + dddd, 94)
        print_colorful("index  :  " + f'{index}', 94)

        print("将要执行以下命令: ", " ".join(command))

        print("\033[91m================PFAI================\033[0m")

        if weitiao == 1:
            execute_command(command, '0')
        else:
            print_colorful("将不执行微调，继续执行后续命令", 91)

        print("\033[91m================PFAI================\033[0m")
        print_colorful("模型临时文件已保存到目录" + dddd, 91)



        yaml_file_path = './PFAI/llama3_lora_sft.yaml'
        # 读取YAML文件
        with open(yaml_file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        data['model_name_or_path'] = f'{model_name_or_path}'
        data['adapter_name_or_path'] = f'{dddd}'
        data['export_dir'] = config.export_dir
        if pfai_huida == 1:
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
            print_colorful("模型导出成功在" + f'{test01}目录中', 94)
            model_name_list = [
                f'{model_path_test01}'
            ]
        if wenda == 1:
            print("\033[91m================PFAI 执行问答程序================\033[0m")

            print_colorful("执行文件提取脚本.....", 91)
            print_colorful("将" + f'{dataset}' + f"提取前{config.flie_len}个", 91)
            input_file_name = f'./LLaMA-Factory/data/{dataset}'
            output_file_name = './PFAI/formatted_pqal.json'

            import os
            import json

            with open(os.path.join(os.path.dirname(__file__), config.QnA_dir_path), 'r') as f:
                data2 = json.load(f) 
            mm = 0
            new_data_list = []
            for i in range(len(data2)):
                new_data = {
                    'QUESTION': data2[i]['instruction'],
                    'LONG_ANSWER': data2[i]['output']
                }
                new_data_list.append(new_data)
            with open(config.file_path, 'w', encoding='utf-8') as file:
                json.dump(new_data_list, file, ensure_ascii=False, indent=4)
            i = 0
            def print_output(text, color_code):
                print(f"\033[{color_code}m{text}\033[0m")
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = config.cuda_visible_devices
            print_output("设置CUDA环境变量，使用以下GPU卡ID: " + config.cuda_visible_devices, 91)

            if config.clone_nltk_data == 1:
                clone_nltk_data()

            folder_path = config.folder_path
            create_folder(folder_path)

            out_file = folder_path + "output.txt"
            src_file = folder_path + "src.txt"
            tgt_file = folder_path + "input.txt"

            # 模型初始化量化
            model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_NAME,
                torch_dtype=config.TORCH_DTYPE,
                device_map=config.DEVICE_MAP
            )
            tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
            print_output("读取extracted_qa文件....", 91)
            file_path = config.file_path
            with open(file_path, 'r') as file:
                data = json.load(file)
            with open(tgt_file, 'a') as input_file, open(out_file, 'a') as output_file:
                for item in tqdm(data, desc="处理问题", total=100):  # 添加进度条
                    if i == config.flie_len:
                        break
                    i += 1
                    question = item['QUESTION']
                    print_output("问题:" + question, 91)  # 红色字体输出问题

                    # 将人类给的答案写入input.txt
                    human_output_pro = item['LONG_ANSWER']
                    bytes_written = input_file.write(human_output_pro + '\n')
                    if bytes_written == len(human_output_pro + '\n'):
                        print_output("人类给的答案已成功写入文件。", 32)  # 绿色字体表示成功
                    else:
                        print_output("人类给的答案写入文件失败。", 31)  # 红色字体表示失败
                    """
                    出了问题改回原来样式
                    bytes_written = input_file.write(question + '\n')
                    if bytes_written == len(question + '\n'):
                        print_output("人类给的答案已成功写入文件。", 32)  # 绿色字体表示成功
                    else:
                        print_output("人类给的答案写入文件失败。", 31)  # 红色字体表示失败

                    """
                    prompt = question
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = tokenizer([text], return_tensors="pt").to(config.DEVICE)

                    generated_ids = model.generate(
                        model_inputs.input_ids,
                        max_new_tokens=config.MAX_NEW_TOKENS,
                        temperature=config.TEMPERATURE,
                        top_p=config.TOP_P,
                        top_k=config.TOP_K
                    )

                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in
                        zip(model_inputs.input_ids, generated_ids)
                    ]

                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    response = response.replace('\n', '')

                    # 打印输出，不添加额外的换行符
                    print(response)
                    # 将回答写入output.txt
                    bytes_written = output_file.write(response + '\n')
                    if bytes_written == len(response + '\n'):
                        print_output("------------回答已成功写入文件。", 32)  # 绿色字体表示成功
                    else:
                        print_output("------------回答写入文件失败。", 31)  # 红色字体表示失败


            def eval(out_file, src_file, tgt_file, isDIn=False, num_pairs=500):
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

                encoder.FLOAT_REPR = lambda o: format(o, '.4f')
                res = defaultdict(lambda: [])
                gts = defaultdict(lambda: [])
                for pair in pairs[:]:
                    key = pair['tokenized_sentence']
                    res[key] = [pair['prediction'].encode('utf-8')]
                    gts[key].append(pair['tokenized_question'].encode('utf-8'))

                QGEval = QGEvalCap(gts, res)
                return QGEval.evaluate()


            parser = ArgumentParser()
            parser.add_argument("-type", dest="type", default="PFAI", help="squad or nqg")
            args = parser.parse_args()
            if args.type == 'PFAI':
                out_file = folder_path + "output.txt"
                src_file = folder_path + "src.txt"
                tgt_file = folder_path + "input.txt"
            else:
                out_file = './ans_squad/generated.txt'
                src_file = './ans_squad/src.txt'  # ----
                tgt_file = './ans_squad/golden.txt'
            with open('./scores.txt', 'w') as f:
                print("scores: \n")
                a = eval(out_file, src_file, tgt_file)
        else:
            print_colorful("将不执行问答，程序停止", 91)
            print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
    except Exception as e:
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")
        print("\033[91m================PFAI================\033[0m")


        print("\033[96m引发异常！！！！！！！！！！！\033[0m"+e)




print_colorful("\n✓✓✓ 所有流程执行完成 ✓✓✓", GREEN)
print("\033[?25h")  # 恢复光标显示

