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
MAGENTA = '35'

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
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    output.append(sc)
            else:
                output.append(score)
        return output

def create_folder(folder_path):
    if os.path.exists(folder_path):
        print_colorful(f"文件夹 '{folder_path}' 已经存在", BLUE)
    else:
        os.makedirs(folder_path)
        print_colorful(f"创建 '{folder_path}' 文件夹成功", GREEN)
    with open(f'./{folder_path}/src.txt', 'w') as file:
        for i in range(config.flie_len):
            file.write(f'这是第{i + 1}行\n')
    print_colorful('写入完成', GREEN)

def clone_nltk_data():
    print_colorful("正在克隆nltk_data项目......", BLUE)
    try:
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
            print_colorful("nltk_data项目中不存在packages目录。", RED)
        shutil.rmtree(temp_dir)
        update_progress()
    except subprocess.CalledProcessError as e:
        print_colorful(f"克隆nltk_data项目失败：{e}", RED)
        exit(1)
    except Exception as e:
        print_colorful(f"复制nltk_data项目失败：{e}", RED)
        exit(1)

def execute_command(command, desc, cuda_devices='0'):
    """执行命令并显示进度"""
    draw_interface()
    start = time.time()

    # 打印赛博大佛
    print_colorful(r"""
      __      __  .__                   ____.                  __    
      /  \    /  \ |  |   ____  __ __   \   \ ______  _  __ _/  |_  
      \   \/\/   / |  | _/ __ \|  |  \   \   \\____ \|  |  \\   __\ 
       \        /  |  |_\  ___/|  |  /    \   \  |_> >  |  /|  |    
        \__/\  /   |____/\___  >____/  /\ \__\  .__/>____/ |__|    
           \/             \/           \/   \/                  v1.7
    """, CYAN)

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

print_colorful(f"程序将执行 {num_rows_to_execute} 次", MAGENTA)
print_colorful("共三个问题：1. 是否微调模型？ 2. 是否执行自动化问答？ 3. 是否运行模型蛋白质答题预测和模型导出？ 输入 1 为执行，0 为不执行", MAGENTA)
print_colorful("请准确输入数字！否则程序将出现异常及其 bug，出现 bug 不给予解答！", RED)
input("按任意键确认...")

print_colorful("是否微调模型？", CYAN)
weitiao = int(input("请输入（1/0）: "))
print_colorful("是否执行自动化问答？", CYAN)
wenda = int(input("请输入（1/0）: "))
print_colorful("是否运行模型蛋白质答题预测和模型导出？", CYAN)
pfai_huida = int(input("请输入（1/0）: "))

print_colorful("启动自动评测脚本", BLUE)
print_colorful("执行自动设置环境变量", BLUE)
new_path = "/opt/conda/bin/"
command = ['which', 'python']
result = subprocess.run(['bash', '-c', f'export PATH="{new_path}:$PATH" && {" ".join(command)}'], capture_output=True, text=True)
print_colorful(f"执行结果：{result.stdout}", GREEN)

print_colorful("转换为 LLaMA-Factory 工作目录：/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/LLaMA-Factory", BLUE)
os.chdir('/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/LLaMA-Factory')
print_colorful("启动 LLaMA-Factory 训练过程...", YELLOW)
print_colorful("创建 /mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/LLaMA-Factory/PFAI 文件夹", BLUE)

path = "PFAI"
if not os.path.exists(path):
    os.makedirs(path)
    print_colorful("文件夹创建成功", GREEN)
else:
    print_colorful("文件夹已存在，继续执行后续命令", BLUE)

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

def execute_command(command, desc, cuda_devices='0'):
    """执行命令并显示进度"""
    draw_interface()
    start = time.time()

    # 打印赛博大佛
    print_colorful(r"""
      __      __  .__                   ____.                  __    
      /  \    /  \ |  |   ____  __ __   \   \ ______  _  __ _/  |_  
      \   \/\/   / |  | _/ __ \|  |  \   \   \\____ \|  |  \\   __\ 
       \        /  |  |_\  ___/|  |  /    \   \  |_> >  |  /|  |    
        \__/\  /   |____/\___  >____/  /\ \__\  .__/>____/ |__|    
           \/             \/           \/   \/                  v1.7
    """, CYAN)

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

for index in range(0, num_rows_to_execute):
    try:
        command = [
            arg.format(
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_train_epochs,
                max_grad_norm=config.max_grad_norm,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                output_dir=config.output_dir
            ) for arg in base_command
        ]
        print_colorful(f"训练开始 第 {index + 1} 轮", MAGENTA)
        print_colorful(f"学习率：{config.learning_rate}", CYAN)
        print_colorful(f"训练轮数：{config.num_rows_to_execute}", CYAN)
        print_colorful(f"最大梯度范围：{config.max_grad_norm}", CYAN)
        print_colorful(f"Lora秩：{config.lora_rank}", CYAN)
        print_colorful(f"LoRA 缩放系数：{config.lora_alpha}", CYAN)
        print_colorful(f"输出目录：{config.output_dir}", CYAN)
        print_colorful(f"index : {index}", CYAN)
        print_colorful(f"将要执行以下命令: {' '.join(command)}", BLUE)

        if weitiao == 1:
            execute_command(command, '微调模型')
        else:
            print_colorful("将不执行微调，继续执行后续命令", YELLOW)

        yaml_file_path = './PFAI/llama3_lora_sft.yaml'
        with open(yaml_file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        data['model_name_or_path'] = f'{model_name_or_path}'
        data['adapter_name_or_path'] = f'{config.output_dir}'
        data['export_dir'] = config.export_dir
        if pfai_huida == 1:
            with open(yaml_file_path, 'w', encoding='utf-8') as file:
                yaml.dump(data, file, default_flow_style=False, allow_unicode=True)
            model_path_test01 = data['export_dir']
            print_colorful(f"YAML 文件已更新，模型将导出到 {model_path_test01} 目录中", BLUE)
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
            print_colorful(f"模型导出成功在 {model_path_test01} 目录中", GREEN)

        if wenda == 1:
            print_colorful("执行问答程序", BLUE)
            print_colorful("执行文件提取脚本.....", BLUE)
            print_colorful(f"将 {dataset} 提取前 {config.flie_len} 个", BLUE)
            input_file_name = f'./LLaMA-Factory/data/{dataset}'
            output_file_name = './PFAI/formatted_pqal.json'

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

            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = config.cuda_visible_devices
            print_colorful(f"设置 CUDA 环境变量，使用以下 GPU 卡 ID: {config.cuda_visible_devices}", BLUE)

            if config.clone_nltk_data == 1:
                clone_nltk_data()

            folder_path = config.folder_path
            create_folder(folder_path)

            out_file = folder_path + "output.txt"
            src_file = folder_path + "src.txt"
            tgt_file = folder_path + "input.txt"

            model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_NAME,
                torch_dtype=config.TORCH_DTYPE,
                device_map=config.DEVICE_MAP
            )
            tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
            print_colorful("读取 extracted_qa 文件....", BLUE)
            file_path = config.file_path
            with open(file_path, 'r') as file:
                data = json.load(file)
            with open(tgt_file, 'a') as input_file, open(out_file, 'a') as output_file:
                for item in tqdm(data, desc="处理问题", total=100):
                    if i == config.flie_len:
                        break
                    i += 1
                    question = item['QUESTION']
                    print_colorful(f"问题: {question}", RED)

                    human_output_pro = item['LONG_ANSWER']
                    input_file.write(human_output_pro + '\n')
                    print_colorful("人类给的答案已成功写入文件。", GREEN)

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

                    print(response)
                    output_file.write(response + '\n')
                    print_colorful("------------回答已成功写入文件。", GREEN)

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
                src_file = './ans_squad/src.txt'
                tgt_file = './ans_squad/golden.txt'
            with open('./scores.txt', 'w') as f:
                print("scores: \n")
                a = eval(out_file, src_file, tgt_file)
        else:
            print_colorful("将不执行问答，程序停止", YELLOW)
    except Exception as e:
        print_colorful(f"引发异常: {e}", RED)

print_colorful("\n✓✓✓ 所有流程执行完成 ✓✓✓", GREEN)
print("\033[?25h")  # 恢复光标显示