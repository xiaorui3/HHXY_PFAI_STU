import subprocess
import threading
import os
import shutil

# 定义颜色代码
RED = '91'
GREEN = '32'
YELLOW = '33'
BLUE = '34'
PURPLE = '35'
CYAN = '36'

# 定义一个函数来打印带颜色的文本
def print_colorful(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m")

# 定义一个函数来实时打印输出，并添加前缀和颜色
def print_output(pipe, name, process, color_code):
    color = f"\033[{color_code}m"
    reset = "\033[0m"
    while True:
        output = pipe.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(f"{color}{name}: {output.strip()}{reset}")

# 定义一个函数来执行命令
def run_command(command, description, print_stdout=True, print_stderr=True, env=None, stdout_color='32', stderr_color='91'):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )
    stdout_thread = threading.Thread(target=print_output, args=(process.stdout, 'HHXY_PFAI_OUT', process, stdout_color if print_stdout else None))
    stderr_thread = threading.Thread(target=print_output, args=(process.stderr, 'HHXY_PFAI_ERR', process, stderr_color if print_stderr else None))
    
    if print_stdout:
        stdout_thread.start()
    if print_stderr:
        stderr_thread.start()

    # 打印命令前的分隔线和描述文本
    print("\033[91m================================================================================黑河学院软件项目开发社团PFAI项目组================================================================================\033[0m")
    print("\033[91m================================================================================黑河学院软件项目开发社团PFAI项目组================================================================================\033[0m")
    print("\033[91m================================================================================黑河学院软件项目开发社团PFAI项目组================================================================================\033[0m")
    print_colorful(f"PFAI自动化脚本01即将启动：{description} ({' '.join(command)})", RED)

    process.wait()
    if print_stdout:
        stdout_thread.join()
    if print_stderr:
        stderr_thread.join()
run_command(['pip', 'config', 'set', 'global.index-url', 'https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple'], '设置pip源为清华大学镜像', stdout_color=BLUE, stderr_color=RED)
# 更新apt包索引并安装软件包
run_command(['sudo', 'apt', 'update', '-y'], '更新apt包索引', stdout_color=GREEN, stderr_color=RED)
run_command([
    'sudo', 'apt', 'install', '-y', 'git', 'git-lfs', 'cmake', 'make', 'iftop', 'atop'
], '安装基础软件包', stdout_color=GREEN, stderr_color=RED)

# 安装huggingface_hub
run_command(['pip', 'install', '-U', 'huggingface_hub'], '安装huggingface_hub', stdout_color=BLUE, stderr_color=RED)

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型
run_command([
    'huggingface-cli', 'download', '--resume-download', 'Qwen/Qwen2-7B', '--local-dir', 'qwen27'
], '下载Qwen/Qwen2-7B模型', stdout_color=YELLOW, stderr_color=RED, env=os.environ)


run_command([
    'huggingface-cli', 'download','c', '--resume-download', 'xiaorui1/hhxy_test', '--local-dir', 'data'
], '下载PFAI数据集', stdout_color=YELLOW, stderr_color=RED, env=os.environ)


print("环境变量---------------------------------")

# 克隆LLaMA-Factory仓库并安装依赖
run_command(['git', 'clone', 'https://github.com/hiyouga/LLaMA-Factory.git'], '克隆LLaMA-Factory仓库', stdout_color=CYAN, stderr_color=RED)
os.chdir('LLaMA-Factory')

print("\033[91m================PFAI================\033[0m")
print_colorful(f"PFAI自动化脚本01即将启动：复制程序......", RED)
def dir_exists(dst):
    return os.path.exists(dst) and os.path.isdir(dst)

# 定义一个函数来复制目录
def copy_dir(src, dst):
    if dir_exists(dst):
        print_colorful(f"目录 {dst} 已经存在，跳过复制。", RED)
    else:
        print_colorful(f"正在复制目录：{src} 到 {dst}", RED)
        shutil.copytree(src, dst, dirs_exist_ok=True)

# 源目录和目标目录
src_dir = "/mnt/workspace/HHXY_PFAI/LLaMA-Factory/PFAI/"
dst_dir = "/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/LLaMA-Factory/PFAI/"

# 执行复制命令
copy_dir(src_dir, dst_dir)

run_command([
    'huggingface-cli', 'download', '--repo-type', 'dataset', '--resume-download', 'xiaorui1/hhxy_test', '--local-dir', '/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/LLaMA-Factory/data'
], '下载PFAI数据集', stdout_color=YELLOW, stderr_color=RED, env=os.environ)

def copy_file(src, dst):
    print_colorful(f"正在复制文件：{src} 到 {dst}", RED)
    shutil.copy(src, dst)

# 源文件和目标路径
src_file = "/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/dataset_info.json"
dst_path = "/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/LLaMA-Factory/data/"

# 确保目标路径存在
os.makedirs(dst_path, exist_ok=True)

# 获取源文件的文件名
file_name = os.path.basename(src_file)

# 执行复制文件命令
copy_file(src_file, os.path.join(dst_path, file_name))



run_command(['pip', 'install', '-r', 'requirements.txt'], '安装requirements.txt中列出的依赖', stdout_color=PURPLE, stderr_color=RED)
run_command([
    'pip', 'install', 'transformers_stream_generator', 'bitsandbytes', 'tiktoken', 'auto-gptq', 'optimum', 'autoawq'
], '安装额外的Python依赖', stdout_color=BLUE, stderr_color=RED)
run_command(['pip', 'install', '--upgrade', 'tensorflow'], '升级tensorflow', stdout_color=GREEN, stderr_color=RED)
run_command(['pip', 'install', 'vllm==0.4.3'], '安装vllm', stdout_color=YELLOW, stderr_color=RED)
run_command([
    'pip', 'install', 'torch==2.1.2', 'torchvision==0.16.2', 'torchaudio==2.1.2', '--index-url', 'https://download.pytorch.org/whl/cu121'
], '安装特定版本的PyTorch', stdout_color=CYAN, stderr_color=RED)
run_command(['pip', 'install', 'tensorflow==2.12.0'], '安装tensorflow 2.12.0', stdout_color=PURPLE, stderr_color=RED)

run_command(['pip', 'install', '-e', '.[metrics]'], '安装LLaMA-Factory的metrics模块', stdout_color=GREEN, stderr_color=RED)

run_command([
    'pip', 'install', 'ijson', 'pyaml'
], '安装额外的Python依赖', stdout_color=BLUE, stderr_color=RED)
run_command(['python','../自动化脚本.py'], '启动自动化脚本', stdout_color=GREEN, stderr_color=RED)