import subprocess
import threading
import os
import shutil
import zipfile

# huggingface-cli download --repo-type dataset --resume-download xiaorui1/hhxy_test --local-dir /mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/LLaMA-Factory/data
# /mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap# huggingface-cli download --resume-download Qwen/Qwen2-7B --local-dir qwen27
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
    except subprocess.CalledProcessError as e:
        print_colorful(f"克隆nltk_data项目失败：{e}", RED)
    except Exception as e:
        print_colorful(f"复制nltk_data项目失败：{e}", RED)
    files_to_extract = ['punkt_tab.zip', 'punkt.zip']
    extract_dir = '/root/nltk_data/tokenizers'
    os.makedirs(extract_dir, exist_ok=True)
    for file_name in files_to_extract:
        file_path = os.path.join(extract_dir, file_name)
        if os.path.exists(file_path):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                print(f"正在解压：{file_name}")
                zip_ref.extractall(extract_dir)
        else:
            print(f"文件不存在，跳过解压：{file_path}")
RED = '91'
GREEN = '32'
YELLOW = '33'
BLUE = '34'
PURPLE = '35'
CYAN = '36'
def print_colorful(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m")
def confirm_deployment():
    print_colorful("警告：你即将开始部署。请输入 'y' 确认部署，或任意其他键退出。", RED)
    while True:
        user_input = input("请输入确认字符 (y/n): ")
        if user_input.lower() == 'y':
            print_colorful("部署确认。", GREEN)
            return True
        elif user_input.lower() == 'n':
            print_colorful("部署已取消。", RED)
            return False
        else:
            print_colorful("无效输入，请输入 'y' 或 'n'。", RED)

if confirm_deployment():
    def print_output(pipe, name, process, color_code):
        color = f"\033[{color_code}m"
        reset = "\033[0m"
        while True:
            output = pipe.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"{color}{name}: {output.strip()}{reset}")
    def run_command(command, description, print_stdout=True, print_stderr=True, env=None, stdout_color='32',
                    stderr_color='91'):
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )
        stdout_thread = threading.Thread(target=print_output, args=(
            process.stdout, 'HHXY_PFAI_OUT', process, stdout_color if print_stdout else None))
        stderr_thread = threading.Thread(target=print_output, args=(
            process.stderr, 'HHXY_PFAI_ERR', process, stderr_color if print_stderr else None))

        if print_stdout:
            stdout_thread.start()
        if print_stderr:
            stderr_thread.start()
        print(
            "\033[91m================================================================================黑河学院软件项目开发社团PFAI项目组================================================================================\033[0m")
        print(
            "\033[91m================================================================================黑河学院软件项目开发社团PFAI项目组================================================================================\033[0m")
        print(
            "\033[91m================================================================================黑河学院软件项目开发社团PFAI项目组================================================================================\033[0m")
        print_colorful(f"PFAI自动化脚本01即将启动：{description} ({' '.join(command)})", RED)

        process.wait()
        if print_stdout:
            stdout_thread.join()
        if print_stderr:
            stderr_thread.join()


    run_command(['pip', 'config', 'set', 'global.index-url', 'https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple'],
                '设置pip源为清华大学镜像', stdout_color=BLUE, stderr_color=RED)
    run_command(['sudo', 'apt', 'update', '-y'], '更新apt包索引', stdout_color=GREEN, stderr_color=RED)
    run_command([
        'sudo', 'apt', 'install', '-y', 'git', 'git-lfs', 'cmake', 'make', 'iftop', 'atop'
    ], '安装基础软件包', stdout_color=GREEN, stderr_color=RED)

    run_command(['pip', 'install', '-U', 'huggingface_hub'], '安装huggingface_hub', stdout_color=BLUE, stderr_color=RED)
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print_colorful("下载Qwen/Qwen2-7B模型 如果等待时间较长可自行手动下载     \n命令1:export HF_ENDPOINT=https://hf-mirror.com \n命令2:cd /mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap \n命令3:huggingface-cli download --resume-download <你的模型名字> --local-dir <要存储的路径>",RED)
    print("环境变量---------------------------------")
    run_command(['apt', 'install', '-y', 'openjdk-17-jdk'], '安装apt install openjdk-17-jdk',
                stdout_color=CYAN, stderr_color=RED)
    run_command(['git', 'clone', 'https://github.com/hiyouga/LLaMA-Factory.git'], '克隆LLaMA-Factory仓库',
                stdout_color=CYAN, stderr_color=RED)
    os.chdir('LLaMA-Factory')

    print("\033[91m================PFAI================\033[0m")
    print_colorful(f"PFAI自动化脚本01即将启动：复制程序......", RED)
    def dir_exists(dst):
        return os.path.exists(dst) and os.path.isdir(dst)
    def copy_dir(src, dst):
        if dir_exists(dst):
            print_colorful(f"目录 {dst} 已经存在，跳过复制。", RED)
        else:
            print_colorful(f"正在复制目录：{src} 到 {dst}", RED)
            shutil.copytree(src, dst, dirs_exist_ok=True)
    src_dir = "/mnt/workspace/HHXY_PFAI/LLaMA-Factory/PFAI/"
    dst_dir = "/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/LLaMA-Factory/PFAI/"
    copy_dir(src_dir, dst_dir)

    def copy_file(src, dst):
        print_colorful(f"正在复制文件：{src} 到 {dst}", RED)
        shutil.copy(src, dst)
    src_file = "/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/dataset_info.json"
    dst_path = "/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/LLaMA-Factory/data/"
    os.makedirs(dst_path, exist_ok=True)
    file_name = os.path.basename(src_file)
    copy_file(src_file, os.path.join(dst_path, file_name))
    run_command(['pip', 'install', '-r', 'requirements.txt'], '安装requirements.txt中列出的依赖', stdout_color=PURPLE,
                stderr_color=RED)
    run_command([
        'pip', 'install', 'transformers_stream_generator', 'bitsandbytes', 'tiktoken', 'auto-gptq', 'optimum', 'autoawq'
    ], '安装额外的Python依赖', stdout_color=BLUE, stderr_color=RED)
    run_command(['pip', 'install', '--upgrade', 'tensorflow'], '升级tensorflow', stdout_color=GREEN, stderr_color=RED)
    run_command(['pip', 'install', 'vllm==0.4.3'], '安装vllm', stdout_color=YELLOW, stderr_color=RED)
    run_command([
        'pip', 'install', 'torch==2.1.2', 'torchvision==0.16.2', 'torchaudio==2.1.2', '--index-url',
        'https://download.pytorch.org/whl/cu121'
    ], '安装特定版本的PyTorch', stdout_color=CYAN, stderr_color=RED)
    run_command(['pip', 'install', 'tensorflow==2.12.0'], '安装tensorflow 2.12.0', stdout_color=PURPLE,
                stderr_color=RED)

    run_command(['pip', 'install', '-e', '.[metrics]'], '安装LLaMA-Factory的metrics模块', stdout_color=GREEN,
                stderr_color=RED)

    run_command([
        'pip', 'install', 'ijson', 'pyaml'
    ], '安装额外的Python依赖', stdout_color=BLUE, stderr_color=RED)

    # run_command(['python', '../自动化脚本.py'], '启动自动化脚本', stdout_color=GREEN, stderr_color=RED)
    clone_nltk_data()
    print_colorful(
        "下载Qwen/Qwen2-7B模型 如果等待时间较长可自行手动下载     \n命令1:export HF_ENDPOINT=https://hf-mirror.com \n命令2:cd /mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap \n命令3:huggingface-cli download --resume-download <你的模型名字> --local-dir <要存储的路径>",
        RED)
else:
    print_colorful("脚本已退出。", RED)
    print_colorful(
        "下载Qwen/Qwen2-7B模型 如果等待时间较长可自行手动下载     \n命令1:export HF_ENDPOINT=https://hf-mirror.com \n命令2:cd /mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap \n命令3:huggingface-cli download --resume-download <你的模型名字> --local-dir <要存储的路径>",
        RED)
    exit(0)



