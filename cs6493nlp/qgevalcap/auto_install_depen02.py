import subprocess
import threading
import os
import shutil
import zipfile
import time
from datetime import datetime

# 颜色代码
RED = '91'
GREEN = '32'
YELLOW = '33'
BLUE = '34'
PURPLE = '35'
CYAN = '36'

# 基础路径配置
BASE_DIR = "/HHXY_PFAI/cs6493nlp/qgevalcap"
LLAMA_FACTORY_DIR = os.path.join(BASE_DIR, "LLaMA-Factory")

# 进度跟踪全局变量
current_step = 0
start_time = time.time()
steps = [
    {"name": "            ", "duration": 0, "status": "pending"},
    {"name": "权限验证和目录准备", "duration": 5, "status": "pending"},
    {"name": "设置pip镜像源", "duration": 10, "status": "pending"},
    {"name": "更新apt索引", "duration": 60, "status": "pending"},
    {"name": "安装基础系统依赖", "duration": 180, "status": "pending"},
    {"name": "安装huggingface_hub", "duration": 30, "status": "pending"},
    {"name": "克隆LLaMA-Factory", "duration": 120, "status": "pending"},
    {"name": "目录结构配置", "duration": 60, "status": "pending"},
    {"name": "安装Python依赖", "duration": 600, "status": "pending"},
    {"name": "克隆nltk_data", "duration": 120, "status": "pending"},
    {"name": "模型下载准备", "duration": 0, "status": "pending"}
]


def get_formatted_time(seconds):
    """将秒转换为 时:分:秒.毫秒 格式"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03d}"


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
        print_colorful(f"▶ 当前步骤: {current_step_name} | 启动时间: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}",
                       CYAN)
        remaining_time = sum(s["duration"] for s in steps[current_step:])
        print_colorful(f"⏳ 预计剩余时间: {get_formatted_time(remaining_time)}", YELLOW)
    else:
        print_colorful("✅ 所有步骤已完成", GREEN)

    # 步骤列表区
    print("\n\033[K步骤进度:")
    for idx, step in enumerate(steps):
        status_icon = "✓" if idx < current_step else "◻"
        color = GREEN if idx < current_step else RED
        duration = f"{get_formatted_time(step['duration'])}" if step['duration'] > 0 else ""
        print(f"\033[{color}m {status_icon} {step['name']} {duration}\033[0m")

    # 时间统计区
    print_colorful(f"\n⏱ 总运行时间: {get_formatted_time(elapsed)}", PURPLE)
    print("-" * term_width)


def update_progress():
    global current_step
    if current_step < len(steps):
        steps[current_step]["status"] = "completed"
    current_step += 1
    draw_interface()


def check_root_and_prepare():
    try:
        if os.geteuid() != 0:
            print_colorful("✗ 请使用sudo权限运行此脚本！", RED)
            exit(1)

        if not os.path.exists(BASE_DIR):
            print_colorful(f"✗ 基础目录不存在: {BASE_DIR}", RED)
            exit(1)

        os.chdir(BASE_DIR)
        print_colorful(f"✓ 工作目录已设置为: {os.getcwd()}", GREEN)
    except Exception as e:
        print_colorful(f"✗ 初始化失败: {e}", RED)
        exit(1)
    update_progress()


def confirm_deployment():
    print_colorful("警告：即将开始系统级部署操作！", RED)
    print_colorful("请输入 'y' 确认部署，其他键退出：", YELLOW)
    while True:
        user_input = input("确认部署 (y/n): ").lower()
        if user_input == 'y':
            return True
        elif user_input == 'n':
            print_colorful("✗ 部署已取消", RED)
            return False
        print_colorful("无效输入，请输入 y/n", RED)


def print_output(pipe, name, process, color_code):
    color = f"\033[{color_code}m"
    reset = "\033[0m"
    while True:
        output = pipe.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            print(f"{color}{name}: [{timestamp}] {output.strip()}{reset}")


def run_command(command, description, **kwargs):
    draw_interface()
    start = time.time()

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ
        )
    except Exception as e:
        print_colorful(f"✗ 命令执行失败: {str(e)}", RED)
        exit(1)

    stdout_thread = threading.Thread(
        target=print_output,
        args=(process.stdout, 'OUT', process, kwargs.get('stdout_color', GREEN)))
    stderr_thread = threading.Thread(
        target=print_output,
        args=(process.stderr, 'ERR', process, kwargs.get('stderr_color', RED)))

    stdout_thread.start()
    stderr_thread.start()

    # 实时界面刷新线程
    refresh_active = True

    def refresh_interface():
        while refresh_active:
            draw_interface()
            time.sleep(0.1)  # 每秒刷新10次

    refresh_thread = threading.Thread(target=refresh_interface)
    refresh_thread.start()

    process.wait()
    refresh_active = False  # 停止刷新

    if process.returncode != 0:
        print_colorful(f"✗ 步骤失败: {description} (退出码: {process.returncode})", RED)
        # exit(1)

    actual_duration = time.time() - start
    if current_step < len(steps):
        steps[current_step]["duration"] = max(steps[current_step]["duration"], int(actual_duration))

    stdout_thread.join()
    stderr_thread.join()
    refresh_thread.join()
    update_progress()


def clone_nltk_data():
    try:
        run_command(['git', 'clone', 'https://github.com/nltk/nltk_data.git'],
                    '克隆nltk_data仓库', stdout_color=BLUE)
        temp_dir = os.path.join(os.getcwd(), 'nltk_data')
        target_dir = '/root/nltk_data'
        os.makedirs(target_dir, exist_ok=True)

        packages_dir = os.path.join(temp_dir, 'packages')
        if os.path.exists(packages_dir):
            shutil.copytree(packages_dir, target_dir, dirs_exist_ok=True)
        shutil.rmtree(temp_dir)

        extract_dir = '/root/nltk_data/tokenizers'
        os.makedirs(extract_dir, exist_ok=True)
        for zip_file in ['punkt_tab.zip', 'punkt.zip']:
            zip_path = os.path.join(extract_dir, zip_file)
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(extract_dir)
    except Exception as e:
        print_colorful(f"✗ nltk操作失败: {str(e)}", RED)
        exit(1)


def directory_operations():
    src_dir = "/HHXY_PFAI/LLaMA-Factory/PFAI/"
    dst_dir = os.path.join(LLAMA_FACTORY_DIR, "PFAI")
    if not os.path.exists(dst_dir):
        try:
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        except Exception as e:
            print_colorful(f"✗ 目录复制失败: {str(e)}", RED)
            # exit(1)

    config_src = "/HHXY_PFAI/cs6493nlp/qgevalcap/dataset_info.json"
    config_dst = os.path.join(LLAMA_FACTORY_DIR, "data/dataset_info.json")
    os.makedirs(os.path.dirname(config_dst), exist_ok=True)
    try:
        shutil.copy2(config_src, config_dst)
    except Exception as e:
        print_colorful(f"✗ 配置文件复制失败: {str(e)}", RED)
        exit(1)
    update_progress()


def show_model_download_tip():
    draw_interface()
    print_colorful("\n⚠️ 模型下载提示 ⚠️", RED)
    print("export HF_ENDPOINT=https://hf-mirror.com")
    print(f"cd {BASE_DIR}")
    print("huggingface-cli download --resume-download Qwen/Qwen2-7B --local-dir qwen27\n")
    input("按回车键继续...")


def main_deployment():
    run_command(['pip', 'config', 'set', 'global.index-url',
                 'https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple'],
                '设置pip镜像源', stdout_color=BLUE)

    run_command(['sudo', 'apt', 'update', '-y'], '更新apt索引', stdout_color=GREEN)

    run_command(['sudo', 'apt', 'install', '-y', 'git', 'git-lfs', 'cmake',
                 'make', 'iftop', 'atop', 'openjdk-17-jdk'], '安装系统依赖', stdout_color=GREEN)

    run_command(['pip', 'install', '-U', 'huggingface_hub'], '安装huggingface_hub', stdout_color=BLUE)

    run_command(['git', 'clone', 'https://github.com/hiyouga/LLaMA-Factory.git'],
                '克隆LLaMA-Factory仓库', stdout_color=CYAN)

    if not os.path.exists("LLaMA-Factory"):
        print_colorful("✗ LLaMA-Factory仓库克隆失败！", RED)
        # exit(1)

    try:
        os.chdir('LLaMA-Factory')
        print_colorful(f"✓ 已进入目录: {os.getcwd()}", GREEN)
    except Exception as e:
        print_colorful(f"✗ 目录切换失败: {str(e)}", RED)
        exit(1)

    directory_operations()

    dependency_commands = [
        (['pip', 'install', '-r', 'requirements.txt'], '安装requirements.txt'),
        (['pip', 'install', 'transformers_stream_generator', 'bitsandbytes',
          'tiktoken', 'auto-gptq', 'optimum', 'autoawq'], '安装推理依赖'),
        (['pip', 'install', '--upgrade', 'tensorflow'], '升级TensorFlow'),
        (['pip', 'install', 'vllm==0.4.3'], '安装vLLM'),
        (['pip', 'install', 'torch==2.1.2', 'torchvision==0.16.2',
          'torchaudio==2.1.2', '--index-url', 'https://download.pytorch.org/whl/cu121'],
         '安装PyTorch'),
        (['pip', 'install', 'tensorflow==2.12.0'], '安装TensorFlow 2.12.0'),
        (['pip', 'install', '-e', '.[metrics]'], '安装Metrics模块'),
        (['pip', 'install', 'ijson', 'pyaml'], '安装辅助工具')
    ]

    for cmd, desc in dependency_commands:
        run_command(cmd, desc)

    clone_nltk_data()
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    show_model_download_tip()


if __name__ == "__main__":
    try:
        print("\033[?25l")  # 隐藏光标
        if confirm_deployment():
            check_root_and_prepare()
            main_deployment()
            print_colorful("\n✓✓✓ 部署成功完成 ✓✓✓", GREEN)
        else:
            print_colorful("✗ 用户取消部署", YELLOW)
    except Exception as e:
        print_colorful(f"\n✗✗✗ 部署失败: {str(e)} ✗✗✗", RED)
    finally:
        print("\033[?25h")  # 恢复光标显示