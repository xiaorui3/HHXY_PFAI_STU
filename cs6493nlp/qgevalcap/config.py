# 模型路径
model_name_or_path = "/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/qwen27/"

# 数据集名称
dataset = "data_temp2025-1-11"  # llamafactory的data名字


# 生成参数
TEMPERATURE = 0.95
TOP_P = 0.95
TOP_K = 50
MAX_NEW_TOKENS = 200
cuda_visible_devices = "0"# 0,1,2,3,4,5,6,7
clone_nltk_data = 0 # 是否下载nltk程序f
file_path = '/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/folder/extracted_qa.json' # 使用的json文件
flie_len = 300# file_path文件有多少个问题答案 # 一定要准确无误
# 输入配置
PROMPT = ""

result_test=0 # 是否进行问答模型

folder_path="./named_extracted_qa_test3/" # 问题的目录

QnA_dir_path="/mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap/folder/250104英文100.json"


model_name_="qwen27"
# 输出目录
output_dir_base = './PFAI/Qwen2-7B/lora/'   # 无用，但是不能删，删了就跑不起来了


# 其他参数
num_rows_to_execute = 1 # 程序执行次数
learning_rate=0.00008 # 模型微调的学习率
num_train_epochs=3 # 模型微调的训练轮数
max_grad_norm=1.0 # 模型微调的最大梯度范围
lora_rank=16 # 模型微调的Lora 秩
lora_alpha=16# 模型微调的LoRA 缩放系数
output_dir=f'PFAI/Qwen2-7B/lora/train__{model_name_}{lora_rank}_{lora_alpha}_{num_train_epochs}' # 模型训练临时文件路径 可以自己设置
export_dir = f'./PFAI/models/' + f'{model_name_}{lora_rank}_{lora_alpha}_{num_train_epochs}' # 模型导出路径 可以自己设置

MODEL_NAME = export_dir  # /home/netzone22/hhxy/LLaMA-Factory/PFAI/models/30_8_16_0.00011   #./qwen27
DEVICE = "cuda"
TORCH_DTYPE = "auto"
DEVICE_MAP = "auto"