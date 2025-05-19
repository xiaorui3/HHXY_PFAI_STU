import os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def multichoice(model_name):
    QnA_dir_path = 'ProteinLMBench.json'
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

        with open(f'result/raw_result__{model_name}.json', 'w') as jj:
            json.dump(after, jj)

        with open(f'result/rst_compar__{model_name}.txt', 'w') as results:
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

    with open(f'result/raw_result__{model_name}.json', 'w') as jj:
        json.dump(after, jj)

    with open(f'result/rst_compar__{model_name}.txt', 'w') as results:
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

model_name_list = [
    '/home/netzone22/hhxy/model/Qwen_Lora_1w_exist_rslora_no_dora_no_pissa_matrix_len_6e-5_fp32_new'# Qwen_Lora_1w_exist_rslora_no_dora_no_pissa_matrix_len_6e5
]
for model in model_name_list:
    try:
        acc = multichoice(model)
        print(f"Acc of {model} is: {acc}")
    except Exception as e:
        print(e)
        continue
