import os
import re
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope.hub.snapshot_download import snapshot_download
from tqdm import tqdm
import torch

device = "cuda"
model_dir = snapshot_download("Qwen/Qwen2-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
clean_prompts = [
    "这个句子是否正确使用了标点符号并清晰地进行了断句？如果是，请输出“1”；如果没有，请输出“0”:",
    "阅读下面的句子。如果句子没有语法错误并且内容完整，请输出“1”。如果句子语法不正确或含义不清，请输出“0”:",
    "作为一个语言分析助手，我需要你帮忙判断下面的句子是否符合语言规范。如果符合请回复“1”，不符合请回复“0”:",
    "请检查以下句子中的信息是否完整。如果句子信息完整，请输出“1”。如果句子信息不完整或缺失关键细节，请输出“0”:",
    "请你判断下列句子是否语法正确且有语义？如果是则输出“1”,否 则输出“0"
]
reconstruct_prompt = "请修改或重述下面的句子，使其使用正确的标点符号并清晰地进行断句，且有明确语义。确保句子不缺失关键信息（如日期、数据等等）。仅输出修改后的句子即可，不要输出多余的内容："
unit_pattern = r'(km|cm|mm|µm|nm|Å|kg|mg|µg|ms|µs|ns|°C|°F|eV|Pa|bar|atm|m/s|km/h|m³|mL|m²|km²|cm²|mm²|mol|cd|Hz|Bq|lx|lm|C/kg|Wb|V·s|V/m|F/m|Ω·m|J/mol|m/s²|rad|sr|db|dB|m/s³|kg/m³|T|G|S|g|s|K|A|V|Ω|C|W|J|F|T|H|N|L|m)'

# Define the task-specific function for "clean"
def clean_sentences(args):
    # Define paths based on mode
    base_path = '/home/wbx/LLaMA-Factory/data-generator/'
    if args.mode == 'train':
        txt_folder = os.path.join(base_path, 'train', 'txt')
        invalid_sentences_file = os.path.join(base_path, 'train', 'invalid.txt')
        cleaned_sentences_file = os.path.join(base_path, 'train', 'cleaned.txt')
    elif args.mode == 'eval':
        txt_folder = os.path.join(base_path, 'eval', 'txt')
        invalid_sentences_file = os.path.join(base_path, 'eval', 'invalid.txt')
        cleaned_sentences_file = os.path.join(base_path, 'eval', 'cleaned.txt')
    else:
        raise ValueError("Invalid mode. Please choose either 'train' or 'eval'.")

    # The device to load the model onto

    # 确保 pad_token 和 eos_token 不相同
    tokenizer.pad_token = tokenizer.eos_token  # 可以设置成其他标记，如果需要

    # 获取该文件夹下所有 .txt 文件
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]

    # 确保 cleaned.txt 存在
    if not os.path.exists(cleaned_sentences_file):
        with open(cleaned_sentences_file, 'w', encoding='utf-8') as cleaned_file:
            cleaned_file.write("")  # 初始化文件为空

    # 遍历所有的 .txt 文件
    for txt_file in tqdm(txt_files, desc="Processing Files", unit="file"):
        file_path = os.path.join(txt_folder, txt_file)  # 获取每个文件的完整路径
        
        # 打开并读取当前文件的内容
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 使用 tqdm 包装 lines，显示进度条
        invalid_flag = False  # 标志变量，确保文件名只输出一次
        for line_num, line in tqdm(enumerate(lines, 1), desc=f"Processing {txt_file}", unit="sentence"):
            sentence = line.strip()  # 去除多余的空白字符
            if not sentence:
                continue  # 跳过空行
            score_count = 0
            total_trials = len(clean_prompts)
            trial_results = []  # 用于记录每次评估的结果
            for match in re.finditer(unit_pattern, sentence):
                preceding_char = sentence[match.start() - 1]  # Get the character before the unit
                if not (preceding_char.isdigit() or preceding_char.isascii()):  # If not a number or letter
                    if len(sentence) > match.end():
                        if sentence[match.end()].isascii():
                            continue
                    score_count = -5  # Apply the penalty
                    break
            if re.search(r"第(章|节|页|步|条|项|段|行|部分|款|篇|点|期)", sentence):
                score_count = -10
            if score_count == 0:
                for prompt in clean_prompts:
                    prompt_text = f"{prompt}\n\n{sentence}"
                    
                    messages = [
                        {"role": "system", "content": "你是一个智能助手，负责检索中文文本的有效性。"},
                        {"role": "user", "content": prompt_text}
                    ]
                    
                    # 构建输入并调整为模型需要的格式
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    model_inputs = tokenizer([text], return_tensors="pt").to(device)

                    # Ensure pad_token_id and attention_mask are properly set
                    model_inputs['attention_mask'] = model_inputs.get('attention_mask', torch.ones(model_inputs['input_ids'].shape, device=device))
                    model_inputs['pad_token_id'] = tokenizer.pad_token_id  # Specify pad_token_id if it's not set by default

                    generated_ids = model.generate(model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pad_token_id=model_inputs.pad_token_id, max_new_tokens=5)

                    # 处理输出并解码
                    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    # 记录每次评估的结果
                    trial_results.append(response)

                    if response == '1':
                        score_count += 1
                # 如果评分符合要求，将句子保存到 cleaned.txt 文件
            if score_count == total_trials:
                with open(cleaned_sentences_file, 'a', encoding='utf-8') as cleaned_file:
                    cleaned_file.write(sentence + "\n")
            else:
                # 如果不符合，将句子写入到 invalid_sentences.txt 文件
                if not invalid_flag:
                    with open(invalid_sentences_file, 'a', encoding='utf-8') as invalid_file:
                        invalid_file.write(f"=============================================================\n")
                        invalid_file.write(f"文件: {txt_file}\n")
                        invalid_file.write("=============================================================\n")
                    invalid_flag = True  # 设置标志，确保文件名只输出一次

                with open(invalid_sentences_file, 'a', encoding='utf-8') as invalid_file:
                    if score_count == -5:
                        invalid_file.write(f"行号: {line_num:4} <units  error>  | {sentence}\n")
                    elif score_count == -10:
                        invalid_file.write(f"行号: {line_num:4} <pinyin error>  | {sentence}\n")
                    else:
                        invalid_file.write(f"行号: {line_num:4} {' '.join(trial_results)} | {score_count}/{total_trials} | {sentence}\n")


def reconstruct_sentences(args):
    mode = args.mode
    with open(f'/home/wbx/LLaMA-Factory/data-generator/{mode}/cleaned.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        # Open the 'reconstruct.txt' file in append mode to store the output
        with open(f'/home/wbx/LLaMA-Factory/data-generator/{mode}/reconstruct.txt', 'a', encoding='utf-8') as output_file:
            for line in tqdm(lines, desc="Processing sentences", unit="sentence"):
                sentence = line.strip()  # 去除多余的空白字符
                messages = [
                    {"role": "system", "content": "你是一个智能助手，负责检索中文文本的有效性。"},
                    {"role": "user", "content": f"{reconstruct_prompt} : \n\n{sentence}"}
                ]

                # 构造输入并调整为模型需要的格式
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = tokenizer([text], return_tensors="pt").to(device)

                # Ensure pad_token_id and attention_mask are properly set
                model_inputs['attention_mask'] = model_inputs.get('attention_mask', torch.ones(model_inputs['input_ids'].shape, device=device))
                model_inputs['pad_token_id'] = tokenizer.pad_token_id  # Specify pad_token_id if it's not set by default

                generated_ids = model.generate(model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pad_token_id=model_inputs.pad_token_id, max_new_tokens=1024)

                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Write the response to 'reconstruct.txt'
                output_file.write(response + '\n')  # Add a newline to separate sentences


def delete_juzi(args):
    mode = args.mode
    # 使用os.path.join来保证路径正确
    input_file = os.path.join('/home/wbx/LLaMA-Factory/data-generator', mode, 'cleaned.txt')
    output_file = os.path.join('/home/wbx/LLaMA-Factory/data-generator', mode, 'final.txt')
    
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            if not line or '句子' in line:
                continue
            outfile.write(line + '\n')
    print(f"处理完成，结果已写入 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files with a language model.")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True, help="Choose the mode: 'train' or 'eval'.")
    parser.add_argument('--task', type=str, choices=['clean', 'reconstruct', 'delete'], required=True, help="Choose the task: 'clean','reconstruct' or 'delete'.")
    args = parser.parse_args()
    if args.task == "clean":
        clean_sentences(args)
    elif args.task == "reconstruct":
        reconstruct_sentences(args)
    elif args.task == "delete":
        delete_juzi(args)
    else:
        raise ValueError("Invalid task. Please choose either 'clean' or 'reconstruct'.")
