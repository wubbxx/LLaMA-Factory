import os
import random
import json
import pandas as pd
from cscd.pseudo.build import add_noise_to_sentence
from tqdm import tqdm  # 导入 tqdm 库
import argparse

# 添加噪声并生成 noised.txt 文件
def add_noise_to_sentences(input_file, output_file):
    # 读取 merged_output.txt 文件内容
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 计算所有句子的总数，用于进度条
    total_sentences = len(lines)

    # 打开 noised.txt 文件用于写入
    with open(output_file, 'w', encoding='utf-8') as output_file:
        # 使用 tqdm 包装整个句子处理的进度条
        with tqdm(total=total_sentences, desc="Processing all sentences", unit="sentence") as pbar:
            # 处理每一行，给句子添加噪声
            for line in lines:
                sentence = line.strip()  # 去除多余的空白字符
                if sentence:
                    # 生成一个 0 到 1 之间的随机数
                    if random.random() < 0.6:  # 60%的概率加入噪声
                        noise = add_noise_to_sentence(sentence)["noise"]
                        output_file.write(f"{sentence}\t{noise}\n")
                    else:
                        output_file.write(f"{sentence}\t{sentence}\n")
                
                    # 更新进度条
                    pbar.update(1)

    print(f"所有句子已处理并保存到 {output_file.name}")

# 将 noised.txt 转换为 JSON 格式
def convert_tsv_to_json(input_file, output_file):
    # 使用 pandas 读取 TSV 文件
    data = pd.read_csv(input_file, sep="\t", header=None, names=["label", "input"])
    
    # 构建 JSON 数据结构
    json_data = []
    
    # 使用 tqdm 添加进度条
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Converting to JSON", ncols=100):
        instruction = "你是一个优秀的中文拼写纠错模型，中文拼写纠错模型即更正用户输入句子中的拼写错误。"
        input_text = (
            "你需要识别并纠正用户输入的句子中可能的错别字并输出正确的句子，"
            "在纠正错别字的同时尽可能减少对原句子的改动。只输出没有错别字的句子，"
            "不要添加任何其他解释或说明。如果句子没有错别字，就直接输出和输入相同的句子。\n\n" + str(row["input"])
        )
        output_text = str(row["label"])
        
        json_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })
    
    # 写入 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据已成功转换并保存到 {output_file}")

# 主函数：依次执行噪声添加和转换为 JSON
def main(args):
    # Define paths based on mode
    base_path = '/home/wbx/LLaMA-Factory/data-generator/'
    if args.mode == 'train':
        merged_output_file = os.path.join(base_path, 'train', 'reconstruct.txt')
        noised_output_file = os.path.join(base_path, 'train', 'noised.txt')
        json_output_file = './train/cnki.json'
    elif args.mode == 'eval':
        merged_output_file = os.path.join(base_path, 'eval', 'reconstruct.txt')
        noised_output_file = os.path.join(base_path, 'eval', 'noised.txt')
        json_output_file = './eval/cnki.json'
    else:
        raise ValueError("Invalid mode. Please choose either 'train' or 'eval'.")

    # 第一步：给句子添加噪声并保存为 noised.txt
    add_noise_to_sentences(merged_output_file, noised_output_file)
    
    # 第二步：将 noised.txt 转换为 JSON 格式并保存
    convert_tsv_to_json(noised_output_file, json_output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files and add noise.")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True, help="Choose the mode: 'train' or 'eval'.")
    args = parser.parse_args()
    main(args)
