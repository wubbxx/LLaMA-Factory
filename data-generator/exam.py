import os
import random
import argparse

def print_random_sentences(input_file, num_sentences=1000):
    """
    从合并的文件中随机抽取指定数量的句子并打印。
    """
    # 读取合并后的文件内容
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 如果文件中的句子少于要求的数量，使用所有句子
    num_sentences = min(num_sentences, len(lines))
    
    # 随机选择句子
    random_sentences = random.sample(lines, num_sentences)
    
    # 打印选中的句子
    i = 0
    for sentence in random_sentences:
        i += 1
        print(f"{i}: ", end="")
        print(sentence.strip())  # .strip() 去除句子两端的多余空白字符


def main():
    parser = argparse.ArgumentParser(description="文件操作程序")
    parser.add_argument('--input_file', type=str, default='/home/wbx/LLaMA-Factory/data-generator/train/cleaned.txt', help="输入文件路径，抽取模式需要此参数")
    parser.add_argument('--num_sentences', type=int, default=10, help="抽取的句子数量，默认为1000")
    
    args = parser.parse_args()
    print_random_sentences(args.input_file, args.num_sentences)

if __name__ == '__main__':
    main()
