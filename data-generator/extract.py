import fitz  # PyMuPDF
import re
import os
from tqdm import tqdm  # 导入tqdm模块

# 用于判断文本是否是公式的函数
def is_formula(text):
    # 检测常见的公式符号
    formula_patterns = [
        r'\$',  # LaTeX公式符号
        r'\$\$',  # 多行LaTeX公式
        r'\\',  # 换行符，用于LaTeX公式
        r'[=+\-*/]',  # 数学运算符
        r'\\sum', r'\\int', r'\\frac', r'\\sqrt',  # LaTeX命令
        r'\[.*\]',  # 方括号包裹的公式
        r'\(.*\)'   # 小括号包裹的公式
    ]
    
    # 如果文本中有符合公式模式的字符，则认为它是公式
    return any(re.search(pattern, text) for pattern in formula_patterns)

# 用于判断是否为参考文献的函数
def is_reference(text):
    # 检测参考文献部分常见的标识符
    reference_patterns = [
        r'\[\d+\]',   # 如 [1]、[2] 等
        r'\(\d+\)',   # 如 (1)、(2) 等
        r'\bReferences\b',  # 标题 "References" 或 "参考文献"
        r'\b参考文献\b',     # 中文标题 "参考文献"
    ]
    
    # 如果文本中有符合参考文献模式的字符，则认为它是参考文献
    return any(re.search(pattern, text) for pattern in reference_patterns)

def extract_text_from_pdf(pdf_path, output_txt_path):
    # 打开PDF文件
    doc = fitz.open(pdf_path)
    all_lines = []
    
    # 遍历每一页
    for page_num in range(len(doc)):
        # 获取当前页
        page = doc.load_page(page_num)
        # 提取当前页的文本，按行拆分
        page_lines = page.get_text().splitlines()
        all_lines.extend(page_lines)
    doc.close()

    # 合并属于同一句话但换行的内容
    merged_text = ""
    for index, line in enumerate(all_lines):
        if index < len(all_lines) - 1:
            next_line = all_lines[index + 1]
            if next_line.strip():  # 先判断next_line是否为空字符串
                if not (next_line.strip()[0] in ["?", "!", '"', "'", "(", "[", "{", "。"]):
                    merged_text += line + " "
                else:
                    merged_text += line + "\n"
            else:
                merged_text += line + "\n"
        else:
            merged_text += line + "\n"

    # 按照常见标点符号分割文本为句子列表，并去除不符合要求的句子
    sentences = []
    for delimiter in ["。", "?", "!"]:
        merged_text = merged_text.replace(delimiter, delimiter + "|")
    parts = merged_text.split("|")
    
    for part in parts:
        part = part.strip()
        if part:
            # 如果是公式，跳过
            if is_formula(part):
                continue

            # 如果是参考文献，跳过
            if is_reference(part):
                continue

            # 只保留汉字、中文标点、英文字母，并且长度大于等于6
            clean_part = re.sub(r'[^\u4e00-\u9fff\u3002\uff1b\uff0c\uff1a\u201c\u201d\u2018\u2019a-zA-Z]', '', part)
            # 检查是否包含中文字符，并且长度大于等于6
            if len(clean_part) >= 6 and re.search(r'[\u4e00-\u9fff]', clean_part):
                # 检查是否有连续超过7个英文字母
                if re.search(r'[a-zA-Z]{8,}', clean_part):
                    continue  # 如果有连续英文字母超过7个，则跳过该句子
                # 检查句子中是否有连续重复的标点符号
                if re.search(r'[。！？\.,，]{2,}', clean_part):
                    continue  # 如果有连续的标点符号，则跳过该句子
                # 检查是否有无意义的字母组合（例如全英文字母或乱码）
                if re.search(r'[a-zA-Z]{3,}', clean_part) and not re.search(r'[\u4e00-\u9fff]', clean_part):
                    continue  # 如果只有英文字符且没有中文，则跳过该句子
                # 检查英文字母和中文字符的比例，过高的英文字母比例可能是无意义字母组合
                english_chars = len(re.findall(r'[a-zA-Z]', clean_part))
                chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', clean_part))
                if english_chars > chinese_chars and english_chars / len(clean_part) > 0.5:
                    continue  # 如果英文字母比例超过50%，则跳过该句子

                sentences.append(clean_part)

    # 使用 tqdm 显示进度条，写入文件
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for sentence in tqdm(sentences, desc="写入文件", unit="句子"):
            f.write(sentence + "\n")

# 遍历 pdf_folder 文件夹下的所有 PDF 文件，并生成对应的 TXT 文件
pdf_folder = "./cnki_eval"
out_folder = "./txt_eval"

# 确保输出目录存在
os.makedirs(out_folder, exist_ok=True)

# 获取 pdf_folder 中所有 PDF 文件
for pdf_filename in os.listdir(pdf_folder):
    if pdf_filename.lower().endswith(".pdf"):  # 只处理PDF文件
        pdf_path = os.path.join(pdf_folder, pdf_filename)
        output_txt_path = os.path.join(out_folder, pdf_filename.replace(".pdf", ".txt"))
        extract_text_from_pdf(pdf_path, output_txt_path)
        print(f"处理完毕: {pdf_filename}")
