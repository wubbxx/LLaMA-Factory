import os
import re
from prettytable import PrettyTable

# 指定主目录路径
base_dir = "/home/wbx/LLaMA-Factory/saves/"

# 提取用户输入句子
def extract_user_input(prompt):
    # 匹配 \n\n 之后到 \nassistant 之间的内容
    match = re.search(r"\n\n(.*?)\nassistant", prompt, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_deepseek_input(prompt):
    # 匹配 \n\n 之后到 \nassistant 之间的内容
    match = re.search(r"\n\n(.*?)\n\nAssistant", prompt, re.DOTALL)
    return match.group(1).strip() if match else ""

# 遍历目录并处理每个文件
def process_predictions(base_dir):
    results = []  # 存储每个文件的统计结果

    # 遍历目录下的所有子文件夹
    for root, dirs, files in os.walk(base_dir):
        if "generated_predictions.jsonl" in files:
            eval_path = os.path.join(root, "generated_predictions.jsonl")
            
            # 获取最后一个文件夹名
            folder_name = os.path.basename(root)
            if "-" in folder_name:
                train_dataset, test_dataset = folder_name.split("-", 1)
            else:
                train_dataset, test_dataset = folder_name, "unknown"

            # 初始化统计变量
            TP = FP1 = FP2 = TN = FN = 0  # 统计四种分类
            total_errors = 0
            total = 0  # 总样本数
            model = "Qwen"
            # 解析文件并统计
            with open(eval_path, "r", encoding="utf-8") as file:
                for line in file:
                    entry = eval(line.strip())
                    label = entry.get("label", "")
                    predict = entry.get("predict", "")
                    prompt = entry.get("prompt", "")
                    user_input = None
                    # 提取用户输入的句子
                    if "deepseek" in eval_path:
                        user_input = extract_deepseek_input(prompt)
                        model = "deepseek"
                    else:
                        user_input = extract_user_input(prompt)

                    # 分类统计
                    if user_input == label:  # 原句正确
                        if predict == label:
                            TN += 1  # 模型未修改，输出正确
                        elif predict != label:
                            FP1 += 1  # 模型误改正确的句子
                    else:  # 原句错误
                        if predict == label:
                            TP += 1  # 模型正确修改错误的句子
                        elif predict == user_input:
                            FN += 1  # 模型未修改错误的句子
                        else:
                            FP2 += 1  # 模型尝试修正错误的句子但改错了
                        total_errors += 1
                    total += 1  # 总样本数增加
            
            # 计算统计指标
            total_predictions = TP + FP1 + FP2  # 总的模型修改数
            total_errors = TP + FN + FP2  # 数据集中总错误数
            accuracy = (TP + TN) / total if total > 0 else 0
            precision = TP / total_predictions if total_predictions > 0 else 0
            recall = TP / total_errors if total_errors > 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # 记录结果
            results.append({
                "base_model": model,
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
                "total": total,
                "total_errors": total_errors,
                "total_correct": total - total_errors,
                "TP": TP,
                "TN": TN,
                "FP1": FP1,
                "FP2": FP2,
                "FN": FN,
                "accuracy": f"{accuracy:.2%}",
                "precision": f"{precision:.2%}",
                "recall": f"{recall:.2%}",
                "f1_score": f"{f1_score:.2%}"
            })
    
    return results

# 主函数调用
if __name__ == "__main__":
    all_results = process_predictions(base_dir)

    # 按照训练集（train_dataset）和测试集（test_dataset）字母排序
    sorted_results = sorted(all_results, key=lambda x: (x["test_dataset"], x["train_dataset"]))

    # 创建表格
    table = PrettyTable()
    table.field_names = [
        "Base Model",
        "训练集", "测试集", "总数", "错误总数", "正确总数", "正确修改 (TP)", "正确保持 (TN)", 
        "误改 (FP1)", "错改 (FP2)", "未改 (FN)", 
        "Accuracy", "Precision", 
        "Recall", "F1-score"
    ]

    # 填充表格数据
    for result in sorted_results:
        table.add_row([ 
            result["base_model"],
            result["train_dataset"],
            result["test_dataset"],
            result["total"],
            result["total_errors"],
            result["total_correct"],
            result["TP"],
            result["TN"],
            result["FP1"],
            result["FP2"],
            result["FN"],
            result["accuracy"],
            result["precision"],
            result["recall"],
            result["f1_score"]
        ])

    # 输出表格
    print(table)
