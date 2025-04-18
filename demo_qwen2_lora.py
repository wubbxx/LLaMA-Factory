from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 定义任务提示
prompts = "你需要识别并纠正用户输入的句子中可能的错别字并输出正确的句子，在纠正错别字的同时尽可能减少对原句子的改动。只输出没有错别字的句子，不要添加任何其他解释或说明。如果句子没有错别字，就直接输出和输入相同的句子。"
device = "cuda"

# 使用本地路径加载模型
model_dir = "/home/wbx/LLaMA-Factory/saves/qwen/lora/sft-7B/merged/cnki-train2"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

lines = [
    "具有四轮、转矩独立可控、转矩响应快速、传动小吕高以及控制灵活等特点。",
    "尤其在车辆的横总想稳定性控制、制动能量回首以及主动安全控制等领域，分布式驱动线控底盘显示出了巨大的潜力。",
    "本文以分布式驱动线控底盘为研究对象，将差动转向机制作为线控转向系统失效时的容错机制，并铜锅设计控制策略，结合行驶状态观测器，进一步研究分布式驱动线控底盘的轨迹跟踪问题。"
]

# 逐行处理每个句子
for line in lines:
    sentence = line.strip()  # 去除多余的空白字符
    if not sentence:
        continue  # 跳过空行

    # 将句子拼接到 prompts 之后
    prompt_text = f"{prompts}\n\n{sentence}"
    
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

    generated_ids = model.generate(model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pad_token_id=model_inputs.pad_token_id, max_new_tokens=(len(sentence) + 5))

    # 处理输出并解码
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = (tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]).strip()

    # 打印结果到命令行
    print(f"原始句: {sentence}")
    print(f"纠正句: {response}")
    print("-" * 50)  # 分隔线