import sys
import os
import uuid
from flask import Flask, request, jsonify
from src.llamafactory.chat.chat_model import run_chat, ChatModel
from src.llamafactory.extras.misc import torch_gc

app = Flask(__name__)
chat_model = None
messages = {}

def generate_user_id():
    """生成唯一的用户ID"""
    return str(uuid.uuid4())

def start_chat_model(port=5000):
    """
    启动聊天模型服务
    :param port: 服务端口号，默认5000
    """
    global chat_model
    
    # 模拟命令行参数
    sys.argv = [
        "chat",  # 固定为chat命令
        "--model_name_or_path", "/home/wbx/.cache/modelscope/hub/Qwen/Qwen2-7B-Instruct",
        "--template", "qwen",
    ]
    
    if os.name != "nt":
        try:
            import readline  # noqa: F401
        except ImportError:
            print("Install `readline` for a better experience.")
    
    chat_model = ChatModel()
    print(f"Chat model started. Server running on port {port}")
    
    # 启动Flask应用
    app.run(port=port)

@app.route('/chat', methods=['POST'])
def chat():
    """
    处理用户输入并返回模型响应
    """
    global chat_model, messages
    
    if not chat_model:
        return jsonify({"error": "Chat model not initialized"}), 500
    
    data = request.json
    user_id = data.get('user_id')
    query = data.get('query', '')
    
    # 如果没有提供user_id，则生成一个新的
    if not user_id:
        user_id = generate_user_id()
        return jsonify({
            "new_user_id": user_id,
            "message": "No user_id provided. A new one has been generated. Please use this for future requests."
        })
    
    if not query:
        return jsonify({"error": "Empty query"}), 400
    
    # 初始化用户消息历史
    if user_id not in messages:
        messages[user_id] = []
    
    # 处理特殊命令
    if query.strip().lower() == "clear":
        messages[user_id] = []
        torch_gc()
        return jsonify({"response": "History has been cleared"})
    
    # 添加用户消息
    messages[user_id].append({"role": "user", "content": query})
    
    # 获取模型响应
    response = ""
    for new_text in chat_model.stream_chat(messages[user_id]):
        response += new_text
    
    # 添加助手响应
    messages[user_id].append({"role": "assistant", "content": response})
    
    return jsonify({"response": response, "user_id": user_id})

@app.route('/new_session', methods=['POST'])
def new_session():
    """创建一个新的会话并返回user_id"""
    user_id = generate_user_id()
    messages[user_id] = []  # 初始化空消息历史
    return jsonify({
        "user_id": user_id,
        "message": "New session created"
    })

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """
    关闭聊天模型服务
    """
    global chat_model
    
    if not chat_model:
        return jsonify({"error": "Chat model not initialized"}), 500
    
    # 清理资源
    torch_gc()
    chat_model = None
    messages.clear()
    
    # 获取关闭服务器的函数
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        return jsonify({"error": "Not running with the Werkzeug Server"}), 500
    
    func()
    return jsonify({"message": "Server shutting down..."})

if __name__ == '__main__':
    start_chat_model()