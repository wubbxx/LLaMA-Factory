import requests
import sys

# Flask 服务器的地址和端口
SERVER_URL = "http://127.0.0.1:5000"

class ChatClient:
    def __init__(self):
        self.user_id = None
        self.session_started = False
    
    def start_session(self):
        """自动创建新会话并获取user_id"""
        try:
            response = requests.post(f"{SERVER_URL}/new_session")
            response.raise_for_status()
            self.user_id = response.json().get("user_id")
            self.session_started = True
            print("\n欢迎使用聊天助手！已自动创建新会话。")
            print("输入 'clear' 清除对话历史，输入 'exit' 退出程序。\n")
            return True
        except requests.exceptions.RequestException as e:
            print(f"创建会话失败: {e}")
            return False
    
    def send_query(self, query):
        """发送查询并获取响应"""
        if not self.session_started:
            print("会话未初始化，请先启动会话")
            return None
        
        try:
            response = requests.post(
                f"{SERVER_URL}/chat",
                json={"user_id": self.user_id, "query": query},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"请求服务器出错: {e}")
            return None

def check_server():
    """检查服务器是否在线"""
    try:
        requests.get(f"{SERVER_URL}/chat", timeout=2)
        return True
    except requests.exceptions.RequestException:
        print("错误: Flask服务器未运行，请先启动服务器！")
        return False

def main():
    if not check_server():
        sys.exit(1)
    
    client = ChatClient()
    if not client.start_session():
        sys.exit(1)
    
    while True:
        try:
            query = input("你: ").strip()
        except KeyboardInterrupt:
            print("\n检测到中断，正在退出...")
            break
        except EOFError:
            print("\n检测到文件结束，正在退出...")
            break

        if not query:
            continue

        if query.lower() == "exit":
            break

        if query.lower() == "clear":
            client.send_query("clear")
            print("助手: 对话历史已清除")
            continue

        print("助手: ", end="", flush=True)
        response = client.send_query(query)
        if response is not None:
            print(response)

if __name__ == "__main__":
    main()