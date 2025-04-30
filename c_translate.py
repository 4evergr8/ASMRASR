import os
import pysrt
from getconfig import get_config
import google.genai as genai

# 配置 API 密钥
genai.configure(api_key="YOUR_API_KEY")

# 创建模型实例
model = genai.GenerativeModel("gemini-pro")

# 启动聊天会话
chat = model.start_chat()

# 进行多轮对话
while True:
    user_input = input("你：")
    if user_input.lower() in ["exit", "quit", "退出"]:
        break

    response = chat.send_message(user_input)
    print("Gemini：", response.text)





def translate(config):
    for root, dirs, files in os.walk(config["asr_path"]):
        for filename in files:
            if filename.endswith((".srt")):
                srt_path = os.path.join(root, filename)
                subs = pysrt.open(srt_path, encoding='utf-8')
                for i in range(0, len(subs), 100):
                    chunk = subs[i:i + 100]
                    for sub in chunk:
                        # 在这里处理每个 subtitle 对象
                        print(sub.text)


if __name__ == "__main__":
    translate(get_config())
