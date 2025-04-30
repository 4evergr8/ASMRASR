import os
import pysrt
from getconfig import get_config
import google.genai
from google.genai import types

# 初始化客户端
client = google.genai.Client(api_key="your-api-key")

# 创建聊天会话
chat = client.chats.create(model="gemini-2.0-flash",config=types.GenerateContentConfig(
        system_instruction="You are a cat. Your name is Neko."),)

# 发送第一条消息
response = chat.send_message("我家里有两只狗。")
print("模型回答:", response.text)

# 发送第二条消息
response = chat.send_message("那我家里总共有多少只爪子？")
print("模型回答:", response.text)

# 输出聊天历史
for message in chat.get_history():
    print(f'{message.role}: {message.parts[0].text}')



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
