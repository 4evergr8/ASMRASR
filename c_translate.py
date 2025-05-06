import os
import time
import pysrt
from google import genai
from getconfig import get_config
from google.genai import types





def translate(config):
    for filename in os.listdir(config["asr_path"]):
        if not filename.endswith(".srt"):
            continue

        src_path = os.path.join(config["asr_path"], filename)
        dst_path = os.path.join(config["tsl_path"], filename)

        if not os.path.isfile(src_path):
            continue

        original_subs = pysrt.open(src_path, encoding='utf-8')

        # 检查是否存在中途翻译文件
        if os.path.exists(dst_path):
            try:
                translated_subs = pysrt.open(dst_path, encoding='utf-8')
                translated_count = sum(1 for sub in translated_subs if sub.text.strip())
                print(f"检测到已有翻译进度：{translated_count} 行，将从此处继续")
            except Exception as e:
                print(f"读取中途翻译文件失败，重新翻译：{e}")
                translated_count = 0
                translated_subs = original_subs[:]
                for sub in translated_subs:
                    sub.text = ''
        else:
            translated_count = 0
            translated_subs = original_subs[:]
            for sub in translated_subs:
                sub.text = ''

        client = genai.Client(api_key=config["api_key"])
        chat = client.chats.create(
            model=config['translate'],
            config=types.GenerateContentConfig(
                system_instruction=config["prompt"]
            )
        )

        # 分批处理未翻译部分
        for i in range(translated_count, len(original_subs), 100):
            chunk = original_subs[i:i + 100]

            for j in range(0, len(chunk), 10):
                sub_chunk = chunk[j:j + 10]
                prompt = "\n".join(f"{i + j + k + 1}|{sub.text}" for k, sub in enumerate(sub_chunk))

                while True:
                    try:
                        response = chat.send_message(prompt)
                        lines = response.text.strip().splitlines()
                        valid_lines = [line for line in lines if "|" in line]

                        if len(valid_lines) == len(sub_chunk):
                            for k, line in enumerate(valid_lines):
                                _, text = line.split("|", 1)
                                translated_subs[i + j + k].text = text.strip()
                            time.sleep(3)
                            break
                        else:
                            print("行数不匹配，等待重试...")
                            time.sleep(2)
                    except Exception as e:
                        print(f"异常：{e}")
                        time.sleep(3)

            # 每100条写入一次（覆盖写）
            translated_subs.save(dst_path, encoding='utf-8')



if __name__ == "__main__":
    translate(get_config())
