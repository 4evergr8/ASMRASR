import os
import time
import pysrt
from google import genai
from getconfig import get_config
from google.genai import types





def translate(config):
    for filename in os.listdir(config["pre_path"]):
        if filename.endswith(".srt"):
            srt_path = os.path.join(config["pre_path"], filename)
            if os.path.isfile(srt_path):
                subs = pysrt.open(srt_path, encoding='utf-8')

                for i in range(0, len(subs), 100):
                    chunk = subs[i:i + 100]

                    client = genai.Client(api_key=config["api_key"])
                    chat = client.chats.create(
                        model=config['translate'],
                        config=types.GenerateContentConfig(
                            system_instruction=config["prompt"] + "\n" + "\n".join(sub.text for sub in chunk),
                        )
                    )

                    for j in range(0, len(chunk), 10):
                        sub_chunk = chunk[j:j + 10]
                        prompt = "\n".join(f"{k + 1}|{sub.text}" for k, sub in enumerate(sub_chunk))

                        while True:
                            try:
                                #print(prompt)
                                response = chat.send_message(prompt)
                                #print(response.text)
                                lines = response.text.strip().splitlines()
                                valid_lines = [line for line in lines if "|" in line]

                                if len(valid_lines) == len(sub_chunk):
                                    for k, line in enumerate(valid_lines):
                                        parts = line.split("|", 1)
                                        if len(parts) == 2:
                                            sub_chunk[k].text = parts[1].strip()
                                    time.sleep(3)  # 成功后稍微等待
                                    break
                                else:
                                    print("返回行数与原始字幕不一致，等待重试...")
                                    time.sleep(2)

                            except Exception as e:
                                print(f"异常：{e}")
                                time.sleep(3)

                srt_path = os.path.join(config['tsl_path'], filename)
                subs.save(srt_path, encoding='utf-8')


if __name__ == "__main__":
    translate(get_config())
