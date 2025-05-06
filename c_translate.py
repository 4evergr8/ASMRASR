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
        basename = os.path.splitext(filename)[0]
        src_path = os.path.join(config["asr_path"], filename)
        dst_path = os.path.join(config["tsl_path"], filename)

        original_subs = pysrt.open(src_path, encoding='utf-8')

        try:
            translated_subs = pysrt.open(dst_path, encoding='utf-8')
        except:
            translated_subs = pysrt.SubRipFile()

        # 如果翻译的字幕与原字幕一样长，跳过
        if len(translated_subs) == len(original_subs):
            print(f"文件 {filename} 已经全部翻译完，跳过.")
            continue

        # 计算剩余未翻译的部分
        original_subs = original_subs[len(translated_subs):]

        # 翻译剩余部分
        for i in range(0, len(original_subs), 100):
            chunk = original_subs[i:i + 100]

            client = genai.Client(api_key=config["api_key"])
            chat = client.chats.create(
                model=config['translate'],
                config=types.GenerateContentConfig(
                    system_instruction=config["prompt"].format(basename=basename) + "\n" + "\n".join(sub.text for sub in chunk),
                )
            )

            for j in range(0, len(chunk), 10):
                sub_chunk = chunk[j:j + 10]
                prompt = "\n".join(f"{k + 1}|{sub.text}" for k, sub in enumerate(sub_chunk))

                while True:
                    try:
                        response = chat.send_message(prompt)
                        lines = response.text.strip().splitlines()
                        valid_lines = [line for line in lines if "|" in line]

                        if len(valid_lines) == len(sub_chunk):
                            for k, line in enumerate(valid_lines):
                                parts = line.split("|", 1)
                                if len(parts) == 2:
                                    sub_chunk[k].text = parts[1].strip()
                            translated_subs.extend(sub_chunk)  # 将翻译结果添加到 translated_subs

                            # 提示翻译成功
                            print(f"翻译成功: {len(translated_subs)} 行已翻译.")
                            time.sleep(3)  # 成功后稍微等待
                            break
                        else:
                            print("返回行数与原始字幕不一致，等待重试...")
                            time.sleep(5)

                    except Exception as e:
                        print(f"异常：{e}")
                        time.sleep(5)

            # 保存翻译后的字幕，只写入 translated_subs
            srt_path = os.path.join(config['tsl_path'], filename)
            translated_subs.save(srt_path, encoding='utf-8')
            print(f"字幕已保存到: {srt_path}")


if __name__ == "__main__":
    translate(get_config())
