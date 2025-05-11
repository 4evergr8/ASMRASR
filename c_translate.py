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

        original_subs = pysrt.open(src_path, encoding='utf-8')

        # 创建新的 SubRipFile 并重新编号
        original_subs = pysrt.SubRipFile(items=[sub for sub in original_subs if "默认占位" not in sub.text])

        # 合并相邻且文本相同的字幕
        merged_subs = pysrt.SubRipFile()
        i = 0
        while i < len(original_subs):
            current = original_subs[i]
            while i + 1 < len(original_subs) and original_subs[i + 1].text == current.text:
                current.end = original_subs[i + 1].end  # 延长结束时间
                i += 1
            merged_subs.append(current)
            i += 1

        original_subs = merged_subs
        original_subs.clean_indexes()

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
        for i in range(0, len(original_subs), 200):
            chunk = original_subs[i:i + 200]

            client = genai.Client(api_key=config["api_key"])
            chat = client.chats.create(
                model=config['translate'],
                config=types.GenerateContentConfig(
                    system_instruction=config["prompt"] + "\n" + "\n".join(
                        sub.text for sub in chunk),
                )
            )

            for j in range(0, len(chunk), 10):
                sub_chunk = chunk[j:j + 10]
                prompt = "要翻译的部分\n"+"\n".join(f"{k + 1}|{sub.text}" for k, sub in enumerate(sub_chunk))

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
                            srt_path = os.path.join(config['tsl_path'], filename)
                            translated_subs.save(srt_path, encoding='utf-8')
                            print(f"字幕已保存到: {srt_path}")
                            time.sleep(3)  # 成功后稍微等待
                            break
                        else:
                            print("返回行数与原始字幕不一致，等待重试...")
                            print(f"发送：{prompt}")
                            print(f"接收：{response}")
                            time.sleep(5)

                    except Exception as e:
                        print(f"异常：{e}")
                        print(f"发送：{prompt}")
                        print(f"接收：{response}")
                        time.sleep(5)

            # 保存翻译后的字幕，只写入 translated_subs
            srt_path = os.path.join(config['tsl_path'], filename)
            translated_subs.save(srt_path, encoding='utf-8')
            print(f"字幕已保存到: {srt_path}")


if __name__ == "__main__":
    translate(get_config())
