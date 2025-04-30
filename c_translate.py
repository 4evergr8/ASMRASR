import os
import pysrt
from getconfig import get_config



def translate(config):
    for root, dirs, files in os.walk(config["asr_path"]):
        for filename in files:
            if filename.endswith((".srt")):
                srt_path = os.path.join(root, filename)
                subs = pysrt.open(srt_path, encoding='utf-8')
                for i in range(0, len(subs), 10):
                    chunk = subs[i:i + 10]
                    for sub in chunk:
                        # 在这里处理每个 subtitle 对象
                        print(sub.text)


if __name__ == "__main__":
    translate(get_config())
