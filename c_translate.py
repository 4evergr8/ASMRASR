import os
import shutil
import subprocess
from getconfig import get_config



def translate(config):
    for root, dirs, files in os.walk(config["asr_path"]):
        for filename in files:
            if filename.endswith((".srt")):
                srt_path = os.path.join(root, filename)
                subs = pysrt.open(srt_path, encoding='utf-8')


if __name__ == "__main__":
    translate(get_config())
