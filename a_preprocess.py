import os
import shutil
import subprocess
from getconfig import get_config


def preprocess(config):
    os.system("chcp 65001")
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(config["pre_path"]):
        for filename in files:
            if filename.endswith((".mp4", ".mkv", ".avi", ".mov", ".flv")):
                video_path = os.path.join(root, filename)

                basename = os.path.splitext(filename)[0]
                audio_output_path = os.path.join(config["cut_path"], f"{basename}.wav")

                command = [
                    "ffmpeg", "-i", video_path,  # 输入视频文件
                    "-vn",  # 不处理视频流，只提取音频
                    "-acodec", "mp3",  # 设置音频编码器为 mp3
                    "-ar", "16000",  # 设置音频采样率
                    "-ac", "1",  # 设置音频为立体声
                    audio_output_path  # 输出路径
                ]

                subprocess.run(command)

                print(f"已提取音频并保存至：{audio_output_path}")



if __name__ == "__main__":
    preprocess(get_config())
