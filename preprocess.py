import os
import subprocess

from getconfig import get_config


def preprocess(config):
    os.system("chcp 65001")

    os.environ['TRANSFORMERS_CACHE'] = config["model_path"]
    os.environ['HTTP_PROXY'] = config["proxy"]
    os.environ['HTTPS_PROXY'] = config["proxy"]

    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(config["pre_path"]):
        for filename in files:
            if filename.endswith((".mp4", ".mkv", ".avi", ".mov", ".flv")):
                video_path = os.path.join(root, filename)

                # 定义输出音频文件的路径
                basename = os.path.splitext(filename)[0]
                audio_output_path = os.path.join(root, f"{basename}.wav")

                # 使用 ffmpeg 提取音频
                command = [
                    "ffmpeg", "-i", video_path,  # 输入视频文件
                    "-vn",  # 不处理视频流，只提取音频
                    "-acodec", "mp3",  # 设置音频编码器为 mp3
                    "-ar", "44100",  # 设置音频采样率
                    "-ac", "2",  # 设置音频为立体声
                    audio_output_path  # 输出路径
                ]

                subprocess.run(command)

                print(f"已提取音频并保存至：{audio_output_path}")





    for root, dirs, files in os.walk(config["pre_path"]):
        for filename in files:
            if filename.endswith((".wav", ".mp3", ".flac")):
                input_audio = os.path.join(root, filename)


                command = [
                    "demucs", "--two-stems", "vocals",  # 只分离人声，或用 --out=output_dir 直接分离全部轨道
                    "-o", config["work_path"],
                    input_audio
                ]

                subprocess.run(command)

                print(f"提取人声保存至：{input_audio}")


if __name__ == "__main__":
    preprocess(get_config())
