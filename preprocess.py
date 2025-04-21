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
                audio_output_path = os.path.join(root, f"{basename}.wav")

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

    # 假设 config 是已经定义好的字典，包含路径等配置信息
    for root, dirs, files in os.walk(config["pre_path"]):
        for filename in files:
            if filename.endswith((".wav", ".mp3", ".flac")):
                input_audio = os.path.join(root, filename)

                audio_basename = os.path.splitext(os.path.basename(input_audio))[0]

                # Step 1: 运行 demucs 分离人声
                command = [
                    "demucs","-n", "mdx_extra_q", "--two-stems", "vocals",  # 只分离人声
                    "-o", config["work_path"],  # 输出到 work_path 文件
                    "--segment", "60",

                    input_audio
                ]
                subprocess.run(command, check=True)

                # Step 2: 获取 demucs 输出目录路径 (这里 demucs 会创建一个 htdemucs 文件夹)
                subfolder = os.path.join(config["work_path"], "htdemucs", audio_basename)
                vocals_path = os.path.join(subfolder, "vocals.wav")

                # Step 3: 确保将提取的 vocals.wav 文件移动到目标文件夹
                final_output_path = os.path.join(config["work_path"], f"{audio_basename}_vocals.wav")

                if os.path.exists(vocals_path):
                    shutil.move(vocals_path, final_output_path)  # 将文件移动到目标文件夹
                    print(f"已提取人声：{final_output_path}")

                # 删除 demucs 创建的临时文件夹
                htdemucs_folder = os.path.join(config["work_path"], "htdemucs")
                shutil.rmtree(htdemucs_folder, ignore_errors=True)  # 删除 htdemucs 文件夹
                print(f"已删除临时文件夹：{htdemucs_folder}")


if __name__ == "__main__":
    preprocess(get_config())
