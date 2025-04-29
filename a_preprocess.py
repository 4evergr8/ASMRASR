import os
import subprocess
from getconfig import get_config
from imageio_ffmpeg import get_ffmpeg_exe
from audio_separator.separator import Separator


def preprocess(config):
    ffmpeg = get_ffmpeg_exe()
    os.system("chcp 65001")

    for root, dirs, files in os.walk(config["pre_path"]):
        for filename in files:
            if filename.endswith((".mp4", ".mkv", ".avi", ".mov", ".flv")):
                video_path = os.path.join(root, filename)

                basename = os.path.splitext(filename)[0]
                audio_output_path = os.path.join(config["pre_path"], f"{basename}.wav")

                command = [
                    ffmpeg, "-i", video_path,  # 输入视频文件
                    "-vn",  # 不处理视频流，只提取音频
                    "-acodec", "pcm_s16le",  # 设置音频编码器为 mp3
                    "-ar", "16000",  # 设置音频采样率
                    "-ac", "1",  # 设置音频为立体声
                    audio_output_path  # 输出路径
                ]

                subprocess.run(command)

                print(f"已提取音频并保存至：{audio_output_path}")



        separator = Separator(
            output_dir=config["work_path"],
            model_file_dir=config["model_path"],
            output_single_stem="vocals"
        )
        separator.load_model(model_filename=config["separator"])
        output_files = separator.separate(config["pre_path"])
        print(f"<UNK>{len(output_files)}")



        for filename in files:
            if filename.endswith((".wav", ".mp3", ".flac" )):
                audio_path = os.path.join(root, filename)

if __name__ == "__main__":
    preprocess(get_config())
