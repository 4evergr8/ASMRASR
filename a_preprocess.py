import os
import shutil
import subprocess

import torch

from getconfig import get_config
from imageio_ffmpeg import get_ffmpeg_exe
from audio_separator.separator import Separator


def preprocess(config):
    ffmpeg = get_ffmpeg_exe()
    os.system("chcp 65001")


        # 处理 audio_path

    for filename in os.listdir(config["pre_path"]):
        if filename.endswith((".mp4", ".mkv", ".avi", ".mov", ".flv")):
            video_path = os.path.join(config["pre_path"], filename)
            if os.path.isfile(video_path):

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

                subprocess.run(command, check=True)



                print(f"已提取音频并保存至：{audio_output_path}")


    if not torch.cuda.is_available():
        exit(0)




    for filename in os.listdir(config["pre_path"]):
        if filename.endswith((".wav", ".mp3", ".flac")):
            basename = os.path.splitext(filename)[0]
            slice_path = os.path.join(config["pre_path"], f"{basename}-slice")
            split_path = os.path.join(config["pre_path"], f"{basename}-split")
            audio_path = os.path.join(config["pre_path"], filename)
            if os.path.isfile(audio_path):

                if os.path.exists(slice_path):
                    shutil.rmtree(slice_path)  # 删除原文件夹及其内容
                os.makedirs(slice_path)  # 创建新文件夹

                segment_length = 1200  # 20 分钟 = 1200 秒
                command = [
                    ffmpeg, "-i", audio_path,  # 输入音频文件
                    "-f", "segment",  # 使用 segment 格式进行切割
                    "-segment_time", str(segment_length),  # 设置每段的时长（单位：秒）
                    "-c", "copy",  # 保持原始编码（无损切割）
                    os.path.join(slice_path, "%03d.wav")  # 输出文件的命名格式
                ]
                subprocess.run(command, check=True)







                separator = Separator(
                    model_file_dir=config["model_path"],
                    output_dir=split_path,
                    output_single_stem="vocals",
                    sample_rate=16000,
                )
                separator.load_model(model_filename=config["separator"])
                for filename in os.listdir(slice_path):
                    if filename.endswith(".wav"):
                        slice_basename = os.path.splitext(filename)[0]  # 比如 '001'
                        exists = any(name.startswith(slice_basename) for name in os.listdir(split_path))
                        if not exists:
                            output_files = separator.separate(os.path.join(slice_path, filename))
                            print(f"<UNK>{len(output_files)}")
                        else:
                            print(f"已存在分离结果，跳过：{filename}")




                file_list = sorted(
                    [f for f in os.listdir(split_path) if f.endswith(".wav")],
                    key=lambda x: int(x[:3])
                )
                with open(os.path.join(config["pre_path"], f"{basename}_list.txt"), "w", encoding="utf-8") as f:
                    for f_name in file_list:
                        full_path = os.path.join(split_path, f_name)
                        f.write(f"file '{full_path}'\n")

                output_path = os.path.join(config["work_path"], f"{basename}.wav")
                command = [
                    ffmpeg,
                    "-f", "concat",
                    "-safe", "0",
                    "-i", os.path.join(config["pre_path"], f"{basename}_list.txt"),
                    "-c", "copy",
                    output_path
                ]
                subprocess.run(command, check=True)








if __name__ == "__main__":
    preprocess(get_config())
