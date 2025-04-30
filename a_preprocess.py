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
    for root, dirs, files in os.walk(config["pre_path"]):
        for filename in files:
            if filename.endswith((".wav", ".mp3", ".flac")):
                audio_path = os.path.join(root, filename)



                slice_dir = os.path.join(config["pre_path"], "slice")
                os.makedirs(slice_dir, exist_ok=True)
                segment_length = 120  # 20 分钟 = 1200 秒
                command = [
                    "ffmpeg", "-i", audio_path,  # 输入音频文件
                    "-f", "segment",  # 使用 segment 格式进行切割
                    "-segment_time", str(segment_length),  # 设置每段的时长（单位：秒）
                    "-c", "copy",  # 保持原始编码（无损切割）
                    os.path.join(slice_dir, "%03d.wav")  # 输出文件的命名格式
                ]
                subprocess.run(command)



                separator = Separator(
                    output_dir=config["work_path"],
                    output_single_stem="vocals",
                    sample_rate=16000
                )
                separator.load_model(model_filename=config["separator"])
                output_files = separator.separate(audio_path)
                print(f"<UNK>{len(output_files)}")





if __name__ == "__main__":
    preprocess(get_config())
