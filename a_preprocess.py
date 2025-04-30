import os
import shutil
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

                if os.path.exists(os.path.join(config["pre_path"], "slice")):
                    shutil.rmtree(os.path.join(config["pre_path"], "slice"))  # 删除原文件夹及其内容
                os.makedirs(os.path.join(config["pre_path"], "slice"))  # 创建新文件夹

                segment_length = 120  # 20 分钟 = 1200 秒
                command = [
                    "ffmpeg", "-i", audio_path,  # 输入音频文件
                    "-f", "segment",  # 使用 segment 格式进行切割
                    "-segment_time", str(segment_length),  # 设置每段的时长（单位：秒）
                    "-c", "copy",  # 保持原始编码（无损切割）
                    os.path.join(os.path.join(config["pre_path"], "slice"), "%03d.wav")  # 输出文件的命名格式
                ]
                subprocess.run(command)




                if os.path.exists(os.path.join(config["pre_path"], "split")):
                    shutil.rmtree(os.path.join(config["pre_path"], "split"))  # 删除原文件夹及其内容
                os.makedirs(os.path.join(config["pre_path"], "split"))  # 创建新文件夹
                separator = Separator(
                    model_file_dir=config["model_path"],
                    output_dir=os.path.join(config["pre_path"],'split'),
                    output_single_stem="vocals",
                    sample_rate=16000,
                    mdxc_params={"segment_size": 256, "override_model_segment_size": False, "batch_size": config["batch_size"],
                                 "overlap": 8, "pitch_shift": 0}
                )
                separator.load_model(model_filename=config["separator"])
                output_files = separator.separate(os.path.join(config["pre_path"], "slice"))
                print(f"<UNK>{len(output_files)}")




                file_list = sorted(
                    [f for f in os.listdir(os.path.join(config["pre_path"],'split')) if f.endswith(".wav")],
                    key=lambda x: int(x[:3])
                )
                concat_input = "|".join([os.path.join(os.path.join(config["pre_path"],'split'), f) for f in file_list])

                basename = os.path.splitext(filename)[0]
                output_path = os.path.join(config["work_path"], f"{basename}.wav")

                command = [
                    "ffmpeg",
                    "-i", f"concat:{concat_input}",
                    "-c", "copy",
                    output_path
                ]

                subprocess.run(command)
                print(f"合并完成：{output_path}")


if __name__ == "__main__":
    preprocess(get_config())
