import sys
import ffmpeg_downloader
import os
from audio_separator.separator import Separator
import subprocess
from settings import config

# subprocess.run(
#     [sys.executable, "-m", "ffmpeg_downloader", "install", "8.0@full-shared"],
#     input="y\n",
#     text=True
# )
ffmpeg_downloader.add_path()
if not os.path.exists(config["path"]["pre"]) or not os.listdir(config["path"]["pre"]):
    print(f"错误：目录不存在或为空")
    exit(0)
separator = Separator(
    model_file_dir=config["path"]["model"],
    output_dir=config["path"]["audio"],  # 分离后的音频存放在 1audio
    output_single_stem="vocals",  # 仅提取人声
    sample_rate=16000,  # 识别专用采样率，极大减小文件体积
    output_format="MP3",  # 采样 MP3 格式进一步优化空间
    use_autocast=True,
    chunk_duration=config["sep"]["chunk_duration"],
    mdxc_params={
        "segment_size": 256,
        "override_model_segment_size": True,
        "batch_size": 8,
        "overlap": 8,
        "pitch_shift": 0
    },
    demucs_params={
        "segment_size": "Default",
        "shifts": 2,
        "overlap": 0.25,
        "segments_enabled": True
    },

)
print(f"正在加载模型: {config['model']['sep']}")
separator.load_model(model_filename=config["model"]["sep"])
print(f"开始批量处理目录 ...")
separator.separate(config["path"]["pre"])
print("\n所有文件人声分离处理完成！")
