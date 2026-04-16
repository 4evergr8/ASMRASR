import sys
import ffmpeg_downloader
import os
import glob
from pathlib import Path
import subprocess
from settings import config

# subprocess.run(
#     [sys.executable, "-m", "ffmpeg_downloader", "install", "8.0@full-shared"],
#     input="y\n",
#     text=True
# )
ffmpeg_downloader.add_path()
extensions = ("mp4", "mkv")
pattern = os.path.join(config["path"]["pre"], "*.*")

files = [f for f in glob.glob(pattern) if f.lower().endswith(extensions)]

for file_path in files:
    file = Path(file_path)
    output = file.with_suffix(".flac")

    print(f"提取中: {file.name} -> {output.name}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(file),
        "-vn",
        "-acodec", "flac",
        "-ac", "2",
        str(output)
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("全部处理完成")
