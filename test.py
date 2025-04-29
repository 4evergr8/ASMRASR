import os
from imageio_ffmpeg import get_ffmpeg_exe
from getconfig import get_config
config=get_config()
# 设置自定义的 ffmpeg 路径
get_config()
os.environ["IMAGEIO_FFMPEG_EXE"] = config["model_path"]

# 获取并打印 `ffmpeg` 路径
ffmpeg_path = get_ffmpeg_exe()
print("自定义 FFmpeg 路径：", ffmpeg_path)