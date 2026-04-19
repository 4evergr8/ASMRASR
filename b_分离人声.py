import os
from audio_separator.separator import Separator
from settings import config


if not os.path.exists(config["path"]["pre"]) or not os.listdir(config["path"]["pre"]):
    print(f"错误：目录不存在或为空")
    exit(0)
separator = Separator(
    model_file_dir=config["path"]["model"],
    output_dir=config["path"]["audio"],  # 分离后的音频存放在 1audio
    output_single_stem="vocals",  # 仅提取人声
    sample_rate=config["sep"]["sample_rate"],  # 44100
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
