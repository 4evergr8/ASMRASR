import os
import glob
from pathlib import Path
import torch
import pysrt
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from settings import config

extensions = ("wav", "mp3", "flac")
pattern = os.path.join(config["path"]["audio"], "*.*")
audio_files = [
    f for f in glob.glob(pattern)
    if f.lower().endswith(extensions)
]

if not audio_files:
    print(f"在 {config['path']['audio']} 中没有找到可处理的音频文件")
    exit(0)

print('设备:', config["device"])

# 2. 初始化 VAD 模型
vad = Model.from_pretrained(
    checkpoint=config["model"]["vad"],
    cache_dir=config["path"]["model"]
)
vad.to(torch.device(config["device"]))

vad_pipeline = VoiceActivityDetection(segmentation=vad)
# 注意：修正了之前代码中的变量名错误 config.vad_min.duration_off -> config["vad"]["min_duration_off"]
vad_pipeline.instantiate({
    "min_duration_on": config["vad"]["min_duration_on"],
    "min_duration_off": config["vad"]["min_duration_off"],
})

# 3. 开始循环处理
for audio_path in audio_files:
    print(f"\n处理音频: {audio_path}")

    file_obj = Path(audio_path)
    basename = file_obj.stem
    vad_log_path = os.path.join(config["path"]["vad"], f"{basename}.srt")



    # 4. 执行 VAD 识别
    vad_result = vad_pipeline(audio_path)

    # 5. 生成 SRT
    srt = pysrt.SubRipFile()
    for idx, (segment, _, _) in enumerate(vad_result.itertracks(yield_label=True), start=1):
        sub_item = pysrt.SubRipItem(
            index=idx,
            start=pysrt.SubRipTime.from_ordinal(int(segment.start * 1000)),
            end=pysrt.SubRipTime.from_ordinal(int(segment.end * 1000)),
            text=f"Speech_{idx}"
        )
        srt.append(sub_item)

    srt.save(vad_log_path, encoding='utf-8')
    print(f"VAD记录写入完成: {vad_log_path}")

print("\n所有音频 VAD 处理完毕！")