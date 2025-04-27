import gc
import torch
import numpy as np
from pyannote.audio import Model
import os
from faster_whisper import WhisperModel
import librosa
from pyannote.audio.pipelines import VoiceActivityDetection
from dataclasses import dataclass

# 配置
path = '/content/gdrive/MyDrive/ASR'
config = {
    "work_path": "1work",
    "asr_path": "1work",
    "log_path": "log",
    "model_path": "model",

    "prompt": "",
    "language": "ja",
    "space": 3,
    "min_duration_on": 0.0,
    "min_duration_off": 0.2,

    "asr": "large-v2",
    "vad": "4evergr8/pyannote-segmentation-3.0",

    "output": ["lrc", "srt", "vtt"]
}

config["work_path"] = os.path.join(path, config["work_path"])
config["asr_path"] = os.path.join(path, config["asr_path"])

# 数据结构
@dataclass
class AudioSegmentInfo:
    start_in_concat: float
    end_in_concat: float
    start_in_origin: float
    end_in_origin: float

@dataclass
class AudioSegmentGroup:
    audio: np.ndarray
    segments: list
    offset: float  # 每组的起始时间（单位：秒）

# 工具函数
def timestamp_to_srt(ts: float) -> str:
    hours = int(ts // 3600)
    minutes = int((ts % 3600) // 60)
    seconds = int(ts % 60)
    millis = int((ts - int(ts)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

def map_back_time(time_in_concat, segments):
    for seg in segments:
        if seg.start_in_concat <= time_in_concat <= seg.end_in_concat:
            relative = time_in_concat - seg.start_in_concat
            return seg.start_in_origin + relative
    return segments[-1].end_in_origin

# 硬件
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
print('设备:', device, '类型:', compute_type)

# 初始化ASR模型
asr_model = WhisperModel(
    config["asr"],
    device=device,
    compute_type=compute_type,
    download_root=config["model_path"]
)

# 遍历所有音频
for root, dirs, files in os.walk(config["work_path"]):
    for filename in files:
        if not filename.endswith(('.wav', '.mp3')):
            continue

        audio_path = os.path.join(root, filename)
        basename = os.path.splitext(filename)[0]
        print(f"\n处理音频: {audio_path}")

        # Step 1: 加载音频
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)


        gc.collect()
        torch.cuda.empty_cache()

        vad_model = Model.from_pretrained(checkpoint=config["vad"], cache_dir=config["model_path"])
        vad_model.to(torch.device(device))
        vad_pipeline = VoiceActivityDetection(segmentation=vad_model)
        vad_pipeline.instantiate({
            "min_duration_on": config["min_duration_on"],
            "min_duration_off": config["min_duration_off"],
        })

        vad_result = vad_pipeline(str(audio_path))

        # VAD结束，释放模型
        del vad_pipeline, vad_model
        gc.collect()
        torch.cuda.empty_cache()

        # Step 3: 切人声片段，并按30分钟分组
        timeline = vad_result.get_timeline()
        silence = np.zeros(int(sr * config["space"]), dtype=audio.dtype)

        groups = []
        current_audio = []
        current_segments = []
        current_time_concat = 0.0
        current_group_start = 0.0  # 当前组的起点时间（秒）
        group_time_limit = 30 * 60  # 每组30分钟

        for segment in timeline:
            if segment.start >= current_group_start + 1800:
                # 保存当前组
                if current_segments:
                    final_audio = np.concatenate(current_audio[:-1])  # 去掉最后一个silence
                    groups.append(AudioSegmentGroup(audio=final_audio, segments=current_segments, offset=current_group_start))

                # 开新组
                current_audio = []
                current_segments = []
                current_time_concat = 0.0
                current_group_start += group_time_limit

            # 将当前segment加到当前组
            start_sample = int(segment.start * sr)
            end_sample = int(segment.end * sr)

            audio_seg = audio[start_sample:end_sample]
            duration = (end_sample - start_sample) / sr

            segment_info = AudioSegmentInfo(
                start_in_concat=current_time_concat,
                end_in_concat=current_time_concat + duration,
                start_in_origin=segment.start - current_group_start,
                end_in_origin=segment.end - current_group_start
            )

            current_segments.append(segment_info)
            current_audio.append(audio_seg)
            current_audio.append(silence)
            current_time_concat += duration + config["space"]

        # 最后一组也要加进去
        if current_segments:
            final_audio = np.concatenate(current_audio[:-1])
            groups.append(AudioSegmentGroup(audio=final_audio, segments=current_segments, offset=current_group_start))

        if not groups:
            print(f"{audio_path} 没有有效人声段，跳过")
            continue

        # Step 4: ASR转写
        srt_segments = []
        subtitle_index = 1

        for group in groups:
            segments, _ = asr_model.transcribe(
                audio=group.audio,
                beam_size=2,
                vad_filter=False,
                initial_prompt=basename,
                language=config['language'],
                word_timestamps=False
            )

            for seg in segments:
                start_real = map_back_time(seg.start, group.segments) + group.offset
                end_real = map_back_time(seg.end, group.segments) + group.offset

                srt_segments.append((subtitle_index, start_real, end_real, seg.text.strip()))
                subtitle_index += 1

        # Step 5: 写出srt
        os.makedirs(config["asr_path"], exist_ok=True)
        srt_path = os.path.join(config["asr_path"], f"{basename}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            for idx, start, end, text in srt_segments:
                f.write(f"{idx}\n")
                f.write(f"{timestamp_to_srt(start)} --> {timestamp_to_srt(end)}\n")
                f.write(f"{text}\n\n")
        print(f"字幕写入: {srt_path}")

        # 处理完一个音频彻底回收
        del audio, groups, timeline, srt_segments
        gc.collect()
        torch.cuda.empty_cache()

# 处理结束后，释放ASR模型
del asr_model
gc.collect()
torch.cuda.empty_cache()
