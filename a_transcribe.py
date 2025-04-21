import gc

import torch
import numpy as np
from pyannote.audio import Model
import os
from faster_whisper import WhisperModel
import librosa
from pyannote.audio.pipelines import VoiceActivityDetection
from dataclasses import dataclass
from getconfig import get_config


def format_lrc_time(time_str):
    time_str = time_str.replace(',', '.')  # 先替换逗号为点
    h, m, s = time_str.strip().split(':')
    s, ms = s.split('.') if '.' in s else (s, '00')
    total_min = int(h) * 60 + int(m)
    return f"[{total_min:02}:{int(s):02}.{ms[:2]:0<2}]"


def timestamp_2_srt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def srt_2_timestamp(srt_time: str) -> float:
    h, m, s_ms = srt_time.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def calculate_overlap(s1, e1, s2, e2):
    latest_start = max(s1, s2)
    earliest_end = min(e1, e2)
    return max(0.0, earliest_end - latest_start)


@dataclass
class SubtitleSegment:
    index: int
    start: str  # SRT格式时间戳
    end: str
    text: str


def transcribe(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print('设备：', device, '类型：', compute_type)

    for root, dirs, files in os.walk(config["work_path"]):
        for filename in files:
            if filename.endswith((".wav", ".mp3")):
                audio_path = os.path.join(root, filename)  # 获取音频文件的完整路径
                print(audio_path)
                basename = os.path.splitext(filename)[0]
                gc.collect()
                vad_model = Model.from_pretrained(checkpoint=config["vad"], cache_dir=config["model_path"])
                vad_model.to(torch.device(device))
                vad_pipeline = VoiceActivityDetection(segmentation=vad_model)
                vad_pipeline.instantiate({

                    "min_duration_on": config["min_duration_on"],
                    "min_duration_off": config["min_duration_off"]})
                vad_result = vad_pipeline(str(audio_path))
                del vad_model
                gc.collect()

                audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
                audios = []
                timeline = vad_result.get_timeline()

                silence = np.zeros(int(sr * config["space"]), dtype=audio.dtype)  # 静音段

                for i, segment in enumerate(timeline):
                    start_sample = int(segment.start * sr)
                    end_sample = int(segment.end * sr)
                    audio_segment = audio[start_sample:end_sample]

                    audios.append(audio_segment)
                    audios.append(silence)

                audio = np.concatenate(audios)
                audio = audio.astype(np.float32)
                print(len(audio))

                pynotebefore = []
                pynoteafter = []

                # Step 1：从 VAD 结果中读取原始时间段
                for idx, segment in enumerate(vad_result.get_timeline(), 1):
                    start = timestamp_2_srt(segment.start)
                    end = timestamp_2_srt(segment.end)
                    text = "..."
                    pynotebefore.append(SubtitleSegment(index=idx, start=start, end=end, text=text))

                # Step 2：逐段计算“拼接后”的新时间
                accumulated_time = 0.0  # 总累计时间
                for idx, segment in enumerate(pynotebefore, 1):
                    start_seconds = srt_2_timestamp(segment.start)
                    end_seconds = srt_2_timestamp(segment.end)
                    duration = end_seconds - start_seconds

                    new_start = accumulated_time
                    new_end = new_start + duration
                    accumulated_time = new_end + config["space"]  # 下一个段的起点：当前段结束 + 5秒间隔

                    new_segment = SubtitleSegment(
                        index=idx,
                        start=timestamp_2_srt(new_start),
                        end=timestamp_2_srt(new_end),
                        text=segment.text
                    )
                    pynoteafter.append(new_segment)

                before_log = os.path.join(config["log_path"], f"before-{basename}.txt")

                with open(before_log, "w", encoding="utf-8") as f:
                    for subtitle in pynotebefore:
                        f.write(f"{subtitle.start}\n")
                        f.write(f"{subtitle.end}\n")
                        f.write("\n")  # 元素之间空一行
                print(before_log)
                after_log = os.path.join(config["log_path"], f"after-{basename}.txt")
                with open(after_log, "w", encoding="utf-8") as f:
                    for subtitle in pynoteafter:
                        f.write(f"{subtitle.start}\n")
                        f.write(f"{subtitle.end}\n")
                        f.write("\n")  # 元素之间空一行
                print(after_log)

                gc.collect()
                asr_model = WhisperModel(
                    config["asr"], device=device,
                    compute_type=compute_type,
                    download_root=config["model_path"]
                )

                segments, _ = asr_model.transcribe(
                    audio=audio,
                    log_progress=True,
                    beam_size=2,
                    vad_filter=False,
                    initial_prompt=basename,
                    language=config['language'],
                    word_timestamps= True
                )

                del asr_model
                gc.collect()

                subtitle_segments = []
                for idx, segment in enumerate(segments, 1):
                    subtitle_segments.append(SubtitleSegment(
                        index=idx,
                        start=timestamp_2_srt(segment.start),  # 转换为 SRT 时间格式
                        end=timestamp_2_srt(segment.end),  # 转换为 SRT 时间格式
                        text=segment.text
                    ))

                asr_log = os.path.join(config["log_path"], f"asr-{basename}.txt")
                with open(asr_log, "w", encoding="utf-8") as f:
                    for subtitle in subtitle_segments:
                        f.write(f"{subtitle.start}\n")
                        f.write(f"{subtitle.end}\n")
                        f.write(f"{subtitle.text}\n")  # 元素之间空一行
                print(asr_log)
                for pynote_seg in pynoteafter:
                    start1 = srt_2_timestamp(pynote_seg.start)
                    end1 = srt_2_timestamp(pynote_seg.end)

                    matched_text = ""
                    for subtitle in subtitle_segments:
                        start2 = srt_2_timestamp(subtitle.start)
                        end2 = srt_2_timestamp(subtitle.end)
                        overlap = calculate_overlap(start1, end1, start2, end2)

                        if overlap > 0:
                            matched_text = matched_text + subtitle.text
                            print('匹配:', start1, end1, start2, end2, overlap)

                    if matched_text == "...":
                        print('不匹配:', start1, end1)
                    pynote_seg.text = matched_text

                # print(subtitle_segments)

                for i in range(min(len(pynotebefore), len(pynoteafter))):
                    pynotebefore[i].text = pynoteafter[i].text


                if 'srt' in config["output"]:
                    srt_path = os.path.join(config["asr_path"], f"{basename}.srt")
                    with open(srt_path, 'w', encoding='utf-8') as f:
                        # 插入开头字幕
                        f.write("0\n")
                        f.write(f"00:00:00,000 --> {pynotebefore[0].start}\n")
                        f.write(f"{basename}\n\n")

                        for idx, item in enumerate(pynotebefore, start=1):
                            f.write(f"{idx}\n")
                            f.write(f"{item.start} --> {item.end}\n")
                            f.write(f"{item.text}\n\n")

                if 'lrc' in config["output"]:
                    lrc_path = os.path.join(config["asr_path"], f"{basename}.lrc")
                    with open(lrc_path, 'w', encoding='utf-8') as f:
                        # 插入开头字幕
                        start = format_lrc_time("00:00:00.00")
                        end = format_lrc_time(pynotebefore[0].start)
                        f.write(f"{start}{basename}\n")
                        f.write(f"{end}\n")

                        for item in pynotebefore:
                            start = format_lrc_time(item.start)
                            end = format_lrc_time(item.end)
                            f.write(f"{start}{item.text}\n")
                            f.write(f"{end}\n")

                if 'vtt' in config["output"]:
                    vtt_path = os.path.join(config["asr_path"], f"{basename}.vtt")
                    with open(vtt_path, 'w', encoding='utf-8') as f:
                        # 插入头部
                        f.write("WEBVTT\n\n")

                        # 插入开头字幕
                        f.write(f"00:00:00.000 --> {pynotebefore[0].start.replace(',', '.')}\n")
                        f.write(f"{basename}\n\n")

                        for item in pynotebefore:
                            start = item.start.replace(',', '.')
                            end = item.end.replace(',', '.')
                            f.write(f"{start} --> {end}\n")
                            f.write(f"{item.text}\n\n")



if __name__ == "__main__":
    transcribe(get_config())
