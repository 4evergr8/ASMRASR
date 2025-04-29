import gc
import os
import torch
import numpy as np
import librosa

from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from dataclasses import dataclass
from faster_whisper import WhisperModel
from getconfig import get_config

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



def transcribe(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print('设备:', device, '类型:', compute_type)

    for root, dirs, files in os.walk(config["work_path"]):
        for filename in files:
            if not filename.endswith(('.wav', '.mp3')):
                continue

            audio_path = os.path.join(root, filename)
            basename = os.path.splitext(filename)[0]
            print(f"处理音频: {audio_path}")

            gc.collect()
            # 加载 VAD 模型
            vad_model = Model.from_pretrained(checkpoint=config["vad"], cache_dir=config["model_path"])
            vad_model.to(torch.device(device))
            vad_pipeline = VoiceActivityDetection(segmentation=vad_model)
            vad_pipeline.instantiate({
                "min_duration_on": config["min_duration_on"],
                "min_duration_off": config["min_duration_off"],
            })

            vad_result = vad_pipeline(str(audio_path))
            del vad_model
            gc.collect()


            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

            silence = np.zeros(int(sr * config["space"]), dtype=audio.dtype)

            groups = []  # AudioSegmentGroup数组
            current_audio = []
            current_segments = []
            current_time_concat = 0.0

            timeline = vad_result.get_timeline()
            group_start_limit = 30 * 60  # 30分钟

            for segment in timeline:
                if segment.start >= group_start_limit:
                    break

                start_sample = int(segment.start * sr)
                end_sample = int(segment.end * sr)

                audio_seg = audio[start_sample:end_sample]
                duration = (end_sample - start_sample) / sr

                # 记录映射关系
                segment_info = AudioSegmentInfo(
                    start_in_concat=current_time_concat,
                    end_in_concat=current_time_concat + duration,
                    start_in_origin=segment.start,
                    end_in_origin=segment.end
                )

                current_segments.append(segment_info)
                current_audio.append(audio_seg)
                current_audio.append(silence)
                current_time_concat += duration + config["space"]

            if current_segments:
                final_audio = np.concatenate(current_audio[:-1])  # 去掉最后一个多余的silence
                groups.append(AudioSegmentGroup(audio=final_audio, segments=current_segments))

            if not groups:
                print(f"{audio_path} 没有有效人声段，跳过")
                continue

            # 初始化 ASR
            asr_model = WhisperModel(
                config["asr"],
                device=device,
                compute_type=compute_type,
                download_root=config["model_path"]
            )

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
                    # 把asr输出的时间戳映射回原音频
                    start_real = map_back_time(seg.start, group.segments)
                    end_real = map_back_time(seg.end, group.segments)

                    srt_segments.append((subtitle_index, start_real, end_real, seg.text.strip()))
                    subtitle_index += 1

            del asr_model
            gc.collect()

            # 写出srt
            srt_path = os.path.join(config["asr_path"], f"{basename}.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                for idx, start, end, text in srt_segments:
                    f.write(f"{idx}\n")
                    f.write(f"{timestamp_to_srt(start)} --> {timestamp_to_srt(end)}\n")
                    f.write(f"{text}\n\n")
            print(f"字幕写入: {srt_path}")

def map_back_time(time_in_concat, segments):
    for seg in segments:
        if seg.start_in_concat <= time_in_concat <= seg.end_in_concat:
            relative = time_in_concat - seg.start_in_concat
            return seg.start_in_origin + relative
    # 没找到直接返回最后一个
    return segments[-1].end_in_origin

if __name__ == "__main__":
    transcribe(get_config())
