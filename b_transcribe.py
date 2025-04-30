import gc
import pysrt
import torch
import numpy as np
from pyannote.audio import Model
from faster_whisper import WhisperModel
import librosa
from pyannote.audio.pipelines import VoiceActivityDetection
from dataclasses import dataclass
import os
from getconfig import get_config


# 数据结构
@dataclass
class AudioSegmentInfo:
    start: float
    end: float
    group_start: float
    group_end: float
    text: str = "..."

@dataclass
class AudioData:
    audio_array: np.ndarray
    segment_info_list: list


def transcribe(config):

    # 硬件
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print('设备:', device, '类型:', compute_type)

    # 只初始化一次ASR模型（不会在每个音频内循环初始化）


    # 遍历所有音频
    for root, dirs, files in os.walk(config["work_path"]):
        for filename in files:
            audio_path = os.path.join(root, filename)
            basename = os.path.splitext(filename)[0]
            print(f"\n处理音频: {audio_path}")

            # Step 1: 加载音频
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)


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


            del vad_pipeline, vad_model
            gc.collect()


            timeline = vad_result.get_timeline()

            audio_groups = []
            group_start_limit = 30 * 60  # 每组音频的时间限制，30分钟
            silence_duration = config["space"]
            silence = np.zeros(int(sr * silence_duration), dtype=audio.dtype)

            for segment in timeline:
                segment_start = segment.start
                segment_end = segment.end

                # 所属分组
                group_index = int(segment_end // group_start_limit)

                # 创建新分组（如果尚不存在）
                while len(audio_groups) <= group_index:
                    audio_groups.append(AudioData(audio_array=np.array([]), segment_info_list=[]))

                # 添加 segment_info 到对应组
                audio_groups[group_index].segment_info_list.append(
                    AudioSegmentInfo(start=segment_start, end=segment_end, group_start=0.0, group_end=0.0)
                )

            # 对每组音频进行拼接处理
            for audio_group in audio_groups:
                group_audio = []
                current_group_end = 0.0

                for i, segment in enumerate(audio_group.segment_info_list):
                    segment_start = segment.start
                    segment_end = segment.end

                    # 计算拼接后的位置
                    group_start = current_group_end
                    group_end = group_start + (segment_end - segment_start)

                    # 更新 group_start 和 group_end
                    segment.group_start = group_start
                    segment.group_end = group_end

                    # 提取音频段
                    start_sample = int(segment_start * sr)
                    end_sample = int(segment_end * sr)
                    audio_seg = audio[start_sample:end_sample]

                    # 加入静音（除首段）
                    if i > 0:
                        group_audio.append(silence)
                    group_audio.append(audio_seg)

                    # 更新下一段的起点
                    current_group_end = group_end + (
                        silence_duration if i < len(audio_group.segment_info_list) - 1 else 0)

                if group_audio:
                    audio_group.audio_array = np.concatenate(group_audio)
                else:
                    audio_group.audio_array = np.array([])

            del audio
            print(len(audio_groups))
            subs = pysrt.SubRipFile()
            for audio_group in audio_groups:
                for segment in audio_group.segment_info_list:
                    sub = pysrt.SubRipItem(
                        index=len(subs) + 1,  # 字幕索引
                        start=pysrt.SubRipTime.from_ordinal(int(segment.group_start * 1000)),  # 转换 start 为 SRT 时间格式
                        end=pysrt.SubRipTime.from_ordinal(int(segment.group_end * 1000)),  # 转换 end 为 SRT 时间格式
                        text=segment.text  # 字幕内容
                    )
                    subs.append(sub)


            srt_path = os.path.join(config["log_path"], f"before-{basename}.srt")
            subs.save(srt_path)
            print(f"log写入: {srt_path}")




            gc.collect()
            asr_model = WhisperModel(
                config["asr"],
                device=device,
                compute_type=compute_type,
                download_root=config["model_path"],
                num_workers=config["num_workers"]

            )
            subs = pysrt.SubRipFile()
            for audio_group in audio_groups:
                segments, _ = asr_model.transcribe(
                    audio=audio_group.audio_array,
                    beam_size=2,
                    vad_filter=False,
                    initial_prompt=basename,
                    language=config['language']
                )

                for seg in segments:
                    seg_start = seg.start
                    seg_end = seg.end
                    seg_text = seg.text.strip()

                    best_match = None
                    max_overlap = 0.0

                    subtitle = pysrt.SubRipItem(
                        index=len(subs) + 1,
                        start=pysrt.SubRipTime.from_ordinal(int(seg_start * 1000)),  # 转换为毫秒
                        end=pysrt.SubRipTime.from_ordinal(int(seg_end * 1000)),  # 转换为毫秒
                        text=seg_text
                    )
                    subs.append(subtitle)

                    for segment_info in audio_group.segment_info_list:
                        # 求开始时间的最大值和结束时间的最小值
                        overlap_start = max(seg_start, segment_info.group_start)
                        overlap_end = min(seg_end, segment_info.group_end)

                        # 如果重合时间大于零，计算重合时长
                        overlap_duration = max(0.0, overlap_end - overlap_start)

                        # 只有当重合时长大于零时，才可能是一个有效的匹配
                        if overlap_duration >= max_overlap:
                            max_overlap = overlap_duration
                            best_match = segment_info

                    if best_match and max_overlap > 0:
                        best_match.text = seg_text

            srt_path = os.path.join(config["log_path"], f"asr-{basename}.srt")
            subs.save(srt_path)


            del asr_model
            gc.collect()

            for audio_group in audio_groups:
                subs = pysrt.SubRipFile()
                for segment in audio_group.segment_info_list:
                    # 将每个 segment 信息转换为 SRT 格式
                    sub = pysrt.SubRipItem(
                        index=len(subs) + 1,  # 字幕索引
                        start=pysrt.SubRipTime.from_ordinal(int(segment.start * 1000)),  # 转换 start 为 SRT 时间格式
                        end=pysrt.SubRipTime.from_ordinal(int(segment.end * 1000)),  # 转换 end 为 SRT 时间格式
                        text=segment.text  # 字幕内容
                    )
                    subs.append(sub)

                # 设置输出 SRT 文件路径

            srt_path = os.path.join(config["asr_path"], f"{basename}.srt")


            subs.save(srt_path)
            print(f"字幕写入: {srt_path}")




if __name__ == "__main__":
    transcribe(get_config())

