import time

import soundfile

from getconfig import get_config
import pysrt
import torch
import numpy as np
from pyannote.audio import Model
from faster_whisper import WhisperModel
import librosa
from pyannote.audio.pipelines import VoiceActivityDetection
import os

def vad(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print('设备:', device, '类型:', compute_type)


    for filename in os.listdir(config["work_path"]):
        if not filename.endswith((".wav", ".mp3", ".flac")):
            continue
        audio_path = os.path.join(config["work_path"], filename)
        print(f"\n处理音频: {audio_path}")
        basename = os.path.splitext(filename)[0]


        vad_log_path = os.path.join(config["log_path"], f"vad-{basename}.srt")
        if not os.path.exists(vad_log_path) or config["overwrite_vad"]:
            vad_model = Model.from_pretrained(checkpoint=config["vad"], cache_dir=config["model_path"])
            vad_model.to(torch.device(device))
            vad_pipeline = VoiceActivityDetection(segmentation=vad_model)
            vad_pipeline.instantiate({
                "min_duration_on": config["min_duration_on"],
                "min_duration_off": config["min_duration_off"],
            })


            vad_result = vad_pipeline(str(audio_path))
            del vad_pipeline, vad_model

            timeline = vad_result.get_timeline()
            vad_log = pysrt.SubRipFile()

            for i, segment in enumerate(timeline, start=1):
                if segment.end - segment.start < config["filter"]:
                    continue  # 跳过持续时间小于设定值的段

                sub = pysrt.SubRipItem(
                    index=i,
                    start=pysrt.SubRipTime.from_ordinal(int(segment.start * 1000)),
                    end=pysrt.SubRipTime.from_ordinal(int(segment.end * 1000)),
                    text="默认占位" + str(i)
                )
                vad_log.append(sub)

            vad_log.save(vad_log_path)
            print(f"VAD记录写入: {vad_log_path}")


def slice(config):
    for filename in os.listdir(config["work_path"]):
        if not filename.endswith((".wav", ".mp3", ".flac")):
            continue
        audio_path = os.path.join(config["work_path"], filename)
        print(f"\n处理音频: {audio_path}")
        basename = os.path.splitext(filename)[0]
        vad_log_path = os.path.join(config["log_path"], f"vad-{basename}.srt")
        vad_log = pysrt.open(vad_log_path)


        slice_log = pysrt.SubRipFile()  # 用于存储调整后的字幕
        silence_duration = config["space"]  # 获取配置中的静音时间
        current_group_end = 0.0  # 当前组的结束时间

        for subtitle in vad_log:
            segment_start = subtitle.start.ordinal / 1000  # 转换为秒
            segment_end = subtitle.end.ordinal / 1000  # 转换为秒

            # 如果是该组的第一个段（序号能被1000整除）
            if subtitle.index == 1:
                group_start = 0.0  # 设置为0，确保每组的第一个段从00:00:00开始
            else:
                group_start = current_group_end  # 否则按照上一段的结束时间进行处理

            group_end = group_start + (segment_end - segment_start)  # 保持时间段的长度不变

            # 更新字幕的开始和结束时间戳
            subtitle.start = pysrt.SubRipTime.from_ordinal(int(group_start * 1000))  # 转换为毫秒并设置新的开始时间
            subtitle.end = pysrt.SubRipTime.from_ordinal(int(group_end * 1000))  # 设置新的结束时间

            # 更新当前组的结束时间
            current_group_end = group_end + silence_duration  # 下一组的开始时间是当前组的结束时间+静音间隔

            # 将处理后的字幕项添加到slice_log中
            slice_log.append(subtitle)

        slice_log_path = os.path.join(config["log_path"], f"slice-{basename}.srt")
        slice_log.save(slice_log_path)
        print(f"slice记录写入: {slice_log_path}")

def asr(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print('设备:', device, '类型:', compute_type)

    for filename in os.listdir(config["work_path"]):
        if not filename.endswith((".wav", ".mp3", ".flac")):
            continue
        audio_path = os.path.join(config["work_path"], filename)
        print(f"\n处理音频: {audio_path}")
        basename = os.path.splitext(filename)[0]
        vad_log_path = os.path.join(config["log_path"], f"vad-{basename}.srt")



        asr_log_path = os.path.join(config["log_path"], f"asr-{basename}.srt")
        if not os.path.exists(asr_log_path) or config["overwrite_asr"]:
            vad_log = pysrt.open(vad_log_path)
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

            audios = []  # 所有音频段顺序拼接
            silence_duration = config["space"]  # 静音间隔
            silence = np.zeros(int(sr * silence_duration), dtype=np.float32)  # 静音段

            for i, subtitle in enumerate(vad_log):
                segment_start = subtitle.start.ordinal / 1000  # 秒
                segment_end = subtitle.end.ordinal / 1000  # 秒

                start_sample = int(segment_start * sr)
                end_sample = int(segment_end * sr)
                audio_seg = audio[start_sample:end_sample]

                if i > 0:
                    audios.append(silence)
                audios.append(audio_seg)

            final_audio = np.concatenate(audios) if audios else np.array([], dtype=np.float32)
            soundfile.write(os.path.join(config["log_path"], f"{basename}.wav"), final_audio, 16000)

            print("音频拼接完成，不再分组。")
            del audio



def match(config):
    for filename in os.listdir(config["work_path"]):
        if not filename.endswith((".wav", ".mp3", ".flac")):
            continue
        audio_path = os.path.join(config["work_path"], filename)
        print(f"\n处理音频: {audio_path}")
        basename = os.path.splitext(filename)[0]
        vad_log_path = os.path.join(config["log_path"], f"vad-{basename}.srt")
        vad_log = pysrt.open(vad_log_path)
        slice_log_path = os.path.join(config["log_path"], f"slice-{basename}.srt")
        slice_log = pysrt.open(slice_log_path)
        asr_log_path = os.path.join(config["log_path"], f"{basename}_原文.srt")
        asr_log = pysrt.open(asr_log_path)





        for slice_sub in slice_log:
            segment_start = slice_sub.start.ordinal / 1000
            segment_end = slice_sub.end.ordinal / 1000


            max_overlap = 0.0
            best_match = None

            # 查找当前组中所有 asr_log 字幕（同一千段内）
            for asr_sub in asr_log:

                seg_start = asr_sub.start.ordinal / 1000
                seg_end = asr_sub.end.ordinal / 1000

                # 重合检测
                overlap_start = max(seg_start, segment_start)
                overlap_end = min(seg_end, segment_end)
                overlap_duration = overlap_end - overlap_start

                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_match = asr_sub

            if best_match is not None:
                slice_sub.text = best_match.text

        # 保存更新后的 slice_log
        match_path = os.path.join(config["log_path"], f"match-{basename}.srt")
        slice_log.save(match_path)
        print(f"match结果写入: {match_path}")



        for vad_sub in vad_log:
            for slice_sub in slice_log:
                idx = slice_sub.index

                if vad_sub.index == idx:
                    vad_sub.text = slice_sub.text
                    break  # 找到对应的字幕后可以停止循环，避免多次匹配



        result_path = os.path.join(config["asr_path"], f"{basename}.srt")
        vad_log.save(result_path)
        print(f"最终结果写入: {result_path}")










if __name__ == "__main__":
    config = get_config()
    vad(config)
    slice(config)
    asr(config)
    match(config)


