import pysrt
import torch
import numpy as np
from pyannote.audio import Model
from faster_whisper import WhisperModel
import librosa
from pyannote.audio.pipelines import VoiceActivityDetection
import os
from getconfig import get_config





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


            group_duration = 1800  # 每组时长：30分钟
            for segment in timeline:
                group_index = int(segment.end // group_duration)
                group_base_idx = 1000 + group_index * 1000
                sub_index = group_base_idx + len(
                    [s for s in vad_log if group_base_idx <= s.index < group_base_idx + 1000])

                sub = pysrt.SubRipItem(
                    index=sub_index,
                    start=pysrt.SubRipTime.from_ordinal(int(segment.start * 1000)),
                    end=pysrt.SubRipTime.from_ordinal(int(segment.end * 1000)),
                    text="默认占位"+str(sub_index)
                )
                vad_log.append(sub)

            vad_log.save(vad_log_path)
            print(f"VAD记录写入: {vad_log_path}")
        else:
            print('VAD记录存在，跳过')

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
            if subtitle.index % 1000 == 0:
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

            audios = []  # 每个元素是一个 numpy 音频数组
            silence_duration = config["space"]  # 静音间隔
            silence = np.zeros(int(sr * silence_duration), dtype=np.float32)  # 静音段

            group_dict = {}  # 存储各组的音频段，key 是组号（如 1, 2, 3）

            # 处理字幕，根据字幕 index 决定所属组
            for subtitle in vad_log:
                segment_start = subtitle.start.ordinal / 1000  # 秒
                segment_end = subtitle.end.ordinal / 1000  # 秒
                index = subtitle.index
                group_id = index // 1000  # 1000~1999 为第 1 组，2000~2999 为第 2 组，以此类推

                start_sample = int(segment_start * sr)
                end_sample = int(segment_end * sr)
                audio_seg = audio[start_sample:end_sample]

                if group_id not in group_dict:
                    group_dict[group_id] = []
                group_dict[group_id].append(audio_seg)

            # 拼接每组音频，并插入静音段
            for group_id in sorted(group_dict):
                group_audio = []
                for i, segment in enumerate(group_dict[group_id]):
                    if i > 0:
                        group_audio.append(silence)
                    group_audio.append(segment)

                group_array = np.concatenate(group_audio) if group_audio else np.array([], dtype=np.float32)
                audios.append(group_array)

            print(f"音频分组完成，共 {len(audios)} 组。")
            del audio


            asr_model = WhisperModel(
                model_size_or_path=config["asr"],
                device=device,
                compute_type=compute_type,
                download_root=config["model_path"],

            )

            asr_log = pysrt.SubRipFile()
            base_index = 1000  # 初始组编号起点

            for group_idx, audio in enumerate(audios, start=1):
                start_index = group_idx * base_index

                segments, _ = asr_model.transcribe(
                    audio=audio,
                    language=config['language'],
                    task="transcribe",
                    log_progress=True,
                    beam_size=5,
                    best_of=5,
                    patience=1,
                    length_penalty=1,
                    repetition_penalty=1.1,  # 稍微提高抑制重复的倾向
                    no_repeat_ngram_size=3,  # 阻止 ngram 重复
                    temperature=[0.2, 0.4, 0.6, 0.8, 1.0],  # 稍高起步避免死循环
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=True,  # 保留上下文
                    prompt_reset_on_temperature=0.5,
                    prefix=None,
                    suppress_blank=True,
                    suppress_tokens=[-1],
                    without_timestamps=False,
                    max_initial_timestamp=1.0,
                    word_timestamps=True,
                    hallucination_silence_threshold=1.5,
                    prepend_punctuations="\"'“¿([{-",
                    append_punctuations="\"'.。,，!！?？:：”)]}、",
                    multilingual=False,
                    vad_filter=False,
                    clip_timestamps="0",
                    language_detection_threshold=None,
                    language_detection_segments=1,
                    hotwords='イッたんだよ やだ まって…やばい… 恥ずかしい あっ ああ んんっ あぅ はっ やっ はぁ はっはっ はうっ ふぅ くぅ'
                )






                for i, seg in enumerate(segments):
                    seg_start = seg.start
                    seg_end = seg.end
                    seg_text = seg.text.strip()

                    subtitle = pysrt.SubRipItem(
                        index=start_index + i,
                        start=pysrt.SubRipTime.from_ordinal(int(seg_start * 1000)),
                        end=pysrt.SubRipTime.from_ordinal(int(seg_end * 1000)),
                        text=seg_text
                    )
                    asr_log.append(subtitle)

            # 保存识别结果

                asr_log.save(asr_log_path)
                print(f"ASR记录写入: {asr_log_path}")
        else:
            asr_log = pysrt.open(asr_log_path)
            print('ASR记录存在，跳过')

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
        asr_log_path = os.path.join(config["log_path"], f"asr-{basename}.srt")
        asr_log = pysrt.open(asr_log_path)





        for slice_sub in slice_log:
            segment_start = slice_sub.start.ordinal / 1000
            segment_end = slice_sub.end.ordinal / 1000
            segment_index = slice_sub.index
            group_prefix = (segment_index // 1000) * 1000

            max_overlap = 0.0
            best_match = None

            # 查找当前组中所有 asr_log 字幕（同一千段内）
            for asr_sub in asr_log:
                if (asr_sub.index // 1000) * 1000 != group_prefix:
                    continue

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

        for i, sub in enumerate(vad_log, 1):
            sub.index = i

        result_path = os.path.join(config["asr_path"], f"{basename}.srt")
        vad_log.save(result_path)
        print(f"最终结果写入: {result_path}")










if __name__ == "__main__":
    config = get_config()
    vad(config)
    slice(config)
    asr(config)
    match(config)

