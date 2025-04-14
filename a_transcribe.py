import os
import subprocess
import torch

from getconfig import get_config


def transcribe(config):
    audio_paths = []

    os.system("chcp 65001")

    os.environ['HF_HOME'] = config["model_path"]
    os.environ['HTTP_PROXY'] = config["proxy"]
    os.environ['HTTPS_PROXY'] = config["proxy"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print('设备：', device, '类型：', compute_type)

    for root, dirs, files in os.walk(config["work_path"]):
        for filename in files:
            if filename.endswith((".wav", ".mp3")):
                audio_path = os.path.join(root, filename)  # 获取音频文件的完整路径
                print(f"找到: {audio_path}")
                audio_paths.append(audio_path)

    if audio_paths:
        command = ["whisperx"] + audio_paths + [
            "--model", config["model"],
            "--model_cache_only", str(config["model_cache_only"]),
            "--device_index", str(config["device_index"]),
            "--device", device,
            "--batch_size", str(config["batch_size"]),
            "--compute_type", compute_type,
            "--output_dir", config["asr_path"],
            "--output_format", config["output_format"],
            "--verbose", str(config["verbose"]),
            "--task", config["task"],
            "--language", config["language"],
            "--align_model", config["align_model"],
            "--interpolate_method", config["interpolate_method"],
            "--vad_method", config["vad_method"],
            "--vad_onset", str(config["vad_onset"]),
            "--vad_offset", str(config["vad_offset"]),
            "--chunk_size", str(config["chunk_size"]),
            "--temperature", str(config["temperature"]),
            "--best_of", str(config["best_of"]),
            "--beam_size", str(config["beam_size"]),
            "--patience", str(config["patience"]),
            "--length_penalty", str(config["length_penalty"]),
            "--suppress_tokens", str(config["suppress_tokens"]),
            "--condition_on_previous_text", str(config["condition_on_previous_text"]),
            "--fp16", str(config["fp16"]),
            "--temperature_increment_on_fallback", str(config["temperature_increment_on_fallback"]),
            "--compression_ratio_threshold", str(config["compression_ratio_threshold"]),
            "--logprob_threshold", str(config["logprob_threshold"]),
            "--no_speech_threshold", str(config["no_speech_threshold"]),
            "--highlight_words", str(config["highlight_words"]),
            "--segment_resolution", config["segment_resolution"],
            "--threads", str(config["threads"]),
            "--print_progress", str(config["print_progress"])
        ]

        if config.get("no_align"):
            command += ["--no_align"]
        if config.get("return_char_alignments"):
            command += ["--return_char_alignments"]
        if config.get("diarize"):
            command += ["--diarize"]
        if config.get("suppress_numerals"):
            command += ["--suppress_numerals"]

        if config.get("min_speakers"):
            command += ["--min_speakers", str(config["min_speakers"])]
        if config.get("max_speakers"):
            command += ["--max_speakers", str(config["max_speakers"])]
        if config.get("initial_prompt"):
            command += ["--initial_prompt", config["initial_prompt"]]
        if config.get("max_line_width"):
            command += ["--max_line_width", str(config["max_line_width"])]
        if config.get("max_line_count"):
            command += ["--max_line_count", str(config["max_line_count"])]
        if config.get("hf_token"):
            command += ["--hf_token", config["hf_token"]]

        subprocess.run(command)


if __name__ == "__main__":
    transcribe(get_config())
