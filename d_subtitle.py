import json
import os

from getconfig import get_config


def seconds_to_lrc_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    sec = seconds % 60
    return f"{minutes:02}:{sec:05.2f}"


def seconds_to_srt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def subtitle(config):
    for root, dirs, files in os.walk(config["zh_path"]):
        for filename in files:
            if filename.endswith(".json"):
                json_path = os.path.join(root, filename)  # 获取音频文件的完整路径
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                segments = data.get("segments", [])
                lrc_lines = []
                for seg in segments:
                    start = seconds_to_lrc_timestamp(seg["start"])
                    end = seconds_to_lrc_timestamp(seg["end"])
                    text = seg["text"].strip()

                    lrc_lines.append(f"[{start}]{text}")
                    lrc_lines.append(f"[{end}]")

                basename = os.path.splitext(filename)[0]
                output_path = os.path.join(root, f"{basename}.lrc")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lrc_lines))

                print(f"已生成 LRC 文件：{output_path}")




    for root, dirs, files in os.walk(config["zh_path"]):
        for filename in files:
            if (filename.startswith("slice-") or filename.startswith("zh-")) and filename.endswith(".json"):
                json_path = os.path.join(root, filename)
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                segments = data.get("segments", [])

                srt_lines = []
                for idx, seg in enumerate(segments, 1):
                    start = seconds_to_srt_timestamp(seg["start"])
                    end = seconds_to_srt_timestamp(seg["end"])
                    text = seg["text"].strip()

                    srt_lines.append(f"{idx}")
                    srt_lines.append(f"{start} --> {end}")
                    srt_lines.append(text)
                    srt_lines.append("")  # 空行分隔

                basename = os.path.splitext(filename)[0]
                output_path = os.path.join(root, f"{basename}.srt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(srt_lines))

                print(f"已生成 SRT 文件：{output_path}")


if __name__ == "__main__":
    subtitle(get_config())