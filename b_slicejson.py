import json
import os
import spacy

from getconfig import get_config


def slicejson(config):
    os.environ['HF_HOME'] = config["model_path"]
    os.environ['HTTP_PROXY'] = config["proxy"]
    os.environ['HTTPS_PROXY'] = config["proxy"]


    nlp = spacy.load("ja_ginza_electra")
    for root, dirs, files in os.walk(config["asr_path"]):
        for filename in files:
            if filename.endswith(".json"):
                print(f"切分中: {filename}")
                json_path = os.path.join(root, filename)  # 获取音频文件的完整路径
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                output = {
                    "segments": [],
                    "language": "ja"
                }

                # 遍历每个原始语音段
                for segment in data["segments"]:
                    text = segment["text"]
                    words = segment["words"]

                    # 将整段文本进行分句
                    doc = nlp(text)

                    # 构造字符位置对应时间戳的索引（按顺序累积）
                    char_times = []
                    for word in words:
                        char = word["word"]
                        start = word.get("start", None)
                        end = word.get("end", None)
                        char_times.append({
                            "char": char,
                            "start": start,
                            "end": end
                        })

                    current_char_index = 0  # 用于遍历 char_times

                    # 遍历分好的每个句子
                    for sent in doc.sents:
                        sentence = sent.text.strip()
                        if not sentence:
                            continue

                        sent_chars = list(sentence)
                        start_time = None
                        end_time = None
                        matched_indices = []

                        # 逐字符匹配，记录时间戳索引范围
                        i = 0
                        while i < len(sent_chars) and current_char_index < len(char_times):
                            target_char = sent_chars[i]
                            current_char = char_times[current_char_index]["char"]
                            if target_char == current_char:
                                matched_indices.append(current_char_index)
                                i += 1
                            current_char_index += 1

                        # 有匹配上的字符，找第一个有时间戳和最后一个有时间戳的
                        for idx in matched_indices:
                            if start_time is None and char_times[idx]["start"] is not None:
                                start_time = char_times[idx]["start"]
                            if char_times[idx]["end"] is not None:
                                end_time = char_times[idx]["end"]

                        if start_time is not None and end_time is not None:
                            output["segments"].append({
                                "text": sentence,
                                "start": start_time,
                                "end": end_time
                            })
                print(f"切分完毕: {filename}")
                basename = os.path.splitext(filename)[0]
                output_path = os.path.join(config["slice_path"], f"{basename}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    slicejson(get_config())