import json
import os
import time
import google.generativeai

from getconfig import get_config


def translate(config):
    os.environ['HF_HOME'] = config["model_path"]
    os.environ['HTTP_PROXY'] = config["proxy"]
    os.environ['HTTPS_PROXY'] = config["proxy"]


    for root, dirs, files in os.walk(config["slice_path"]):
        for filename in files:
            if filename.endswith(".json"):
                print(f"正在翻译: {filename}")
                json_path = os.path.join(root, filename)
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                before = [segment["text"] for segment in data["segments"]]
                after = []

                try:

                    print(len(before))
                    google.generativeai.configure(api_key=config["genimi_token"])
                    model = google.generativeai.GenerativeModel('gemini-2.0-flash')
                    chat = model.start_chat()
                    response = chat.send_message(config["first_prompt"])
                    print(f"初始回复：", response.text)

                    chunk_size = 30
                    for i in range(0, len(before), chunk_size):
                        chunk = before[i:i + chunk_size]

                        joined_text = "\n".join(chunk)

                        try:
                            response = chat.send_message(f'{config["common_prompt"]}{len(chunk)}\n{joined_text}')
                            #print(f"第{i // chunk_size + 1}组：", response.text)
                            after.extend(response.text.splitlines())


                        except Exception as e:
                            print(f"第{i // chunk_size + 1}组出错：{e}")
                            after.append('11111111111')
                        time.sleep(5)


                except Exception as e:
                    print("出错了：", e)
                finally:
                    basename = os.path.splitext(filename)[0]
                    output_path = os.path.join(root, f"{basename}.txt")
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write("\n".join(after) + "\n")

                print(len(after))

                for i, segment in enumerate(data["segments"]):
                    segment["text"] = after[i]
                print(f"翻译完成: {filename}")
                basename = os.path.splitext(filename)[0]
                output_path = os.path.join(config["zh_path"], f"{basename}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    translate(get_config())