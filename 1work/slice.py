import os
from pydub import AudioSegment

# 获取当前目录
path = os.path.dirname(os.path.abspath(__file__))

# 遍历当前目录下的所有wav文件
for root, dirs, files in os.walk(path):
    for filename in files:
        if filename.endswith(".wav"):
            # 获取文件的完整路径
            audio_path = os.path.join(root, filename)
            print(f"处理音频: {audio_path}")

            # 加载音频文件
            audio = AudioSegment.from_wav(audio_path)

            # 计算新的时长（原始时长的八分之一）
            new_duration = len(audio) // 64*8  # len(audio) 返回毫秒

            # 切割音频
            audio_segment = audio[:new_duration]

            # 输出文件路径
            output_path = os.path.join(root, f"cut_{filename}")

            # 导出切割后的音频
            audio_segment.export(output_path, format="wav")

            print(f"切割后的音频保存为: {output_path}")
