from audio_separator.separator import Separator

# 自定义输出路径
separator = Separator(output_dir="your/output/folder")

# 加载模型
separator.load_model(model_filename="UVR-Mgfbnfgbgfd")

# 开始分离音频
output_files = separator.separate("path/to/audio_directory")
