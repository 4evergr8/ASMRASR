from audio_separator.separator import Separator


separator = Separator(output_dir="your/output/folder",model_file_dir='your/output/folder')
separator.load_model(model_filename="htdemucs_ft.yaml")
output_files = separator.separate("path/to/audio_directory")
