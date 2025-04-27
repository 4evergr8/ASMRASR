from audio_separator.separator import Separator
from getconfig import get_config


config=get_config()
# Initialize the Separator class (with optional configuration properties, below)
separator = Separator(model_file_dir=config["model_path"],output_dir=config["work_path"],output_single_stem=True,sample_rate=16000,use_autocast=True)

# Load a machine learning model (if unspecified, defaults to 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
separator.load_model('MDX23C-8KFFT-InstVoc_HQ_2.ckpt')

# Perform the separation on specific audio files without reloading the model
output_files = separator.separate('D:\Project\ASMRASR\MIDV-771.wav')

print(f"Separation complete! Output file(s): {' '.join(output_files)}")