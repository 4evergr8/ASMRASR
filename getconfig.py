import os
import sys

import yaml





def get_config():
    path: str = os.path.dirname(sys.argv[0])
    with open(os.path.join(path, '0config.yaml'), 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    def get_path(config_value):
        path = os.path.join(os.getcwd(), config_value)
        os.makedirs(path, exist_ok=True)
        return path

    config["pre_path"] = get_path(config["pre_path"])
    config["work_path"] = get_path(config["work_path"])
    config["asr_path"] = get_path(config["asr_path"])
    config["tsl_path"] = get_path(config["tsl_path"])
    config["model_path"] = get_path(config["model_path"])
    config["log_path"] = get_path(config["log_path"])


    return config


if __name__ == "__main__":
    get_config()