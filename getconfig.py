import os
import yaml

path: str = os.path.dirname(os.path.abspath(__file__))


def get_config():
    with open(os.path.join(path, '0config.yaml'), 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    config["pre_path"] = os.path.join(path, config["pre_path"])
    config["work_path"] = os.path.join(path, config["work_path"])
    config["asr_path"] = os.path.join(path, config["asr_path"])
    config["slice_path"] = os.path.join(path, config["slice_path"])
    config["zh_path"] = os.path.join(path, config["zh_path"])
    config["result_path"] = os.path.join(path, config["result_path"])
    config["model_path"] = os.path.join(path, config["model_path"])
    return config


if __name__ == "__main__":
    get_config()