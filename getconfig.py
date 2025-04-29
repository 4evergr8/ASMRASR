import os
import yaml


path: str = os.path.dirname(os.path.abspath(__file__))


def get_config():
    with open(os.path.join(path, '0config.yaml'), 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    def get_path(config_value):
        """
        根据传入的路径值创建绝对路径并创建文件夹
        :param config_value: 配置字典中的相对路径
        :return: 返回创建的绝对路径
        """

        path = os.path.join(os.getcwd(), config_value)
        os.makedirs(path, exist_ok=True)
        print(f"创建文件夹: {path}")
        return path

    config["pre_path"] = get_path(config["pre_path"])
    config["work_path"] = get_path(config["work_path"])
    config["asr_path"] = get_path(config["asr_path"])
    config["merge_path"] = get_path(config["merge_path"])
    config["model_path"] = get_path(config["model_path"])
    config["log_path"] = get_path(config["log_path"])

    return config


if __name__ == "__main__":
    get_config()