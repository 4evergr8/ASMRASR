import os
import urllib.request
import zipfile
import subprocess
import sys
import shutil

def getdependency():

    # 获取当前脚本所在的目录作为保存目录
    save_dir = os.path.dirname(os.path.abspath(__file__))

    zip_url = f"https://github.com/m-bain/whisperX/archive/refs/heads/main.zip"
    zip_name = f"whisperX-main.zip"
    zip_path = os.path.join(save_dir, zip_name)

    print(f"📥 正在下载：{zip_url}")
    urllib.request.urlretrieve(zip_url, zip_path)
    print(f"✅ 下载完成：{zip_path}")

    # 解压
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)
    extracted_dir = os.path.join(save_dir,'whisperX-main')
    print(f"📦 解压完成：{extracted_dir}")

    print(f"📦 正在使用 pip 安装 {extracted_dir}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", extracted_dir])
    print("✅ 安装完成")

    # 检查是否有 requirements.txt 文件，若有则安装依赖
    requirements_path = os.path.join(save_dir,"requirements.txt")
    if os.path.exists(requirements_path):
        print(f"📦 检测到 requirements.txt，正在安装依赖...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("✅ 依赖安装完成")

    # 删除 zip 文件
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"🧹 已删除压缩包：{zip_path}")

    # 删除解压后的目录
    if os.path.exists(extracted_dir):
        shutil.rmtree(extracted_dir)
        print(f"🧹 已删除文件夹：{extracted_dir}")

if __name__ == "__main__":
    getdependency()
