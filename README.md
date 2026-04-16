
<div align="center">
  <img src="https://api.iconify.design/material-symbols:movie.svg" height="36" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:music-note.svg" height="36" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:noise-control-on.svg" height="36" style="margin: 0 0px;">

  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 10px;">
  <img src="https://api.iconify.design/material-symbols:language-chinese-pinyin.svg" height="36" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:subtitles.svg" height="36" style="margin: 0 0px;">
</div>


<div align="center">
  <img src="https://api.iconify.design/material-symbols:music-note.svg" height="36" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:noise-control-on.svg" height="36" style="margin: 0 0px;">

  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:language-chinese-pinyin.svg" height="36" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:subtitles.svg" height="36" style="margin: 0 0px;">

</div>


<h3 align="center">ASMRASR</h3>

  <p align="center">
    用机器学习为DL上的日语音频生成中文字幕
    <br>
    同样适用于日语视频的汉化
    <br>
    <a href="https://github.com/4evergr8/asmrasr/issues/new">🐞故障报告</a>
    ·
    <a href="https://github.com/4evergr8/asmrasr/issues/new">🏹功能请求</a>
  </p>




## 使用方法
### 本地运行（无需独显，但是会很慢）
* 安装Python3.10
* 打包下载此仓库，使用Pycharm打开项目
* 0pre文件夹放置待处理视频和混合音频,1audio文件夹存放纯净人声音频
* 安装依赖后依次运行所有代码
* 在3asr文件夹中获取识别结果


### 独显运行（NVIDIA GPU 加速，推荐）
*   **要求**：拥有 NVIDIA 显卡并安装了显卡驱动。
*   **安装步骤**：
    1.  按照上述“本地运行”步骤初始化虚拟环境。
    2.  运行项目根目录下的 `install_gpu.bat`。
    3.  在弹出菜单中选择与你驱动相匹配的 CUDA 版本（即使你的驱动显示为 13.x，也可以选择 12.1 或 12.4，它们通常向后兼容）。
    4.  安装完成后，再次运行 `运行.bat` 即可享受硬件加速。
*   **优势**：处理速度可提升 5-10 倍，并支持更高精度的模型推理。

### 云端运行（不稳定，调试中）
* 点击<a href="https://colab.research.google.com/github/4evergr8/ASMRASR/blob/main/笔记本.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" width="80">
</a>在Colab中打开项目
* 选择GPU运行时，点击全部运行，并允许访问云盘文件
* 将要处理的混合音频和视频上传至0pre文件夹
* 将要处理的纯净音频上传至1audio文件夹
* 要等狠久

## 主要功能
* 为日语音频生成中文字幕
* 支持视频提取音频,人声背景分离
* 无需独显，核显运行
* 支持MP3和WAV



## 项目原理
* 使用demucs提取人声
* 用Pyannote裁剪出人声,写入srt文件
* 使用faster-whisper进行带时间戳转写,对其中的VAD实现进行替换,换成之前的识别结果
* 输出srt文件
## 感谢
* chickenrice0721的Whisper微调模型
* faster-whisper和Pyannote
* Google的Colab帮了我很大的忙，Kaggle就是一坨

