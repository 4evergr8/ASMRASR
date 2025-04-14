
<div align="center">
  <img src="https://api.iconify.design/material-symbols:movie.svg" height="36" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:music-note.svg" height="36" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:noise-control-on.svg" height="36" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:language-japanese-kana.svg" height="36" style="margin: 0 0px;">
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
  <img src="https://api.iconify.design/material-symbols:language-japanese-kana.svg" height="36" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:language-chinese-pinyin.svg" height="36" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:subtitles.svg" height="36" style="margin: 0 0px;">

</div>


<h3 align="center">ASMRASR</h3>

  <p align="center">
    用机器学习为DL上的音频生成中文字幕
    <br>
    同样适用于JAV
    <br>
    <a href="https://github.com/4evergr8/asmrasr/issues/new">🐞故障报告</a>
    ·
    <a href="https://github.com/4evergr8/asmrasr/issues/new">🏹功能请求</a>
  </p>




## 使用方法
### 本地运行（无需独显，但是会很慢）
* 安装Python3.10
* 打包下载此仓库，使用Pycharm打开项目
* 在0config.yaml中配置代理和token
* 将要处理的混合音频和视频放入0pre文件夹
* 将要处理的纯净音频放入1work文件夹
* 运行main等待处理完成
* 输出字幕在5result文件夹

### 云端运行（不稳定，调试中）
* 将整个项目解压到GoogleDrive并命名为ASMRASR
* 将要处理的混合音频和视频上传至0pre文件夹
* 将要处理的纯净音频上传至1work文件夹
* 在0config.yaml中配置token
* 点击<a href="https://colab.research.google.com/github/yourusername/yourrepository/blob/main/your_notebook.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" width="80">
</a>在Colab中打开项目
* 选择GPU运行时，点击全部运行，并允许访问云盘文件

## 主要功能
* 为音频生成中文字幕和日语字幕
* 本地识别，云端翻译
* 无需独显，核显运行
* 支持MP3和WAV



## 项目原理
* 使用demucs提取人声
* 使用WhisperX进行转写和对齐
* 用ja_ginza模型对长句进行拆分
* 用Google的Gemini进行在线翻译
* 转化lrc和srt文件
## 感谢
* 感谢Gemini提供了免费使用的api
* 在线翻译提示词来自于NEKOparapa/AiNiee
* Google的Colab帮了我很大的忙，Kaggle就是一坨
