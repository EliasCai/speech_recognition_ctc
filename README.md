# speech_recognition_ctc（中文语音识别）
### 数据来源
- [清华30小时中文语音库](http://www.openslr.org/18/ )
- [AISHELL-ASR0009-OS1录音时长178小时](http://www.aishelltech.com/kysjcp )

### 代码说明

- 1-1-generate.py: 生成MFCC特征矩阵，方便2-2和2-3的代码调用（一次性载入内存）
- 2-2-ctc_speech_thchs30.py：使用清华语音库进行训练，输入为一次性加载，输出为文字
- 2-3-ctc_speech_thchs30_pinyin.py：使用清华语音库进行训练，输入为一次性加载，输出为拼音
- 2-4-ctc_speech_aishell.py：使用希尔贝壳语音库进行训练，输入为生成器，输出为文字
- 2-5-ctc_speech_thchs.py：使用清华语音库进行训练，输入为生成器，输出为文字
- 2-6-ctc_speech_both.py：使用清华和希尔贝壳语音库进行训练，输入为生成器，输出为文字
- 2-7-ctc_speech_pinyin.py：使用清华和希尔贝壳语音库进行训练，输入为生成器，输出为拼音
- utils.py：包含数据的读取和处理，由2-4调用