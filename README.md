#### 这是一个运行funasr的后端代码

main.py 运行在9000端口上

通过curl -X POST http://localhost:9000/asr -F "file=@./audio.mp3"测试

audio_in文件地址 = 'https://modelscope.cn/api/v1/models/damo/speech_campplus_speaker-diarization_common/repo?Revision=master&FilePath=examples/2speakers_example.wav'
