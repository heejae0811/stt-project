import whisper

model = whisper.load_model("base")  # 또는 "small", "medium", "large"

# .m4a 파일 분석
result = model.transcribe("./data/01-1.m4a", language="ko")  # 한국어 음성일 경우

print(result['text'])  # 인식된 텍스트 출력
