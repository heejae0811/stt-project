import whisper
import csv
import json
import os

# 모델 로드
model = whisper.load_model("base")  # 또는 "small", "medium", "large"

# 분석할 음성 파일 경로
audio_path = "./data/02-2.m4a"

# Whisper로 음성 → 텍스트 변환
result = model.transcribe(audio_path, language="ko")

# 저장 디렉토리 만들기 (없으면 자동 생성)
save_dir = "output"
os.makedirs(save_dir, exist_ok=True)

# ① 전체 텍스트 TXT로 저장
with open(os.path.join(save_dir, "02-2.txt"), "w", encoding="utf-8") as f:
    f.write(result["text"])

# ② 구간별 내용 CSV로 저장
with open(os.path.join(save_dir, "02-2.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "start", "end", "text"])
    for seg in result["segments"]:
        writer.writerow([seg["id"], seg["start"], seg["end"], seg["text"]])

# ③ 전체 구조 JSON으로 저장
with open(os.path.join(save_dir, "02-2.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Whisper 분석 및 저장 완료: TXT / CSV / JSON")
