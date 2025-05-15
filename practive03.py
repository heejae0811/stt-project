import pandas as pd
from transformers import pipeline
import glob
import os

# 감정 분석 파이프라인 (한글용 모델 사용)
sentiment = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC", tokenizer="snunlp/KR-FinBert-SC")

# 분석할 CSV 경로들 (폴더 경로 맞게 수정)
csv_files = glob.glob("./output/*.csv")  # 또는 ["data/01-1.csv", "data/02-1.csv", ...]

# 결과 저장 리스트
all_results = []

# 각 파일별로 분석 수행
for file_path in csv_files:
    df = pd.read_csv(file_path)
    filename = os.path.basename(file_path)

    # 감정 분석
    df["sentiment"] = df["text"].apply(lambda x: sentiment(x)[0]["label"] if isinstance(x, str) else "error")

    # 분석 대상 식별용 열 추가
    df["source_file"] = filename

    # 결과 저장
    all_results.append(df)

# 전체 결과 결합
combined_df = pd.concat(all_results, ignore_index=True)

# 저장 (선택)
combined_df.to_csv("all_segments_with_sentiment.csv", index=False, encoding="utf-8-sig")

# 간단한 요약 출력
summary = combined_df.groupby(["source_file", "sentiment"]).size().unstack(fill_value=0)
print(summary)
