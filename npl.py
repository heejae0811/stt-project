# NLP 통합 분석 코드 (한국어 전용, Unnamed: 2 열 기반 분석, 불용어 제거 포함)

import pandas as pd
from pathlib import Path
from transformers import pipeline
from krwordrank.word import KRWordRank
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# ✅ 한국어 불용어 리스트 정의
KOREAN_STOPWORDS = [
    "그", "이", "저", "것", "수", "좀", "더", "등", "및", "의", "가", "를", "은", "는", "에", "에서", "으로",
    "그리고", "그러나", "하지만", "그래서", "즉", "또는", "또한", "때문에", "하지만", "하거나", "해서", "된다",
    "합니다", "있습니다", "없습니다", "있는", "하는", "하여", "된다", "하면", "했다", "했다", "하게"
]

# 1. CSV에서 'Unnamed: 2' 열의 텍스트만 추출 (불용어 제거용)
def load_text_from_csv(file_path):
    df = pd.read_csv(file_path)
    if 'Unnamed: 2' not in df.columns:
        return []
    texts = df['Unnamed: 2'].dropna().astype(str).tolist()
    return [t.strip() for t in texts if len(t.strip()) > 1]

# 2. 감정 분석 (KoBERT 기반)
sentiment_model = pipeline("sentiment-analysis", model="snunlp/KR-FinBERT-SC")

def analyze_sentiment(text):
    try:
        result = sentiment_model(text[:512])[0]
        return result["label"], float(result["score"])
    except:
        return "unknown", 0.0

# 3. 의도 분류 (간단한 규칙 기반)
def classify_intent(text):
    if "?" in text:
        return "질문"
    elif "해주세요" in text or "바랍니다" in text:
        return "요청"
    elif "좋겠어요" in text or "했으면" in text:
        return "제안"
    elif "싫어요" in text or "불편" in text:
        return "불만"
    else:
        return "진술"

# 4. 키워드 추출 (KRWordRank + 불용어 제거)
def extract_keywords(text_list, top_n=10):
    wordrank_extractor = KRWordRank(min_count=1, max_length=10)
    keywords, _, _ = wordrank_extractor.extract(text_list, beta=0.85, max_iter=10)
    filtered = [(word, score) for word, score in keywords.items() if word not in KOREAN_STOPWORDS]
    return sorted(filtered, key=lambda x: -x[1])[:top_n]

# 5. 요약 (KoBART 기반)
tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")
summary_model = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")

def summarize_text(text):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summary_model.generate(inputs, max_length=50, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 6. 토픽 모델링 (LDA + 불용어 제거)
def topic_modeling(text_list, num_topics=3):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=KOREAN_STOPWORDS)
    doc_term_matrix = vectorizer.fit_transform(text_list)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)

    terms = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_terms = [terms[i] for i in topic.argsort()[-5:][::-1]]
        topics.append((f"Topic {idx+1}", top_terms))
    return topics

# 7. 전체 실행 함수
def full_nlp_analysis(folder_path):
    all_texts = []
    output_rows = []

    for file in Path(folder_path).glob("*.csv"):
        texts = load_text_from_csv(file)
        all_texts.extend(texts)

        for text in texts:
            sentiment, score = analyze_sentiment(text)
            intent = classify_intent(text)
            summary = summarize_text(text)

            output_rows.append({
                "파일명": file.name,
                "텍스트": text,
                "감정": sentiment,
                "감정점수": score,
                "의도": intent,
                "요약": summary
            })

    df_result = pd.DataFrame(output_rows)
    df_result.to_csv("NLP_통합분석_결과.csv", index=False, encoding='utf-8-sig')

    # 키워드 저장
    keywords = extract_keywords(all_texts)
    pd.DataFrame(keywords, columns=["키워드", "중요도"]).to_csv("NLP_키워드.csv", index=False, encoding='utf-8-sig')

    # 토픽 저장
    topics = topic_modeling(all_texts)
    pd.DataFrame(topics, columns=["토픽번호", "주요단어"]).to_csv("NLP_토픽모델링.csv", index=False, encoding='utf-8-sig')

    print("✅ 모든 한국어 NLP 분석 완료! 결과 CSV 3종 생성 (불용어 제거 포함, Unnamed: 2 기준)")

# 8. 실행
if __name__ == "__main__":
    full_nlp_analysis("data/text")
