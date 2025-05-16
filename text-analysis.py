import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline, PreTrainedTokenizerFast, BartForConditionalGeneration
from krwordrank.word import KRWordRank

# 결과 저장 폴더 생성
os.makedirs("result", exist_ok=True)

# 한국어 불용어 리스트
KOREAN_STOPWORDS = [
    "그", "이", "저", "것", "수", "좀", "더", "등", "및", "의", "가", "를", "은", "는", "에", "에서", "으로",
    "그리고", "그러나", "하지만", "그래서", "즉", "또는", "또한", "때문에", "하지만", "하거나", "해서", "된다",
    "합니다", "있습니다", "없습니다", "있는", "하는", "하여", "된다", "하면", "했다", "했다", "하게"
]

# 텍스트 불러오기
def load_texts_from_csv_folder(folder_path):
    all_texts = []
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        if 'Unnamed: 2' in df.columns:
            texts = df['Unnamed: 2'].dropna().astype(str).tolist()
            all_texts.extend([t.strip() for t in texts if len(t.strip()) > 1])
    return all_texts

# 감정 분석 (KoBERT)
sentiment_model = pipeline("sentiment-analysis", model="snunlp/KR-FinBERT-SC")
def analyze_sentiment(text):
    try:
        result = sentiment_model(text[:512])[0]
        return result["label"], float(result["score"])
    except:
        return "unknown", 0.0

# 의도 분류 (룰 기반)
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

# 요약 (KoBART)
tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")
summary_model = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")
def summarize_text(text):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summary_model.generate(inputs, max_length=50, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 키워드 추출 (KRWordRank)
def extract_keywords(text_list, top_n=10):
    wordrank_extractor = KRWordRank(min_count=1, max_length=10)
    keywords, _, _ = wordrank_extractor.extract(text_list, beta=0.85, max_iter=10)
    filtered = [(word, score) for word, score in keywords.items() if word not in KOREAN_STOPWORDS]
    return sorted(filtered, key=lambda x: -x[1])[:top_n]

# 토픽 모델링 (LDA)
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

# 분석 실행
def run_korean_nlp_pipeline(folder_path):
    texts = load_texts_from_csv_folder(folder_path)

    # 1. 키워드 추출
    keywords = extract_keywords(texts)
    pd.DataFrame(keywords, columns=["키워드", "점수"]).to_csv("result/result_keywords.csv", index=False, encoding="utf-8-sig")

    # 2. 텍스트 요약 + 감정 + 의도
    summary_data = []
    for text in texts:
        sentiment, score = analyze_sentiment(text)
        intent = classify_intent(text)
        summary = summarize_text(text) if len(text) > 20 else ""
        summary_data.append({
            "원문": text,
            "감정": sentiment,
            "감정점수": score,
            "의도": intent,
            "요약": summary
        })
    pd.DataFrame(summary_data).to_csv("result/result_summary.csv", index=False, encoding="utf-8-sig")

    # 3. 워드클라우드 시각화
    text_blob = " ".join(texts)
    wordcloud = WordCloud(font_path="/System/Library/Fonts/Supplemental/AppleGothic.ttf", background_color='white').generate(text_blob)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("WordCloud", fontsize=15)
    plt.savefig("result/result_wordcloud.png")
    plt.show()

    # 4. 텍스트 분류 모델 평가
    y = [classify_intent(t) for t in texts]
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv("result/result_classification_report.csv", encoding="utf-8-sig")

    # 5. 토픽 모델링
    topics = topic_modeling(texts)
    pd.DataFrame(topics, columns=["토픽", "주요단어"]).to_csv("result/result_topics.csv", index=False, encoding="utf-8-sig")

# 실행
if __name__ == "__main__":
    run_korean_nlp_pipeline("data/text")
