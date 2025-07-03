import pandas as pd
from konlpy.tag import Okt
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# 한글 폰트 설정
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv('./body_friend_data_summary.csv')

# 형태소 분석기
okt = Okt()

# 불용어 정의
stopwords = set([
    "걔", "거", "건", "것", "게", "그", "그거", "그게", "그냥", "그렇고", "그렇다", "근데", "그래서", "같다",
    "나", "내", "너", "네", "도", "듯", "등", "막", "만", "만하", "뭐", "뭐라다", "번",
    "보다", "뿐", "수", "수도", "안", "않다", "어떻다", "에", "에게", "여기", "예", "오다",
    "우리", "으로", "이", "이게", "이렇게", "이렇다", "이제", "있다", "저", "저거", "저게",
    "저기", "점", "정도", "정말", "제", "조금", "주다", "지금", "진짜", "좀", "쪽", "차다", "하다", "하고",
    "또", "또한", "때", "되다", "더", "얘", "데", "구체", "해", "아주", "나다", "약간", "원래", "걸", "왜",
])

# 텍스트 분석 함수
def top_words_by_satisfaction(df, text_columns, label_col='pnf_satisfaction2', top_n=20):
    results = []
    for text_col in text_columns:
        for group_val, group_df in df.groupby(label_col):
            texts = group_df[text_col].dropna().tolist()
            words_all = []

            for text in texts:
                morphs = okt.pos(text, stem=True)
                words = [
                    word for word, pos in morphs
                    if pos in ['Noun', 'Adjective', 'Verb']
                    and word not in stopwords
                ]
                words_all.extend(words)

            # 빈도수
            word_counts = Counter(words_all)
            top_words = dict(word_counts.most_common(top_n))

            # TF-IDF
            vectorizer = TfidfVectorizer(tokenizer=lambda x: [
                word for word, pos in okt.pos(x, stem=True)
                if pos in ['Noun', 'Adjective', 'Verb'] and word not in stopwords
            ])
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.mean(axis=0).A1
            tfidf_dict = dict(zip(feature_names, tfidf_scores))

            for word, count in top_words.items():
                results.append({
                    'text_column': text_col,
                    'pnf_satisfaction2': "만족" if group_val == 0 else "불만족",
                    'word': word,
                    'count': count,
                    'tfidf': round(tfidf_dict.get(word, 0), 4)
                })

    return pd.DataFrame(results)

# 실행
text_columns = ['text1', 'text2', 'text3', 'text4']
top_words_df = top_words_by_satisfaction(df, text_columns, label_col='pnf_satisfaction2', top_n=20)

column_titles = {
    'text1': '안마의자에서 하신 운동이 어떠셨나요?',
    'text2': '운동 중 마사지가 어떠셨나요?',
    'text3': 'PNF 스트레칭의 어떤 점이 만족스러우셨나요?',
    'text4': 'PNF의 스트레칭의 어떤 점이 불만족스러우셨나요?'
}

# 터미널 출력
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(top_words_df)

# 엑셀 저장
top_words_df.to_excel('./results/pnf_satisfaction_results.xlsx', index=False)

# 시각화
for (text_col, group_val), subset in top_words_df.groupby(['text_column', 'pnf_satisfaction2']):
    # 빈도수 + TF-IDF
    plt.figure(figsize=(12, 6))
    plt.bar(subset['word'], subset['count'], label='빈도수')
    plt.plot(subset['word'], subset['tfidf'] * max(subset['count']), color='red', marker='o', label='TF-IDF (스케일)')
    plt.title(f'{column_titles.get(text_col, text_col)} - {group_val} 그룹 상위 단어', pad=10)
    plt.ylabel('빈도수 + TF-IDF')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', color='lightgray', linestyle='--', linewidth=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 워드 클라우드
    word_freqs = dict(zip(subset['word'], subset['count']))
    wc = WordCloud(font_path='/System/Library/Fonts/Supplemental/AppleGothic.ttf', background_color='white', width=1000, height=500)
    plt.figure(figsize=(12, 6))
    wc.generate_from_frequencies(word_freqs)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{column_titles.get(text_col, text_col)} - {group_val} 그룹 워드클라우드', pad=20)
    plt.show()
