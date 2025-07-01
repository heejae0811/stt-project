import pandas as pd
from konlpy.tag import Okt
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import rc

# 한글 폰트 설정
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv('./body_friend_data_summary.csv')

# 형태소 분석기
okt = Okt()

# 불용어 정의
stopwords = set([
    # 조사 및 의미 없는 단어
    "그냥", "이렇게", "조금", "좀", "거", "것", "근데", "이거", "저거", "그거",
    "그렇고", "그래서", "정도", "약간", "그", "이", "저", "또", "막", "좀", "때",
    "게", "데", "더", "되게", "에서", "으로", "까지", "도", "만", "든", "뿐", "중",
    "요", "듯", "수", "등", "의", "과", "및", "에서", "에게", "와", "랑", "하고",

    # 동사/형용사 중 의미 약한 것들
    "하다", "있다", "같다", "되다", "그렇다", "이렇다", "오다", "보다", "주다", "차다", "않다", "어떻다", "그렇다",

    # 내가 추가
    "또한", "점", "건", "뭐", "번", "쪽", "해", "에", "예", "내", "네", "제", "안", "얘", "걔", "쟤", "왜", "이제", "이게", "그게", "저게",
    "뭐라다", "워낙", "걸", "수도", "만하", "여기", "저기", "악", "아주", "나", "너", "우리", "구체", "지금", "점", "진짜", "정말"
])

# 텍스트 분석 함수
def top_words_per_text_and_group(df, text_columns, label_col='pnf_satisfaction2', top_n=20):
    results = []
    for text_col in text_columns:
        for group_val, group_df in df.groupby(label_col):
            texts = group_df[text_col].dropna().tolist()
            words_all = []

            for text in texts:
                morphs = okt.pos(text, stem=True)  # 전체 품사 분석
                words = [
                    word for word, pos in morphs
                    if pos in ['Noun', 'Adjective', 'Verb']
                    and word not in stopwords
                ]
                words_all.extend(words)

            word_counts = Counter(words_all)
            top_words = word_counts.most_common(top_n)

            for word, count in top_words:
                results.append({
                    'text_column': text_col,
                    'pnf_satisfaction': "만족" if group_val == 0 else "불만족",
                    'word': word,
                    'count': count
                })

    return pd.DataFrame(results)

# 실행
text_columns = ['text1', 'text2', 'text3', 'text4']
top_words_df = top_words_per_text_and_group(df, text_columns, top_n=20)
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
for (text_col, group_val), subset in top_words_df.groupby(['text_column', 'pnf_satisfaction']):
    plt.figure(figsize=(10, 5))
    plt.bar(subset['word'], subset['count'])
    plt.title(f'{column_titles.get(text_col, text_col)} - {group_val} 그룹 상위 단어')
    plt.xlabel('단어')
    plt.ylabel('빈도수')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', color='lightgray', linestyle='--', linewidth=0.2)
    plt.tight_layout()
    plt.show()