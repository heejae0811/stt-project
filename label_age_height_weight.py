import pandas as pd
from konlpy.tag import Okt
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

# 한글 폰트 설정
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv("body_friend_data_summary.csv")

# 형태소 분석기
okt = Okt()

# 불용어
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

# 텍스트 병합
df["combined_text"] = df[["text1", "text2", "text3", "text4"]].fillna("").agg(" ".join, axis=1)

# 연령/키/체중 그룹화
df["age_group"] = pd.cut(df["age"], bins=[40, 50, 60, 70, 100], labels=["40대", "50대", "60대", "70대 이상"], right=False)
df["height_group"] = pd.cut(df["height"], bins=[140, 150, 160, 170, 200], labels=["140cm", "150cm", "160cm", "170cm 이상"], right=False)
df["weight_group"] = pd.cut(df["weight"], bins=[40, 50, 60, 70, 100], labels=["40kg", "50kg", "60kg", "70kg 이상"], right=False)

# 그룹별 단어 분석 함수
def extract_top_words_by_group(df, group_col, top_n=10):
    results = []
    for group, group_df in df.groupby(group_col):
        texts = group_df["combined_text"].dropna().tolist()
        words_all = []

        for text in texts:
            morphs = okt.pos(text, stem=True)
            words = [
                word for word, pos in morphs
                if pos in ['Noun', 'Adjective', 'Verb']
                and word not in stopwords
            ]
            words_all.extend(words)

        word_counts = Counter(words_all)
        for word, count in word_counts.most_common(top_n):
            results.append({
                "group_type": group_col,
                "group": group,
                "word": word,
                "count": count
            })

    return pd.DataFrame(results)

# 실행
age_words = extract_top_words_by_group(df, "age_group")
height_words = extract_top_words_by_group(df, "height_group")
weight_words = extract_top_words_by_group(df, "weight_group")

# 터미널 출력
combined = pd.concat([age_words, height_words, weight_words], ignore_index=True)
print(combined)

# 엑셀 저장
combined.to_excel("./results/age_height_weight_results.xlsx", index=False)

# 시각화 함수
def plot_top_words(df, group_type):
    subset = df[df["group_type"] == group_type]
    subset = subset[subset["group"].notna()]  # NaN 그룹 제거
    groups = subset["group"].unique()

    for group in groups:
        data = subset[subset["group"] == group].sort_values(by="count", ascending=False)
        plt.figure(figsize=(10, 5))
        sns.barplot(x="word", y="count", data=data)
        plt.title(f"{group_type} - {str(group)} 그룹 상위 단어")
        plt.xlabel("단어")
        plt.ylabel("빈도수")
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', linestyle='--', color='lightgray', linewidth=0.2)
        plt.tight_layout()
        plt.show()

# 각 그룹 시각화
plot_top_words(combined, "age_group")
plot_top_words(combined, "height_group")
plot_top_words(combined, "weight_group")