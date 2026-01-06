## Importing Libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.font_manager as fm
import platform
from matplotlib import rc


# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 깃허브 리눅스 기준
if platform.system() == 'Linux':
    fontname = './NanumGothic.ttf'
    font_files = fm.findSystemFonts(fontpaths=fontname)
    fm.fontManager.addfont(fontname)
    fm._load_fontmanager(try_read_cache=False)
    rc('font', family='NanumGothic')

# Page Config
st.set_page_config(page_title='강원대 2026 캠프,
                   page_icon='😊',
                   layout='wide',
                   initial_sidebar_state='auto')  # 페이지 너비에 따라서 달라짐

## loading data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('tips.csv')  # CSV 파일 로드 시도
        return df                     # 성공 시 DataFrame 반환
    except FileNotFoundError:
        st.error("🚨 'tips.csv' 파일이 존재하지 않습니다. 파일을 업로드하거나 경로를 확인하세요.")
        return pd.DataFrame()         # 실패 시 빈 DataFrame 반환


df = load_data()
if df.empty:
    st.stop()

df.rename(columns={'sex': '성별'}, inplace=True)


# ---------------------
# Sidebar Filters
st.sidebar.header('🔍 Filters')
day_filter = st.sidebar.multiselect('요일 선택', options=df['day'].unique(), default=df['day'].unique())
sex_filter = st.sidebar.multiselect('성별 선택', options=df['성별'].unique(), default=df['성별'].unique())
smoker_filter = st.sidebar.multiselect('흡연 여부 선택', options=df['smoker'].unique(), default=df['smoker'].unique())
time_filter = st.sidebar.multiselect('식사 시간대 선택', options=df['time'].unique(), default=df['time'].unique())

# ---------------------
# 필터링
filtered_df = df[
    (df['day'].isin(day_filter)) &
    (df['성별'].isin(sex_filter)) &
    (df['smoker'].isin(smoker_filter)) &
    (df['time'].isin(time_filter))
]

st.sidebar.write('')

# 시각화 옵션
st.sidebar.header('🔍 시각화 옵션')
figure_type = st.sidebar.selectbox('시각화 형태 선택', ['px.scatter', 'px.bar', 'px.pie', 'px.donut'])
x_data = st.sidebar.selectbox('X축 데이터 선택', ['성별','smoker','day','size','time', 'total_bill'])
y_data = st.sidebar.selectbox('Y축 데이터 선택', ['total_bill','tip'])

st.sidebar.write('')



## body
st.title('💡 Streamlit을 활용한 Dashboard')
st.write('')
st.write('')

st.subheader('🍽️ Tips 데이터셋 소개')

with st.expander('📋 Tips 데이터 설명'):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("- **total_bill**: 전체 결제 금액")
        st.markdown("- **tip**: 팁 금액")
        st.markdown("- **성별**: 고객 성별 (Male/Female)")
        st.markdown("- **smoker**: 흡연자 여부 (Yes/No)")
        st.markdown("- **day**: 요일 (Thur, Fri, Sat, Sun)")

    with col2:
        st.markdown("- **time**: 식사 시간대 (Lunch/Dinner)")
        st.markdown("- **size**: 동반 인원 수")
        st.markdown("- **Payer Name**: 결제자 이름")
        st.markdown("- **price_per_person**: 1인당 결제 금액")

    st.markdown("💡 위 변수들을 기반으로 다양한 시각화 및 통계 분석을 할 수 있습니다.")



# ---------------------
# Raw Data
with st.expander(f'📊 필터링된 데이터 (총 {filtered_df.shape[0]}개 행)'):
    st.dataframe(filtered_df)

st.write('')
st.write('')

# ---------------------
# Metrics
st.subheader('📈 요약 통계')

col1, col2, col3, col4 = st.columns(4)
col1.metric('💰 총 결제금액 합계', round(filtered_df['total_bill'].sum(), 2))
col2.metric('💵 팁 평균', round(filtered_df['tip'].mean(), 2))
col3.metric('👥 평균 인원 수', round(filtered_df['size'].mean(), 2))
col4.metric('🍽️ 총 팁 수령 합계', round(filtered_df['tip'].sum(), 2))

st.write('')
st.write('')




# ---------------------
# 시각화
st.subheader('📉 시각화')
    
if figure_type == 'px.scatter':
    fig = px.scatter(filtered_df, x=x_data, y=y_data, color=x_data, size=y_data,
                     title=f'{x_data} vs {y_data} (산점도)')
    st.plotly_chart(fig, use_container_width=True)
elif figure_type == 'px.bar':
    fig = px.bar(filtered_df, x=x_data, y=y_data, color=x_data,
                 title=f'{x_data} 별 {y_data} (막대 그래프)')
    st.plotly_chart(fig, use_container_width=True)
elif figure_type == 'px.pie':
    fig = px.pie(filtered_df, names=x_data, values=y_data,
                 title=f'{x_data} 비율 (파이차트)')
    st.plotly_chart(fig, use_container_width=True)
elif figure_type == 'px.donut':
    fig = px.pie(filtered_df, names=x_data, values=y_data, hole=0.4,
                 title=f'{x_data} 비율 (도넛 차트)')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info('그래프 유형을 선택하세요.')


# ---------------------
# EDA 시각화 선택
st.subheader("🔍 탐색적 데이터 분석")  

eda_option = st.selectbox(
    "분석 항목을 선택하세요:",
    [
        "결제 금액과 팁 간의 상관관계",
        "요일별 평균 팁",
        "흡연 여부에 따른 팁 차이",
        "식사 시간대에 따른 결제 금액 차이",
        "동반 인원 수에 따른 팁 변화"
    ]
)

if eda_option == "결제 금액과 팁 간의 상관관계":
    fig = px.scatter(filtered_df, x='total_bill', y='tip', color='성별',
                     trendline='ols', title='Total Bill vs Tip')
    st.plotly_chart(fig, use_container_width=True)
    correlation = filtered_df['total_bill'].corr(filtered_df['tip'])
    st.markdown(f"📌 **결제 금액과 팁 간의 상관계수**: `{correlation:.2f}`")
    st.info("결제 금액이 높을수록 팁 금액도 증가하는 경향이 있습니다. 상관계수가 양의 값을 가지며, 두 변수 간의 선형 관계를 시사합니다.")

elif eda_option == "요일별 평균 팁":
    avg_tip_by_day = filtered_df.groupby('day')['tip'].mean().reset_index()
    fig = px.bar(avg_tip_by_day, x='day', y='tip', color='day',
                 title='요일별 평균 팁 금액')
    st.plotly_chart(fig, use_container_width=True)
    st.info("요일에 따라 고객이 남기는 팁 금액에 차이가 있습니다. 예를 들어, 주말에는 손님이 많거나 분위기가 더 좋아 팁이 많을 수 있습니다.")

elif eda_option == "흡연 여부에 따른 팁 차이":
    fig = px.box(filtered_df, x='smoker', y='tip', color='smoker',
                 title='흡연 여부에 따른 팁 분포 (Boxplot)')
    st.plotly_chart(fig, use_container_width=True)
    st.info("흡연자와 비흡연자 그룹 간의 팁 분포를 비교해볼 수 있습니다. 중간값과 이상치를 시각적으로 확인할 수 있습니다.")

elif eda_option == "식사 시간대에 따른 결제 금액 차이":
    fig = px.violin(filtered_df, x='time', y='total_bill', color='time',
                    box=True, points='all',
                    title='식사 시간대별 결제 금액 분포 (Violin Plot)')
    st.plotly_chart(fig, use_container_width=True)
    st.info("점심과 저녁 시간대의 결제 금액 분포를 비교해볼 수 있습니다. 일반적으로 저녁 시간대의 결제 금액이 더 높은 경향이 있습니다.")

elif eda_option == "동반 인원 수에 따른 팁 변화":
    avg_tip_by_size = filtered_df.groupby('size')['tip'].mean().reset_index()
    fig = px.line(avg_tip_by_size, x='size', y='tip', markers=True,
                  title='인원 수에 따른 평균 팁 변화')
    st.plotly_chart(fig, use_container_width=True)
    st.info("동반 인원 수가 많아질수록 팁이 증가하는 경향을 볼 수 있습니다. 하지만 일정 인원 수 이후에는 팁이 정체되거나 감소할 수도 있습니다.")