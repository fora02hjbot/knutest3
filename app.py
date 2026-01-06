import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# 페이지 설정
st.set_page_config(
    page_title="고객 이탈 예측 시스템",
    page_icon="📊",
    layout="wide"
)

# 제목
st.title("📊 통신사 고객 이탈 예측 시스템")
st.markdown("---")

# 모델 로드
@st.cache_resource
def load_model():
    """모델과 인코더를 로드하는 함수"""
    try:
        with open('churn_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ churn_model.pkl 파일을 찾을 수 없습니다. 먼저 analysis.ipynb에서 모델을 학습시켜주세요.")
        st.stop()

# 모델 로드
model_data = load_model()
rf_model = model_data['model']
label_encoders = model_data['label_encoders']
feature_names = model_data['feature_names']

# 사이드바에 모델 정보 표시
with st.sidebar:
    st.header("📈 모델 정보")
    st.metric("정확도", f"{model_data['accuracy']:.2%}")
    st.metric("F1-Score", f"{model_data['f1_score']:.4f}")
    st.markdown("---")
    st.info("💡 고객 정보를 입력하고 '이탈 예측하기' 버튼을 클릭하세요.")

# 메인 영역
col1, col2 = st.columns(2)

with col1:
    st.header("👤 고객 정보 입력")
    
    # 기본 정보
    st.subheader("기본 정보")
    gender = st.selectbox("성별", ["Male", "Female"])
    senior_citizen = st.selectbox("시니어 고객", [0, 1], format_func=lambda x: "예" if x == 1 else "아니오")
    partner = st.selectbox("파트너", ["Yes", "No"], format_func=lambda x: "있음" if x == "Yes" else "없음")
    dependents = st.selectbox("부양가족", ["Yes", "No"], format_func=lambda x: "있음" if x == "Yes" else "없음")
    
    st.markdown("---")
    
    # 서비스 정보
    st.subheader("서비스 정보")
    phone_service = st.selectbox("전화 서비스", ["Yes", "No"], format_func=lambda x: "사용" if x == "Yes" else "미사용")
    multiple_lines = st.selectbox("다중 회선", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("인터넷 서비스", ["DSL", "Fiber optic", "No"])
    
    # 인터넷 서비스가 있는 경우에만 부가 서비스 표시
    if internet_service != "No":
        online_security = st.selectbox("온라인 보안", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("온라인 백업", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("기기 보호", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("기술 지원", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("스트리밍 TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("스트리밍 영화", ["Yes", "No", "No internet service"])
    else:
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"
    
    st.markdown("---")
    
    # 계약 및 결제 정보
    st.subheader("계약 및 결제 정보")
    contract = st.selectbox("계약 형태", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("무인쇄 청구서", ["Yes", "No"], format_func=lambda x: "사용" if x == "Yes" else "미사용")
    payment_method = st.selectbox("결제 방법", [
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ])

with col2:
    st.header("💰 요금 정보")
    
    # 수치형 변수 - 슬라이더 사용
    tenure = st.slider(
        "가입 기간 (개월)", 
        min_value=0, 
        max_value=72, 
        value=12,
        help="고객이 서비스를 사용한 개월 수"
    )
    
    monthly_charges = st.slider(
        "월 요금 ($)", 
        min_value=18.0, 
        max_value=120.0, 
        value=65.0,
        step=0.1,
        help="월간 청구 금액"
    )
    
    total_charges = st.number_input(
        "총 요금 ($)", 
        min_value=0.0, 
        max_value=10000.0, 
        value=monthly_charges * tenure,
        step=0.1,
        help="고객이 지금까지 지불한 총 금액"
    )
    
    st.markdown("---")
    
    # 예측 버튼
    predict_button = st.button(
        "🔮 이탈 예측하기", 
        type="primary",
        use_container_width=True
    )
    
    # 예측 결과 표시 영역
    if predict_button:
        # 입력 데이터 준비
        input_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # DataFrame으로 변환
        input_df = pd.DataFrame([input_data])
        
        # 범주형 변수 인코딩 (모델이 기대하는 순서대로)
        encoded_data = {}
        for col in feature_names:
            if col in input_df.columns:
                if col in label_encoders:
                    # LabelEncoder를 사용하여 인코딩
                    le = label_encoders[col]
                    try:
                        input_value = input_df[col].iloc[0]
                        # 입력값이 학습 시 사용된 값인지 확인
                        if input_value in le.classes_:
                            encoded_data[col] = le.transform([input_value])[0]
                        else:
                            # 새로운 값인 경우 첫 번째 클래스로 매핑 (기본값)
                            st.warning(f"⚠️ {col}의 값 '{input_value}'이 모델 학습 시 사용되지 않았습니다. 기본값을 사용합니다.")
                            encoded_data[col] = 0
                    except Exception as e:
                        st.error(f"❌ {col} 인코딩 오류: {str(e)}")
                        encoded_data[col] = 0
                else:
                    # 수치형 변수는 그대로 사용
                    encoded_data[col] = float(input_df[col].iloc[0])
            else:
                st.error(f"❌ 필수 특성 '{col}'이 입력 데이터에 없습니다.")
                st.stop()
        
        # 예측을 위한 데이터 준비 (모델이 기대하는 순서대로)
        prediction_input = pd.DataFrame([encoded_data])
        prediction_input = prediction_input[feature_names]  # 특성 순서 맞추기
        
        # 예측 수행
        churn_probability = rf_model.predict_proba(prediction_input)[0]
        no_churn_prob = churn_probability[0]  # 유지 확률
        yes_churn_prob = churn_probability[1]  # 이탈 확률
        
        # 예측 결과 표시
        st.markdown("---")
        st.header("📊 예측 결과")
        
        # 확률 표시
        col_prob1, col_prob2 = st.columns(2)
        
        with col_prob1:
            st.metric(
                "유지 확률", 
                f"{no_churn_prob:.2%}",
                delta=f"{(no_churn_prob - 0.5)*100:.1f}%p" if no_churn_prob > 0.5 else None
            )
        
        with col_prob2:
            st.metric(
                "이탈 확률", 
                f"{yes_churn_prob:.2%}",
                delta=f"{(yes_churn_prob - 0.5)*100:.1f}%p" if yes_churn_prob > 0.5 else None,
                delta_color="inverse"
            )
        
        # 진행 바
        st.progress(yes_churn_prob)
        st.caption(f"이탈 위험도: {yes_churn_prob:.1%}")
        
        # 경고 메시지
        if yes_churn_prob >= 0.7:
            st.error("🚨 **위험**: 이탈 위험이 매우 높습니다! 즉시 고객 관리가 필요합니다.")
        elif yes_churn_prob >= 0.5:
            st.warning("⚠️ **주의**: 이탈 위험이 높습니다. 고객 만족도 조사 및 개선 조치를 권장합니다.")
        elif yes_churn_prob >= 0.3:
            st.info("ℹ️ **관찰**: 이탈 위험이 보통 수준입니다. 정기적인 모니터링이 필요합니다.")
        else:
            st.success("✅ **안전**: 이탈 위험이 낮습니다. 현재 고객 관리 상태를 유지하세요.")
        
        # 상세 정보
        with st.expander("📋 상세 정보 보기"):
            st.write("**입력된 고객 정보:**")
            st.json(input_data)
            st.write("**예측 확률:**")
            st.write(f"- 유지: {no_churn_prob:.4f}")
            st.write(f"- 이탈: {yes_churn_prob:.4f}")

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>통신사 고객 이탈 예측 시스템 | Powered by Random Forest</p>
    </div>
    """,
    unsafe_allow_html=True
)

