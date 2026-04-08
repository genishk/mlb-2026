# 🏆 MLB Analytics Performance Tracker - 배포 가이드

## 📋 개요

성과 트래커를 공개 웹사이트로 배포하는 가이드입니다.

## 🚀 배포 방법

### 1. **Streamlit Cloud 배포 (추천)**

#### 준비사항

- GitHub 계정
- Streamlit Cloud 계정 (무료)

#### 배포 단계

1. **GitHub 리포지토리 생성**

   ```bash
   # 새 리포지토리 생성 (GitHub에서)
   # 이름: mlb-performance-tracker
   ```

2. **필요한 파일들 업로드**

   ```
   mlb-performance-tracker/
   ├── src/
   │   ├── public_performance_tracker.py  # 메인 앱
   │   ├── simple_model_analyzer.py       # 분석 모듈
   │   └── odds/data/matched/            # 예측 데이터
   ├── data/records/                     # 히스토리 데이터
   ├── requirements.txt                  # 의존성
   ├── .streamlit/config.toml           # 설정
   └── README.md
   ```

3. **Streamlit Cloud에서 배포**
   - https://share.streamlit.io 접속
   - GitHub 연결
   - 리포지토리 선택: `mlb-performance-tracker`
   - 메인 파일: `src/public_performance_tracker.py`
   - 배포 클릭

#### 결과

- 공개 URL: `https://your-app-name.streamlit.app`
- 자동 업데이트: GitHub 푸시 시 자동 재배포

### 2. **Heroku 배포**

#### 추가 파일 필요

```bash
# Procfile 생성
echo "web: streamlit run src/public_performance_tracker.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# runtime.txt 생성
echo "python-3.9.18" > runtime.txt
```

#### 배포 명령

```bash
# Heroku CLI 설치 후
heroku create mlb-performance-tracker
git push heroku main
```

### 3. **로컬 테스트**

```bash
# 로컬에서 실행
streamlit run src/public_performance_tracker.py

# 브라우저에서 확인
# http://localhost:8501
```

## 📊 공개되는 정보

### ✅ 공개 정보

- 모든 모델의 성과 통계
- 일별 성과 분석
- 투명한 ROI, 승률 데이터
- 히스토리컬 퍼포먼스

### ❌ 비공개 정보

- 실제 픽 추천 (Statistical Insights)
- Zone 분석 로직
- 텔레그램 메시지 생성
- 모델 가중치 설정

## 🔧 커스터마이징

### 브랜딩 변경

```python
# public_performance_tracker.py 수정
page_title="🏆 [Your Brand] - Performance Tracker"
main_header="🏆 [Your Brand] Performance Tracker"
```

### 도메인 연결

- Streamlit Cloud: 커스텀 도메인 연결 가능
- Heroku: 쉬운 도메인 연결

## 📈 마케팅 활용

### 1. **SEO 최적화**

```python
# 메타 태그 추가
st.markdown("""
<meta name="description" content="Real-time MLB prediction model performance tracking">
<meta name="keywords" content="MLB predictions, machine learning, sports analytics">
""", unsafe_allow_html=True)
```

### 2. **소셜 미디어 공유**

- 성과 스크린샷 정기 공유
- 베스트 모델 하이라이트
- 투명성 강조

### 3. **신뢰성 구축**

- 모든 데이터 공개
- 실시간 업데이트 강조
- 독립적 검증 가능

## 🚨 주의사항

### 데이터 보안

- 유료 정보 절대 포함 금지
- 모델 로직 비공개
- 개인정보 보호

### 법적 고려사항

- 면책 조항 포함
- 교육 목적 명시
- 투자 조언 아님 표시

## 🔄 자동화

### GitHub Actions으로 자동 배포

```yaml
# .github/workflows/deploy.yml
name: Deploy to Streamlit
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Streamlit
        run: echo "Auto-deployed to Streamlit Cloud"
```

## 📞 지원

배포 과정에서 문제가 발생하면:

1. GitHub Issues 확인
2. Streamlit Community 포럼
3. Discord 지원 채널

---

## 🎯 다음 단계

1. 공개 성과 트래커 배포
2. 텔레그램 채널 연결
3. 구독자 유치 시작
4. 성과 기반 마케팅 진행
