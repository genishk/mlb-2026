import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import sys
from pathlib import Path
import json

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.simple_model_analyzer import SimpleModelAnalyzer
import glob
import re

# 페이지 설정
st.set_page_config(
    page_title="🏆 MLB Analytics - Prediction Analytics",
    page_icon="🏆", 
    layout="wide"
)

# 모바일 반응형 CSS 추가 (가로 스크롤 제거)
st.markdown("""
<style>
/* 뷰포트 및 전체 컨테이너 설정 */
html, body {
    max-width: 100vw !important;
    overflow-x: hidden !important;
}

/* Streamlit 메인 컨테이너 */
.main .block-container {
    max-width: 100vw !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* 모든 이미지 반응형 */
img {
    max-width: 100% !important;
    height: auto !important;
}

/* 테이블 스크롤 처리 */
.stDataFrame {
    overflow-x: auto !important;
    max-width: 100% !important;
}

/* 차트 반응형 */
.js-plotly-plot {
    max-width: 100% !important;
}

/* 모바일 환경에서 패딩 조정 */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100vw !important;
    }
    
    /* 테이블 폰트 크기 조정 */
    .stDataFrame table {
        font-size: 12px !important;
    }
    
    /* 칼럼 간격 조정 */
    .css-1r6slb0 {
        gap: 0.5rem !important;
    }
}

/* 강제로 가로 스크롤 방지 */
* {
    box-sizing: border-box !important;
}

.element-container {
    max-width: 100% !important;
}

/* 부드러운 스크롤 (모든 브라우저 지원) */
html {
    scroll-behavior: smooth;
}

* {
    scroll-behavior: smooth;
}

/* Webkit 브라우저 (Safari, Chrome) 부드러운 스크롤 */
:root {
    scroll-behavior: smooth;
}

/* 모든 요소에 부드러운 스크롤 적용 */
.main, .block-container {
    scroll-behavior: smooth;
}

/* 맨 위로 이동 버튼 v2 */
.scroll-to-top {
    position: fixed !important;
    bottom: 80px !important;
    right: 20px !important;
    background: rgba(66, 133, 244, 0.9);
    color: white;
    border: none;
    border-radius: 50%;
    width: 45px;
    height: 45px;
    font-size: 16px;
    cursor: pointer;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    z-index: 1000;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    text-decoration: none;
}

.scroll-to-top:hover {
    background: rgba(66, 133, 244, 1);
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

@media (max-width: 768px) {
    .scroll-to-top {
        width: 40px !important;
        height: 40px !important;
        font-size: 14px !important;
        bottom: 75px !important;
        right: 15px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# 복리 재투자 계산 함수 추가
def calculate_compound_growth(daily_roi_data, initial_investment=1000):
    """일별 ROI 데이터를 기반으로 복리 재투자 효과 계산"""
    portfolio_value = initial_investment
    daily_values = []
    
    for date, roi_percent in daily_roi_data:
        # ROI를 소수로 변환 (8.5% -> 0.085)
        daily_return = roi_percent / 100
        # 복리 적용: 현재 자산 * (1 + 일일수익률)
        portfolio_value = portfolio_value * (1 + daily_return)
        daily_values.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'daily_roi': roi_percent,
            'cumulative_return': ((portfolio_value - initial_investment) / initial_investment) * 100
        })
    
    return daily_values

def find_best_model_compound_performance(results):
    """최고 성과 모델의 복리 성과 찾기"""
    if not results.get('model_performances') or not results.get('daily_performances'):
        return None, None, None
    
    # 최고 ROI 모델 찾기
    best_model = None
    best_roi = -float('inf')
    
    for model_name, performance in results['model_performances'].items():
        model_roi = performance.get('actual_roi', 0)
        if model_roi > best_roi:
            best_roi = model_roi
            best_model = model_name
    
    if not best_model or best_model not in results['daily_performances']:
        return None, None, None
    
    # 해당 모델의 일별 성과 데이터 가져오기
    daily_data = results['daily_performances'][best_model]
    
    # 날짜순으로 정렬된 일별 ROI 리스트 생성
    daily_roi_list = []
    for date in sorted(daily_data.keys()):
        roi = daily_data[date].get('roi', 0)
        daily_roi_list.append((date, roi))
    
    if not daily_roi_list:
        return None, None, None
    
    # 복리 계산
    compound_data = calculate_compound_growth(daily_roi_list)
    
    return best_model, compound_data, daily_roi_list

def create_compound_growth_chart(compound_data):
    """복리 성장 차트 생성 - 모바일/데스크탑 최적화"""
    if not compound_data:
        return None
    
    df = pd.DataFrame(compound_data)
    
    # 성장 차트 - 더 매력적인 색상과 스타일링
    fig = px.line(
        df, 
        x='date', 
        y='portfolio_value',
        title='💰 Compound Growth: Daily Reinvestment Strategy',
        labels={
            'date': 'Date',
            'portfolio_value': 'Portfolio Value ($)',
        }
    )
    
    # 시작점과 끝점 강조
    fig.add_hline(
        y=1000, 
        line_dash="dash", 
        line_color="rgba(255, 255, 255, 0.6)", 
        annotation_text="Initial $1,000",
        annotation_position="top left"
    )
    
    # 현대적이고 세련된 차트 스타일링
    fig.update_layout(
        height=450,
        showlegend=False,
        title={
            'text': '💰 Compound Growth',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': '#ffffff', 'family': 'Arial Black'}
        },
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='#1a1a2e',  # 어두운 네이비
        font=dict(color='white'),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)',
            showline=True,
            linecolor='rgba(255,255,255,0.5)',
            title_font=dict(size=14, color='#ffffff')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)',
            showline=True,
            linecolor='rgba(255,255,255,0.5)',
            title_font=dict(size=14, color='#ffffff'),
            tickformat='$,.0f'
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        # 모바일 반응형
        autosize=True
    )
    
    # 그라데이션 라인과 마커 스타일링  
    fig.update_traces(
        line=dict(
            width=4, 
            color='#00ff88',
            shape='spline',  # 부드러운 곡선
        ),
        mode='lines+markers',
        marker=dict(
            size=8,
            color='#00ff88',
            line=dict(width=2, color='rgba(255,255,255,0.9)')
        ),
        fill='tozeroy',  # 바닥부터 채우기
        fillcolor='rgba(0, 255, 136, 0.15)'
    )
    
    # 호버 템플릿 개선
    fig.update_traces(
        hovertemplate='<b>Date:</b> %{x}<br>' +
                    '<b>Portfolio Value:</b> $%{y:,.0f}<br>' +
                    '<extra></extra>'
    )
    
    return fig

def generate_social_media_content(best_model_name, compound_data, daily_roi_list, platform="twitter"):
    """SNS 홍보용 콘텐츠 생성"""
    if not best_model_name or not compound_data:
        return "No data available for social media content generation."
    
    initial_value = 1000
    final_value = compound_data[-1]['portfolio_value']
    total_return_percent = compound_data[-1]['cumulative_return']
    multiplier = final_value / initial_value
    days_count = len(compound_data)
    start_date = compound_data[0]['date']
    end_date = compound_data[-1]['date']
    
    # 베스트/워스트 데이 찾기
    best_day = max(compound_data, key=lambda x: x['daily_roi'])
    worst_day = min(compound_data, key=lambda x: x['daily_roi'])
    
    # 평균 일별 ROI 계산 (LinkedIn에서 사용)
    avg_daily_roi = sum(item['daily_roi'] for item in compound_data) / len(compound_data)
    
    if platform == "twitter":
        content = f"""🚨 AI DESTROYED THE MARKET 🚨

{best_model_name.upper()} AI: $1,000 → ${final_value:,.0f} in {days_count} days!
That's {multiplier:.1f}x growth with daily reinvestment 🚀

📈 +{total_return_percent:.0f}% return
🔥 Best day: +{best_day['daily_roi']:.1f}%

25 AI models + compound magic ✨

🤖 mlb-analytics-pro.streamlit.app
💎 dubclub.win/MLBAnalyticsPro

#MLB #AI #Compound"""

    elif platform == "instagram":
        content = f"""🚨 AI DEMOLISHED THE MARKET 🚨

Check this out - our {best_model_name.upper()} AI model just went VIRAL:

💰 Started with $1,000
📈 Ended with ${final_value:,.0f}
🔥 That's {multiplier:.1f}x growth in {days_count} days!

🎯 HOW?
✅ 25 different AI models working 24/7
✅ Daily compound reinvestment strategy  
✅ Complete transparency - every prediction tracked
✅ {total_return_percent:.0f}% total return!

📊 BEST DAY: +{best_day['daily_roi']:.1f}% 
📉 WORST DAY: {worst_day['daily_roi']:.1f}%

This is the power of AI + compound interest.

🔗 Free Dashboard: mlb-analytics-pro.streamlit.app
💎 VIP Channel: dubclub.win/MLBAnalyticsPro

⚠️ Educational content only. Past results don't guarantee future performance.

#MLB #AI #SportsAnalytics #MachineLearning #CompoundGrowth #DataScience #TechFinance"""

    elif platform == "reddit":
        content = f"""**{best_model_name.upper()} AI Model Results: $1,000 → ${final_value:,.0f} in {days_count} Days**

I've been tracking 25 different ML models for MLB predictions and wanted to share some transparent results. Here's what happened when I simulated daily compound reinvestment with our best-performing model:

**The Numbers:**
- Initial Investment: $1,000
- Final Value: ${final_value:,.0f}
- Growth Multiplier: {multiplier:.1f}x
- Total Return: +{total_return_percent:.0f}%
- Time Period: {start_date} to {end_date}
- Trading Days: {days_count}

**Daily Performance Highlights:**
- Best Day: +{best_day['daily_roi']:.1f}% on {best_day['date']}
- Worst Day: {worst_day['daily_roi']:.1f}% on {worst_day['date']}

**What Made This Work:**
1. 25 different ML models (CatBoost, LightGBM, XGBoost, Neural Networks, etc.)
2. Complete transparency - every prediction is tracked and verified
3. Daily compound reinvestment strategy
4. Focus on statistical edge over long term

**Important Disclaimers:**
- This is educational analysis only
- Past performance doesn't guarantee future results  
- All data is transparently available on our dashboard
- Not financial or betting advice

You can see the complete transparent analysis at: mlb-analytics-pro.streamlit.app

Happy to answer questions about the methodology or tech stack!"""

    elif platform == "linkedin":
        content = f"""🤖 AI & Compound Growth: A {days_count}-Day Case Study

I recently completed an analysis of 25 machine learning models applied to MLB predictions, and the compound growth results were remarkable.

**Key Findings:**
→ Model: {best_model_name.upper()}
→ Simulated Period: {start_date} to {end_date}
→ Compound Strategy: Daily reinvestment of all returns
→ Result: $1,000 → ${final_value:,.0f} ({total_return_percent:.0f}% return)

**Technical Approach:**
• Ensemble of 25 ML models (CatBoost, LightGBM, XGBoost, Neural Networks)
• Complete performance transparency and tracking
• Statistical edge identification through model consensus
• Risk management through diversified model portfolio

**The Compound Effect:**
Daily reinvestment turned an {avg_daily_roi:.1f}% average daily ROI into {multiplier:.1f}x portfolio growth. This demonstrates the mathematical power of compound returns when applied consistently.

**Volatility Analysis:**
• Best single day: +{best_day['daily_roi']:.1f}%
• Worst single day: {worst_day['daily_roi']:.1f}%
• Shows both the potential and risks involved

This analysis reinforces how data science and compound mathematics can create powerful outcomes when applied systematically.

Full transparent methodology available at: mlb-analytics-pro.streamlit.app

*Educational analysis - not investment advice. Past performance doesn't predict future results.*

#DataScience #MachineLearning #Analytics #CompoundGrowth #TechInnovation"""

    else:  # generic/tiktok
        content = f"""🤯 AI MODEL WENT VIRAL: $1,000 → ${final_value:,.0f}!

This {best_model_name.upper()} AI just proved compound interest is INSANE:

💰 Starting: $1,000
🚀 Ending: ${final_value:,.0f}  
📊 Growth: {multiplier:.1f}x in {days_count} days!
📈 Return: +{total_return_percent:.0f}%

🔥 Best day: +{best_day['daily_roi']:.1f}%
📉 Worst day: {worst_day['daily_roi']:.1f}%

25 AI models + daily reinvestment = COMPOUND MAGIC ✨

See the full transparent results:
🔗 mlb-analytics-pro.streamlit.app
💎 dubclub.win/MLBAnalyticsPro

⚠️ Educational only - past results ≠ future results!

#AI #Compound #Growth #MLB #Analytics"""

    return content

# 스타일 설정
st.markdown("""
<style>
    /* 프리미엄 메인 헤더 스타일링 - 이모티콘 보존 */
    .main-header {
        font-size: 3.2rem;
        font-weight: 900;
        text-align: center;
        color: #1d4ed8;
        margin-bottom: 2rem;
        text-shadow: 0 4px 8px rgba(29, 78, 216, 0.2);
        position: relative;
        padding: 1rem 0;
        animation: header-glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes header-glow {
        0% { filter: drop-shadow(0 0 10px rgba(29, 78, 216, 0.3)); }
        100% { filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.5)); }
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 60%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #3b82f6, transparent);
        border-radius: 1px;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 60%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #06b6d4, transparent);
        border-radius: 1px;
    }
    
    .subtitle {
        font-size: 1.4rem;
        font-weight: 600;
        text-align: center;
        color: #4b5563;
        margin-bottom: 3rem;
        padding: 0.5rem 0;
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        border: 1px solid #e5e7eb;
    }
    
    .footer {
        text-align: center;
        color: #6b7280;
        margin-top: 3rem;
        padding: 2rem;
        border-top: 2px solid #e5e7eb;
    }
    
    /* Color Focus - 트렌디한 색상 적용 */
    
         /* 메트릭 카드 임팩트 업그레이드 */
     .stMetric {
         background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
         border-radius: 16px;
         padding: 2.5rem 2rem;
         box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
         border: 1px solid #e2e8f0;
         transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
         margin: 1rem 0;
         position: relative;
         overflow: hidden;
     }
     
     .stMetric::before {
         content: '';
         position: absolute;
         top: 0;
         left: 0;
         right: 0;
         height: 4px;
         background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4, #10b981);
         background-size: 300% 100%;
         animation: gradient-shift 3s ease infinite;
     }
     
     @keyframes gradient-shift {
         0%, 100% { background-position: 0% 50%; }
         50% { background-position: 100% 50%; }
     }
     
     .stMetric:hover {
         transform: translateY(-4px) scale(1.02);
         box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
         border-color: #3b82f6;
     }
     
     /* 메트릭 값 임팩트 강화 */
     .stMetric [data-testid="metric-container"] > div:first-child {
         font-size: 3.5rem !important;
         font-weight: 800 !important;
         color: #1e293b !important;
         line-height: 1 !important;
         margin-bottom: 0.5rem !important;
         text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
     }
     
     /* 메트릭 라벨 개선 */
     .stMetric [data-testid="metric-container"] > div:last-child {
         font-size: 1rem !important;
         font-weight: 600 !important;
         color: #6b7280 !important;
         text-transform: uppercase;
         letter-spacing: 0.05em;
         margin-top: 0.5rem !important;
     }
     
     /* 메트릭 변화값 스타일링 */
     .stMetric [data-testid="metric-container"] .metric-delta {
         font-size: 1.1rem !important;
         font-weight: 600 !important;
         margin-top: 0.25rem !important;
     }
    
         /* 섹션 헤더 개선 - 메트릭 박스와 차별화 */
     h1, h2, h3 {
         color: #1e293b !important;
         font-weight: 700 !important;
     }
     
     h2 {
         color: #1e293b !important;
         font-size: 1.9rem !important;
         font-weight: 800 !important;
         margin-top: 3rem !important;
         margin-bottom: 2rem !important;
         padding-bottom: 0.8rem !important;
         border-bottom: 2px solid #e2e8f0 !important;
         letter-spacing: 0.02em !important;
         position: relative !important;
     }
     
     h2::after {
         content: '' !important;
         position: absolute !important;
         bottom: -2px !important;
         left: 0 !important;
         width: 80px !important;
         height: 2px !important;
         background: linear-gradient(90deg, #374151, #6b7280) !important;
         border-radius: 1px !important;
     }
     
     h3 {
         color: #374151 !important;
         font-size: 1.5rem !important;
         font-weight: 700 !important;
         margin-top: 2rem !important;
         margin-bottom: 1.2rem !important;
         padding-left: 1rem !important;
         border-left: 3px solid #94a3b8 !important;
         position: relative !important;
     }
     
     h3::before {
         content: '' !important;
         position: absolute !important;
         left: -3px !important;
         top: 0 !important;
         bottom: 0 !important;
         width: 3px !important;
         background: linear-gradient(180deg, #64748b, #94a3b8) !important;
         border-radius: 1px !important;
     }
     
     /* Footer 헤더는 박스 스타일 제거 */
     .footer h3 {
         background: none !important;
         border: none !important;
         box-shadow: none !important;
         padding: 0 !important;
         margin: 0.5rem 0 !important;
         text-align: center !important;
     }
     
     .footer h3::before {
         display: none !important;
     }
    
    /* 베스트 모델 특별 강조 */
    .element-container:has(.stMarkdown:contains("Top Performer")) {
        background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    
    /* 텔레그램/프리미엄 링크 트렌디한 스타일 */
    .stMarkdown a[href*="t.me"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white !important;
        padding: 10px 20px;
        border-radius: 20px;
        text-decoration: none !important;
        font-weight: 600;
        display: inline-block;
        margin: 5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stMarkdown a[href*="t.me"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    .stMarkdown a[href*="coff.ee"] {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white !important;
        padding: 10px 20px;
        border-radius: 20px;
        text-decoration: none !important;
        font-weight: 600;
        display: inline-block;
        margin: 5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
    }
    
    .stMarkdown a[href*="coff.ee"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.4);
    }
    
    /* 성과 테이블 프리미엄 스타일링 */
    .stDataFrame {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08) !important;
        border: 1px solid #e2e8f0 !important;
        margin: 1rem 0 !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stDataFrame::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 3px !important;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4, #10b981) !important;
        background-size: 300% 100% !important;
        animation: gradient-shift 3s ease infinite !important;
    }
    
    .stDataFrame table {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06) !important;
        border: none !important;
        background: white !important;
    }
    
    .stDataFrame th {
        background: linear-gradient(135deg, #1e293b 0%, #374151 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 18px 16px !important;
        font-size: 0.95rem !important;
        text-align: center !important;
        border: none !important;
        letter-spacing: 0.025em !important;
        text-transform: uppercase !important;
    }
    
    .stDataFrame td {
        padding: 14px 12px !important;
        text-align: center !important;
        border: none !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        position: relative !important;
    }
    
    .stDataFrame tr:nth-child(even) {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%) !important;
    }
    
    .stDataFrame tr:nth-child(odd) {
        background: white !important;
    }
    
    .stDataFrame tr:hover {
        background: linear-gradient(135deg, #e2e8f0 0%, #f1f5f9 100%) !important;
        transform: scale(1.005) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stDataFrame tbody tr {
        border-radius: 8px !important;
        margin: 2px 0 !important;
    }
    
         /* 날짜 입력 필드 프리미엄 스타일링 */
    .stDateInput > div[data-testid="stDateInput"] > div {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        border-radius: 12px !important;
        padding: 0.8rem 1.2rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06) !important;
        border: 1px solid #e2e8f0 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stDateInput > div[data-testid="stDateInput"] > div::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 3px !important;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4) !important;
        background-size: 200% 100% !important;
        animation: gradient-shift 2s ease infinite !important;
    }
    
    .stDateInput > div[data-testid="stDateInput"] > div:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1) !important;
        border-color: #3b82f6 !important;
    }
    
    .stDateInput input {
        font-weight: 600 !important;
        color: #1e293b !important;
        font-size: 1rem !important;
        border: none !important;
        background: transparent !important;
    }
    
    /* 텔레그램 픽 예시 카드 스타일링 */
    .pick-preview-section {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    .pick-detail {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        border: 2px solid #00d4ff;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.2);
        color: #ffffff;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .pick-detail:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.3);
        border-color: #00ff88;
    }
    
    .pick-detail strong {
        color: #00ff88;
        font-size: 1.1rem;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
    }
    
    .example-disclaimer {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        color: #92400e;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #fbbf24;
        margin-bottom: 1.5rem;
        font-weight: 600;
        text-align: center;
    }
    
    /* 차트 높이 최적화 - 데스크톱 & 모바일 */
    .plotly-graph-div {
        height: 350px !important;
    }
    
    /* Plotly 차트 컨테이너 최적화 */
    div[data-testid="stPlotlyChart"] {
        margin-bottom: 1rem !important;
    }
    
    /* 차트 제목 스타일링 */
    .plotly .plotlytitle {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    /* 중요한 숫자들 강조 */
    .stMarkdown strong {
        color: #1e293b;
        font-weight: 700;
    }
    
    /* ROI 양수/음수 색상 구분 - 강력한 적용 + 굵기 차등 */
    .stDataFrame td:contains("+"),
    .stDataFrame tbody td:contains("+"),
    table td:contains("+"),
    .dataframe td:contains("+") {
        color: #059669 !important;
        font-weight: 600 !important;
        background-color: rgba(5, 150, 105, 0.1) !important;
    }
    
    .stDataFrame td:contains("-"),
    .stDataFrame tbody td:contains("-"),
    table td:contains("-"),
    .dataframe td:contains("-") {
        color: #dc2626 !important;
        font-weight: 500 !important;
        background-color: rgba(220, 38, 38, 0.1) !important;
    }
    
    /* 낮은 양수 ROI (1-3%) - 보통 굵기 */
    .stDataFrame td:contains("+0."),
    .stDataFrame td:contains("+1."),
    .stDataFrame td:contains("+2."),
    .stDataFrame td:contains("+3.") {
        color: #16a34a !important;
        font-weight: 600 !important;
        background-color: rgba(22, 163, 74, 0.1) !important;
    }
    
    /* 중간 ROI (4-9%) - 더 굵게 */
    .stDataFrame td:contains("+4"),
    .stDataFrame td:contains("+5"),
    .stDataFrame td:contains("+6"),
    .stDataFrame td:contains("+7"),
    .stDataFrame td:contains("+8"),
    .stDataFrame td:contains("+9") {
        color: #15803d !important;
        font-weight: 700 !important;
        background-color: rgba(21, 128, 61, 0.15) !important;
    }
    
    /* 높은 ROI (10-19%) - 매우 굵게 */
    .stDataFrame td:contains("+1"),
    .stDataFrame td:contains("+2"),
    .stDataFrame td:contains("+3") {
        color: #166534 !important;
        font-weight: 800 !important;
        background-color: rgba(22, 163, 74, 0.2) !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* 매우 높은 ROI (20%+) - 최고 굵기 + 테두리 */
    .stDataFrame td:contains("+2"),
    .stDataFrame td:contains("+3"),
    .stDataFrame td:contains("+4"),
    .stDataFrame td:contains("+5") {
        color: #14532d !important;
        font-weight: 900 !important;
        background-color: rgba(21, 128, 61, 0.25) !important;
        border: 2px solid rgba(21, 128, 61, 0.4) !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
        border-radius: 4px !important;
    }
    
    /* 큰 손실 ROI (-10% 이하) - 더 굵게 강조 */
    .stDataFrame td:contains("-1"),
    .stDataFrame td:contains("-2"),
    .stDataFrame td:contains("-3"),
    .stDataFrame td:contains("-4"),
    .stDataFrame td:contains("-5") {
        color: #991b1b !important;
        font-weight: 800 !important;
        background-color: rgba(153, 27, 27, 0.15) !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* 에러/경고 메시지 트렌디한 색상 */
    .stAlert > div {
        border-radius: 8px !important;
    }
    
    .stSuccess > div {
        background-color: #dcfce7 !important;
        border-color: #16a34a !important;
        color: #15803d !important;
    }
    
    .stError > div {
        background-color: #fef2f2 !important;
        border-color: #dc2626 !important;
        color: #dc2626 !important;
    }
    
    .stWarning > div {
        background-color: #fefce8 !important;
        border-color: #ca8a04 !important;
        color: #a16207 !important;
    }
    
    .stInfo > div {
        background-color: #dbeafe !important;
        border-color: #3b82f6 !important;
        color: #1d4ed8 !important;
    }
    
    /* 스피너 색상 */
    .stSpinner > div {
        color: #3b82f6 !important;
    }
    
    /* 진행바 색상 */
    .stProgress .progress-bar {
        background-color: #3b82f6 !important;
    }
    
    /* 드롭다운 & 입력 필드 스타일링 - 임팩트 업그레이드 */
    
    /* 날짜 입력 필드 컨테이너 - 메트릭 카드 스타일 적용 */
    .stDateInput {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08) !important;
        border: 1px solid #e2e8f0 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin: 1rem 0 !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stDateInput::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 3px !important;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4, #10b981) !important;
        background-size: 300% 100% !important;
        animation: gradient-shift 3s ease infinite !important;
    }
    
    .stDateInput:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12) !important;
        border-color: #3b82f6 !important;
    }
    
    /* 날짜 입력 라벨 - 임팩트 스타일링 */
    .stDateInput > label {
        font-weight: 700 !important;
        color: #1e293b !important;
        margin-bottom: 12px !important;
        font-size: 1.1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* 날짜 입력 필드 - 프리미엄 스타일 */
    .stDateInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 18px 24px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        background: linear-gradient(135deg, #ffffff 0%, #fefefe 100%) !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.02) !important;
        height: 60px !important;
        line-height: 1.5 !important;
        min-height: 60px !important;
        color: #1e293b !important;
    }
    
    .stDateInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.15), inset 0 2px 4px rgba(0, 0, 0, 0.02) !important;
        outline: none !important;
        transform: scale(1.02) !important;
        background: #ffffff !important;
    }
    
    .stDateInput > div > div > input:hover {
        border-color: #6366f1 !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15), inset 0 2px 4px rgba(0, 0, 0, 0.02) !important;
        background: #ffffff !important;
    }
    
    /* 셀렉트박스 컨테이너 - 메트릭 카드 스타일 적용 */
    .stSelectbox {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08) !important;
        border: 1px solid #e2e8f0 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin: 1rem 0 !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stSelectbox::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 3px !important;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4, #10b981) !important;
        background-size: 300% 100% !important;
        animation: gradient-shift 3s ease infinite !important;
    }
    
    .stSelectbox:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12) !important;
        border-color: #3b82f6 !important;
    }
    
    /* 셀렉트박스 라벨 - 임팩트 스타일링 */
    .stSelectbox > label {
        font-weight: 700 !important;
        color: #1e293b !important;
        margin-bottom: 12px !important;
        font-size: 1.1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* 셀렉트박스 드롭다운 - 프리미엄 스타일 */
    .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        background: linear-gradient(135deg, #ffffff 0%, #fefefe 100%) !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.02) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        min-height: 60px !important;
        height: auto !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #6366f1 !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15), inset 0 2px 4px rgba(0, 0, 0, 0.02) !important;
        background: #ffffff !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.15), inset 0 2px 4px rgba(0, 0, 0, 0.02) !important;
        transform: scale(1.02) !important;
        background: #ffffff !important;
    }
    
    /* 셀렉트박스 내부 텍스트 - 프리미엄 스타일링 */
    .stSelectbox > div > div > div {
        padding: 18px 24px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        line-height: 1.5 !important;
        min-height: 24px !important;
        display: flex !important;
        align-items: center !important;
    }
    
    /* 드롭다운 화살표 */
    .stSelectbox > div > div > div:after {
        border-color: #6b7280 transparent transparent transparent !important;
    }
    
    /* 드롭다운 옵션 리스트 */
    .stSelectbox ul {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        background: white !important;
    }
    
    .stSelectbox ul li {
        padding: 12px 16px !important;
        font-size: 1rem !important;
        color: #374151 !important;
        transition: background-color 0.15s ease !important;
    }
    
    .stSelectbox ul li:hover {
        background-color: #f1f5f9 !important;
        color: #1e293b !important;
    }
    
    /* 입력 필드 라벨 개선 */
    .stDateInput > label,
    .stSelectbox > label {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #374151 !important;
        margin-bottom: 8px !important;
    }
    
    /* 컬럼 헤더에서 입력 필드들 간격 개선 */
    .stDateInput,
    .stSelectbox {
        margin-bottom: 1.5rem !important;
    }
    
    /* 🎬 Netflix-style Loading & Micro-interactions */
    .stSpinner > div > div {
        border-color: #667eea #e5e7eb #e5e7eb #e5e7eb !important;
        border-width: 3px !important;
        animation: pulse 1.5s ease-in-out infinite alternate !important;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        100% { transform: scale(1.05); opacity: 0.7; }
    }
    
    /* Enhanced Section Dividers */
    hr {
        border: none !important;
        height: 3px !important;
        background: linear-gradient(90deg, transparent, #667eea, transparent) !important;
        margin: 3rem 0 2rem 0 !important;
        border-radius: 2px !important;
    }
    
    /* Premium Section Headers */
    .main .block-container h2,
    .main .block-container h3 {
        background: linear-gradient(135deg, #1e293b, #334155);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 2rem 0 1.5rem 0;
        box-shadow: 0 8px 25px rgba(30, 41, 59, 0.3);
        border-left: 4px solid #667eea;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    /* Smooth Page Transitions */
    .main .block-container {
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Interactive Elements Enhancement */
    .stSelectbox:hover,
    .stDateInput:hover,
    .stButton:hover {
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }
    
    /* Premium Table Styling */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    /* Enhanced Info Boxes */
    .stInfo {
        background: linear-gradient(135deg, #dbeafe, #bfdbfe) !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15) !important;
        border-left: 4px solid #3b82f6 !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7, #fde68a) !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.15) !important;
        border-left: 4px solid #f59e0b !important;
    }
    
    /* Plotly Chart Enhancement */
    .js-plotly-plot {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1) !important;
        transition: transform 0.3s ease !important;
    }
    
    .js-plotly-plot:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 20px 45px rgba(0,0,0,0.15) !important;
    }
    
    /* 모바일 반응형 스타일 - 완전히 새로운 프로페셔널 모바일 UX */
    @media (max-width: 768px) {
        /* === 기본 타이포그래피 === */
        body, .main, .stApp {
            font-size: 16px !important; /* 모바일 기본 읽기 가능한 크기 */
            line-height: 1.5 !important;
        }
        
        /* 헤더들 - 더 큰 크기로 */
        .main-header {
            font-size: 2.5rem !important;
            margin-bottom: 1rem !important;
            line-height: 1.2 !important;
        }
        
        .subtitle {
            font-size: 1rem !important;
            margin-bottom: 1.5rem !important;
            line-height: 1.4 !important;
        }
        
        h1 { font-size: 2rem !important; line-height: 1.2 !important; }
        h2 { font-size: 1.6rem !important; line-height: 1.3 !important; }
        h3 { font-size: 1.4rem !important; line-height: 1.3 !important; }
        h4 { font-size: 1.2rem !important; line-height: 1.3 !important; }
        
        /* 일반 텍스트 - 충분한 크기 */
        p, div, span {
            font-size: 1rem !important;
            line-height: 1.5 !important;
        }
        
        /* === 메트릭 카드 최적화 === */
        div[data-testid="column"] {
            flex: 0 0 48% !important;
            max-width: 48% !important;
            margin-bottom: 1.5rem !important;
        }
        
        .stMetric {
            padding: 1.5rem 1rem !important;
            margin: 0.8rem 0 !important;
            border-radius: 12px !important;
        }
        
        /* 메트릭 값 (숫자) - 잘 보이게 */
        .stMetric [data-testid="metric-container"] > div:first-child,
        .stMetric div[data-testid="metric-value"] {
            font-size: 2.2rem !important;
            line-height: 1.1 !important;
            font-weight: 700 !important;
        }
        
        /* 메트릭 라벨 - 읽기 좋게 */
        .stMetric [data-testid="metric-container"] > div:last-child,
        .stMetric div[data-testid="metric-label"] {
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            line-height: 1.4 !important;
            margin-top: 0.5rem !important;
        }
        
        /* === 테이블 최적화 - 실제 읽을 수 있는 크기 === */
        .stDataFrame th,
        .stDataFrame thead th,
        table th,
        .dataframe th {
            font-size: 0.9rem !important;
            padding: 12px 8px !important;
            font-weight: 700 !important;
            line-height: 1.4 !important;
            min-height: 44px !important; /* 터치 영역 확보 */
        }
        
        .stDataFrame td,
        .stDataFrame tbody td,
        table td,
        .dataframe td {
            font-size: 0.85rem !important;
            padding: 10px 8px !important;
            line-height: 1.5 !important;
            min-height: 40px !important;
        }
        
        .stDataFrame,
        .stDataFrame table,
        table,
        .dataframe {
            font-size: 0.85rem !important;
        }
        
        /* === UI 컨트롤 최적화 - 터치 친화적 === */
        .stSelectbox > div > div,
        .stSelectbox select {
            font-size: 1rem !important;
            line-height: 1.5 !important;
            padding: 12px 16px !important;
            min-height: 48px !important; /* Apple/Google 권장 터치 영역 */
        }
        
        .stDateInput > div > div > input {
            font-size: 1rem !important;
            padding: 12px 16px !important;
            min-height: 48px !important;
        }
        
        .stButton > button {
            font-size: 1rem !important;
            padding: 12px 24px !important;
            min-height: 48px !important;
            border-radius: 8px !important;
        }
        
        /* === 픽 카드 최적화 === */
        .pick-detail {
            font-size: 0.95rem !important;
            padding: 1.5rem !important;
            line-height: 1.6 !important;
            margin: 1rem 0 !important;
            border-radius: 12px !important;
        }
        
        .pick-detail span,
        .pick-detail strong {
            font-size: 1rem !important;
            line-height: 1.5 !important;
        }
        
        /* === 적절한 간격 === */
        .stSelectbox, 
        .stDateInput {
            margin-bottom: 1.5rem !important;
        }
        
        .stDateInput + .stDateInput {
            margin-top: 0.5rem !important;
        }
        
        div[data-testid="column"] .stMetric {
            margin: 1rem 0 !important;
        }
        
        /* === 정보 박스들 === */
        .stInfo, .stWarning, .stError, .stSuccess {
            font-size: 0.95rem !important;
            line-height: 1.5 !important;
            padding: 1rem !important;
        }
        
        /* === 차트 컨테이너 === */
        .js-plotly-plot {
            margin: 1rem 0 !important;
        }
        
        /* === 반응형 그리드 개선 === */
        .metric-container {
            gap: 1.5rem !important;
            margin: 1rem 0 !important;
        }
        
        /* === 스크롤 개선 === */
        .stDataFrame {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }
        
        /* === 포커스 상태 개선 === */
        .stSelectbox select:focus,
        .stDateInput input:focus,
        .stButton button:focus {
            outline: 2px solid #3b82f6 !important;
            outline-offset: 2px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_analyzer():
    """Load analyzer (cached)"""
    return SimpleModelAnalyzer()

@st.cache_data
def run_analysis(start_date, end_date):
    """Run analysis (cached)"""
    analyzer = load_analyzer()
    return analyzer.analyze(start_date, end_date)

class DailyPicksAnalyzer:
    """일별 픽 성과 분석기"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.picks_dir = self.project_root / "data" / "picks"
        self.records_dir = self.project_root / "data" / "records"
    
    def get_available_pick_dates(self):
        """픽이 있는 날짜들 반환"""
        try:
            pattern = str(self.picks_dir / "daily_picks_*.json")
            files = glob.glob(pattern)
            
            dates = []
            for file in files:
                # daily_picks_20250717_192053.json에서 날짜 추출
                match = re.search(r'daily_picks_(\d{8})_\d{6}\.json', file)
                if match:
                    date_str = match.group(1)
                    # YYYYMMDD -> YYYY-MM-DD 변환
                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    dates.append(formatted_date)
            
            return sorted(list(set(dates)))  # 중복 제거 후 정렬
            
        except Exception as e:
            st.error(f"픽 날짜 로드 오류: {e}")
            return []
    
    def load_picks_for_date(self, target_date):
        """특정 날짜의 픽 데이터 로드"""
        try:
            # YYYY-MM-DD -> YYYYMMDD 변환
            date_str = target_date.replace('-', '')
            pattern = str(self.picks_dir / f"daily_picks_{date_str}_*.json")
            files = glob.glob(pattern)
            
            if not files:
                return None
            
            # 가장 최신 파일 선택 (시간순)
            latest_file = max(files, key=lambda x: re.search(r'(\d{6})\.json', x).group(1))
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            st.error(f"픽 데이터 로드 오류: {e}")
            return None
    
    def load_historical_records_for_date(self, target_date):
        """특정 날짜의 실제 경기 결과 로드 (가장 최신 파일 사용)"""
        try:
            # 가장 최신 historical records 파일 사용 (SimpleModelAnalyzer와 동일)
            pattern = str(self.records_dir / "mlb_historical_records_*.json")
            files = glob.glob(pattern)
            
            if not files:
                return None
            
            # 가장 최신 파일 선택 (파일 수정 시간 기준)
            latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            return None
    
    def match_picks_with_results(self, picks_data, historical_data):
        """픽과 실제 결과 매칭"""
        if not picks_data or not historical_data or not picks_data.get('picks'):
            return picks_data
        
        # historical_data 인덱싱 (SimpleModelAnalyzer와 동일한 방식)
        historical_index = {}
        for record in historical_data:
            if all(key in record for key in ['date', 'home_team_name', 'away_team_name']):
                key = f"{record['date']}_{record['home_team_name']}_{record['away_team_name']}"
                historical_index[key] = record
        
        # 실제 결과 매칭
        matched_count = 0
        total_picks = len(picks_data['picks'])
        
        for pick in picks_data['picks']:
            # 기본값 설정
            pick['actual_result'] = 'No Data'
            pick['actual_roi'] = 0
            pick['is_correct'] = None
            
            game_info = pick.get('game_info', '')  # "Seattle Mariners @ Detroit Tigers"
            game_date = pick.get('game_date', '')  # "2025-07-13"
            predicted_team = pick.get('predicted_team', '')
            selection_odds = pick.get('selection_odds', 0)
            
            if not game_info or not predicted_team or not game_date:
                continue
            
            # game_info를 파싱해서 매칭 키 생성: "2025-07-13_Detroit Tigers_Seattle Mariners"
            try:
                parts = game_info.split(' @ ')
                if len(parts) != 2:
                    continue
                away_team = parts[0].strip()
                home_team = parts[1].strip()
                matching_key = f"{game_date}_{home_team}_{away_team}"
            except Exception as e:
                continue
            
            # 안전한 타입 변환
            try:
                selection_odds = float(selection_odds) if selection_odds is not None else 0
            except (ValueError, TypeError):
                selection_odds = 0
            
            # 히스토리컬 데이터에서 매칭 찾기
            if matching_key in historical_index:
                matched_count += 1
                record = historical_index[matching_key]
                
                # SimpleModelAnalyzer와 동일한 방식으로 home_win 확인
                if 'home_win' not in record:
                    continue
                
                actual_result = record['home_win']  # 이미 0 또는 1 값
                
                # 예측 정확도 확인
                if predicted_team == 'home':
                    is_correct = actual_result == 1
                elif predicted_team == 'away':
                    is_correct = actual_result == 0
                else:
                    continue  # 잘못된 predicted_team
                
                # 실제 ROI 계산 (SimpleModelAnalyzer 로직 참조)
                if is_correct and selection_odds != 0:
                    if selection_odds > 0:
                       # 양수 배당: +150 -> 1.5배
                       actual_roi = (selection_odds / 100) * 100
                    else:
                       # 음수 배당: -150 -> 0.67배
                       actual_roi = (100 / abs(selection_odds)) * 100
                else:
                    actual_roi = -100  # 틀렸거나 배당이 0인 경우 전액 손실
                
                # 결과 업데이트
                pick['actual_result'] = 'Win' if is_correct else 'Loss'
                pick['actual_roi'] = actual_roi
                pick['is_correct'] = is_correct
        
        # 매칭 통계 추가
        picks_data['matching_stats'] = {
            'total_picks': total_picks,
            'matched_picks': matched_count,
            'unmatched_picks': total_picks - matched_count,
            'historical_records_count': len(historical_data),
            'unique_game_keys': len(historical_index)
        }
        
        return picks_data
    
    def analyze_daily_picks_performance(self, target_date):
        """특정 날짜 픽 성과 분석"""
        # 픽 데이터 로드
        picks_data = self.load_picks_for_date(target_date)
        if not picks_data:
            return None
        
        # 픽에서 실제 게임 날짜들 추출
        game_dates = set()
        for pick in picks_data.get('picks', []):
            if pick.get('game_date'):
                game_dates.add(pick['game_date'])
        
        # 모든 게임 날짜의 historical records 로드
        historical_data = []
        for game_date in game_dates:
            game_historical = self.load_historical_records_for_date(game_date)
            if game_historical:
                historical_data.extend(game_historical)
        
        # 픽과 결과 매칭
        matched_data = self.match_picks_with_results(picks_data, historical_data)
        
        # 성과 통계 계산
        picks = matched_data.get('picks', [])
        total_picks = len(picks)
        
        if total_picks == 0:
            return matched_data
        
        # 결과가 있는 픽들만 분석 (안전한 필터링)
        analyzed_picks = []
        for p in picks:
            actual_result = p.get('actual_result')
            if actual_result and actual_result not in [None, 'No Data']:
                analyzed_picks.append(p)
        
        if analyzed_picks:
            # 안전한 계산
            correct_picks = 0
            total_predicted_roi = 0
            total_actual_roi = 0
            
            for p in analyzed_picks:
                # is_correct 안전하게 체크
                if p.get('is_correct') is True:
                    correct_picks += 1
                
                # ROI 안전하게 계산
                predicted_roi = p.get('predicted_roi', 0)
                actual_roi = p.get('actual_roi', 0)
                
                try:
                    total_predicted_roi += float(predicted_roi) if predicted_roi is not None else 0
                    total_actual_roi += float(actual_roi) if actual_roi is not None else 0
                except (ValueError, TypeError):
                    continue
            
            accuracy = (correct_picks / len(analyzed_picks)) * 100
            avg_predicted_roi = total_predicted_roi / len(analyzed_picks) if len(analyzed_picks) > 0 else 0
            avg_actual_roi = total_actual_roi / len(analyzed_picks) if len(analyzed_picks) > 0 else 0
            
            # 통계 추가
            matched_data['performance_stats'] = {
                'total_picks': total_picks,
                'analyzed_picks': len(analyzed_picks),
                'correct_picks': correct_picks,
                'accuracy': accuracy,
                'avg_predicted_roi': avg_predicted_roi,
                'avg_actual_roi': avg_actual_roi,
                'total_predicted_roi': total_predicted_roi,
                'total_actual_roi': total_actual_roi
            }
        else:
            # 결과 데이터가 없는 경우
            matched_data['performance_stats'] = {
                'total_picks': total_picks,
                'analyzed_picks': 0,
                'correct_picks': 0,
                'accuracy': 0,
                'avg_predicted_roi': 0,
                'avg_actual_roi': 0,
                'total_predicted_roi': 0,
                'total_actual_roi': 0
            }
        
        return matched_data

@st.cache_data
def load_daily_picks_analyzer():
    """픽 분석기 로드 (캐시됨)"""
    return DailyPicksAnalyzer()

def analyze_model_segments(matched_data, model_name):
    """개별 모델의 구간별 성과 분석 (심플 대시보드 기준과 동일)"""
    prob_key = f'{model_name}_probability'
    
    # 해당 모델의 데이터만 필터링
    model_data = [r for r in matched_data if r.get(prob_key) is not None and r.get(prob_key) > 0]
    
    if not model_data:
        return None
    
    # 분석 데이터 준비 (앙상블 분석과 동일한 방식)
    analysis_data = []
    for record in model_data:
        prob = record[prob_key]
        actual_home_win = record.get('actual_home_win', 0)
        home_odds = record.get('home_odds')
        away_odds = record.get('away_odds')
        
        if home_odds is None or away_odds is None:
            continue
        
        try:
            home_odds = float(home_odds)
            away_odds = float(away_odds)
        except (ValueError, TypeError):
            continue
        
        # 예측 결정
        predicted_home_win = 1 if prob > 0.5 else 0
        predicted_team = 'home' if predicted_home_win else 'away'
        
        # 예측 정보 계산
        selection_odds = home_odds if predicted_home_win else away_odds
        unit_amount = 100  # 심플 대시보드와 동일한 분석 단위
        
        # 예측 ROI 계산 (배당률 기반)
        if selection_odds > 0:
            decimal_odds = (selection_odds / 100) + 1
        else:
            decimal_odds = (100 / abs(selection_odds)) + 1
        
        # 예측한 팀의 승률
        if predicted_home_win:
            win_prob = prob
        else:
            win_prob = 1 - prob
        
        predicted_roi = (win_prob * decimal_odds - 1) * 100
        
        # 실제 ROI 계산
        is_correct = predicted_home_win == actual_home_win
        if is_correct:
            if selection_odds > 0:
                profit = unit_amount * (selection_odds / 100)
            else:
                profit = unit_amount * (100 / abs(selection_odds))
            actual_roi = (profit / unit_amount) * 100
        else:
            actual_roi = -100  # 전액 손실
        
        confidence = abs(prob - 0.5)  # 심플 대시보드와 동일한 신뢰도 계산
        
        analysis_data.append({
            'ensemble_prob': prob,
            'predicted_roi': predicted_roi,
            'actual_roi': actual_roi,
            'confidence': confidence,
            'selection_odds': selection_odds,
            'predicted_team': predicted_team,
            'actual_win': actual_home_win,
            'home_odds': home_odds,
            'away_odds': away_odds
        })
    
    if not analysis_data:
        return None
    
    # 구간별 분석 수행 (심플 대시보드와 동일)
    segments = {}
    
    # 1. 예측 ROI 구간별 분석
    segments['predicted_roi'] = analyze_predicted_roi_segments(analysis_data)
    
    # 2. 배당률 구간별 분석
    segments['odds'] = analyze_odds_segments(analysis_data)
    
    # 3. 신뢰도 구간별 분석
    segments['confidence'] = analyze_confidence_segments(analysis_data)
    
    # 4. Kelly Criterion 분석
    segments['kelly'] = analyze_kelly_segments(analysis_data)
    
    # 5. 시장 vs 모델 괴리도 분석
    segments['market_divergence'] = analyze_market_divergence_segments(analysis_data)
    
    return segments

def analyze_predicted_roi_segments(analysis_data):
    """예측 ROI 구간별 분석 (심플 대시보드와 동일)"""
    segments = {
        'Very Negative (<-20%)': [],
        'Negative (-20% ~ 0%)': [],
        'Positive (0% ~ 20%)': [],
        'Very Positive (20% ~ 60%)': [],
        'Extremely Positive (>60%)': []
    }
    
    for data in analysis_data:
        pred_roi = data['predicted_roi']
        if pred_roi < -20:
            segments['Very Negative (<-20%)'].append(data)
        elif pred_roi < 0:
            segments['Negative (-20% ~ 0%)'].append(data)
        elif pred_roi < 20:
            segments['Positive (0% ~ 20%)'].append(data)
        elif pred_roi < 60:
            segments['Very Positive (20% ~ 60%)'].append(data)
        else:
            segments['Extremely Positive (>60%)'].append(data)
    
    return calculate_segment_performance(segments)

def analyze_confidence_segments(analysis_data):
    """신뢰도 구간별 분석 (심플 대시보드와 동일)"""
    segments = {
        'Low Confidence (0-0.05)': [],
        'Medium Confidence (0.05-0.15)': [],
        'High Confidence (0.15-0.25)': [],
        'Very High Confidence (>0.25)': []
    }
    
    for data in analysis_data:
        confidence = data['confidence']
        
        if confidence < 0.05:
            segments['Low Confidence (0-0.05)'].append(data)
        elif confidence < 0.15:
            segments['Medium Confidence (0.05-0.15)'].append(data)
        elif confidence < 0.25:
            segments['High Confidence (0.15-0.25)'].append(data)
        else:
            segments['Very High Confidence (>0.25)'].append(data)
    
    return calculate_segment_performance(segments)

def analyze_odds_segments(analysis_data):
    """배당률별 성과 분석"""
    segments = {
        'Heavy Favorite (< -200)': [],
        'Favorite (-200 ~ -120)': [],
        'Pick Em (-120 ~ +120)': [],
        'Underdog (+120 ~ +150)': [],
        'Strong Underdog (+150 ~ +300)': [],
        'Heavy Underdog (> +300)': []
    }
    
    for data in analysis_data:
        odds = data['selection_odds']
        
        if odds < -200:
            segments['Heavy Favorite (< -200)'].append(data)
        elif odds < -120:
            segments['Favorite (-200 ~ -120)'].append(data)
        elif -120 <= odds <= 120:
            segments['Pick Em (-120 ~ +120)'].append(data)
        elif odds <= 150:
            segments['Underdog (+120 ~ +150)'].append(data)
        elif odds <= 300:
            segments['Strong Underdog (+150 ~ +300)'].append(data)
        else:
            segments['Heavy Underdog (> +300)'].append(data)
    
    return calculate_segment_performance(segments)

def analyze_market_divergence_segments(analysis_data):
    """시장 vs 모델 괴리도 분석 (심플 대시보드와 동일)"""
    segments = {
        'Model Much More Optimistic (+10%+)': [],
        'Model Slightly Optimistic (+5% ~ +10%)': [],
        'Market Aligned (-5% ~ +5%)': [],
        'Model Slightly Pessimistic (-10% ~ -5%)': [],
        'Model Much More Pessimistic (-10%--)': []
    }
    
    for data in analysis_data:
        selection_odds = data['selection_odds']
        ensemble_prob = data['ensemble_prob']
        predicted_team = data['predicted_team']
        
        # 시장 배당률을 확률로 변환
        if selection_odds > 0:
            market_implied_prob = 100 / (selection_odds + 100)
        else:
            market_implied_prob = abs(selection_odds) / (abs(selection_odds) + 100)
        
        # 모델 확률 (예측 팀 기준으로 조정)
        if predicted_team == 'home':
            model_prob = ensemble_prob
        else:
            model_prob = 1 - ensemble_prob
        
        # 괴리도 계산 (모델 확률 - 시장 확률)
        divergence = model_prob - market_implied_prob
        
        # 구간 분류
        if divergence >= 0.10:
            segments['Model Much More Optimistic (+10%+)'].append(data)
        elif divergence >= 0.05:
            segments['Model Slightly Optimistic (+5% ~ +10%)'].append(data)
        elif -0.05 <= divergence < 0.05:
            segments['Market Aligned (-5% ~ +5%)'].append(data)
        elif divergence >= -0.10:
            segments['Model Slightly Pessimistic (-10% ~ -5%)'].append(data)
        else:
            segments['Model Much More Pessimistic (-10%--)'].append(data)
    
    return calculate_segment_performance(segments)

def analyze_kelly_segments(analysis_data):
    """Kelly Criterion 분석 (심플 대시보드와 동일)"""
    segments = {
        'No Selection (Kelly ≤ 0%)': [],
        'Low Confidence (0% < Kelly ≤ 5%)': [],
        'Medium Confidence (5% < Kelly ≤ 15%)': [],
        'High Confidence (15% < Kelly ≤ 25%)': [],
        'Very High Confidence (25% ~ 60%)': [],
        'Extremely High Confidence (Kelly > 60%)': []
    }
    
    for data in analysis_data:
        ensemble_prob = data['ensemble_prob']
        predicted_team = data['predicted_team']
        
        # 선택한 팀의 확률과 배당률
        if predicted_team == 'home':
            win_prob = ensemble_prob
            selection_odds = data['home_odds']
        else:
            win_prob = 1 - ensemble_prob
            selection_odds = data['away_odds']
        
        # 배당률을 decimal odds로 변환
        if selection_odds > 0:
            decimal_odds = (selection_odds / 100) + 1
        else:
            decimal_odds = (100 / abs(selection_odds)) + 1
        
        # Kelly Criterion 계산
        p = win_prob
        q = 1 - win_prob
        b = decimal_odds - 1
        
        kelly_fraction = (p * b - q) / b if b > 0 else 0
        kelly_percentage = max(0, kelly_fraction) * 100
        
        if kelly_percentage <= 0:
            segments['No Selection (Kelly ≤ 0%)'].append(data)
        elif kelly_percentage <= 5:
            segments['Low Confidence (0% < Kelly ≤ 5%)'].append(data)
        elif kelly_percentage <= 15:
            segments['Medium Confidence (5% < Kelly ≤ 15%)'].append(data)
        elif kelly_percentage <= 25:
            segments['High Confidence (15% < Kelly ≤ 25%)'].append(data)
        elif kelly_percentage <= 60:
            segments['Very High Confidence (25% ~ 60%)'].append(data)
        else:
            segments['Extremely High Confidence (Kelly > 60%)'].append(data)
    
    return calculate_segment_performance(segments)

def calculate_segment_performance(segments):
    """구간별 성과 계산 (심플 대시보드와 동일)"""
    results = {}
    
    for segment_name, segment_data in segments.items():
        if not segment_data:
            results[segment_name] = {
                'games': 0,
                'predicted_roi': 0,
                'actual_roi': 0,
                'roi_difference': 0,
                'win_rate': 0,
                'accuracy': 0
            }
            continue
        
        games = len(segment_data)
        predicted_roi = sum(d['predicted_roi'] for d in segment_data) / games
        actual_roi = sum(d['actual_roi'] for d in segment_data) / games
        roi_difference = actual_roi - predicted_roi
        
        # 승률 계산 (실제 수익이 난 경우)
        wins = sum(1 for d in segment_data if d['actual_roi'] > 0)
        win_rate = (wins / games) * 100 if games > 0 else 0
        
        # 정확도 계산 (예측한 팀이 실제로 이긴 경우)
        correct_predictions = 0
        for d in segment_data:
            if d['predicted_team'] == 'home' and d['actual_win'] == 1:
                correct_predictions += 1
            elif d['predicted_team'] == 'away' and d['actual_win'] == 0:
                correct_predictions += 1
        
        accuracy = (correct_predictions / games) * 100 if games > 0 else 0
        
        results[segment_name] = {
            'games': games,
            'predicted_roi': predicted_roi,
            'actual_roi': actual_roi,
            'roi_difference': roi_difference,
            'win_rate': win_rate,
            'accuracy': accuracy
        }
    
    return results

def display_segment_performance_table(segment_data, segment_type):
    """구간별 성과 테이블 표시 (심플 대시보드와 동일)"""
    if not segment_data:
        st.warning(f"No {segment_type} data available.")
        return
    
    # 데이터 정리
    table_data = []
    for segment_name, data in segment_data.items():
        if data['games'] > 0:
            table_data.append({
                segment_type: segment_name,
                'Games': data['games'],
                'Predicted ROI (%)': data['predicted_roi'],
                'Actual ROI (%)': data['actual_roi'],
                'ROI Difference (%)': data['roi_difference'],
                'Win Rate (%)': data['win_rate'],
                'Accuracy (%)': data['accuracy']
            })
    
    if not table_data:
        st.warning(f"No {segment_type} data with games available.")
        return
    
    df = pd.DataFrame(table_data)
    
    # ROI에 따른 색상 함수
    def highlight_segment_roi(val):
        if val > 5:
            return 'background-color: #d4edda; color: #155724'
        elif val < -5:
            return 'background-color: #f8d7da; color: #721c24'
        else:
            return 'background-color: #fff3cd; color: #856404'
    
    # 테이블 스타일링 (프리미엄 스타일 + 칼럼 폭 최적화)
    styled_df = df.style.map(
        highlight_segment_roi, 
        subset=['Actual ROI (%)']
    ).format({
        'Predicted ROI (%)': '{:.2f}%',
        'Actual ROI (%)': '{:.2f}%',
        'ROI Difference (%)': '{:.2f}%',
        'Win Rate (%)': '{:.1f}%',
        'Accuracy (%)': '{:.1f}%'
    }).set_table_styles([
        # 테이블 전체 스타일
        {'selector': '', 'props': [
            ('width', '100%'),
            ('table-layout', 'auto'),
            ('border-collapse', 'collapse'),
            ('border-radius', '12px'),
            ('overflow', 'hidden'),
            ('box-shadow', '0 4px 20px rgba(0,0,0,0.08)')
        ]},
        # 헤더 스타일
        {'selector': 'th', 'props': [
            ('background', 'linear-gradient(135deg, #f8fafc, #e2e8f0)'),
            ('color', '#1e293b'),
            ('font-weight', '700'),
            ('padding', '12px 8px'),
            ('text-align', 'center'),
            ('border', 'none'),
            ('font-size', '0.9rem'),
            ('white-space', 'nowrap'),
            ('min-width', 'fit-content')
        ]},
        # 데이터 셀 스타일
        {'selector': 'td', 'props': [
            ('padding', '10px 8px'),
            ('text-align', 'center'),
            ('border', 'none'),
            ('font-size', '0.85rem'),
            ('white-space', 'nowrap'),
            ('min-width', 'fit-content')
        ]},
        # 짝수 행 스타일
        {'selector': 'tr:nth-child(even)', 'props': [
            ('background-color', '#f8fafc')
        ]},
        # 호버 효과
        {'selector': 'tr:hover', 'props': [
            ('background-color', '#e2e8f0'),
            ('transition', 'all 0.2s ease'),
            ('transform', 'scale(1.001)')
        ]},
        # 첫 번째 컬럼 (세그먼트 이름) 좀 더 넓게
        {'selector': 'th:first-child, td:first-child', 'props': [
            ('text-align', 'left'),
            ('padding-left', '12px'),
            ('min-width', '180px'),
            ('max-width', '220px')
        ]},
        # 숫자 컬럼들 컴팩트하게
        {'selector': 'th:not(:first-child), td:not(:first-child)', 'props': [
            ('min-width', '80px'),
            ('max-width', '100px')
        ]}
    ])
    
    st.dataframe(styled_df, hide_index=True, use_container_width=True)

def main():
    # 페이지 상단 앵커 추가
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    
    # 배너 이미지 추가
    try:
        st.image("images/KakaoTalk_20250727_162246740.jpg", 
                caption="MLB Analytics Pro - Advanced Statistical Analysis", 
                use_container_width=True)
    except:
        pass  # 이미지가 없어도 계속 진행
    
    # 배너와 헤더 사이 간격
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 헤더
    st.markdown('<div class="main-header">🏆 MLB Prediction Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Real-time transparent analysis of 25 ML prediction models</div>', unsafe_allow_html=True)
    
    # 텔레그램 연결
    st.markdown("---")
    st.markdown("""
    ### 📱 Join Our Community
    
    **🔔 Get Real-time Updates & Analysis**
    
    <div style="display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap;">
        <a href="https://t.me/mlbanalytics_free" target="_blank" style="text-decoration: none; margin: 0; padding: 0;">
            <div style="background: linear-gradient(135deg, #4285f4, #2563eb); color: white; padding: 0.75rem 1.5rem; border-radius: 25px; display: inline-block; font-weight: 600; box-shadow: 0 4px 12px rgba(66, 133, 244, 0.3); transition: all 0.3s ease; border: none; min-width: 150px; text-align: center; margin: 0;">
                📊 Free Previews
            </div>
        </a>
        <a href="https://t.me/mlbanalytics_community" target="_blank" style="text-decoration: none; margin: 0; padding: 0;">
            <div style="background: linear-gradient(135deg, #4285f4, #2563eb); color: white; padding: 0.75rem 1.5rem; border-radius: 25px; display: inline-block; font-weight: 600; box-shadow: 0 4px 12px rgba(66, 133, 244, 0.3); transition: all 0.3s ease; border: none; min-width: 150px; text-align: center; margin: 0;">
                💬 Community Discussion
            </div>
        </a>
        <a href="https://dubclub.win/MLBAnalyticsPro/" target="_blank" style="text-decoration: none;">
            <div style="background: linear-gradient(135deg, #dc2626, #b91c1c); color: white; padding: 0.75rem 1.5rem; border-radius: 25px; display: inline-block; font-weight: 600; box-shadow: 0 4px 12px rgba(220, 38, 38, 0.4); transition: all 0.3s ease; border: none; min-width: 150px; text-align: center;">
                💎 Premium VIP Access
            </div>
        </a>
        <a href="https://buymeacoffee.com/mlbanalyticspro" target="_blank" style="text-decoration: none;">
            <div style="background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 0.75rem 1.5rem; border-radius: 25px; display: inline-block; font-weight: 600; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4); transition: all 0.3s ease; border: none; min-width: 150px; text-align: center;">
                ☕ Buy Me A Coffee
            </div>
        </a>
    </div>
    
    • **Free Channel**: Medium-confidence prediction analysis and insights  
    • **Premium VIP**: High-confidence predictions with detailed methodology  
    • **Community**: Discuss predictions and connect with fellow analysts  
    • **Support**: Help keep this project running with a coffee donation
    """, unsafe_allow_html=True)
    
    # 텔레그램 픽 예시 섹션
    st.markdown("---")
    st.markdown("### 🎯 Sample Daily Prediction Format")
    
    # 📱 Telegram 스타일 CSS 추가
    st.markdown("""
    <style>
    .telegram-pick-detail {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        border: 2px solid #00d4ff;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.2);
        color: #ffffff;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .telegram-pick-detail:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.3);
        border-color: #00ff88;
    }
    
    .telegram-pick-detail strong {
        color: #ffffff;
        font-size: 1.1rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .telegram-analysis-result {
        color: #00ff88;
        font-weight: bold;
        font-size: 1.1rem;
        text-shadow: 0 0 15px rgba(0, 255, 136, 0.5);
        background: rgba(0, 255, 136, 0.1);
        padding: 4px 8px;
        border-radius: 6px;
        border-left: 3px solid #00ff88;
        margin: 4px 0;
        display: block;
        word-break: keep-all;
        overflow-wrap: break-word;
    }
    
    .telegram-pick-detail {
        line-height: 1.6;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.info("""
    📱 **Join our Telegram channels to receive daily predictions like this!**
    
    ⚠️ These are sample formats only. For today's actual predictions, visit our Telegram channels.
    
    💡 Not financial advice - Educational/analytical content for demonstration purposes.
    """)
    
    # 예시 픽 카드들 - Telegram Dashboard 스타일로 수정
    st.markdown("""
    <div class="telegram-pick-detail">
        <strong>#1 Seattle Mariners @ Detroit Tigers</strong><br>
        <div class="telegram-analysis-result">
            🎯 Analysis Result: Seattle Mariners (-108.0)
        </div>
        🎲 Statistical Probability: 62.8% | 📈 ROI Projection: +20.9% | 
        💡 Confidence Level: 0.128<br>
        ⏰ 2025-02-25 13:40:00 EST
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="telegram-pick-detail">
        <strong>#2 Atlanta Braves @ St. Louis Cardinals</strong><br>
        <div class="telegram-analysis-result">
            🎯 Analysis Result: Atlanta Braves (+136.0)
        </div>
        🎲 Statistical Probability: 52.8% | 📈 ROI Projection: +24.7% | 
        💡 Confidence Level: 0.028<br>
        ⏰ 2025-02-25 14:15:00 EST
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="telegram-pick-detail">
        <strong>#3 Los Angeles Dodgers @ San Francisco Giants</strong><br>
        <div class="telegram-analysis-result">
            🎯 Analysis Result: Los Angeles Dodgers (-145.0)
        </div>
        🎲 Statistical Probability: 68.4% | 📈 ROI Projection: +18.2% | 
        💡 Confidence Level: 0.184<br>
        ⏰ 2025-02-25 16:05:00 EST
    </div>
    """, unsafe_allow_html=True)
    
    # 설명
    st.markdown("---")
    st.markdown("""
    ### 📊 About This Tracker
    
    This is a **completely transparent** prediction analysis system that monitors 25 machine learning models 
    for MLB game analysis. All results are independently verified and updated in real-time.
    
    - **🔍 100% Transparency**: Every prediction tracked and recorded
    - **📈 Real Performance**: No cherry-picking, all data included  
    - **🤖 25 ML Models**: SVM, Neural Networks, CatBoost, LightGBM, XGBoost variants
    - **📅 Historical Data**: Complete track record since inception
    """)
    
    # 날짜 선택
    st.markdown("---")
    st.markdown("## 📅 Select Analysis Period")
    
    col1, col2 = st.columns(2)
    
    try:
        analyzer = load_analyzer()
        available_dates = analyzer.get_available_dates()
        
        if available_dates:
            min_date = datetime.strptime(available_dates[0], '%Y-%m-%d').date()
            max_date = datetime.strptime(available_dates[-1], '%Y-%m-%d').date()
            
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )
                
            with col2:
                end_date = st.date_input(
                    "End Date", 
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )
        else:
            st.error("No data available")
            return
            
    except Exception as e:
        st.error(f"Error loading dates: {e}")
        return
    
    # 분석 실행
    if start_date <= end_date:
        with st.spinner("📊 Analyzing model performance..."):
            try:
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                
                results = run_analysis(start_date_str, end_date_str)
                
                if not results or not results.get('model_performances'):
                    st.error("No performance data available for the selected period.")
                    return
                
                # 기본 통계
                st.markdown("---")
                st.markdown("## 📊 Performance Summary")
                
                total_games = results.get('analysis_summary', {}).get('total_games', 0)
                analysis_period = (end_date - start_date).days
                models_count = len(results.get('model_performances', {}))
                best_roi = max([model.get('actual_roi', 0) for model in results['model_performances'].values()], default=0)
                
                # 🔥 Netflix-style Impact Metrics Cards
                st.markdown("""
                <style>
                .impact-metric-card {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem 1.5rem;
                    border-radius: 20px;
                    color: white;
                    text-align: center;
                    box-shadow: 0 20px 40px rgba(102, 126, 234, 0.4);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    backdrop-filter: blur(10px);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    margin: 0.5rem;
                }
                
                .impact-metric-card:hover {
                    transform: translateY(-8px);
                    box-shadow: 0 30px 60px rgba(102, 126, 234, 0.6);
                }
                
                .metric-number {
                    font-size: 2.8rem;
                    font-weight: 900;
                    line-height: 0.9;
                    margin-bottom: 0.5rem;
                    text-shadow: 0 4px 8px rgba(0,0,0,0.3);
                    background: linear-gradient(45deg, #ffffff, #f0f9ff);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    word-break: keep-all;
                    overflow: hidden;
                }
                
                .metric-label {
                    font-size: 1rem;
                    font-weight: 600;
                    opacity: 0.9;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-top: 0.5rem;
                }
                
                .metric-games { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
                .metric-period { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
                .metric-models { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
                .metric-roi { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
                
                @media (max-width: 768px) {
                    .impact-metric-card {
                        padding: 1.5rem 1rem;
                        margin: 0.3rem;
                    }
                                         .metric-number {
                         font-size: 2.2rem;
                     }
                    .metric-label {
                        font-size: 0.9rem;
                    }
                }
                </style>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="impact-metric-card metric-games">
                        <div class="metric-number">{total_games:,}</div>
                        <div class="metric-label">Total Games Analyzed</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                    <div class="impact-metric-card metric-period">
                        <div class="metric-number">{analysis_period}</div>
                        <div class="metric-label">Analysis Period (Days)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"""
                    <div class="impact-metric-card metric-models">
                        <div class="metric-number">{models_count}</div>
                        <div class="metric-label">AI Models Tracked</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col4:
                    roi_color = "#43e97b" if best_roi > 0 else "#f5576c"
                    st.markdown(f"""
                    <div class="impact-metric-card metric-roi">
                        <div class="metric-number" style="color: {roi_color};">+{best_roi:.1f}%</div>
                        <div class="metric-label">Best Model ROI</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 🔥 복리 재투자 효과 섹션 추가
                st.markdown("---")
                best_model_name, compound_data, daily_roi_list = find_best_model_compound_performance(results)
                
                if best_model_name and compound_data:
                    st.markdown("## 🚀 **COMPOUND GROWTH POWER: What If You Reinvested Everything?**")
                    
                    # 핵심 수치들 계산
                    initial_value = 1000
                    final_value = compound_data[-1]['portfolio_value']
                    total_return_percent = compound_data[-1]['cumulative_return']
                    multiplier = final_value / initial_value
                    days_count = len(compound_data)
                    start_date_comp = compound_data[0]['date']
                    end_date_comp = compound_data[-1]['date']
                    
                    # 🔥 프리미엄 헤더 카드 - 모바일 반응형
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                               padding: 1.5rem 1rem; border-radius: 20px; margin: 1rem 0 1.5rem 0; 
                               box-shadow: 0 20px 40px rgba(102, 126, 234, 0.4);
                               border: 1px solid rgba(255, 255, 255, 0.2);
                               backdrop-filter: blur(10px);">
                       <h3 style="color: white; text-align: center; margin: 0; font-size: clamp(1.2rem, 4vw, 1.8rem);
                                  font-weight: 900; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                           💰 FOLLOWING {best_model_name.upper()} WITH DAILY REINVESTMENT
                       </h3>
                       <p style="color: rgba(255, 255, 255, 0.9); text-align: center; margin: 0.5rem 0 0 0; 
                                 font-size: clamp(0.9rem, 3vw, 1.2rem); font-weight: 500;">
                           {start_date_comp} to {end_date_comp} • {days_count} Trading Days
                       </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 반응형 컬럼 - 모바일에서는 2x2, 데스크탑에서는 1x4
                    st.markdown("""
                    <style>
                    @media (max-width: 768px) {
                       .metric-container {
                           display: grid !important;
                           grid-template-columns: 1fr 1fr !important;
                           gap: 1rem !important;
                           margin: 0.2rem 0 !important;
                       }
                    }
                    @media (min-width: 769px) {
                       .metric-container {
                           display: grid !important;
                           grid-template-columns: repeat(4, 1fr) !important;
                           gap: 1rem !important;
                           margin: 0.2rem 0 !important;
                       }
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-container" style="margin: 0.5rem 0 1rem 0;">
                       <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                                   padding: 1.5rem 1rem; border-radius: 16px; text-align: center; color: white;
                                   box-shadow: 0 12px 28px rgba(16, 185, 129, 0.35);
                                   border: 1px solid rgba(255, 255, 255, 0.2);
                                   transition: transform 0.3s ease;">
                           <div style="font-size: clamp(0.7rem, 2.5vw, 0.9rem); font-weight: 600; 
                                       letter-spacing: 1px; margin-bottom: 0.5rem; opacity: 0.9;">
                               INITIAL INVESTMENT
                           </div>
                           <div style="font-size: clamp(1.4rem, 4vw, 1.8rem); font-weight: 900; 
                                       text-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                               ${initial_value:,.0f}
                           </div>
                       </div>
                       
                       <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                                   padding: 1.5rem 1rem; border-radius: 16px; text-align: center; color: white;
                                   box-shadow: 0 12px 28px rgba(245, 158, 11, 0.35);
                                   border: 1px solid rgba(255, 255, 255, 0.2);
                                   transition: transform 0.3s ease;">
                           <div style="font-size: clamp(0.7rem, 2.5vw, 0.9rem); font-weight: 600; 
                                       letter-spacing: 1px; margin-bottom: 0.5rem; opacity: 0.9;">
                               FINAL VALUE
                           </div>
                           <div style="font-size: clamp(1.4rem, 4vw, 1.8rem); font-weight: 900; 
                                       text-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                               ${final_value:,.0f}
                           </div>
                       </div>
                       
                       <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                                   padding: 1.5rem 1rem; border-radius: 16px; text-align: center; color: white;
                                   box-shadow: 0 12px 28px rgba(239, 68, 68, 0.35);
                                   border: 1px solid rgba(255, 255, 255, 0.2);
                                   transition: transform 0.3s ease;">
                           <div style="font-size: clamp(0.7rem, 2.5vw, 0.9rem); font-weight: 600; 
                                       letter-spacing: 1px; margin-bottom: 0.5rem; opacity: 0.9;">
                               GROWTH MULTIPLIER
                           </div>
                           <div style="font-size: clamp(1.4rem, 4vw, 1.8rem); font-weight: 900; 
                                       text-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                               {multiplier:.1f}x
                           </div>
                       </div>
                       
                       <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); 
                                   padding: 1.5rem 1rem; border-radius: 16px; text-align: center; color: white;
                                   box-shadow: 0 12px 28px rgba(139, 92, 246, 0.35);
                                   border: 1px solid rgba(255, 255, 255, 0.2);
                                   transition: transform 0.3s ease;">
                           <div style="font-size: clamp(0.7rem, 2.5vw, 0.9rem); font-weight: 600; 
                                       letter-spacing: 1px; margin-bottom: 0.5rem; opacity: 0.9;">
                               TOTAL RETURN
                           </div>
                           <div style="font-size: clamp(1.4rem, 4vw, 1.8rem); font-weight: 900; 
                                       text-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                               +{total_return_percent:.0f}%
                           </div>
                       </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 복리 성장 차트
                    st.markdown("""
                    <div style="margin: 0.5rem 0;">
                        <h3 style="color: #ffffff; margin: 0; font-size: clamp(1.2rem, 4vw, 1.6rem);">
                            📈 Compound Growth Visualization
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    compound_chart = create_compound_growth_chart(compound_data)
                    if compound_chart:
                       st.plotly_chart(compound_chart, use_container_width=True)
                    
                    # 강력한 마케팅 메시지
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); 
                               padding: 1.5rem; border-radius: 15px; margin: 1rem 0;
                               box-shadow: 0 10px 25px rgba(251, 191, 36, 0.3);">
                       <h3 style="color: #92400e; text-align: center; margin: 0; font-size: 1.4rem;">
                           🎯 THIS IS THE POWER OF COMPOUND REINVESTMENT
                       </h3>
                       <p style="color: #92400e; text-align: center; margin: 0.75rem 0; font-size: 1.1rem; font-weight: bold;">
                           While most people bet the same amount every day, our AI models show what happens 
                           when you reinvest your winnings daily - turning $1,000 into ${final_value:,.0f}!
                       </p>
                       <p style="color: #92400e; text-align: center; margin: 0; font-size: 1rem;">
                           ⚠️ Past performance doesn't guarantee future results. For educational analysis only.
                       </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 컴파운드 섹션 완료
                
                # 모델별 성과 테이블
                st.markdown("---")
                st.markdown("## 🏆 Model Performance Rankings")
                
                # 모델 이름 축약 함수
                def shorten_model_name(model_name):
                    """모델 이름을 간단하게 축약"""
                    name_map = {
                       'model_svm': 'SVM',
                       'model_nn': 'Neural Net',
                       'model_rf': 'Random Forest',
                       'model_advanced_catboost': 'Adv.CatBoost',
                       'model_advanced_catboost_basic': 'CatBoost Basic',
                       'model_advanced_lgbm': 'Adv.LightGBM',
                       'model_advanced_lgbm_basic': 'LightGBM Basic',
                       'model_advanced_xgboost': 'Adv.XGBoost',
                       'model_advanced_xgboost_basic': 'XGBoost Basic',
                       'model_advanced_nn': 'Adv.Neural Net',
                       'model_advanced_rf': 'Adv.Random Forest',
                       'model_advanced_svm': 'Adv.SVM',
                       'model1_extended_lgbm': 'Model1-LGBM',
                       'model2_extended_catboost': 'Model2-CatBoost',
                       'model3_extended_xgboost': 'Model3-XGBoost'
                    }
                    
                    if model_name in name_map:
                       return name_map[model_name]
                    
                    # 일반적인 패턴 처리
                    if model_name.startswith('model'):
                       # model1, model2 등
                       if model_name[5:].isdigit():
                           return f"Model {model_name[5:]}"
                       # model_으로 시작하는 경우
                       return model_name.replace('model_', '').replace('_', ' ').title()
                    
                    return model_name
                
                # 데이터 정리
                model_data = []
                for model_name, model_info in results['model_performances'].items():
                    short_name = shorten_model_name(model_name)
                    model_data.append({
                       'Model': short_name,
                       'Full_Model': model_name,  # 원본 이름 보관
                       'ROI (%)': model_info.get('actual_roi', 0),
                       'Win Rate (%)': model_info.get('win_rate', 0),
                       'Total Predictions': model_info.get('total_bets', 0),
                       'Wins': model_info.get('correct_predictions', 0),
                       'Losses': model_info.get('total_bets', 0) - model_info.get('correct_predictions', 0),
                       'Profit/Loss ($)': model_info.get('profit_loss', 0),
                       'Total Invested ($)': model_info.get('total_invested', 0)
                    })
                
                df = pd.DataFrame(model_data)
                df = df.sort_values('ROI (%)', ascending=False)
                
                # 베스트 모델 하이라이트
                best_model = df.iloc[0]
                st.markdown(f"### 🏆 **{best_model['Model']} - Top Performer**")
                
                # 🏆 Top Performer Metrics Cards
                st.markdown("""
                <style>
                .top-performer-card {
                    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                    padding: 1.6rem 1.2rem;
                    border-radius: 16px;
                    color: white;
                    text-align: center;
                    box-shadow: 0 15px 35px rgba(245, 158, 11, 0.4);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    transition: all 0.3s ease;
                    margin: 0.3rem;
                    position: relative;
                    overflow: hidden;
                }
                
                .top-performer-card::before {
                    content: '🏆';
                    position: absolute;
                    top: 0.5rem;
                    right: 0.5rem;
                    font-size: 1.2rem;
                    opacity: 0.3;
                }
                
                .top-performer-card:hover {
                    transform: translateY(-4px) scale(1.01);
                    box-shadow: 0 25px 50px rgba(245, 158, 11, 0.6);
                }
                
                .top-performer-number {
                    font-size: 2.2rem;
                    font-weight: 800;
                    line-height: 1;
                    margin-bottom: 0.6rem;
                    text-shadow: 0 3px 6px rgba(0,0,0,0.3);
                }
                
                .top-performer-label {
                    font-size: 0.8rem;
                    font-weight: 600;
                    opacity: 0.9;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                
                .roi-champion { color: #fbbf24; }
                .winrate-champion { color: #34d399; }
                .predictions-champion { color: #60a5fa; }
                .profit-champion { color: #a78bfa; }
                
                @media (max-width: 768px) {
                    .top-performer-card {
                        padding: 1.2rem 0.9rem;
                        margin: 0.2rem;
                    }
                    .top-performer-number {
                        font-size: 1.8rem;
                    }
                    .top-performer-label {
                        font-size: 0.75rem;
                    }
                }
                </style>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="top-performer-card">
                        <div class="top-performer-number roi-champion">{best_model['ROI (%)']:.2f}%</div>
                        <div class="top-performer-label">ROI</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                    <div class="top-performer-card">
                        <div class="top-performer-number winrate-champion">{best_model['Win Rate (%)']:.1f}%</div>
                        <div class="top-performer-label">Win Rate</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"""
                    <div class="top-performer-card">
                        <div class="top-performer-number predictions-champion">{best_model['Total Predictions']}</div>
                        <div class="top-performer-label">Total Predictions</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col4:
                    profit_loss = best_model['Profit/Loss ($)']
                    st.markdown(f"""
                    <div class="top-performer-card">
                        <div class="top-performer-number profit-champion">${profit_loss:.1f}</div>
                        <div class="top-performer-label">Profit/Loss</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 전체 모델 성과 테이블 (개선된 스타일링)
                st.markdown("### 📊 Complete Model Rankings")
                
                # 더 모던한 ROI 하이라이팅
                def highlight_roi(val):
                    if val > 10:
                       return 'background: linear-gradient(135deg, #d1fae5, #a7f3d0); color: #065f46; font-weight: 600; border-radius: 6px; padding: 4px 8px;'
                    elif val > 0:
                       return 'background: linear-gradient(135deg, #ecfdf5, #d1fae5); color: #047857; font-weight: 500; border-radius: 6px; padding: 4px 8px;'
                    elif val > -5:
                       return 'background: linear-gradient(135deg, #fef3c7, #fde68a); color: #92400e; font-weight: 500; border-radius: 6px; padding: 4px 8px;'
                    else:
                       return 'background: linear-gradient(135deg, #fee2e2, #fecaca); color: #991b1b; font-weight: 600; border-radius: 6px; padding: 4px 8px;'
                
                # Full_Model 컬럼 제거하고 스타일링
                display_df = df.drop('Full_Model', axis=1)
                
                styled_df = display_df.style.map(
                    highlight_roi, 
                    subset=['ROI (%)']
                ).format({
                    'ROI (%)': '{:.2f}%',
                    'Win Rate (%)': '{:.1f}%',
                    'Profit/Loss ($)': '${:.1f}',
                    'Total Invested ($)': '${:.0f}'
                }).set_table_styles([
                    # 테이블 전체 스타일
                    {'selector': '', 'props': [
                       ('width', '100%'),
                       ('table-layout', 'auto'),
                       ('border-collapse', 'collapse'),
                       ('border-radius', '12px'),
                       ('overflow', 'hidden'),
                       ('box-shadow', '0 4px 20px rgba(0,0,0,0.08)')
                    ]},
                    # 헤더 스타일
                    {'selector': 'th', 'props': [
                       ('background', 'linear-gradient(135deg, #f8fafc, #e2e8f0)'),
                       ('color', '#1e293b'),
                       ('font-weight', '700'),
                       ('padding', '12px 8px'),
                       ('text-align', 'center'),
                       ('border', 'none'),
                       ('font-size', '0.9rem'),
                       ('white-space', 'nowrap')
                    ]},
                    # 데이터 셀 스타일
                    {'selector': 'td', 'props': [
                       ('padding', '10px 8px'),
                       ('text-align', 'center'),
                       ('border', 'none'),
                       ('font-size', '0.85rem'),
                       ('white-space', 'nowrap')
                    ]},
                    # 짝수 행 스타일
                    {'selector': 'tr:nth-child(even)', 'props': [
                       ('background-color', '#f8fafc')
                    ]},
                    # 호버 효과
                    {'selector': 'tr:hover', 'props': [
                       ('background-color', '#e2e8f0'),
                       ('transition', 'all 0.2s ease'),
                       ('transform', 'scale(1.001)')
                    ]},
                    # 모델 이름 컬럼 (첫 번째) 좀 더 넓게
                    {'selector': 'th:first-child, td:first-child', 'props': [
                       ('text-align', 'left'),
                       ('padding-left', '12px'),
                       ('min-width', '120px'),
                       ('max-width', '180px')
                    ]},
                    # ROI 컬럼 강조
                    {'selector': 'th:nth-child(2), td:nth-child(2)', 'props': [
                       ('min-width', '80px'),
                       ('font-weight', '600')
                    ]},
                    # 다른 숫자 컬럼들 컴팩트하게
                    {'selector': 'th:nth-child(n+3), td:nth-child(n+3)', 'props': [
                       ('min-width', '75px'),
                       ('max-width', '110px')
                    ]}
                ])
                
                st.dataframe(styled_df, hide_index=True, use_container_width=True)
                
                # 차트 (개선된 시각화)
                st.markdown("---")
                st.markdown("### 📈 Performance Visualization")
                
                # 차트용 데이터 준비 (상위 10개 모델만)
                chart_df = df.head(10).copy()
                
                # ROI 차트 - 개선된 스타일링
                fig_roi = px.bar(
                    chart_df, 
                    x='Model', 
                    y='ROI (%)',
                    title='Top Models ROI Comparison',
                    color='ROI (%)',
                    color_continuous_scale='RdYlGn',
                    text='ROI (%)'
                )
                
                # 차트 스타일링 개선
                fig_roi.update_layout(
                    height=500,
                    xaxis_tickangle=-30,  # 각도를 덜 가파르게
                    title={
                       'text': 'Top Models ROI Comparison',
                       'x': 0.5,
                       'xanchor': 'center',
                       'font': {'size': 18, 'color': '#1e293b'}
                    },
                    xaxis={
                       'title': {'text': 'Model', 'font': {'size': 14, 'color': '#374151'}},
                       'tickfont': {'size': 11, 'color': '#4b5563'},
                       'gridcolor': '#e5e7eb',
                       'linecolor': '#d1d5db'
                    },
                    yaxis={
                       'title': {'text': 'ROI (%)', 'font': {'size': 14, 'color': '#374151'}},
                       'tickfont': {'size': 12, 'color': '#4b5563'},
                       'gridcolor': '#e5e7eb',
                       'linecolor': '#d1d5db'
                    },
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=50, r=50, t=60, b=100)
                )
                
                # 바 위에 값 표시 개선
                fig_roi.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside',
                    textfont_size=10,
                    textfont_color='#374151',
                    marker_line_color='rgba(0,0,0,0.2)',
                    marker_line_width=1
                )
                
                st.plotly_chart(fig_roi, use_container_width=True)
                
                # 📊 추가 분석 차트들 (2열 배치)
                col1, col2 = st.columns(2)
                
                with col1:
                    # Win Rate vs ROI 산점도
                    fig_scatter = px.scatter(
                       chart_df, 
                       x='Win Rate (%)', 
                       y='ROI (%)',
                       size='Total Predictions',
                       color='ROI (%)',
                       hover_data=['Model', 'Profit/Loss ($)'],
                       title='Win Rate vs ROI Analysis',
                       color_continuous_scale='RdYlGn',
                       text='Model'
                    )
                    
                    # 산점도 스타일링 개선
                    fig_scatter.update_layout(
                       height=450,
                       title={
                           'text': 'Win Rate vs ROI Analysis',
                           'x': 0.5,
                           'xanchor': 'center',
                           'font': {'size': 16, 'color': '#1e293b'}
                       },
                       xaxis={
                           'title': {'text': 'Win Rate (%)', 'font': {'size': 12, 'color': '#374151'}},
                           'tickfont': {'size': 10, 'color': '#4b5563'},
                           'gridcolor': '#e5e7eb'
                       },
                       yaxis={
                           'title': {'text': 'ROI (%)', 'font': {'size': 12, 'color': '#374151'}},
                           'tickfont': {'size': 10, 'color': '#4b5563'},
                           'gridcolor': '#e5e7eb'
                       },
                       plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    # 점들 스타일 개선
                    fig_scatter.update_traces(
                       marker_line_color='rgba(0,0,0,0.3)',
                       marker_line_width=1,
                       textposition='top center',
                       textfont_size=8
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with col2:
                    # 추가 차트: Profit/Loss 가로 막대 차트
                    fig_profit = px.bar(
                       chart_df.sort_values('Profit/Loss ($)', ascending=True), 
                       x='Profit/Loss ($)',
                       y='Model',
                       orientation='h',
                       title='Model Profit/Loss Comparison',
                       color='Profit/Loss ($)',
                       color_continuous_scale='RdYlGn',
                       text='Profit/Loss ($)'
                    )
                    
                    fig_profit.update_layout(
                       height=450,
                       title={
                           'text': 'Profit/Loss Comparison',
                           'x': 0.5,
                           'xanchor': 'center',
                           'font': {'size': 16, 'color': '#1e293b'}
                       },
                       xaxis={
                           'title': {'text': 'Profit/Loss ($)', 'font': {'size': 12, 'color': '#374151'}},
                           'tickfont': {'size': 10, 'color': '#4b5563'},
                           'gridcolor': '#e5e7eb'
                       },
                       yaxis={
                           'title': {'text': '', 'font': {'size': 12, 'color': '#374151'}},
                           'tickfont': {'size': 10, 'color': '#4b5563'},
                           'gridcolor': '#e5e7eb'
                       },
                       plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    fig_profit.update_traces(
                       texttemplate='$%{text:.0f}',
                       textposition='auto',
                       textfont_size=9,
                       marker_line_color='rgba(0,0,0,0.2)',
                       marker_line_width=1
                    )
                    
                    st.plotly_chart(fig_profit, use_container_width=True)
                
                # 일별 성과 분석
                if 'daily_performances' in results and results['daily_performances']:
                    st.markdown("---")
                    st.markdown("## 📅 Daily Performance Analysis")
                    
                    daily_data = results['daily_performances']
                    available_models = sorted(daily_data.keys())
                    
                    # 모델 이름 축약해서 표시 
                    model_display_options = []
                    model_name_mapping = {}
                    for model in available_models:
                       short_name = shorten_model_name(model)
                       model_display_options.append(short_name)
                       model_name_mapping[short_name] = model
                    
                    best_model_full_name = best_model['Full_Model']
                    best_model_short = shorten_model_name(best_model_full_name)
                    default_index = model_display_options.index(best_model_short) if best_model_short in model_display_options else 0
                    
                    selected_model_short = st.selectbox(
                       "Select model for daily analysis:",
                       model_display_options,
                       index=default_index
                    )
                    
                    selected_model = model_name_mapping[selected_model_short]
                    
                    if selected_model in daily_data:
                       model_daily_data = daily_data[selected_model]
                       
                       # 일별 데이터 정리
                       daily_df = []
                       for date, performance in model_daily_data.items():
                           daily_df.append({
                               'Date': date,
                               'ROI (%)': performance.get('roi', 0),
                               'Win Rate (%)': performance.get('win_rate', 0),
                               'Games': performance.get('games_with_odds', 0),
                               'Profit/Loss ($)': performance.get('profit_loss', 0)
                           })
                       
                       daily_df = pd.DataFrame(daily_df)
                       daily_df = daily_df.sort_values('Date')
                       
                       if not daily_df.empty:
                           # 일별 성과 요약
                           # 🎯 Enhanced Daily Performance Metrics
                           st.markdown("""
                           <style>
                           .daily-metric-card {
                               background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                               padding: 1.8rem 1.3rem;
                               border-radius: 16px;
                               color: white;
                               text-align: center;
                               box-shadow: 0 15px 35px rgba(15, 52, 96, 0.4);
                               border: 1px solid rgba(255, 255, 255, 0.1);
                               transition: all 0.3s ease;
                               margin: 0.4rem;
                               position: relative;
                               overflow: hidden;
                           }
                           
                           .daily-metric-card::before {
                               content: '';
                               position: absolute;
                               top: 0;
                               left: 0;
                               right: 0;
                               height: 3px;
                               background: linear-gradient(90deg, #00f2fe, #4facfe, #00f2fe);
                           }
                           
                           .daily-metric-card:hover {
                               transform: translateY(-5px);
                               box-shadow: 0 25px 50px rgba(15, 52, 96, 0.6);
                           }
                           
                                                       .daily-number {
                                font-size: 2.4rem;
                                font-weight: 800;
                                line-height: 0.9;
                                margin-bottom: 0.6rem;
                                text-shadow: 0 3px 6px rgba(0,0,0,0.4);
                                word-break: keep-all;
                                overflow: hidden;
                            }
                           
                           .daily-label {
                               font-size: 0.9rem;
                               font-weight: 500;
                               opacity: 0.85;
                               text-transform: uppercase;
                               letter-spacing: 0.5px;
                           }
                           
                           .positive { color: #10b981; }
                           .negative { color: #ef4444; }
                           .neutral { color: #6b7280; }
                           
                           @media (max-width: 768px) {
                               .daily-metric-card {
                                   padding: 1.3rem 1rem;
                                   margin: 0.2rem;
                               }
                                                               .daily-number {
                                    font-size: 1.8rem;
                                }
                               .daily-label {
                                   font-size: 0.8rem;
                               }
                           }
                           </style>
                           """, unsafe_allow_html=True)
                           
                           col1, col2, col3, col4 = st.columns(4)
                           
                           profitable_days = len(daily_df[daily_df['ROI (%)'] > 0])
                           profit_rate = profitable_days/len(daily_df)*100
                           avg_roi = daily_df['ROI (%)'].mean()
                           total_pl = daily_df['Profit/Loss ($)'].sum()
                           
                           with col1:
                               st.markdown(f"""
                               <div class="daily-metric-card">
                                   <div class="daily-number">{len(daily_df)}</div>
                                   <div class="daily-label">Total Days</div>
                               </div>
                               """, unsafe_allow_html=True)
                               
                           with col2:
                               profit_color = "positive" if profit_rate > 50 else "neutral"
                               st.markdown(f"""
                               <div class="daily-metric-card">
                                   <div class="daily-number {profit_color}">{profitable_days}</div>
                                   <div class="daily-label">Profitable Days ({profit_rate:.1f}%)</div>
                               </div>
                               """, unsafe_allow_html=True)
                               
                           with col3:
                               roi_color = "positive" if avg_roi > 0 else "negative"
                               st.markdown(f"""
                               <div class="daily-metric-card">
                                   <div class="daily-number {roi_color}">{avg_roi:+.2f}%</div>
                                   <div class="daily-label">Avg Daily ROI</div>
                               </div>
                               """, unsafe_allow_html=True)
                               
                           with col4:
                               pl_color = "positive" if total_pl > 0 else "negative"
                               st.markdown(f"""
                               <div class="daily-metric-card">
                                   <div class="daily-number {pl_color}">${total_pl:+.1f}</div>
                                   <div class="daily-label">Total P/L</div>
                               </div>
                               """, unsafe_allow_html=True)
                           
                           # 일별 ROI 차트 - 개선된 스타일링
                           fig_daily = px.line(
                               daily_df, 
                               x='Date', 
                               y='ROI (%)',
                               title=f'{selected_model_short} - Daily ROI Performance',
                               markers=True
                           )
                           
                           # 차트 스타일링 개선
                           fig_daily.update_layout(
                               height=400,
                               title={
                                   'text': f'{selected_model_short} - Daily ROI Performance',
                                   'x': 0.5,
                                   'xanchor': 'center',
                                   'font': {'size': 18, 'color': '#1e293b'}
                               },
                               xaxis={
                                   'title': {'text': 'Date', 'font': {'size': 14, 'color': '#374151'}},
                                   'tickfont': {'size': 11, 'color': '#4b5563'},
                                   'gridcolor': '#e5e7eb',
                                   'linecolor': '#d1d5db'
                               },
                               yaxis={
                                   'title': {'text': 'ROI (%)', 'font': {'size': 14, 'color': '#374151'}},
                                   'tickfont': {'size': 12, 'color': '#4b5563'},
                                   'gridcolor': '#e5e7eb',
                                   'linecolor': '#d1d5db'
                               },
                               plot_bgcolor='rgba(0,0,0,0)',
                               paper_bgcolor='rgba(0,0,0,0)',
                               margin=dict(l=50, r=50, t=60, b=60)
                           )
                           
                           # 라인 스타일 개선
                           fig_daily.update_traces(
                               line=dict(color='#3b82f6', width=3),
                               marker=dict(color='#1d4ed8', size=6, line=dict(color='white', width=2)),
                               hovertemplate='<b>%{x}</b><br>ROI: %{y:.2f}%<extra></extra>'
                           )
                           
                           # 기준선 추가 (0% 라인)
                           fig_daily.add_hline(
                               y=0, 
                               line_dash="dash", 
                               line_color="#dc2626", 
                               line_width=2,
                               opacity=0.7,
                               annotation_text="Break-even",
                               annotation_position="bottom right",
                               annotation_font_color="#dc2626"
                           )
                           
                           st.plotly_chart(fig_daily, use_container_width=True)
                           
                           # 일별 상세 데이터 테이블 - 개선된 스타일링
                           st.markdown("### 📋 Daily Performance Details")
                           
                           # 날짜 포맷 변경 및 추가 컬럼
                           daily_df['Date'] = pd.to_datetime(daily_df['Date']).dt.strftime('%Y-%m-%d')
                           
                           # 더 정교한 ROI 하이라이팅
                           def highlight_daily_roi(val):
                               if val > 15:
                                   return 'background: linear-gradient(135deg, #d1fae5, #a7f3d0); color: #065f46; font-weight: 700; border-radius: 6px; padding: 6px 10px;'
                               elif val > 5:
                                   return 'background: linear-gradient(135deg, #ecfdf5, #d1fae5); color: #047857; font-weight: 600; border-radius: 6px; padding: 6px 10px;'
                               elif val > 0:
                                   return 'background: linear-gradient(135deg, #f0fdf4, #dcfce7); color: #15803d; font-weight: 500; border-radius: 6px; padding: 6px 10px;'
                               elif val > -5:
                                   return 'background: linear-gradient(135deg, #fefce8, #fef3c7); color: #a16207; font-weight: 500; border-radius: 6px; padding: 6px 10px;'
                               elif val > -15:
                                   return 'background: linear-gradient(135deg, #fed7aa, #fdba74); color: #c2410c; font-weight: 600; border-radius: 6px; padding: 6px 10px;'
                               else:
                                   return 'background: linear-gradient(135deg, #fee2e2, #fecaca); color: #991b1b; font-weight: 700; border-radius: 6px; padding: 6px 10px;'
                           
                           # Status 컬럼 제거 - ROI 색상으로만 충분히 표현
                           display_cols = ['Date', 'ROI (%)', 'Win Rate (%)', 'Games', 'Profit/Loss ($)']
                           daily_display_df = daily_df[display_cols]
                           
                           # 프리미엄 테이블 스타일링
                           styled_daily_df = daily_display_df.style.map(
                               highlight_daily_roi, 
                               subset=['ROI (%)']
                           ).format({
                               'ROI (%)': '{:.2f}%',
                               'Win Rate (%)': '{:.1f}%',
                               'Profit/Loss ($)': '${:.1f}'
                           }).set_table_styles([
                               # 테이블 전체 스타일
                               {'selector': '', 'props': [
                                   ('width', '100%'),
                                   ('table-layout', 'auto'),
                                   ('border-collapse', 'collapse'),
                                   ('border-radius', '12px'),
                                   ('overflow', 'hidden'),
                                   ('box-shadow', '0 6px 25px rgba(0,0,0,0.1)')
                               ]},
                               # 헤더 스타일
                               {'selector': 'th', 'props': [
                                   ('background', 'linear-gradient(135deg, #1e293b, #374151)'),
                                   ('color', 'white'),
                                   ('font-weight', '700'),
                                   ('padding', '16px 8px'),
                                   ('text-align', 'center'),
                                   ('border', 'none'),
                                   ('text-transform', 'uppercase'),
                                   ('letter-spacing', '0.025em'),
                                   ('font-size', '0.85rem'),
                                   ('white-space', 'nowrap')
                               ]},
                               # 데이터 셀 스타일
                               {'selector': 'td', 'props': [
                                   ('padding', '12px 8px'),
                                   ('text-align', 'center'),
                                   ('border', 'none'),
                                   ('font-size', '0.85rem'),
                                   ('font-weight', '500'),
                                   ('white-space', 'nowrap')
                               ]},
                               # 짝수 행 스타일
                               {'selector': 'tr:nth-child(even)', 'props': [
                                   ('background', 'linear-gradient(135deg, #f8fafc, #ffffff)')
                               ]},
                               # 호버 효과
                               {'selector': 'tr:hover', 'props': [
                                   ('background', 'linear-gradient(135deg, #e2e8f0, #f1f5f9)'),
                                   ('transform', 'scale(1.002)'),
                                   ('transition', 'all 0.2s ease')
                               ]},
                               # Date 컬럼
                               {'selector': 'th:first-child, td:first-child', 'props': [
                                   ('min-width', '100px'),
                                   ('max-width', '120px')
                               ]},
                               # ROI 컬럼 강조 (두 번째)
                               {'selector': 'th:nth-child(2), td:nth-child(2)', 'props': [
                                   ('min-width', '80px'),
                                   ('max-width', '100px'),
                                   ('font-weight', '600')
                               ]},
                               # 나머지 숫자 컬럼들
                               {'selector': 'th:nth-child(n+3), td:nth-child(n+3)', 'props': [
                                   ('min-width', '75px'),
                                   ('max-width', '110px')
                               ]}
                           ])
                           
                           st.dataframe(styled_daily_df, hide_index=True, use_container_width=True)
                           
                           # 추가 통계 및 차트들
                           st.markdown("### 📊 Additional Statistics")
                           
                           # 🎲 Advanced Analytics Metrics  
                           st.markdown("""
                           <style>
                           .analytics-metric-card {
                               background: linear-gradient(135deg, #2d1b69 0%, #11998e 100%);
                               padding: 1.6rem 1.2rem;
                               border-radius: 14px;
                               color: white;
                               text-align: center;
                               box-shadow: 0 12px 30px rgba(45, 27, 105, 0.35);
                               border: 1px solid rgba(255, 255, 255, 0.15);
                               transition: all 0.3s ease;
                               margin: 0.3rem;
                               position: relative;
                           }
                           
                           .analytics-metric-card::after {
                               content: '';
                               position: absolute;
                               bottom: 0;
                               left: 0;
                               right: 0;
                               height: 2px;
                               background: linear-gradient(90deg, #38f9d7, #43e97b);
                           }
                           
                           .analytics-metric-card:hover {
                               transform: translateY(-4px) scale(1.02);
                               box-shadow: 0 20px 45px rgba(45, 27, 105, 0.5);
                           }
                           
                                                       .analytics-number {
                                font-size: 2.2rem;
                                font-weight: 700;
                                line-height: 0.85;
                                margin-bottom: 0.7rem;
                                text-shadow: 0 2px 5px rgba(0,0,0,0.3);
                                word-break: keep-all;
                                overflow: hidden;
                            }
                           
                           .analytics-label {
                               font-size: 0.85rem;
                               font-weight: 400;
                               opacity: 0.9;
                               text-transform: uppercase;
                               letter-spacing: 0.8px;
                           }
                           
                           .best-performance { color: #43e97b; }
                           .worst-performance { color: #f87171; }
                           .volatility-metric { color: #fbbf24; }
                           .games-total { color: #60a5fa; }
                           
                           @media (max-width: 768px) {
                               .analytics-metric-card {
                                   padding: 1.2rem 0.9rem;
                                   margin: 0.2rem;
                               }
                                                               .analytics-number {
                                    font-size: 1.6rem;
                                }
                               .analytics-label {
                                   font-size: 0.75rem;
                               }
                           }
                           </style>
                           """, unsafe_allow_html=True)
                           
                           col1, col2, col3, col4 = st.columns(4)
                           
                           best_day_roi = daily_df['ROI (%)'].max()
                           worst_day_roi = daily_df['ROI (%)'].min()
                           volatility = daily_df['ROI (%)'].std()
                           total_games = daily_df['Games'].sum()
                           
                           with col1:
                               st.markdown(f"""
                               <div class="analytics-metric-card">
                                   <div class="analytics-number best-performance">+{best_day_roi:.2f}%</div>
                                   <div class="analytics-label">Best Day ROI</div>
                               </div>
                               """, unsafe_allow_html=True)
                               
                           with col2:
                               st.markdown(f"""
                               <div class="analytics-metric-card">
                                   <div class="analytics-number worst-performance">{worst_day_roi:+.2f}%</div>
                                   <div class="analytics-label">Worst Day ROI</div>
                               </div>
                               """, unsafe_allow_html=True)
                               
                           with col3:
                               st.markdown(f"""
                               <div class="analytics-metric-card">
                                   <div class="analytics-number volatility-metric">{volatility:.2f}%</div>
                                   <div class="analytics-label">ROI Volatility</div>
                               </div>
                               """, unsafe_allow_html=True)
                               
                           with col4:
                               st.markdown(f"""
                               <div class="analytics-metric-card">
                                   <div class="analytics-number games-total">{total_games:,}</div>
                                   <div class="analytics-label">Total Games</div>
                               </div>
                               """, unsafe_allow_html=True)
                           
                           # 추가 시각화 차트들
                           st.markdown("### 📈 Performance Trend Analysis")
                           
                           chart_col1, chart_col2 = st.columns(2)
                           
                           with chart_col1:
                               # 누적 수익 차트
                               daily_df['Cumulative P/L'] = daily_df['Profit/Loss ($)'].cumsum()
                               
                               fig_cumulative = px.area(
                                   daily_df,
                                   x='Date',
                                   y='Cumulative P/L',
                                   title=f'{selected_model_short} - Cumulative Profit/Loss',
                                   color_discrete_sequence=['#3b82f6']
                               )
                               
                               fig_cumulative.update_layout(
                                   height=350,
                                   title={
                                       'text': f'{selected_model_short} - Cumulative P/L',
                                       'x': 0.5,
                                       'xanchor': 'center',
                                       'font': {'size': 16, 'color': '#1e293b'}
                                   },
                                   xaxis={
                                       'title': {'text': 'Date', 'font': {'size': 12, 'color': '#374151'}},
                                       'tickfont': {'size': 10, 'color': '#4b5563'},
                                       'gridcolor': '#e5e7eb'
                                   },
                                   yaxis={
                                       'title': {'text': 'Cumulative P/L ($)', 'font': {'size': 12, 'color': '#374151'}},
                                       'tickfont': {'size': 10, 'color': '#4b5563'},
                                       'gridcolor': '#e5e7eb'
                                   },
                                   plot_bgcolor='rgba(0,0,0,0)',
                                   paper_bgcolor='rgba(0,0,0,0)'
                               )
                               
                               fig_cumulative.update_traces(
                                   fill='tonexty',
                                   fillcolor='rgba(59, 130, 246, 0.2)',
                                   line=dict(color='#3b82f6', width=2),
                                   hovertemplate='<b>%{x}</b><br>Cumulative P/L: $%{y:.2f}<extra></extra>'
                               )
                               
                               st.plotly_chart(fig_cumulative, use_container_width=True)
                           
                           with chart_col2:
                               # 승률 vs 게임수 산점도 (사이즈 에러 수정)
                               daily_df['Abs_PL'] = daily_df['Profit/Loss ($)'].abs()  # 절댓값 사용
                               
                               fig_games_winrate = px.scatter(
                                   daily_df,
                                   x='Games',
                                   y='Win Rate (%)',
                                   size='Abs_PL',  # 절댓값 사용으로 에러 방지
                                   color='ROI (%)',
                                   title=f'{selected_model_short} - Games vs Win Rate',
                                   color_continuous_scale='RdYlGn',
                                   hover_data=['Date', 'Profit/Loss ($)']
                               )
                               
                               fig_games_winrate.update_layout(
                                   height=350,
                                   title={
                                       'text': f'{selected_model_short} - Games vs Win Rate',
                                       'x': 0.5,
                                       'xanchor': 'center',
                                       'font': {'size': 16, 'color': '#1e293b'}
                                   },
                                   xaxis={
                                       'title': {'text': 'Games Played', 'font': {'size': 12, 'color': '#374151'}},
                                       'tickfont': {'size': 10, 'color': '#4b5563'},
                                       'gridcolor': '#e5e7eb'
                                   },
                                   yaxis={
                                       'title': {'text': 'Win Rate (%)', 'font': {'size': 12, 'color': '#374151'}},
                                       'tickfont': {'size': 10, 'color': '#4b5563'},
                                       'gridcolor': '#e5e7eb'
                                   },
                                   plot_bgcolor='rgba(0,0,0,0)',
                                   paper_bgcolor='rgba(0,0,0,0)'
                               )
                               
                               fig_games_winrate.update_traces(
                                   marker_line_color='rgba(0,0,0,0.3)',
                                   marker_line_width=1
                               )
                               
                               st.plotly_chart(fig_games_winrate, use_container_width=True)
                       else:
                           st.warning("No daily performance data available for the selected model in this date range.")
                
                # 🆕 Daily Picks Performance 섹션 추가
                st.markdown("---")
                st.markdown("## 📱 Daily Picks Performance Analysis")
                st.markdown("Track performance of picks generated from the Telegram Dashboard")
                
                # 픽 분석기 로드
                picks_analyzer = load_daily_picks_analyzer()
                available_pick_dates = picks_analyzer.get_available_pick_dates()
                
                if available_pick_dates:
                    # 날짜 선택
                    col_pick1, col_pick2, col_pick3 = st.columns([1, 1, 2])
                    
                    with col_pick1:
                       selected_pick_date = st.selectbox(
                           "📅 Select Pick Date:",
                           available_pick_dates,
                           index=len(available_pick_dates)-1,  # 최신 날짜 기본 선택
                           key="pick_date_selector"
                       )
                    
                    with col_pick2:
                       st.info(f"📊 {len(available_pick_dates)} pick dates available")
                    
                    if selected_pick_date:
                       # 선택된 날짜의 픽 성과 분석
                       with st.spinner("📊 Analyzing daily picks performance..."):
                           daily_picks_analysis = picks_analyzer.analyze_daily_picks_performance(selected_pick_date)
                       
                       if daily_picks_analysis:
                           # 기본 통계
                           stats = daily_picks_analysis.get('performance_stats', {})
                           config = daily_picks_analysis.get('configuration', {})
                           matching_stats = daily_picks_analysis.get('matching_stats', {})
                           
                           # 성과 요약 메트릭
                           st.markdown(f"### 📊 Pick Performance Summary - {selected_pick_date}")
                           
                           # 🎯 Daily Picks Performance Metrics
                           st.markdown("""
                           <style>
                           .picks-metric-card {
                               background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                               padding: 1.8rem 1.4rem;
                               border-radius: 18px;
                               color: white;
                               text-align: center;
                               box-shadow: 0 16px 38px rgba(139, 92, 246, 0.4);
                               border: 1px solid rgba(255, 255, 255, 0.2);
                               transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                               margin: 0.4rem;
                               position: relative;
                               overflow: hidden;
                           }
                           
                           .picks-metric-card::before {
                               content: '';
                               position: absolute;
                               top: 0;
                               left: 0;
                               width: 100%;
                               height: 100%;
                               background: radial-gradient(circle at top right, rgba(255,255,255,0.1), transparent 50%);
                               pointer-events: none;
                           }
                           
                           .picks-metric-card:hover {
                               transform: translateY(-6px) rotate(1deg);
                               box-shadow: 0 25px 55px rgba(139, 92, 246, 0.6);
                           }
                           
                                                       .picks-number {
                                font-size: 2.4rem;
                                font-weight: 900;
                                line-height: 0.8;
                                margin-bottom: 0.8rem;
                                text-shadow: 0 4px 8px rgba(0,0,0,0.4);
                                position: relative;
                                z-index: 2;
                                word-break: keep-all;
                                overflow: hidden;
                            }
                           
                           .picks-label {
                               font-size: 0.9rem;
                               font-weight: 600;
                               opacity: 0.95;
                               text-transform: uppercase;
                               letter-spacing: 1px;
                               position: relative;
                               z-index: 2;
                           }
                           
                           .picks-total { background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); }
                           .picks-accuracy { background: linear-gradient(135deg, #059669 0%, #047857 100%); }
                           .picks-avg-roi { background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); }
                           .picks-total-roi { background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%); }
                           
                           .accuracy-excellent { color: #10b981; }
                           .accuracy-good { color: #f59e0b; }
                           .accuracy-poor { color: #ef4444; }
                           
                           .roi-positive { color: #10b981; }
                           .roi-negative { color: #ef4444; }
                           
                           @media (max-width: 768px) {
                               .picks-metric-card {
                                   padding: 1.4rem 1rem;
                                   margin: 0.2rem;
                               }
                                                               .picks-number {
                                    font-size: 1.8rem;
                                }
                               .picks-label {
                                   font-size: 0.8rem;
                               }
                           }
                           </style>
                           """, unsafe_allow_html=True)
                           
                           col1, col2, col3, col4 = st.columns(4)
                           
                           total_picks = stats.get('total_picks', 0)
                           accuracy = stats.get('accuracy', 0)
                           avg_actual_roi = stats.get('avg_actual_roi', 0)
                           total_actual_roi = stats.get('total_actual_roi', 0)
                           analyzed_picks = stats.get('analyzed_picks', 0)
                           
                           if analyzed_picks > 0:
                               total_roi_percentage = (total_actual_roi / (analyzed_picks * 100)) * 100
                           else:
                               total_roi_percentage = 0
                           
                           with col1:
                               st.markdown(f"""
                               <div class="picks-metric-card picks-total">
                                   <div class="picks-number">{total_picks}</div>
                                   <div class="picks-label">Total Picks</div>
                               </div>
                               """, unsafe_allow_html=True)
                               
                           with col2:
                               accuracy_class = "accuracy-excellent" if accuracy >= 60 else ("accuracy-good" if accuracy >= 50 else "accuracy-poor")
                               st.markdown(f"""
                               <div class="picks-metric-card picks-accuracy">
                                   <div class="picks-number {accuracy_class}">{accuracy:.1f}%</div>
                                   <div class="picks-label">Accuracy</div>
                               </div>
                               """, unsafe_allow_html=True)
                               
                           with col3:
                               avg_roi_class = "roi-positive" if avg_actual_roi > 0 else "roi-negative"
                               st.markdown(f"""
                               <div class="picks-metric-card picks-avg-roi">
                                   <div class="picks-number {avg_roi_class}">{avg_actual_roi:+.1f}%</div>
                                   <div class="picks-label">Avg Actual ROI</div>
                               </div>
                               """, unsafe_allow_html=True)
                               
                           with col4:
                               total_roi_class = "roi-positive" if total_roi_percentage > 0 else "roi-negative"
                               st.markdown(f"""
                               <div class="picks-metric-card picks-total-roi">
                                   <div class="picks-number {total_roi_class}">{total_roi_percentage:+.1f}%</div>
                                   <div class="picks-label">Total ROI</div>
                               </div>
                               """, unsafe_allow_html=True)
                           
                           
                           
                           # 설정 정보 - 간단한 스타일
                           st.markdown("### ⚙️ Configuration Used")
                           
                           col_config1, col_config2 = st.columns(2)
                           
                           with col_config1:
                               st.markdown("**🤖 Active Models:**")
                               model_weights = config.get('model_weights', {})
                               for model, weight in model_weights.items():
                                   display_name = model.replace('model_', '').replace('_', ' ').title()
                                   st.markdown(f"• **{display_name}**: {weight:.2f}")
                           
                           with col_config2:
                               st.markdown("**🎯 Selected Zones:**")
                               selected_zones = config.get('selected_zones', {})
                               for dimension, segments in selected_zones.items():
                                   dim_name = dimension.replace('_', ' ').title()
                                   
                                   # 세그먼트 이름들을 간단하게 표시
                                   if isinstance(segments, list) and segments:
                                       segments_text = ", ".join(segments)
                                       st.markdown(f"• **{dim_name}** ({len(segments)} filters):")
                                       st.markdown(f"  _{segments_text}_")
                                   else:
                                       st.markdown(f"• **{dim_name}**: {len(segments) if segments else 0} filters")
                           
                           # 픽별 상세 성과 테이블
                           st.markdown("### 📋 Individual Pick Performance")
                           
                           picks = daily_picks_analysis.get('picks', [])
                           if picks:
                               # 테이블 데이터 준비
                               table_data = []
                               for pick in picks:
                                   table_data.append({
                                       'Pick #': f"#{pick['pick_number']}",
                                       'Game Date': pick.get('game_date', 'N/A'),
                                       'Game': pick['game_info'],
                                       'Prediction': f"{pick['selection_team_name']} ({pick['selection_odds']:+})",
                                       'Predicted ROI (%)': pick['predicted_roi'],
                                       'Actual Result': pick.get('actual_result', 'Pending'),
                                       'Actual ROI (%)': pick.get('actual_roi', 0),
                                       'Status': '✅' if pick.get('is_correct') else '❌' if pick.get('is_correct') is False else '⏳'
                                   })
                               
                               picks_df = pd.DataFrame(table_data)
                               
                               # 픽 성과 테이블 스타일링 (기존 ROI 스타일 활용)
                               def highlight_pick_status(row):
                                   styles = [''] * len(row)
                                   
                                   # Actual ROI 컬럼 스타일링 (안전한 접근)
                                   try:
                                       actual_roi = float(row['Actual ROI (%)']) if row['Actual ROI (%)'] is not None else 0
                                       
                                       if actual_roi > 50:
                                           styles[5] = 'background: linear-gradient(135deg, #d1fae5, #a7f3d0); color: #065f46; font-weight: 800;'
                                       elif actual_roi > 0:
                                           styles[5] = 'background: linear-gradient(135deg, #ecfdf5, #d1fae5); color: #047857; font-weight: 600;'
                                       elif actual_roi < -50:
                                           styles[5] = 'background: linear-gradient(135deg, #fee2e2, #fecaca); color: #991b1b; font-weight: 800;'
                                       else:
                                           styles[5] = 'background: linear-gradient(135deg, #fef3c7, #fde68a); color: #92400e; font-weight: 500;'
                                   except (ValueError, TypeError, KeyError):
                                       styles[5] = 'background-color: #f8f9fa; color: #6c757d;'
                                   
                                   # Status 컬럼 스타일링 (안전한 접근)
                                   try:
                                       status = str(row['Status'])
                                       if status == '✅':
                                           styles[6] = 'background-color: #d4edda; color: #155724; font-weight: bold; text-align: center;'
                                       elif status == '❌':
                                           styles[6] = 'background-color: #f8d7da; color: #721c24; font-weight: bold; text-align: center;'
                                       else:
                                           styles[6] = 'background-color: #fff3cd; color: #856404; font-weight: bold; text-align: center;'
                                   except (KeyError, TypeError):
                                       styles[6] = 'background-color: #f8f9fa; color: #6c757d; text-align: center;'
                                   
                                   return styles
                               
                               styled_picks_df = picks_df.style.apply(
                                   highlight_pick_status, 
                                   axis=1
                               ).format({
                                   'Predicted ROI (%)': '{:.1f}%',
                                   'Actual ROI (%)': '{:.1f}%'
                               }).set_table_styles([
                                   # 테이블 전체 스타일
                                   {'selector': '', 'props': [
                                       ('width', '100%'),
                                       ('table-layout', 'auto'),
                                       ('border-collapse', 'collapse'),
                                       ('border-radius', '12px'),
                                       ('overflow', 'hidden'),
                                       ('box-shadow', '0 8px 32px rgba(0, 0, 0, 0.08)')
                                   ]},
                                   # 헤더 스타일
                                   {'selector': 'th', 'props': [
                                       ('background', 'linear-gradient(135deg, #1e293b, #374151)'),
                                       ('color', 'white'),
                                       ('font-weight', '700'),
                                       ('padding', '16px 8px'),
                                       ('text-align', 'center'),
                                       ('border', 'none'),
                                       ('font-size', '0.9rem'),
                                       ('text-transform', 'uppercase'),
                                       ('letter-spacing', '0.025em')
                                   ]},
                                   # 데이터 셀 스타일
                                   {'selector': 'td', 'props': [
                                       ('padding', '12px 8px'),
                                       ('text-align', 'center'),
                                       ('border', 'none'),
                                       ('font-size', '0.85rem'),
                                       ('font-weight', '500')
                                   ]},
                                   # 짝수 행 스타일
                                   {'selector': 'tr:nth-child(even)', 'props': [
                                       ('background', 'linear-gradient(135deg, #f8fafc, #ffffff)')
                                   ]},
                                   # 호버 효과
                                   {'selector': 'tr:hover', 'props': [
                                       ('background', 'linear-gradient(135deg, #e2e8f0, #f1f5f9)'),
                                       ('transform', 'scale(1.002)'),
                                       ('transition', 'all 0.2s ease')
                                   ]}
                               ])
                               
                               st.dataframe(styled_picks_df, hide_index=True, use_container_width=True)
                               

                           else:
                               st.warning("No picks found for this date")
                       else:
                           st.warning(f"No pick data found for {selected_pick_date}")
                else:
                    st.info("📭 No daily picks found. Generate picks using the Telegram Dashboard first!")
                
                # 🆕 구간별 성과 분석 추가
                st.markdown("---")
                st.markdown("## 📊 Segment Performance Analysis")
                st.markdown("Detailed analysis of model performance across different prediction confidence levels and market scenarios.")
                
                # 선택된 모델에 대한 구간별 분석 수행
                if selected_model and 'matched_data' in results:
                    matched_data = results['matched_data']
                    segment_analysis = analyze_model_segments(matched_data, selected_model)
                    
                    if segment_analysis:
                       # 구간별 성과 탭 (심플 대시보드와 동일)
                       segment_tabs = st.tabs([
                           "📊 Predicted ROI", 
                           "💰 Odds Ranges", 
                           "🎯 Confidence Levels", 
                           "📈 Market vs Model", 
                           "🎰 Kelly Criterion"
                       ])
                       
                       with segment_tabs[0]:
                           st.markdown("### 📊 Performance by Predicted ROI Ranges")
                           st.markdown("Analysis of how well the model performs across different expected return levels.")
                           
                           if 'predicted_roi' in segment_analysis:
                               display_segment_performance_table(segment_analysis['predicted_roi'], "Predicted ROI Range")
                       
                       with segment_tabs[1]:
                           st.markdown("### 💰 Performance by Market Odds")
                           st.markdown("Analysis of model performance across different odds ranges (favorites vs underdogs).")
                           
                           if 'odds' in segment_analysis:
                               display_segment_performance_table(segment_analysis['odds'], "Odds Range")
                       
                       with segment_tabs[2]:
                           st.markdown("### 🎯 Performance by Confidence Levels")
                           st.markdown("Analysis of how well the model performs at different confidence levels.")
                           
                           if 'confidence' in segment_analysis:
                               display_segment_performance_table(segment_analysis['confidence'], "Confidence Level")
                       
                       with segment_tabs[3]:
                           st.markdown("### 📈 Market vs Model Analysis")
                           st.markdown("Comparison of model predictions vs market expectations (implied probabilities from odds).")
                           
                           if 'market_divergence' in segment_analysis:
                               display_segment_performance_table(segment_analysis['market_divergence'], "Market Divergence")
                               
                               st.info("""
                               **📊 Understanding Market Divergence:**
                               - **Model More Optimistic**: Model predicts higher win probability than market odds suggest
                               - **Market Aligned**: Model and market expectations are similar
                               - **Model More Pessimistic**: Model predicts lower win probability than market odds suggest
                               """)
                       
                       with segment_tabs[4]:
                           st.markdown("### 🎰 Kelly Criterion Analysis")
                           st.markdown("Analysis based on optimal position sizes according to Kelly Criterion.")
                           
                           if 'kelly' in segment_analysis:
                               display_segment_performance_table(segment_analysis['kelly'], "Kelly Confidence Level")
                               
                               st.info("""
                               **🎰 Understanding Kelly Criterion:**
                               - **No Selection**: Kelly suggests no selection (negative expected value)
                               - **Low Confidence**: Kelly suggests 0-5% confidence level
                               - **Medium Confidence**: Kelly suggests 5-15% confidence level
                               - **High Confidence**: Kelly suggests 15-25% confidence level
                               - **Very High Confidence**: Kelly suggests 25%+ confidence level (very high confidence opportunities)
                               """)
                    else:
                       st.warning("No segment analysis data available for the selected model.")
                
                # 면책 조항
                st.markdown("---")
                st.markdown("## ⚠️ Important Legal Disclaimer")
                st.error("""
                **⚠️ IMPORTANT: READ CAREFULLY BEFORE USING THIS SERVICE**
                
                **NOT FINANCIAL OR INVESTMENT ADVICE**
                - This service provides **statistical analysis and educational content only**
                - Information is for **analytical, educational, and research purposes only**
                - This is **NOT financial, investment, or gambling advice**
                - We do **NOT recommend, encourage, or induce any betting or gambling activities**
                
                **NO GUARANTEES & RISK WARNING**
                - Past performance does **NOT guarantee future results**
                - All predictions and analysis may be **completely wrong**
                - Any gambling or betting activity carries **significant risk of financial loss**
                - You could **lose all invested money**
                
                **PERSONAL RESPONSIBILITY**
                - Any decisions you make are **entirely your own responsibility**
                - You must **comply with all applicable laws** in your jurisdiction
                - You are **solely responsible** for determining the legality of any activities
                - **We are not responsible** for any losses, damages, or legal issues
                
                **DATA ACCURACY DISCLAIMER**
                - Data may contain **errors or inaccuracies**
                - Technical issues may cause **incorrect information**
                - Information may be **outdated or incomplete**
                - **Independent verification** is required before making any decisions
                
                **LEGAL COMPLIANCE**
                - You must **comply with all local, state, and federal laws**
                - Service may not be legal in all jurisdictions
                - **Consult legal counsel** if you have questions about legality
                - We make **no representations** about legal compliance
                """)
                
                st.markdown("---")
                st.markdown("## 📋 Terms of Service")
                st.info("""
                **BY USING THIS SERVICE, YOU ACKNOWLEDGE AND AGREE THAT:**
                
                1. **You are using this service at your own risk**
                2. **You will not hold us liable for any losses or damages**
                3. **You understand this is educational/analytical content only**
                4. **You will comply with all applicable laws and regulations**
                5. **You are responsible for your own decisions and actions**
                6. **You understand that gambling/betting can be addictive and harmful**
                
                **If you do not agree with these terms, please discontinue use immediately.**
                """)
                
                st.markdown("---")
                st.markdown("## 🔄 Data Updates")
                st.info("""
                - **Real-time Updates**: Performance data is updated automatically after each game
                - **Complete Transparency**: All predictions are tracked and recorded
                - **No Cherry-picking**: Every prediction is included in the analysis
                - **Independent Verification**: Results are independently tracked and verified
                """)
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.error("Please try selecting a different date range.")
    else:
        st.warning("Please select a valid date range (start date must be before or equal to end date).")
    
    # 📱 소셜미디어 콘텐츠 생성 섹션 (푸터 위에 위치)
    st.markdown("---")
    st.markdown("## 📱 Generate Social Media Content")
    
    # 컴파운드 데이터가 있는 경우에만 표시
    if 'best_model_name' in locals() and 'compound_data' in locals() and 'daily_roi_list' in locals():
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.1); 
                   padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;
                   border: 1px solid rgba(255, 255, 255, 0.2);
                   backdrop-filter: blur(10px);">
        """, unsafe_allow_html=True)
        
        platform = st.selectbox(
            "📱 Choose Platform:",
            ["twitter", "instagram", "reddit", "linkedin", "tiktok"],
            format_func=lambda x: {
                "twitter": "🐦 Twitter/X",
                "instagram": "📸 Instagram", 
                "reddit": "🤖 Reddit",
                "linkedin": "💼 LinkedIn",
                "tiktok": "🎵 TikTok/General"
            }[x]
        )
        
        # 플랫폼 선택 바로 아래에 버튼 배치
        st.markdown('<div style="margin: 0.5rem 0;"></div>', unsafe_allow_html=True)
        
        if st.button("🚀 Generate Content", type="primary", use_container_width=True):
            # 컴파운드 데이터 가져오기
            try:
                best_model_name, compound_data, daily_roi_list = find_best_model_compound_performance(results)
                social_content = generate_social_media_content(
                    best_model_name, compound_data, daily_roi_list, platform
                )
                
                # 결과를 세션 상태에 저장
                st.session_state.generated_content = social_content
                st.session_state.selected_platform = platform
            except:
                st.error("데이터를 불러올 수 없습니다. 페이지를 새로고침해주세요.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 생성된 콘텐츠 표시
        if hasattr(st.session_state, 'generated_content') and st.session_state.generated_content:
           
           st.text_area(
               f"Copy this content for {st.session_state.selected_platform.upper()}:",
               st.session_state.generated_content,
               height=350,
               help="Click inside the box and press Ctrl+A to select all, then Ctrl+C to copy!",
           )
           
           # 글자 수 체크 - 개선된 디자인
           char_count = len(st.session_state.generated_content)
           limits = {
               "twitter": 280,
               "instagram": 2200,
               "reddit": 40000,
               "linkedin": 3000,
               "tiktok": 2200
           }
           
           limit = limits.get(st.session_state.selected_platform, 2200)
           percentage = (char_count / limit) * 100
           
           if char_count <= limit:
               st.success(f"✅ Perfect length: {char_count:,}/{limit:,} characters ({percentage:.1f}%)")
           else:
               st.error(f"❌ Too long: {char_count:,}/{limit:,} characters ({percentage:.1f}%)")
               st.info("💡 Consider shortening for better engagement!")
           
           # 진행률 바
           progress_color = "#10b981" if char_count <= limit else "#ef4444"
           st.markdown(f"""
           <div style="background: rgba(255, 255, 255, 0.2); border-radius: 10px; overflow: hidden; margin: 1rem 0;">
               <div style="background: {progress_color}; height: 8px; width: {min(percentage, 100):.1f}%; 
                           transition: width 0.3s ease; border-radius: 10px;"></div>
           </div>
           """, unsafe_allow_html=True)

    # 푸터 (맨 아래)
    st.markdown("---")
    
    # 푸터 텍스트 먼저
    st.markdown("""
    <div class="footer">
        <h3>🏆 MLB Analytics Performance Tracker</h3>
        <p>Real-time transparent performance tracking of 25 ML models</p>
        <p><em>Updated continuously • 100% Transparent • Independently Verified</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # 프로필 이미지를 완전 중앙에 배치
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 이미지 표시 성공 여부 추적
    image_displayed = False
    
    # 방법 1: Base64 인코딩으로 HTML 표시
    try:
        with open("images/KakaoTalk_20250727_162022080.png", "rb") as f:
            import base64
            img_data = base64.b64encode(f.read()).decode()
            st.markdown(f"""
            <div style="display: flex; justify-content: center; align-items: center; width: 100%; text-align: center;">
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{img_data}" width="120" style="border-radius: 10px;">
                    <p style="font-size: 12px; color: #666; margin-top: 8px;">Analytics by MLB Analytics Pro</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            image_displayed = True
    except:
        pass
    
    # 방법 2: Streamlit 기본 방식 (첫 번째 방법 실패시만)
    if not image_displayed:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            try:
                st.image("images/KakaoTalk_20250727_162022080.png", 
                        caption="Analytics by MLB Analytics Pro", 
                        width=120)
            except:
                pass
    
    # 맨 위로 이동 버튼 추가 (앵커 링크 방식)
    st.markdown("""
    <a href="#top" class="scroll-to-top" style="text-decoration: none; color: white; display: flex; align-items: center; justify-content: center;">
        ↑
    </a>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 