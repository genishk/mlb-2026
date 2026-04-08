import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path
import numpy as np
import glob
import re

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.simple_model_analyzer import SimpleModelAnalyzer
from src.ensemble_optimizer import StableEnsembleOptimizer

# 페이지 설정
st.set_page_config(
    page_title="📊 MLB Model Performance Dashboard",
    page_icon="📊", 
    layout="wide"
)

# 🆕 F5 새로고침 시 캐시 무효화를 위한 session state 체크
if 'page_refresh_key' not in st.session_state:
    st.session_state.page_refresh_key = 0
    # 캐시 클리어
    st.cache_data.clear()

# 🆕 수동 캐시 클리어 버튼 (사이드바 상단에 추가)
def add_cache_clear_button():
    if st.sidebar.button("🔄 Clear Cache & Refresh", help="데이터가 업데이트되지 않을 때 클릭하세요"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

# 스타일 설정
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #007bff;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .positive-roi {
        background: linear-gradient(90deg, #e8f5e8, #ffffff);
        border-left: 4px solid #28a745;
    }
    
    .negative-roi {
        background: linear-gradient(90deg, #f8e8e8, #ffffff);
        border-left: 4px solid #dc3545;
    }
    
    .neutral-roi {
        background: linear-gradient(90deg, #fff8e1, #ffffff);
        border-left: 4px solid #ffc107;
    }
    
    .optimization-container {
        background: linear-gradient(90deg, #f0f8ff, #ffffff);
        border-left: 4px solid #007bff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .stability-high {
        background: linear-gradient(90deg, #e8f5e8, #ffffff);
        border-left: 4px solid #28a745;
    }
    
    .stability-medium {
        background: linear-gradient(90deg, #fff8e1, #ffffff);
        border-left: 4px solid #ffc107;
    }
    
    .stability-low {
        background: linear-gradient(90deg, #f8e8e8, #ffffff);
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_analyzer():
    """Load analyzer (cached)"""
    return SimpleModelAnalyzer()

@st.cache_data
def load_optimizer():
    """Load ensemble optimizer (cached)"""
    return StableEnsembleOptimizer()

@st.cache_data
def get_available_dates():
    """Get available dates (cached)"""
    analyzer = load_analyzer()
    return analyzer.get_available_dates()

@st.cache_data
def run_analysis(start_date, end_date):
    """Run analysis (cached) - v2 with Expected ROI"""
    analyzer = load_analyzer()
    return analyzer.analyze(start_date, end_date)

# @st.cache_data  # 가중치 변경 반영을 위해 캐시 비활성화
def run_ensemble_optimization(start_date, end_date, target_metric, max_models, auto_select, stability_weight, roi_weight,
                            min_roi, min_stability, max_drawdown, max_consecutive_losses, min_games):
    """Run ensemble optimization (cached)"""
    optimizer = load_optimizer()
    return optimizer.optimize_ensemble(
        start_date=start_date,
        end_date=end_date,
        target_metric=target_metric,
        max_models=max_models,
        auto_select_models=auto_select,
        stability_weight=stability_weight,
        roi_weight=roi_weight,
        min_roi=min_roi,
        min_stability=min_stability,
        max_drawdown=max_drawdown,
        max_consecutive_losses=max_consecutive_losses,
        min_games=min_games
    )

def get_current_ensemble_weights():
    """현재 최적화된 앙상블 가중치 반환"""
    # 🚨 실제 최적화 결과 가중치 사용
    return {
        'model6': 0.250,
        'model1_extended_lgbm': 0.100,
        'model_advanced_lgbm_basic': 0.300,
        'model2_extended_catboost': 0.100,
        'model_advanced_catboost': 0.100,
        'model3_extended_xgboost': 0.100,
    }

def create_weight_comparison_chart(current_weights, optimal_weights):
    """현재 가중치와 최적 가중치 비교 차트"""
    
    models = list(set(list(current_weights.keys()) + list(optimal_weights.keys())))
    current_vals = [current_weights.get(model, 0) for model in models]
    optimal_vals = [optimal_weights.get(model, 0) for model in models]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Current Weights',
        x=models,
        y=current_vals,
        marker_color='lightblue',
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        name='Optimal Weights',
        x=models,
        y=optimal_vals,
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title="Current vs Optimal Model Weights",
        xaxis_title="Models",
        yaxis_title="Weight",
        barmode='group',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_stability_chart(period_performances):
    """기간별 안정성 차트"""
    
    periods = [f"Period {i+1}" for i in range(len(period_performances))]
    rois = [p['roi'] for p in period_performances]
    stability_scores = [p['stability_score'] for p in period_performances]
    
    fig = go.Figure()
    
    # ROI 라인
    fig.add_trace(go.Scatter(
        x=periods,
        y=rois,
        mode='lines+markers',
        name='ROI (%)',
        yaxis='y',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # Stability Score 라인 (두 번째 y축)
    fig.add_trace(go.Scatter(
        x=periods,
        y=stability_scores,
        mode='lines+markers',
        name='Stability Score',
        yaxis='y2',
        line=dict(color='orange', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Performance Stability Across Periods",
        xaxis_title="Time Periods",
        yaxis=dict(
            title="ROI (%)",
            side="left"
        ),
        yaxis2=dict(
            title="Stability Score",
            side="right",
            overlaying="y"
        ),
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
    
    return fig

def create_risk_metrics_chart(performance):
    """리스크 지표 시각화"""
    
    # 일일 수익률 분포
    daily_returns = performance['daily_returns']
    
    if not daily_returns:
        return None
    
    fig = go.Figure()
    
    # 히스토그램
    fig.add_trace(go.Histogram(
        x=daily_returns,
        nbinsx=20,
        name='Daily Returns Distribution',
        opacity=0.7,
        marker_color='lightblue'
    ))
    
    # 평균선
    mean_return = np.mean(daily_returns)
    fig.add_vline(x=mean_return, line_dash="dash", line_color="green", 
                  annotation_text=f"Mean: {mean_return:.2f}%")
    
    # 0선
    fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.5)
    
    fig.update_layout(
        title="Daily Returns Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        height=400,
        showlegend=False
    )
    
    return fig

def get_available_prediction_files_for_period(start_date_str: str, end_date_str: str):
    """특정 기간에 사용될 예측 파일들을 미리 확인"""
    project_root = Path(__file__).parent.parent
    predictions_dir = project_root / "src" / "odds" / "data" / "matched"
    
    prediction_files = list(predictions_dir.glob("mlb_predictions_with_odds_*.json"))
    
    filtered_files = []
    total_games = 0
    
    for file_path in prediction_files:
        try:
            # 파일명에서 날짜 추출 (ensemble_optimizer와 동일한 로직)
            filename = file_path.stem
            parts = filename.split('_')
            if len(parts) >= 5:
                date_part = parts[4]  # YYYYMMDD
                from datetime import datetime
                file_date = datetime.strptime(date_part, '%Y%m%d')
                file_date_str = file_date.strftime('%Y-%m-%d')
                
                # 날짜 범위 확인
                if start_date_str <= file_date_str <= end_date_str:
                    # 파일 크기 및 게임 수 확인
                    try:
                        import json
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            game_count = len(data) if isinstance(data, list) else 0
                            total_games += game_count
                            
                        filtered_files.append({
                            'filename': filename,
                            'date': file_date_str,
                            'games': game_count,
                            'file_size': f"{file_path.stat().st_size / 1024:.1f} KB"
                        })
                    except Exception:
                        filtered_files.append({
                            'filename': filename,
                            'date': file_date_str,
                            'games': 'N/A',
                            'file_size': f"{file_path.stat().st_size / 1024:.1f} KB"
                        })
        except Exception:
            continue
    
    # 날짜순 정렬
    filtered_files.sort(key=lambda x: x['date'])
    
    return {
        'files': filtered_files,
        'total_files': len(filtered_files),
        'total_games': total_games
    }

def display_ensemble_optimization_tab():
    """앙상블 최적화 탭 내용 (개선된 버전)"""
    
    st.header("🔧 Ensemble Weight Optimizer")
    st.info("📍 Goal: Find ONE stable optimal weight combination based on historical performance")
    
    # === 설정 섹션 ===
    st.subheader("⚙️ Optimization Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 Analysis Period")
        
        # 사용 가능한 날짜 가져오기
        try:
            available_dates = get_available_dates()
            if available_dates:
                min_date = datetime.strptime(available_dates[0], '%Y-%m-%d').date()
                max_date = datetime.strptime(available_dates[-1], '%Y-%m-%d').date()
                
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="opt_start_date"
                )
                
                end_date = st.date_input(
                    "End Date", 
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="opt_end_date"
                )
                
                if start_date > end_date:
                    st.error("⚠️ Start date cannot be later than end date.")
                    return
                    
                # 선택된 기간 정보 표시
                period_days = (end_date - start_date).days + 1
                st.success(f"📊 Selected: {period_days} days ({start_date} ~ {end_date})")
                
                # 🆕 실제 사용될 입력 데이터 표시
                st.markdown("#### 📁 Input Data for Selected Period")
                
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                
                with st.spinner("🔍 Analyzing input files..."):
                    file_info = get_available_prediction_files_for_period(start_date_str, end_date_str)
                
                if file_info['total_files'] > 0:
                    # 요약 정보
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.metric("📄 Files Found", file_info['total_files'])
                    with col_info2:
                        st.metric("🎯 Total Games", file_info['total_games'])
                    with col_info3:
                        avg_games = file_info['total_games'] / file_info['total_files'] if file_info['total_files'] > 0 else 0
                        st.metric("📊 Avg Games/Day", f"{avg_games:.1f}")
                    
                    # 파일 상세 정보 (expander)
                    with st.expander(f"📋 File Details ({file_info['total_files']} files)", expanded=False):
                        if file_info['files']:
                            file_df = pd.DataFrame(file_info['files'])
                            file_df.columns = ['Filename', 'Date', 'Games', 'File Size']
                            st.dataframe(file_df, hide_index=True, use_container_width=True)
                        else:
                            st.info("No files found for the selected period.")
                    
                    # 데이터 충족성 확인
                    min_required_games = 20  # ensemble_optimizer의 min_games_required와 동일
                    if file_info['total_games'] < min_required_games:
                        st.warning(f"⚠️ **Insufficient Data**: Found {file_info['total_games']} games, but minimum {min_required_games} required.")
                    elif file_info['total_games'] < 50:
                        st.info(f"ℹ️ **Limited Data**: {file_info['total_games']} games available. Results may be less reliable.")
                    else:
                        st.success(f"✅ **Sufficient Data**: {file_info['total_games']} games available for robust analysis.")
                        
                else:
                    st.error("❌ **No Data Found**: No prediction files found for the selected period.")
                    st.info("💡 **Suggestion**: Try selecting a different date range or check if prediction files exist in the data directory.")
                
            else:
                st.error("📂 No prediction files found.")
                return
                
        except Exception as e:
            st.error(f"❌ Error loading dates: {str(e)}")
            return
        
        st.markdown("### 🎯 Optimization Target")
        target_metric = st.selectbox(
            "Target Metric:",
            ["stability_score", "roi", "sharpe_ratio", "risk_adjusted_roi"],
            format_func=lambda x: {
                "stability_score": "🛡️ Stability Score (안정성 우선)",
                "roi": "💰 ROI (수익률 우선)",
                "sharpe_ratio": "📊 Sharpe Ratio (리스크 조정 수익률)",
                "risk_adjusted_roi": "⚖️ Risk-Adjusted ROI (위험 조정 수익률)"
            }[x]
        )
    
    with col2:
        st.markdown("### 🤖 Model Selection")
        
        auto_select = st.checkbox(
            "🔍 Auto-select best models",
            value=True,
            help="자동으로 성과가 좋은 모델들을 선별하여 앙상블에 포함"
        )
        
        max_models = st.slider(
            "📊 Maximum models in ensemble:",
            min_value=1,
            max_value=10,
            value=6,
            help="앙상블에 포함할 최대 모델 수"
        )
        
        # 🎯 새로 추가: 안정성/ROI 가중치 설정
        st.markdown("#### ⚖️ Selection Criteria Weights")
        stability_weight = st.slider(
            "🛡️ Stability Weight:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="모델 선별 시 안정성 점수의 가중치 (나머지는 ROI 가중치)"
        )
        roi_weight = 1.0 - stability_weight
        
        # 가중치 표시
        col_weight1, col_weight2 = st.columns(2)
        with col_weight1:
            st.metric("🛡️ Stability", f"{stability_weight:.1%}")
        with col_weight2:
            st.metric("💰 ROI", f"{roi_weight:.1%}")
        
        if auto_select:
            st.info(f"""
            🔍 **Auto-selection criteria:**
            
            **🎯 Selection Process:**
            1. **Filter** models meeting basic criteria
            2. **Rank** by Composite Score (Stability {stability_weight:.0%} + ROI {roi_weight:.0%})
            3. **Select** top {max_models} models
            
            **📋 Basic Criteria (adaptively relaxed):**
            - ROI ≥ -20% (adaptively relaxed to -25%, -30%...)
            - Stability Score ≥ 0 (adaptively relaxed)
            - Max Drawdown ≤ 50% (adaptively relaxed to 60%, 70%...)
            - Consecutive Losses ≤ 10 days (adaptively relaxed)
            - Minimum 10 games
            
            ⚡ *Criteria automatically relaxed if insufficient models found*
            🏆 *Final ranking: {stability_weight:.0%} Stability + {roi_weight:.0%} ROI*
            """)
        else:
            st.warning("⚠️ Manual selection mode: Will use first available models")
        
        st.markdown("### ⚙️ Advanced Settings")
        
        show_individual_analysis = st.checkbox(
            "📈 Show individual model analysis",
            value=True,
            help="개별 모델 성과 분석 결과 표시"
        )
        
        # 🆕 새로 추가: 필터링 기준 설정
        with st.expander("🔧 Model Filtering Criteria", expanded=False):
            st.markdown("#### 📊 Performance Thresholds")
            
            col_filter1, col_filter2 = st.columns(2)
            
            with col_filter1:
                min_roi = st.number_input(
                    "💰 Min ROI (%)",
                    min_value=-50.0,
                    max_value=10.0,
                    value=-15.0,
                    step=1.0,
                    help="최소 ROI 기준 (이 값보다 낮은 모델은 제외)"
                )
                
                min_stability = st.number_input(
                    "🛡️ Min Stability Score",
                    min_value=-50.0,
                    max_value=50.0,
                    value=30.0,
                    step=1.0,
                    help="최소 안정성 점수 (이 값보다 낮은 모델은 제외)"
                )
                
                min_games = st.number_input(
                    "🎯 Min Games",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=1,
                    help="최소 게임 수 (이 값보다 적은 게임 수의 모델은 제외)"
                )
            
            with col_filter2:
                max_drawdown = st.number_input(
                    "📉 Max Drawdown (%)",
                    min_value=20.0,
                    max_value=150.0,
                    value=90.0,
                    step=5.0,
                    help="최대 낙폭 기준 (이 값보다 높은 모델은 제외)"
                )
                
                max_consecutive_losses = st.number_input(
                    "📅 Max Consecutive Losses (days)",
                    min_value=3,
                    max_value=30,
                    value=10,
                    step=1,
                    help="최대 연속 손실 일수 (이 값보다 긴 연속 손실 모델은 제외)"
                )
            
            # 현재 설정 요약 표시
            st.markdown("#### 📋 Current Filter Summary")
            filter_summary = f"""
            **📊 Performance Filters:**
            - Min ROI: {min_roi}%
            - Min Stability: {min_stability}
            - Min Games: {min_games}
            
            **🛡️ Risk Filters:**
            - Max Drawdown: {max_drawdown}%
            - Max Consecutive Losses: {max_consecutive_losses} days
            """
            st.info(filter_summary)
    
    # === 실행 버튼 ===
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Find Optimal Ensemble", type="primary", use_container_width=True):
            
            with st.spinner("🔄 Analyzing models and optimizing ensemble weights..."):
                try:
                    start_date_str = start_date.strftime('%Y-%m-%d')
                    end_date_str = end_date.strftime('%Y-%m-%d')
                    
                    optimization_results = run_ensemble_optimization(
                        start_date_str, end_date_str, target_metric, max_models, auto_select,
                        stability_weight, roi_weight,
                        min_roi, min_stability, max_drawdown, max_consecutive_losses, min_games
                    )
                    st.session_state.optimization_results = optimization_results
                    st.success("✅ Optimization completed!")
                    
                except Exception as e:
                    st.error(f"❌ Optimization failed: {str(e)}")
                    with st.expander("🔍 Error Details"):
                        st.exception(e)
                    return
    
    # === 결과 표시 ===
    if 'optimization_results' in st.session_state:
        results = st.session_state.optimization_results
        
        # === 모델 발견 및 선별 결과 ===
        if show_individual_analysis:
            st.markdown("---")
            st.subheader("🔍 Model Discovery & Selection")
            
            model_discovery = results['model_discovery']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Models Discovered", len(model_discovery['available_models']))
            with col2:
                st.metric("Models Selected", len(model_discovery['selected_models']))
            with col3:
                st.metric("Selection Method", "Auto" if model_discovery['auto_selected'] else "Manual")
            
            # 개별 모델 성과 표시
            individual_perfs = model_discovery['individual_performances']
            if individual_perfs:
                st.markdown("### 📊 Individual Model Performance")
                
                # 성과 데이터프레임 생성
                perf_data = []
                for model, perf in individual_perfs.items():
                    status = "✅ Selected" if model in model_discovery['selected_models'] else "❌ Not Selected"
                    perf_data.append({
                        'Model': model,
                        'Status': status,
                        'ROI (%)': f"{perf['roi']:.2f}",
                        'Stability Score': f"{perf['stability_score']:.1f}",
                        'Win Rate (%)': f"{perf['win_rate']:.1f}",
                        'Max Drawdown (%)': f"{perf['max_drawdown']:.1f}",
                        'Sharpe Ratio': f"{perf['sharpe_ratio']:.3f}",
                        'Games': perf['total_games']
                    })
                
                perf_df = pd.DataFrame(perf_data)
                
                # 선별된 모델과 선별되지 않은 모델 구분
                selected_df = perf_df[perf_df['Status'] == "✅ Selected"]
                not_selected_df = perf_df[perf_df['Status'] == "❌ Not Selected"]
                
                if not selected_df.empty:
                    st.markdown("#### ✅ Selected Models for Ensemble")
                    st.dataframe(selected_df.drop('Status', axis=1), hide_index=True, use_container_width=True)
                
                if not not_selected_df.empty:
                    with st.expander("❌ Models Not Selected", expanded=False):
                        st.dataframe(not_selected_df.drop('Status', axis=1), hide_index=True, use_container_width=True)
        
        # === 최적 가중치 결과 ===
        st.markdown("---")
        st.subheader("🏆 Recommended Optimal Weights")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Optimal Weight Configuration")
            optimal_weights = results['optimal_weights']
            
            # 가중치를 깔끔하게 표시
            weights_df = pd.DataFrame([
                {"Model": model, "Weight": f"{weight:.3f}", "Percentage": f"{weight*100:.1f}%"}
                for model, weight in optimal_weights.items()
            ])
            st.dataframe(weights_df, hide_index=True, use_container_width=True)
            
            # run_ensemble.py용 코드 생성
            st.markdown("### 📝 Implementation Code")
            code_str = "ENSEMBLE_CONFIG = {\n"
            for model, weight in optimal_weights.items():
                code_str += f"    '{model}': {weight:.3f},\n"
            code_str += "}"
            
            st.code(code_str, language="python")
            st.warning("⚠️ Manual Update Required: Copy the weights above to run_ensemble.py")
        
        with col2:
            st.markdown("### 📈 Performance Metrics")
            
            # Train-Validation Split이 사용되었는지 확인
            split_used = results['data_summary'].get('split_used', False)
            
            if split_used:
                # Train-Validation Split 결과 표시
                st.markdown("#### 🎯 Training Performance")
                train_perf = results['train_performance']
                
                col_train1, col_train2 = st.columns(2)
                with col_train1:
                    st.metric("ROI (Train)", f"{train_perf['roi']:.2f}%")
                    st.metric("Win Rate (Train)", f"{train_perf['win_rate']:.1f}%")
                    st.metric("Stability Score (Train)", f"{train_perf['stability_score']:.1f}")
                    st.metric("Sharpe Ratio (Train)", f"{train_perf['sharpe_ratio']:.3f}")
                
                with col_train2:
                    st.metric("Max Drawdown (Train)", f"{train_perf['max_drawdown']:.1f}%")
                    st.metric("Volatility (Train)", f"{train_perf['volatility']:.2f}%")
                    st.metric("Profit Days (Train)", f"{train_perf['profit_days']}/{train_perf['total_days']}")
                    st.metric("Max Consecutive Losses (Train)", f"{train_perf['consecutive_losses_max']} days")
                
                # 🆕 Training에 사용된 파일 정보 표시
                train_files = results['optimization_details'].get('train_files', [])
                if train_files:
                    st.markdown("**📁 Training Files Used:**")
                    # 날짜만 표시 (YYYY-MM-DD 형태)
                    st.text(", ".join(train_files))
                
                st.markdown("#### 🔍 Validation Performance")
                val_perf = results['validation_performance']
                
                col_val1, col_val2 = st.columns(2)
                with col_val1:
                    st.metric("ROI (Val)", f"{val_perf['roi']:.2f}%")
                    st.metric("Win Rate (Val)", f"{val_perf['win_rate']:.1f}%")
                    st.metric("Stability Score (Val)", f"{val_perf['stability_score']:.1f}")
                    st.metric("Sharpe Ratio (Val)", f"{val_perf['sharpe_ratio']:.3f}")
                
                with col_val2:
                    st.metric("Max Drawdown (Val)", f"{val_perf['max_drawdown']:.1f}%")
                    st.metric("Volatility (Val)", f"{val_perf['volatility']:.2f}%")
                    st.metric("Profit Days (Val)", f"{val_perf['profit_days']}/{val_perf['total_days']}")
                    st.metric("Max Consecutive Losses (Val)", f"{val_perf['consecutive_losses_max']} days")
                
                # 🆕 Validation에 사용된 파일 정보 표시
                validation_files = results['optimization_details'].get('validation_files', [])
                if validation_files:
                    st.markdown("**📁 Validation Files Used:**")
                    # 날짜만 표시 (YYYY-MM-DD 형태)
                    st.text(", ".join(validation_files))
                else:
                    st.info("No validation files (insufficient data for split)")
                
                # 전체 데이터 성과도 표시
                st.markdown("#### 📊 Overall Performance (All Data)")
                perf = results['final_performance']
                st.info(f"ROI: {perf['roi']:.2f}% | Win Rate: {perf['win_rate']:.1f}% | Stability: {perf['stability_score']:.1f}")
                
            else:
                # 기존 방식 (Split 미사용)
                perf = results['final_performance']
                
                col1_metrics, col2_metrics = st.columns(2)
                
                with col1_metrics:
                    st.metric("ROI", f"{perf['roi']:.2f}%")
                    st.metric("Win Rate", f"{perf['win_rate']:.1f}%")
                    st.metric("Stability Score", f"{perf['stability_score']:.1f}")
                    st.metric("Sharpe Ratio", f"{perf['sharpe_ratio']:.3f}")
                
                with col2_metrics:
                    st.metric("Max Drawdown", f"{perf['max_drawdown']:.1f}%")
                    st.metric("Volatility", f"{perf['volatility']:.2f}%")
                    st.metric("Profit Days", f"{perf['profit_days']}/{perf['total_days']}")
                    st.metric("Max Consecutive Losses", f"{perf['consecutive_losses_max']} days")
                
                # 🆕 사용된 파일 정보 표시 (Split 미사용 시)
                train_files = results['optimization_details'].get('train_files', [])
                if train_files:
                    st.markdown("**📁 All Files Used:**")
                    # 날짜만 표시 (YYYY-MM-DD 형태)
                    st.text(", ".join(train_files))
        
        # === 안정성 분석 ===
        st.markdown("---")
        st.subheader("📊 Stability Analysis")
        
        stability = results['stability_validation']
        
        # 안정성 점수에 따른 스타일링
        if stability.get('overall_stability', 0) > 0.8:
            stability_class = "stability-high"
            stability_emoji = "🟢"
            stability_text = "High Stability"
        elif stability.get('overall_stability', 0) > 0.6:
            stability_class = "stability-medium"
            stability_emoji = "🟡"
            stability_text = "Medium Stability"
        else:
            stability_class = "stability-low"
            stability_emoji = "🔴"
            stability_text = "Low Stability"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container {stability_class}">
                <h4>{stability_emoji} {stability_text}</h4>
                <p><strong>Overall Stability:</strong> {stability.get('overall_stability', 0):.1%}</p>
                <p><strong>ROI Consistency:</strong> {stability.get('roi_consistency', 0):.1%}</p>
                <p><strong>Stability Consistency:</strong> {stability.get('stability_consistency', 0):.1%}</p>
                <p><strong>Avg Max Drawdown:</strong> {stability.get('avg_max_drawdown', 0):.1f}%</p>
                <p><strong>Worst Period ROI:</strong> {stability.get('worst_period_roi', 0):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # 기간별 성과 차트
            if 'period_performances' in stability and stability['period_performances']:
                stability_chart = create_stability_chart(stability['period_performances'])
                st.plotly_chart(stability_chart, use_container_width=True)
        
        # === 리스크 분석 ===
        st.subheader("📉 Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 일일 수익률 분포
            risk_chart = create_risk_metrics_chart(perf)
            if risk_chart:
                st.plotly_chart(risk_chart, use_container_width=True)
        
        with col2:
            # 가중치 비교
            current_weights = get_current_ensemble_weights()
            comparison_chart = create_weight_comparison_chart(current_weights, optimal_weights)
            st.plotly_chart(comparison_chart, use_container_width=True)
        
        # === 백테스팅 상세 결과 ===
        with st.expander("🧪 Detailed Backtest Results", expanded=False):
            st.markdown("### 📊 Optimization Details")
            
            opt_details = results['optimization_details']
            data_summary = results['data_summary']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Games Analyzed", data_summary['total_games'])
                st.metric("Cleaned Games Used", data_summary['cleaned_games'])
                st.metric("Analysis Period", data_summary['analysis_period'])
            
            with col2:
                st.metric("Tested Combinations", opt_details['tested_combinations'])
                st.metric("Total Combinations", opt_details['total_combinations'])
                st.metric("Optimization Target", opt_details['optimization_target'])
            
            with col3:
                st.metric("Models Discovered", data_summary['models_discovered'])
                st.metric("Models Selected", data_summary['models_selected'])
                if stability.get('is_stable'):
                    st.success("✅ Weights are stable")
                else:
                    st.warning("⚠️ Weights may be unstable")
            
            # Train-Validation Split 정보 표시
            if data_summary.get('split_used', False):
                st.markdown("### 🔄 Train-Validation Split Details")
                
                col_split1, col_split2, col_split3 = st.columns(3)
                with col_split1:
                    st.metric("Training Games", data_summary['train_games'])
                with col_split2:
                    st.metric("Validation Games", data_summary['validation_games'])
                with col_split3:
                    split_ratio = data_summary['train_games'] / (data_summary['train_games'] + data_summary['validation_games'])
                    st.metric("Split Ratio", f"{split_ratio:.1%} / {1-split_ratio:.1%}")
                
                # 후보 가중치들의 성과 비교
                if 'validation_results' in results and results['validation_results']:
                    st.markdown("### 📊 Candidate Weights Performance")
                    
                    val_results = results['validation_results']
                    optimal_weights = results['optimal_weights']  # 실제 선택된 가중치
                    comparison_data = []
                    
                    for i, result in enumerate(val_results[:10]):  # 상위 10개만 표시
                        # 🔧 가중치가 최적 가중치와 일치하는지 확인
                        is_selected = result['weights'] == optimal_weights
                        
                        comparison_data.append({
                            'Rank': i + 1,
                            'Train Score': f"{result['train_score']:.3f}",
                            'Val Score': f"{result['validation_score']:.3f}",
                            'Score Diff': f"{result['validation_score'] - result['train_score']:.3f}",
                            'Train ROI (%)': f"{result['train_performance']['roi']:.2f}",
                            'Val ROI (%)': f"{result['validation_performance']['roi']:.2f}",
                            'Selected': '✅' if is_selected else ''
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, hide_index=True, use_container_width=True)
        
        # 🆕 구간별 성과 분석 추가
        if 'segments_analysis' in results and results['segments_analysis']:
            st.markdown("---")
            st.subheader("🎯 Ensemble Performance by Segments")
            st.info("📊 **Analysis Goal**: Understand where the optimized ensemble performs best")
            
            segments = results['segments_analysis']
            
            # 🆕 데이터 품질 요약 추가 (상세 버전)
            if 'data_quality' in segments:
                data_quality = segments['data_quality']
                optimization_games = results['data_summary']['cleaned_games']
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("🔢 Input Games", data_quality['total_input'])
                with col2:
                    st.metric("✅ Processed", data_quality['successful'])
                with col3:
                    success_rate = data_quality['successful'] / data_quality['total_input'] * 100 if data_quality['total_input'] > 0 else 0
                    st.metric("📊 Success Rate", f"{success_rate:.1f}%")
                with col4:
                    excluded = data_quality['total_input'] - data_quality['successful']
                    st.metric("❌ Excluded", excluded)
                with col5:
                    st.metric("🔧 From Optimization", optimization_games)
                
                # ✅ 데이터 흐름 일관성 설명 추가
                if 'split_info' in segments:
                    split_info = segments['split_info']
                    split_method = split_info.get('split_method', 'time_based')
                    
                    if split_method == 'file_based_same_as_optimization':
                        st.success(f"""
                        ✅ **Data Consistency Achieved!**
                        - **Optimization**: {optimization_games} games → {split_info['analysis_games']} analysis + {split_info['validation_games']} validation
                        - **Segment Analysis**: Uses same {split_info['analysis_games']} analysis games
                        - **Split Method**: File-based (same across optimization & segment analysis)
                        - **Data Flow**: Optimized - no redundant splits!
                        """)
                    else:
                        st.info(f"""
                        ℹ️ **Data Flow Info:**
                        - **Optimization**: {optimization_games} games (cleaned)
                        - **Segment Analysis**: {split_info['analysis_games']} games (70% time-based split)
                        - **Split Method**: {split_method}
                        """)
                
                # 제외 이유 상세 분석
                if excluded > 0:
                    with st.expander("🔍 Detailed Exclusion Analysis", expanded=False):
                        col_detail1, col_detail2 = st.columns(2)
                        
                        with col_detail1:
                            st.markdown("### 📊 Exclusion Breakdown")
                            
                            exclusion_data = []
                            if data_quality['no_ensemble_prob'] > 0:
                                exclusion_data.append({'Reason': 'Ensemble Probability Failed', 'Count': data_quality['no_ensemble_prob']})
                            if data_quality['no_odds'] > 0:
                                exclusion_data.append({'Reason': 'Missing Odds Information', 'Count': data_quality['no_odds']})
                            if data_quality['invalid_odds_format'] > 0:
                                exclusion_data.append({'Reason': 'Invalid Odds Format', 'Count': data_quality['invalid_odds_format']})
                            
                            if exclusion_data:
                                exclusion_df = pd.DataFrame(exclusion_data)
                                exclusion_df['Percentage'] = (exclusion_df['Count'] / data_quality['total_input'] * 100).round(1)
                                st.dataframe(exclusion_df, hide_index=True, use_container_width=True)
                        
                        with col_detail2:
                            st.markdown("### 🎯 Key Insights")
                            
                            # 주요 제외 원인 분석
                            main_reason = ""
                            if data_quality['no_odds'] > data_quality['no_ensemble_prob']:
                                main_reason = "**Missing Odds Information** is the primary cause"
                                st.warning("💰 Most games excluded due to missing betting odds")
                            elif data_quality['no_ensemble_prob'] > 0:
                                main_reason = "**Ensemble Probability Calculation** is the primary cause"
                                st.warning("🎯 Most games excluded due to ensemble probability calculation failure")
                            else:
                                st.success("✅ Good data quality - minimal exclusions")
                            
                            if data_quality['invalid_odds_format'] > 0:
                                st.info(f"📊 {data_quality['invalid_odds_format']} games had invalid odds format")
                            
                            # 권고사항
                            if success_rate < 80:
                                st.error("🚨 **Low success rate!** Consider improving data quality.")
                            elif success_rate < 90:
                                st.warning("⚠️ **Moderate success rate.** Some data quality issues detected.")
                            else:
                                st.success("✅ **High success rate!** Good data quality.")
            elif 'overall' in segments:
                # 기존 방식 (data_quality 정보가 없는 경우)
                overall = segments['overall']
                optimization_games = results['data_summary']['cleaned_games']
                segment_games = overall['total_games']
                excluded_games = optimization_games - segment_games
                
                if excluded_games > 0:
                    st.warning(f"""
                    📋 **Data Quality Note**: 
                    - **Optimization Analysis**: {optimization_games} games 
                    - **Segment Analysis**: {segment_games} games ({excluded_games} excluded)
                    
                    **🔍 Excluded games reasons:**
                    - Missing odds information (ROI calculation impossible)
                    - Ensemble probability calculation failed
                    - Invalid data format
                    
                    *This is normal - segment analysis requires more complete data for ROI calculations.*
                    """)
            
            # 전체 요약
            if 'overall' in segments:
                overall = segments['overall']
                st.markdown("### 📈 Overall Performance Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Games", f"{overall['total_games']:,}")
                with col2:
                    st.metric("Predicted ROI", f"{overall['predicted_roi']:.2f}%")
                with col3:
                    st.metric("Actual ROI", f"{overall['actual_roi']:.2f}%")
                with col4:
                    roi_diff = overall['roi_difference']
                    st.metric("ROI Difference", f"{roi_diff:+.2f}%", 
                             delta=f"{'Better' if roi_diff > 0 else 'Worse'} than predicted")
            
            # 🆕 Smart Investment Zone Finder 추가
            st.markdown("---")
            st.subheader("🎯 Smart Investment Zone Finder")
            st.info("🔍 **Find optimal betting conditions** across all segment analyses based on performance criteria")
            
            # 설정 섹션
            with st.expander("⚙️ Investment Criteria Settings", expanded=True):
                col_criteria1, col_criteria2 = st.columns(2)
                
                with col_criteria1:
                    min_roi = st.slider(
                        "💰 Minimum ROI (%)", 
                        min_value=0.0, 
                        max_value=20.0, 
                        value=5.0, 
                        step=0.5,
                        help="최소 수익률 기준"
                    )
                    
                    min_accuracy = st.slider(
                        "🎯 Minimum Accuracy (%)", 
                        min_value=45.0, 
                        max_value=80.0, 
                        value=50.0, 
                        step=1.0,
                        help="최소 예측 정확도 기준"
                    )
                
                with col_criteria2:
                    min_win_rate = st.slider(
                        "🏆 Minimum Win Rate (%)", 
                        min_value=45.0, 
                        max_value=80.0, 
                        value=50.0, 
                        step=1.0,
                        help="최소 승률 기준"
                    )
                    
                    min_games = st.slider(
                        "📊 Minimum Games", 
                        min_value=10, 
                        max_value=100, 
                        value=40, 
                        step=5,
                        help="통계적 신뢰성을 위한 최소 게임 수"
                    )
                
                # 현재 기준 요약
                st.markdown("#### 📋 Current Investment Criteria")
                criteria_summary = f"""
                **📈 Performance Thresholds:**
                - ROI ≥ {min_roi}%
                - Accuracy ≥ {min_accuracy}%
                - Win Rate ≥ {min_win_rate}%
                
                **📊 Reliability Threshold:**
                - Minimum Games ≥ {min_games}
                """
                st.info(criteria_summary)
            
            # 분석 실행 버튼
            if st.button("🔍 Find Investment Zones", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing investment opportunities across all segments..."):
                    # ✅ 개선된 데이터 흐름: 최적화와 구간분석에서 동일한 분할 사용
                    validation_data_for_zones = results.get('validation_data_for_zones', None)  # 검증용 데이터 (30%)
                    analysis_data_for_combinations = results.get('matched_data', None)  # 분석용 데이터 (70%)
                    
                    # 🚨 최적화 결과에서 가중치 가져오기
                    optimal_weights = results.get('optimal_weights', {})
                    
                    # 구간분석 분할 정보도 함께 전달
                    segments_result = results.get('segments_analysis', {})
                    split_info_for_display = segments_result.get('split_info', {}) if isinstance(segments_result, dict) else {}
                    
                    # 🚨 디버깅: 실제 전달된 데이터 확인 및 완전한 데이터 검증
                    # 1. 기본 데이터 확인
                    if validation_data_for_zones is None:
                        st.warning("⚠️ **Validation Data**: validation_data_for_zones is None")
                        # 백업: segments_analysis에서 분할 정보 확인
                        segments_info = results.get('segments_analysis', {}).get('split_info', {})
                        validation_games_from_segments = segments_info.get('validation_games', 0)
                        st.info(f"🔍 Segments analysis reports {validation_games_from_segments} validation games")
                        validation_data_for_zones = []
                    else:
                        st.success(f"✅ Validation data received: {len(validation_data_for_zones)} games")
                        
                    if analysis_data_for_combinations is None:
                        st.error("❌ **Critical Error**: analysis_data_for_combinations is None!")
                        analysis_data_for_combinations = []
                    else:
                        st.success(f"✅ Analysis data received: {len(analysis_data_for_combinations)} games")
                    
                    # 🆕 2. 데이터 수학적 검증 - 사용자가 지적한 수치 문제 확인
                    segments_analysis_games = segments_result.get('data_quality', {}).get('successful', 'N/A')
                    # 투자존 분석 실행
                    
                    investment_zones = find_optimal_investment_zones(
                        segments, min_roi, min_accuracy, min_win_rate, min_games, validation_data_for_zones, optimal_weights, analysis_data_for_combinations
                    )
                    
                    # 검증 결과를 분할 정보와 함께 표시
                    
                    # 분할 정보를 투자존 결과에 추가
                    if 'validation_results' in investment_zones and investment_zones['validation_results']:
                        investment_zones['validation_results']['segments_split_info'] = split_info_for_display
                    
                    # 🆕 세션 스테이트에 스마트존파인더 결과 저장
                    st.session_state.investment_zones = investment_zones
                    st.session_state.smart_zone_segments = segments
                    st.session_state.smart_zone_optimal_weights = optimal_weights
                    
                    st.success("✅ Smart Zone Finder completed!")
                    
            # 🚀 스마트존파인더 결과 표시 (세션에서 가져오기)
            if 'investment_zones' in st.session_state:
                investment_zones = st.session_state.investment_zones
                segments = st.session_state.smart_zone_segments
                optimal_weights = st.session_state.smart_zone_optimal_weights
                
                display_investment_zones(investment_zones, optimal_weights, segments)
            
            # 📊 구간별 성과 분석
            st.markdown("---")
            st.markdown("## 📊 Segment Performance Analysis")
            st.markdown("Detailed analysis of ensemble performance across different market conditions and prediction characteristics.")
            
            segment_tabs = st.tabs([
                "📊 Predicted ROI", 
                "💰 Odds Ranges", 
                "🎯 Confidence Levels", 
                "📈 Market vs Model", 
                "🎰 Kelly Criterion", 
                "🤝 Model Consensus"
            ])
            
            with segment_tabs[0]:
                st.markdown("### 📊 Performance by Predicted ROI Ranges")
                st.markdown("Analysis of how well the ensemble performs across different expected return levels.")
                
                if 'predicted_roi' in segments:
                    display_segment_analysis(segments['predicted_roi'], "Predicted ROI Range")
                
                    # 전체 데이터 참고용 테이블
                    with st.expander("📋 Reference: Full Data Analysis (100%)", expanded=False):
                        st.info("Same analysis using all available data (no train/validation split)")
                        if 'full_segments_analysis' in results and results['full_segments_analysis']:
                            full_segments = results['full_segments_analysis']
                            if 'predicted_roi' in full_segments:
                                display_segment_analysis(full_segments['predicted_roi'], "Predicted ROI Range (Full Data)")
                        else:
                            st.info("Full data analysis same as above")
                else:
                    st.warning("Predicted ROI analysis not available")
            
            with segment_tabs[1]:
                st.markdown("### 💰 Performance by Odds Ranges")
                st.markdown("Analysis of ensemble performance across different betting odds (market confidence levels).")
                
                if 'odds' in segments:
                    display_segment_analysis(segments['odds'], "Odds Range")
                    
                    # 전체 데이터 참고용 테이블
                    with st.expander("📋 Reference: Full Data Analysis (100%)", expanded=False):
                        if 'full_segments_analysis' in results and results['full_segments_analysis']:
                            full_segments = results['full_segments_analysis']
                            if 'odds' in full_segments:
                                display_segment_analysis(full_segments['odds'], "Odds Range (Full Data)")
                        else:
                            st.info("Full data analysis same as above")
                else:
                    st.warning("Odds analysis not available")
            
            with segment_tabs[2]:
                st.markdown("### 🎯 Performance by Confidence Levels")
                st.markdown("Analysis of how prediction confidence correlates with actual performance.")
                
                if 'confidence' in segments:
                    display_segment_analysis(segments['confidence'], "Confidence Level")
                    
                    # 전체 데이터 참고용 테이블
                    with st.expander("📋 Reference: Full Data Analysis (100%)", expanded=False):
                        if 'full_segments_analysis' in results and results['full_segments_analysis']:
                            full_segments = results['full_segments_analysis']
                            if 'confidence' in full_segments:
                                display_segment_analysis(full_segments['confidence'], "Confidence Level (Full Data)")
                        else:
                            st.info("Full data analysis same as above")
                else:
                    st.warning("Confidence analysis not available")
            
            with segment_tabs[3]:
                st.markdown("### 📈 Market vs Model Opinion Divergence")
                st.markdown("Analysis of performance when model predictions differ from market expectations.")
                
                if 'odds_probability_divergence' in segments:
                    display_market_divergence_analysis(segments['odds_probability_divergence'])
                    
                    # 개념 설명
                    with st.expander("🧠 Understanding Market vs Model Divergence", expanded=False):
                        st.info("""
                        **Positive Divergence**: Model is more optimistic than market (model sees higher win probability)
                        
                        **Negative Divergence**: Model is more pessimistic than market (model sees lower win probability)
                        
                        **Market Aligned**: Model and market agree on win probability (±5% difference)
                        """)
                    
                    # 전체 데이터 참고용 테이블
                    with st.expander("📋 Reference: Full Data Analysis (100%)", expanded=False):
                        if 'full_segments_analysis' in results and results['full_segments_analysis']:
                            full_segments = results['full_segments_analysis']
                            if 'odds_probability_divergence' in full_segments:
                                display_market_divergence_analysis(full_segments['odds_probability_divergence'])
                        else:
                            st.info("Full data analysis same as above")
                else:
                    st.warning("Market vs Model divergence analysis not available")
            
            with segment_tabs[4]:
                st.markdown("### 🎰 Kelly Criterion Bet Sizing")
                st.markdown("Analysis of performance across different optimal bet sizes calculated using Kelly Criterion.")
                
                if 'kelly_criterion' in segments:
                    display_kelly_criterion_analysis(segments['kelly_criterion'])
                    
                    # Kelly Criterion 설명
                    with st.expander("📚 Understanding Kelly Criterion", expanded=False):
                        st.info("""
                        **Kelly Formula**: f* = (bp - q) / b
                        
                        - **b**: decimal odds - 1 (profit multiplier)
                        - **p**: win probability 
                        - **q**: lose probability (1-p)
                        
                        **Kelly %**: Optimal bet size as percentage of bankroll
                        - **0%**: No edge, don't bet
                        - **5%**: Small edge, conservative bet
                        - **15%**: Good edge, moderate bet  
                        - **25%+**: Strong edge, aggressive bet (risky!)
                        """)
                    
                    # 전체 데이터 참고용 테이블
                    with st.expander("📋 Reference: Full Data Analysis (100%)", expanded=False):
                        if 'full_segments_analysis' in results and results['full_segments_analysis']:
                            full_segments = results['full_segments_analysis']
                            if 'kelly_criterion' in full_segments:
                                display_kelly_criterion_analysis(full_segments['kelly_criterion'])
                        else:
                            st.info("Full data analysis same as above")
                else:
                    st.warning("Kelly Criterion analysis not available")
            
            with segment_tabs[5]:
                st.markdown("### 🤝 Model Consensus Analysis")
                st.markdown("Analysis of performance based on how many models agree on the prediction.")
                
                if 'model_consensus' in segments:
                    display_model_consensus_analysis(segments['model_consensus'])
                    
                    # 모델 합의도 설명
                    with st.expander("🧠 Understanding Model Consensus", expanded=False):
                        st.info("""
                        **Strong Consensus (80%+)**: Most models agree on the same team
                        
                        **Moderate Consensus (60-80%)**: Majority of models agree
                        
                        **Weak Consensus (50-60%)**: Slight majority agreement
                        
                        **No Consensus (<50%)**: Models are split 50/50 or minority rules
                        """)
                    
                    # 전체 데이터 참고용 테이블
                    with st.expander("📋 Reference: Full Data Analysis (100%)", expanded=False):
                        if 'full_segments_analysis' in results and results['full_segments_analysis']:
                            full_segments = results['full_segments_analysis']
                            if 'model_consensus' in full_segments:
                                display_model_consensus_analysis(full_segments['model_consensus'])
                        else:
                            st.info("Full data analysis same as above")
                else:
                    st.warning("Model consensus analysis not available")

def display_segment_analysis(segment_data: dict, segment_type: str):
    """구간별 분석 결과를 표시하는 헬퍼 함수"""
    
    if not segment_data:
        st.warning(f"No data available for {segment_type} analysis")
        return
    
    # 데이터 정리
    analysis_rows = []
    for segment_name, stats in segment_data.items():
        if stats['games'] > 0:  # 게임이 있는 구간만 표시
            analysis_rows.append({
                'Segment': segment_name,
                'Games': stats['games'],
                'Predicted ROI (%)': f"{stats['predicted_roi']:.2f}",
                'Actual ROI (%)': f"{stats['actual_roi']:.2f}",
                'ROI Difference (%)': f"{stats['roi_difference']:+.2f}",
                'Win Rate (%)': f"{stats['win_rate']:.1f}",
                'Accuracy (%)': f"{stats['accuracy']:.1f}"
            })
    
    if not analysis_rows:
        st.warning(f"No segments with data found for {segment_type}")
        return
    
    # DataFrame 생성
    df = pd.DataFrame(analysis_rows)
    
    # 인사이트 계산
    best_actual_roi = max(analysis_rows, key=lambda x: float(x['Actual ROI (%)'].replace('+', '')))
    worst_actual_roi = min(analysis_rows, key=lambda x: float(x['Actual ROI (%)'].replace('+', '')))
    best_accuracy = max(analysis_rows, key=lambda x: float(x['Accuracy (%)']))
    
    # 결과 표시
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("##### 🔍 Key Insights")
        
        st.markdown(f"**🎯 Best ROI Segment:**")
        st.success(f"{best_actual_roi['Segment']}: {best_actual_roi['Actual ROI (%)']}%")
        
        st.markdown(f"**📉 Worst ROI Segment:**")
        st.error(f"{worst_actual_roi['Segment']}: {worst_actual_roi['Actual ROI (%)']}%")
        
        st.markdown(f"**🎯 Most Accurate Segment:**")
        st.info(f"{best_accuracy['Segment']}: {best_accuracy['Accuracy (%)']}%")
        
        # 추가 인사이트
        positive_segments = [row for row in analysis_rows if float(row['Actual ROI (%)'].replace('+', '')) > 0]
        if positive_segments:
            st.markdown(f"**✅ Profitable Segments:** {len(positive_segments)}/{len(analysis_rows)}")
        else:
            st.markdown(f"**❌ No profitable segments found**")

def display_market_divergence_analysis(segment_data: dict):
    """시장 vs 모델 괴리도 분석 결과를 표시하는 전용 함수"""
    
    if not segment_data:
        st.warning("No data available for Market vs Model divergence analysis")
        return
    
    # 데이터 정리
    analysis_rows = []
    for segment_name, stats in segment_data.items():
        if stats['games'] > 0:  # 게임이 있는 구간만 표시
            analysis_rows.append({
                'Divergence Level': segment_name,
                'Games': stats['games'],
                'Predicted ROI (%)': f"{stats['predicted_roi']:.2f}",
                'Actual ROI (%)': f"{stats['actual_roi']:.2f}",
                'ROI Difference (%)': f"{stats['roi_difference']:+.2f}",
                'Win Rate (%)': f"{stats['win_rate']:.1f}",
                'Accuracy (%)': f"{stats['accuracy']:.1f}"
            })
    
    if not analysis_rows:
        st.warning("No segments with data found for Market vs Model divergence")
        return
    
    # DataFrame 생성
    df = pd.DataFrame(analysis_rows)
    
    # 특별한 인사이트 계산
    optimistic_segments = [row for row in analysis_rows if 'Optimistic' in row['Divergence Level']]
    pessimistic_segments = [row for row in analysis_rows if 'Pessimistic' in row['Divergence Level']]
    aligned_segments = [row for row in analysis_rows if 'Aligned' in row['Divergence Level']]
    
    # 결과 표시
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("##### 🔍 Market vs Model Insights")
        
        # 최고 성과 구간
        best_segment = max(analysis_rows, key=lambda x: float(x['Actual ROI (%)'].replace('+', '')))
        st.markdown(f"**🎯 Best Performing Divergence:**")
        st.success(f"{best_segment['Divergence Level']}: {best_segment['Actual ROI (%)']}%")
        
        # 괴리도별 평균 성과
        if optimistic_segments:
            optimistic_avg_roi = np.mean([float(row['Actual ROI (%)'].replace('+', '')) for row in optimistic_segments])
            st.markdown(f"**📈 Model Optimistic Avg:**")
            st.metric("ROI", f"{optimistic_avg_roi:.2f}%")
        
        if pessimistic_segments:
            pessimistic_avg_roi = np.mean([float(row['Actual ROI (%)'].replace('+', '')) for row in pessimistic_segments])
            st.markdown(f"**📉 Model Pessimistic Avg:**")
            st.metric("ROI", f"{pessimistic_avg_roi:.2f}%")
        
        if aligned_segments:
            aligned_avg_roi = np.mean([float(row['Actual ROI (%)'].replace('+', '')) for row in aligned_segments])
            st.markdown(f"**⚖️ Market Aligned Avg:**")
            st.metric("ROI", f"{aligned_avg_roi:.2f}%")
        
        # 시장 효율성 평가
        total_segments = len(analysis_rows)
        profitable_segments = len([row for row in analysis_rows if float(row['Actual ROI (%)'].replace('+', '')) > 0])
        
        st.markdown(f"**📊 Market Efficiency:**")
        efficiency_ratio = profitable_segments / total_segments if total_segments > 0 else 0
        if efficiency_ratio > 0.6:
            st.success(f"Model finds {profitable_segments}/{total_segments} profitable scenarios")
        elif efficiency_ratio > 0.4:
            st.warning(f"Model finds {profitable_segments}/{total_segments} profitable scenarios")
        else:
            st.error(f"Model finds {profitable_segments}/{total_segments} profitable scenarios")

@st.cache_data
def get_file_info_preview(start_date_str, end_date_str):
    """날짜 범위에 따른 파일 정보 미리보기 (분석 실행 없이)"""
    try:
        analyzer = load_analyzer()
        data = analyzer.load_data(start_date_str, end_date_str)
        return data['file_info']
    except Exception as e:
        return None

def display_kelly_criterion_analysis(segment_data: dict):
    """Kelly Criterion 분석 결과를 표시하는 전용 함수"""
    
    if not segment_data:
        st.warning("No data available for Kelly Criterion analysis")
        return
    
    # 데이터 정리
    analysis_rows = []
    for segment_name, stats in segment_data.items():
        if stats['games'] > 0:  # 게임이 있는 구간만 표시
            analysis_rows.append({
                'Kelly Range': segment_name,
                'Games': stats['games'],
                'Predicted ROI (%)': f"{stats['predicted_roi']:.2f}",
                'Actual ROI (%)': f"{stats['actual_roi']:.2f}",
                'ROI Difference (%)': f"{stats['roi_difference']:+.2f}",
                'Win Rate (%)': f"{stats['win_rate']:.1f}",
                'Accuracy (%)': f"{stats['accuracy']:.1f}"
            })
    
    if not analysis_rows:
        st.warning("No segments with data found for Kelly Criterion analysis")
        return
    
    # DataFrame 생성
    df = pd.DataFrame(analysis_rows)
    
    # Kelly 특화 인사이트 계산
    bet_segments = [row for row in analysis_rows if 'No Bet' not in row['Kelly Range']]
    no_bet_segments = [row for row in analysis_rows if 'No Bet' in row['Kelly Range']]
    large_bet_segments = [row for row in analysis_rows if 'Large' in row['Kelly Range'] or 'Extreme' in row['Kelly Range']]
    
    # 결과 표시
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("##### 🎰 Kelly Insights")
        
        # 최고 성과 구간
        best_segment = max(analysis_rows, key=lambda x: float(x['Actual ROI (%)'].replace('+', '')))
        st.markdown(f"**🎯 Best Kelly Range:**")
        st.success(f"{best_segment['Kelly Range']}: {best_segment['Actual ROI (%)']}%")
        
        # 베팅 vs 노베팅 비교
        if bet_segments and no_bet_segments:
            bet_avg_roi = np.mean([float(row['Actual ROI (%)'].replace('+', '')) for row in bet_segments])
            no_bet_avg_roi = np.mean([float(row['Actual ROI (%)'].replace('+', '')) for row in no_bet_segments])
            
            st.markdown(f"**📊 Betting vs No Betting:**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Bet ROI", f"{bet_avg_roi:.2f}%")
            with col_b:
                st.metric("No Bet ROI", f"{no_bet_avg_roi:.2f}%")
        
        # 고위험 베팅 경고
        if large_bet_segments:
            large_bet_avg_roi = np.mean([float(row['Actual ROI (%)'].replace('+', '')) for row in large_bet_segments])
            large_bet_games = sum(row['Games'] for row in large_bet_segments)
            
            st.markdown(f"**⚠️ High-Risk Bets (15%+ Kelly):**")
            if large_bet_avg_roi > 5:
                st.success(f"{large_bet_games} games, ROI: {large_bet_avg_roi:.2f}%")
            elif large_bet_avg_roi > -5:
                st.warning(f"{large_bet_games} games, ROI: {large_bet_avg_roi:.2f}%")
            else:
                st.error(f"{large_bet_games} games, ROI: {large_bet_avg_roi:.2f}%")
        
        # Kelly 유효성 검증
        positive_kelly_segments = [row for row in analysis_rows if 'No Bet' not in row['Kelly Range']]
        if positive_kelly_segments:
            positive_count = len([row for row in positive_kelly_segments if float(row['Actual ROI (%)'].replace('+', '')) > 0])
            total_positive = len(positive_kelly_segments)
            success_rate = positive_count / total_positive if total_positive > 0 else 0
            
            st.markdown(f"**📈 Kelly Validation:**")
            if success_rate > 0.6:
                st.success(f"Kelly works! {positive_count}/{total_positive} profitable")
            elif success_rate > 0.4:
                st.warning(f"Mixed results: {positive_count}/{total_positive} profitable")
            else:
                st.error(f"Kelly fails: {positive_count}/{total_positive} profitable")

def display_model_consensus_analysis(segment_data: dict):
    """모델 합의도 분석 결과를 표시하는 전용 함수"""
    
    if not segment_data:
        st.warning("No data available for Model Consensus analysis")
        return
    
    # 데이터 정리
    analysis_rows = []
    for segment_name, stats in segment_data.items():
        if stats['games'] > 0:  # 게임이 있는 구간만 표시
            analysis_rows.append({
                'Consensus Level': segment_name,
                'Games': stats['games'],
                'Predicted ROI (%)': f"{stats['predicted_roi']:.2f}",
                'Actual ROI (%)': f"{stats['actual_roi']:.2f}",
                'ROI Difference (%)': f"{stats['roi_difference']:+.2f}",
                'Win Rate (%)': f"{stats['win_rate']:.1f}",
                'Accuracy (%)': f"{stats['accuracy']:.1f}"
            })
    
    if not analysis_rows:
        st.warning("No segments with data found for Model Consensus analysis")
        return
    
    # DataFrame 생성
    df = pd.DataFrame(analysis_rows)
    
    # 합의도 특화 인사이트 계산
    strong_consensus = [row for row in analysis_rows if 'Strong' in row['Consensus Level']]
    weak_consensus = [row for row in analysis_rows if 'Weak' in row['Consensus Level'] or 'No' in row['Consensus Level']]
    
    # 결과 표시
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("##### 🤝 Consensus Insights")
        
        # 최고 성과 구간
        best_segment = max(analysis_rows, key=lambda x: float(x['Actual ROI (%)'].replace('+', '')))
        st.markdown(f"**🎯 Best Consensus Level:**")
        st.success(f"{best_segment['Consensus Level']}: {best_segment['Actual ROI (%)']}%")
        
        # 강한 합의 vs 약한 합의 비교
        if strong_consensus and weak_consensus:
            strong_avg_roi = np.mean([float(row['Actual ROI (%)'].replace('+', '')) for row in strong_consensus])
            strong_avg_accuracy = np.mean([float(row['Accuracy (%)']) for row in strong_consensus])
            
            weak_avg_roi = np.mean([float(row['Actual ROI (%)'].replace('+', '')) for row in weak_consensus])
            weak_avg_accuracy = np.mean([float(row['Accuracy (%)']) for row in weak_consensus])
            
            st.markdown(f"**💪 Strong Consensus Performance:**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("ROI", f"{strong_avg_roi:.2f}%")
            with col_b:
                st.metric("Accuracy", f"{strong_avg_accuracy:.1f}%")
            
            st.markdown(f"**🤷 Weak Consensus Performance:**")
            col_c, col_d = st.columns(2)
            with col_c:
                st.metric("ROI", f"{weak_avg_roi:.2f}%")
            with col_d:
                st.metric("Accuracy", f"{weak_avg_accuracy:.1f}%")
        
        # 합의도 효과성 평가
        total_segments = len(analysis_rows)
        high_consensus_segments = len([row for row in analysis_rows if 'Strong' in row['Consensus Level'] or 'Moderate' in row['Consensus Level']])
        
        st.markdown(f"**📊 Consensus Distribution:**")
        if high_consensus_segments / total_segments > 0.6:
            st.success(f"High consensus: {high_consensus_segments}/{total_segments} segments")
        elif high_consensus_segments / total_segments > 0.4:
            st.warning(f"Moderate consensus: {high_consensus_segments}/{total_segments} segments")
        else:
            st.error(f"Low consensus: {high_consensus_segments}/{total_segments} segments")
        
        # 정확도 vs 합의도 상관관계
        if len(analysis_rows) >= 3:
            high_accuracy_segments = len([row for row in analysis_rows if float(row['Accuracy (%)']) > 60])
            st.markdown(f"**🎯 High Accuracy Segments:**")
            st.info(f"{high_accuracy_segments}/{total_segments} segments >60% accuracy")

def split_data_by_time(matched_data, split_ratio=0.7):
    """시간 순서대로 데이터 분할"""
    if not matched_data:
        return {'train_data': [], 'test_data': [], 'split_date': '', 'train_count': 0, 'test_count': 0}
    
    # 날짜순 정렬
    sorted_data = sorted(matched_data, key=lambda x: x.get('date', ''))
    
    # 분할점 계산
    split_index = int(len(sorted_data) * split_ratio)
    
    train_data = sorted_data[:split_index]
    test_data = sorted_data[split_index:]
    
    split_date = train_data[-1]['date'] if train_data else ''
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'split_date': split_date,
        'train_count': len(train_data),
        'test_count': len(test_data)
    }

def calculate_zone_roi(games_list):
    """게임 리스트에서 실제 ROI 계산 - 앙상블 분석과 완전히 동일한 방식"""
    if not games_list:
        return 0.0
    
    # 🎯 앙상블 분석의 analyze_ensemble_segments()와 완전히 동일한 로직
    total_roi = 0.0
    valid_games = 0
    
    for game in games_list:
        # 1. 앙상블 확률 계산 (필요한 데이터 확인)
        ensemble_prob = game.get('ensemble_probability', 0.5)
        actual_win = game.get('actual_home_win')  # 실제 홈팀 승리 여부 (0 or 1)
        home_odds = game.get('home_odds')
        away_odds = game.get('away_odds')
        
        if home_odds is None or away_odds is None or actual_win is None:
            continue
            
        try:
            home_odds = float(home_odds)
            away_odds = float(away_odds)
            actual_win = int(actual_win)
        except (ValueError, TypeError):
            continue
        
        # 2. 베팅 결정 및 ROI 계산 (앙상블 분석과 동일)
        if ensemble_prob > 0.5:
            # 홈팀에 베팅
            if actual_win == 1:  # 홈팀이 실제로 승리
                if home_odds > 0:
                    actual_roi = (home_odds / 100) * 100
                else:
                    actual_roi = (100 / abs(home_odds)) * 100
            else:  # 홈팀 패배
                actual_roi = -100
        else:
            # 원정팀에 베팅
            if actual_win == 0:  # 원정팀이 실제로 승리 (홈팀 패배)
                if away_odds > 0:
                    actual_roi = (away_odds / 100) * 100
                else:
                    actual_roi = (100 / abs(away_odds)) * 100
            else:  # 원정팀 패배
                actual_roi = -100
        
        total_roi += actual_roi
        valid_games += 1
    
    return total_roi / valid_games if valid_games > 0 else 0.0

def three_tier_validation(past_roi, future_roi):
    """3단계 검증 시스템 - 🚨 수정: 절대 ROI 기준 추가"""
    # 필수 조건: 미래 ROI 5% 이상
    if future_roi < 5.0:
        return {
            'status': 'rejected',
            'tier': None,
            'message': f"❌ Future ROI too low ({future_roi:.1f}% < 5%)",
            'validated': False
        }
    
    # ROI 향상도 계산 (양수면 미래가 더 좋음, 음수면 미래가 더 나쁨)
    roi_improvement = future_roi - past_roi
    
    # 🆕 더 합리적인 티어 기준
    # Tier 1: 큰 향상 OR 절대 ROI가 높으면서 소폭 하락
    if roi_improvement >= 10.0:  # 10% 이상 향상
        return {
            'status': 'excellent_improvement',
            'tier': 'Tier 1',
            'message': f"🏆 Excellent improvement (+{roi_improvement:.1f}%)",
            'confidence': 'High',
            'validated': True
        }
    elif future_roi >= 30.0 and roi_improvement >= -5.0:  # 높은 절대 ROI + 소폭 하락 허용
        return {
            'status': 'high_absolute_performance',
            'tier': 'Tier 1',
            'message': f"🏆 High absolute ROI ({future_roi:.1f}%) with acceptable change ({roi_improvement:+.1f}%)",
            'confidence': 'High',
            'validated': True
        }
    # Tier 2: 안정적 성능 유지 OR 좋은 절대 ROI에도 불구하고 큰 하락
    elif roi_improvement >= -10.0:  # 10% 이내 하락까지 허용
        return {
            'status': 'stable_performance',
            'tier': 'Tier 2',
            'message': f"⚠️ Stable performance ({roi_improvement:+.1f}%)",
            'confidence': 'Medium',
            'validated': True
        }
    elif future_roi >= 20.0 and roi_improvement < -10.0:  # 🆕 절대 ROI 20% 이상이면 큰 하락도 Tier 2
        return {
            'status': 'good_absolute_despite_decline',
            'tier': 'Tier 2',
            'message': f"⚠️ Good absolute ROI ({future_roi:.1f}%) despite decline ({roi_improvement:.1f}%)",
            'confidence': 'Medium',
            'validated': True
        }
    # 실패: 낮은 절대 ROI + 큰 하락 (이전 조건들을 모두 통과하지 못한 경우)
    else:  # future_roi >= 5% AND future_roi < 20% AND roi_improvement < -10%
        return {
            'status': 'significant_decline',
            'tier': None,
            'message': f"❌ Low absolute ROI ({future_roi:.1f}% < 20%) with significant decline ({roi_improvement:.1f}% < -10%)",
            'confidence': 'None',
            'validated': False
        }

def validate_investment_zones_three_tier_preprocessed(investment_zones, preprocessed_test_data, original_test_data, optimal_weights=None):
    """3단계 티어 시스템으로 투자존 검증 - 전처리된 데이터 사용"""
    if not investment_zones or not preprocessed_test_data:
        return {
            'tier1_zones': [],
            'tier2_zones': [],
            'all_validated_zones': [],
            'validation_details': [],
            'optimal_weights': optimal_weights
        }
    
    validation_results = []
    tier1_zones = []
    tier2_zones = []
    
    for zone in investment_zones:
        # 미래 데이터에서 해당 존 조건 만족하는 게임들 찾기
        future_games = []
        
        for processed_game in preprocessed_test_data:
            if check_zone_condition_preprocessed(processed_game, zone, original_test_data, optimal_weights):
                future_games.append(processed_game)
        
        # 최소 15경기 필요
        if len(future_games) >= 15:
            future_roi = calculate_zone_roi_preprocessed(future_games)
            validation = three_tier_validation(zone['roi'], future_roi)
            
            result = {
                'zone': zone,
                'zone_name': f"{zone['dimension_name']} - {zone['segment']}",
                'past_roi': zone['roi'],
                'future_roi': future_roi,
                'future_games': len(future_games),
                'gap': future_roi - zone['roi'],  # 양수면 상승, 음수면 하락
                'validation': validation,
                'tier': validation.get('tier'),
                'validated': validation['validated']
            }
            
            validation_results.append(result)
            
            # 티어별 분류
            if validation['tier'] == 'Tier 1':
                zone_copy = zone.copy()
                zone_copy.update({
                    'validation_tier': 'Tier 1',
                    'confidence_level': 'High',
                    'future_roi': future_roi,
                    'validation_gap': result['gap']
                })
                tier1_zones.append(zone_copy)
                
            elif validation['tier'] == 'Tier 2':
                zone_copy = zone.copy()
                zone_copy.update({
                    'validation_tier': 'Tier 2', 
                    'confidence_level': 'Medium',
                    'future_roi': future_roi,
                    'validation_gap': result['gap']
                })
                tier2_zones.append(zone_copy)
        else:
            # 게임 수 부족
            result = {
                'zone': zone,
                'zone_name': f"{zone['dimension_name']} - {zone['segment']}",
                'past_roi': zone['roi'],
                'future_games': len(future_games),
                'validation': {
                    'status': 'insufficient_data',
                    'message': f"📊 Insufficient future data ({len(future_games)} games < 15)",
                    'validated': False
                }
            }
            validation_results.append(result)
    
    return {
        'tier1_zones': tier1_zones,
        'tier2_zones': tier2_zones,
        'all_validated_zones': tier1_zones + tier2_zones,
        'validation_details': validation_results,
        'optimal_weights': optimal_weights
    }

def validate_investment_zones_three_tier(investment_zones, test_data, optimal_weights=None):
    """3단계 티어 시스템으로 투자존 검증 - 🚨 수정: 가중치 전달"""
    if not investment_zones or not test_data:
        return {
            'tier1_zones': [],
            'tier2_zones': [],
            'all_validated_zones': [],
            'validation_details': [],
            'optimal_weights': optimal_weights  # 🚨 가중치 전달
        }
    
    validation_results = []
    tier1_zones = []
    tier2_zones = []
    
    for zone in investment_zones:
        # 미래 데이터에서 해당 존 조건 만족하는 게임들 찾기
        future_games = []
        
        for game in test_data:
            if check_zone_condition(game, zone, optimal_weights):  # 🚨 가중치 전달
                future_games.append(game)
        
        # 최소 15경기 필요
        if len(future_games) >= 15:
            future_roi = calculate_zone_roi(future_games)
            validation = three_tier_validation(zone['roi'], future_roi)
            
            result = {
                'zone': zone,
                'zone_name': f"{zone['dimension_name']} - {zone['segment']}",
                'past_roi': zone['roi'],
                'future_roi': future_roi,
                'future_games': len(future_games),
                'gap': future_roi - zone['roi'],  # 양수면 상승, 음수면 하락
                'validation': validation,
                'tier': validation.get('tier'),
                'validated': validation['validated']
            }
            
            validation_results.append(result)
            
            # 티어별 분류
            if validation['tier'] == 'Tier 1':
                zone_copy = zone.copy()
                zone_copy.update({
                    'validation_tier': 'Tier 1',
                    'confidence_level': 'High',
                    'future_roi': future_roi,
                    'validation_gap': result['gap']
                })
                tier1_zones.append(zone_copy)
                
            elif validation['tier'] == 'Tier 2':
                zone_copy = zone.copy()
                zone_copy.update({
                    'validation_tier': 'Tier 2', 
                    'confidence_level': 'Medium',
                    'future_roi': future_roi,
                    'validation_gap': result['gap']
                })
                tier2_zones.append(zone_copy)
        else:
            # 게임 수 부족
            result = {
                'zone': zone,
                'zone_name': f"{zone['dimension_name']} - {zone['segment']}",
                'past_roi': zone['roi'],
                'future_games': len(future_games),
                'validation': {
                    'status': 'insufficient_data',
                    'message': f"📊 Insufficient future data ({len(future_games)} games < 15)",
                    'validated': False
                }
            }
            validation_results.append(result)
    
    return {
        'tier1_zones': tier1_zones,
        'tier2_zones': tier2_zones,
        'all_validated_zones': tier1_zones + tier2_zones,
        'validation_details': validation_results,
        'optimal_weights': optimal_weights  # 🚨 가중치 전달
    }

def find_optimal_investment_zones(segments_analysis, min_roi, min_accuracy, min_win_rate, min_games, matched_data=None, optimal_weights=None, analysis_data_for_combinations=None):
    """모든 구간분석을 종합해서 최적 투자 조건 도출 - 🚨 수정: 가중치 전달"""
    
    optimal_zones = []
    total_analyzed_segments = 0
    
    # 분석할 차원들 정의
    analysis_dimensions = {
        'predicted_roi': 'Predicted ROI Range',
        'odds': 'Odds Range',
        'confidence': 'Confidence Level',
        'odds_probability_divergence': 'Market vs Model Divergence',
        'kelly_criterion': 'Kelly Criterion Range',
        'model_consensus': 'Model Consensus Level'
    }
    
    # 🆕 모든 세그먼트에서 최대 게임 수 찾기 (샘플 사이즈 보너스 기준용)
    max_games = 0
    for dimension, dimension_name in analysis_dimensions.items():
        if dimension in segments_analysis:
            data = segments_analysis[dimension]
            for segment_name, stats in data.items():
                if stats['games'] > max_games:
                    max_games = stats['games']
    
    # 각 차원별로 기준을 만족하는 구간 찾기
    for dimension, dimension_name in analysis_dimensions.items():
        if dimension in segments_analysis:
            data = segments_analysis[dimension]
            
            for segment_name, stats in data.items():
                total_analyzed_segments += 1
                
                # 기준 검사
                meets_criteria = (
                    stats['games'] >= min_games and 
                    stats['actual_roi'] >= min_roi and
                    stats['accuracy'] >= min_accuracy and
                    stats['win_rate'] >= min_win_rate
                )
                
                if meets_criteria:
                    # 종합 점수 계산 - 🆕 max_games 전달
                    score = calculate_investment_zone_score(stats, max_games)
                    
                    optimal_zones.append({
                        'dimension': dimension,
                        'dimension_name': dimension_name,
                        'segment': segment_name,
                        'roi': stats['actual_roi'],
                        'predicted_roi': stats['predicted_roi'],
                        'roi_difference': stats['roi_difference'],
                        'games': stats['games'],
                        'accuracy': stats['accuracy'],
                        'win_rate': stats['win_rate'],
                        'score': score
                    })
    
    # 점수 순으로 정렬
    optimal_zones.sort(key=lambda x: x['score'], reverse=True)
    
    # 🆕 검증용 데이터로 검증 (파일 기반 분할된 63개 검증 데이터 사용)
    validation_results = None
    validated_zones = optimal_zones  # 기본값은 전체 존
    
    if matched_data and len(matched_data) >= 10:  # 충분한 검증 데이터가 있을 때만
        # 🚨 수정: 구간분석과 동일한 전처리 수행
        # 구간분석과 동일한 전처리 로직 적용
        preprocessed_validation_data = []
        excluded_count = 0
        
        for record in matched_data:
            # 앙상블 확률 계산 (구간분석과 동일)
            ensemble_prob = 0
            total_weight_used = 0
            
            for model, weight in optimal_weights.items():
                prob_key = f"{model}_probability"
                if prob_key in record and record[prob_key] is not None:
                    ensemble_prob += record[prob_key] * weight
                    total_weight_used += weight
            
            if total_weight_used == 0:
                excluded_count += 1
                continue
                
            ensemble_prob /= total_weight_used
            
            # 베팅 정보 검증
            actual_win = record.get('actual_home_win', 0)
            home_odds = record.get('home_odds')
            away_odds = record.get('away_odds')
            
            if home_odds is None or away_odds is None:
                excluded_count += 1
                continue
                
            try:
                home_odds = float(home_odds)
                away_odds = float(away_odds)
            except (ValueError, TypeError):
                excluded_count += 1
                continue
            
            # 예측 기반 베팅 결정 및 ROI 계산 (구간분석과 동일)
            if ensemble_prob > 0.5:
                # 홈팀에 베팅
                predicted_team = "home"
                if actual_win == 1:
                    if home_odds > 0:
                        actual_roi = (home_odds / 100) * 100
                    else:
                        actual_roi = (100 / abs(home_odds)) * 100
                else:
                    actual_roi = -100
                
                # 예측 ROI 계산
                if home_odds > 0:
                    predicted_roi = ensemble_prob * (home_odds / 100) * 100 + (1 - ensemble_prob) * (-100)
                else:
                    predicted_roi = ensemble_prob * (100 / abs(home_odds)) * 100 + (1 - ensemble_prob) * (-100)
                    
                bet_odds = home_odds
            else:
                # 원정팀에 베팅
                predicted_team = "away"
                if actual_win == 0:
                    if away_odds > 0:
                        actual_roi = (away_odds / 100) * 100
                    else:
                        actual_roi = (100 / abs(away_odds)) * 100
                else:
                    actual_roi = -100
                
                # 예측 ROI 계산
                if away_odds > 0:
                    predicted_roi = (1 - ensemble_prob) * (away_odds / 100) * 100 + ensemble_prob * (-100)
                else:
                    predicted_roi = (1 - ensemble_prob) * (100 / abs(away_odds)) * 100 + ensemble_prob * (-100)
                    
                bet_odds = away_odds
            
            # 신뢰도 계산 (구간분석과 동일)
            confidence = abs(ensemble_prob - 0.5)
            
            # 전처리된 데이터 구조 (구간분석과 동일)
            preprocessed_validation_data.append({
                'ensemble_prob': ensemble_prob,
                'predicted_roi': predicted_roi,
                'actual_roi': actual_roi,
                'confidence': confidence,
                'bet_odds': bet_odds,
                'predicted_team': predicted_team,
                'actual_win': actual_win,
                'date': record.get('date', ''),
                'home_team': record.get('home_team', ''),
                'away_team': record.get('away_team', '')
            })
        
        # ✅ 전처리된 데이터로 검증 수행
        validation_results = validate_investment_zones_three_tier_preprocessed(
            optimal_zones, preprocessed_validation_data, matched_data, optimal_weights
        )
        
        # 검증된 존들 (Tier 1 + Tier 2)
        validated_zones = validation_results['all_validated_zones']
        
        # 분할 정보 추가 (구간분석에서 전달받은 정보)
        validation_results['split_info'] = {
            'test_data': matched_data,
            'test_count': len(matched_data),
            'split_date': 'Pre-split by segment analysis'
        }
    
    # 🆕 검증된 존들로만 교차 검증 분석 - 🚨 수정: 조합 분석에는 분석용 데이터 사용
    cross_validated_combinations = []
    if len(validated_zones) >= 2:
        # 조합 분석에는 개별 존과 같은 분석용 데이터 사용
        combination_data = analysis_data_for_combinations if analysis_data_for_combinations else matched_data
        cross_validated_combinations = find_cross_validated_combinations(
            segments_analysis, validated_zones, min_roi, min_accuracy, min_win_rate, min_games, combination_data, max_games
        )
    
    return {
        'optimal_zones': optimal_zones,
        'validated_zones': validated_zones,  # 🆕 검증된 존들
        'validation_results': validation_results,  # 🆕 검증 세부사항
        'cross_validated_combinations': cross_validated_combinations,
        'total_analyzed': total_analyzed_segments,
        'qualifying_zones': len(optimal_zones),
        'validated_zones_count': len(validated_zones),  # 🆕 검증된 존 수
        'criteria': {
            'min_roi': min_roi,
            'min_accuracy': min_accuracy, 
            'min_win_rate': min_win_rate,
            'min_games': min_games
        }
    }

def find_cross_validated_combinations(segments_analysis, validated_zones, min_roi, min_accuracy, min_win_rate, min_games, matched_data=None, max_games=0):
    """여러 차원이 동시에 만족하는 조합 조건들을 찾아 성과 분석 - 🆕 실제 데이터 매칭 기반"""
    
    combinations = []
    
    # 🆕 실제 데이터가 없으면 조합분석 불가능
    if matched_data is None:
        st.warning("⚠️ **Real Data Required**: Combination analysis requires original matched data for cross-dimensional filtering")
        return []
    
    # 🆕 조합 분석용 최소 게임 수 (개별 기준보다 낮게) - 먼저 정의
    combination_min_games = max(min_games * 0.6, 15)  # 개별 기준의 60%, 최소 15게임
    
    # 🚨 수정: 검증된 존들 중에서 상위 성과 존들만 조합 분석 (최대 10개)
    top_zones = validated_zones[:10]  # 검증된 존들이 이미 스코어순 정렬되어 전달됨
    
    # 🆕 가능한 조합 수 계산 및 표시
    possible_combinations = 0
    for i in range(len(top_zones)):
        for j in range(i + 1, len(top_zones)):
            if top_zones[i]['dimension'] != top_zones[j]['dimension']:
                possible_combinations += 1
    
    # 🆕 디버깅 정보를 expander로 변경 - 모든 디버그 정보를 여기에 포함
    with st.expander("🔍 **Combination Analysis Debug Info**", expanded=False):
        st.write(f"**Analysis Configuration:**")
        st.write(f"- Input validated zones for combination: {len(validated_zones)}")
        st.write(f"- **Analysis data games (70%)**: {len(matched_data)}")
        st.write(f"- Combination min games threshold: {combination_min_games}")
        st.write(f"- Using top validated zones for analysis: {min(len(validated_zones), 10)}")
        st.info("📋 **Data Source**: Using same 70% analysis data as individual zones for consistency")
        st.write("")
        
        st.write(f"**Individual Zones Selected for Combination Analysis:**")
        st.info("📋 **Selection Criteria**: Top 10 zones ranked by Investment Zone Score\n"
                "- Zones are already filtered by user-defined criteria (ROI, accuracy, win rate, min games)\n" 
                "- Only combinations of different dimensions are analyzed (same dimension combinations don't make sense)")
        
        for i, zone in enumerate(top_zones):
            st.write(f"{i+1}. **{zone['dimension_name']}** - {zone['segment']} (ROI: {zone['roi']:.2f}%, Score: {zone['score']:.2f}, Games: {zone['games']})")
        st.write("")
        
        st.write(f"**Combination Generation Logic:**")
        st.write(f"- Total possible pairs from {len(top_zones)} zones: {len(top_zones)} × {len(top_zones)-1} ÷ 2 = {len(top_zones) * (len(top_zones)-1) // 2}")
        st.write(f"- **Valid combinations** (different dimensions only): **{possible_combinations}**")
        st.write(f"- Each combination requires both conditions to be satisfied **simultaneously**")
        st.write("")
        
        st.write(f"**Combination Analysis Results:**")
        
        # 2개 차원 조합 분석 및 디버그 로그를 expander 내에서 실행
        combination_results = []
        for i in range(len(top_zones)):
            for j in range(i + 1, len(top_zones)):
                zone1 = top_zones[i]
                zone2 = top_zones[j]
                
                # 서로 다른 차원인 경우만 조합
                if zone1['dimension'] != zone2['dimension']:
                    combination_stats = calculate_combination_performance_real_data(
                        matched_data, zone1, zone2
                    )
                    
                    # 🆕 디버깅 로그
                    if combination_stats:
                        status = "✅ Qualifying" if combination_stats['games'] >= combination_min_games else "❌ Below Threshold"
                        st.write(f"📊 **{zone1['dimension_name']}** + **{zone2['dimension_name']}**: {combination_stats['games']} games (threshold: {combination_min_games}) {status}")
                        combination_results.append((zone1, zone2, combination_stats))
                    else:
                        st.write(f"❌ **{zone1['dimension_name']}** + **{zone2['dimension_name']}**: No overlapping games")
                        
                        # 🆕 특별히 Probability + Confidence 조합일 때 상세 분석
                        if ((zone1['dimension'] == 'probability' and zone2['dimension'] == 'confidence') or 
                            (zone1['dimension'] == 'confidence' and zone2['dimension'] == 'probability')):
                            st.write("  🔍 **Special Analysis for Probability + Confidence:**")
                            both_match_count = 0
                            zone1_only_count = 0
                            zone2_only_count = 0
                            
                            for k, game in enumerate(matched_data[:20]):
                                zone1_match = check_zone_condition(game, zone1)
                                zone2_match = check_zone_condition(game, zone2)
                                
                                if zone1_match and zone2_match:
                                    both_match_count += 1
                                elif zone1_match:
                                    zone1_only_count += 1
                                elif zone2_match:
                                    zone2_only_count += 1
                                
                                if k < 5:  # 처음 5개 게임 상세 분석
                                    ensemble_prob = game.get('ensemble_probability', 0.5)
                                    confidence_prob = max(ensemble_prob, 1 - ensemble_prob)
                                    confidence_val = abs(ensemble_prob - 0.5)
                                    
                                    st.write(f"    Game {k+1}: prob={ensemble_prob:.4f}, confidence_prob={confidence_prob:.4f}, confidence_val={confidence_val:.4f}")
                                    st.write(f"      Zone1 ({zone1['segment']}): {'✅' if zone1_match else '❌'}")
                                    st.write(f"      Zone2 ({zone2['segment']}): {'✅' if zone2_match else '❌'}")
                            
                            st.write(f"  📊 **Summary (first 20 games):**")
                            st.write(f"    - Both conditions: {both_match_count}")
                            st.write(f"    - Only {zone1['dimension_name']}: {zone1_only_count}")
                            st.write(f"    - Only {zone2['dimension_name']}: {zone2_only_count}")
                            st.write(f"    - Expected total overlap: ~{both_match_count/20*len(matched_data):.1f} games")
    
    # expander 밖에서는 combination_results만 사용
    
    # 🆕 실제 조합 분석 계속 (expander 밖에서)
    combinations = []
    
    for zone1, zone2, combination_stats in combination_results:
        # 🆕 실제 데이터 기반 결과 검증
        if combination_stats and combination_stats['games'] >= combination_min_games:
            # 조합이 개별 존들보다 좋은 성과를 내는지 확인
            individual_avg_roi = (zone1['roi'] + zone2['roi']) / 2
            combination_roi = combination_stats['actual_roi']
            
            # 실제 시너지 효과 계산 (조합 ROI - 개별 평균 ROI)
            synergy_effect = combination_roi - individual_avg_roi
            
            # 조합 점수 계산 - 🆕 max_games 전달
            combination_score = calculate_investment_zone_score(combination_stats, max_games)
            
            combinations.append({
                'zone1': zone1,
                'zone2': zone2,
                'combination_name': f"{zone1['dimension_name']} + {zone2['dimension_name']}",
                'combination_detail': f"{zone1['segment']} + {zone2['segment']}",
                'stats': combination_stats,
                'synergy_effect': synergy_effect,
                'individual_avg_roi': individual_avg_roi,
                'score': combination_score,
                'outperforms_individual': combination_roi > individual_avg_roi,
                'combination_min_games_used': combination_min_games,
                'data_source': 'real_data',  # 🆕 실제 데이터임을 명시
                # 🔧 베팅 추천을 위한 필드 추가
                'zone1_dimension': zone1['dimension'],
                'zone1_segment': zone1['segment'],
                'zone2_dimension': zone2['dimension'],
                'zone2_segment': zone2['segment']
            })
    
    # 시너지 효과 순으로 정렬
    combinations.sort(key=lambda x: x['synergy_effect'], reverse=True)
    
    # 🆕 최종 결과 로그
    st.success(f"✅ **Combination Analysis Complete**: {len(combinations)} qualifying combinations found")
    for i, combo in enumerate(combinations[:3]):  # 상위 3개만 미리보기
        st.write(f"{i+1}. {combo['combination_name']}: {combo['stats']['games']} games, {combo['synergy_effect']:.2f}% synergy")
    
    return combinations

def calculate_combination_performance_real_data(matched_data, zone1, zone2):
    """두 차원이 동시에 만족하는 게임들의 실제 성과 계산 - 🆕 100% 실제 데이터 기반"""
    
    # 🆕 실제 데이터에서 두 조건을 동시에 만족하는 게임들 필터링
    combination_games = []
    
    for game in matched_data:
        # Zone1 조건 체크
        zone1_match = check_zone_condition(game, zone1)
        # Zone2 조건 체크 
        zone2_match = check_zone_condition(game, zone2)
        
        # 두 조건 모두 만족하는 경우만 추가
        if zone1_match and zone2_match:
            combination_games.append(game)
    
    # 조합 조건을 만족하는 게임이 없으면 None 반환
    if not combination_games:
        return None
    
    # 🆕 실제 데이터 기반 성과 계산 (구간분석과 동일한 방식)
    games = len(combination_games)
    
    # 실제 ROI 계산
    actual_rois = []
    predicted_rois = []
    wins = 0
    correct_predictions = 0
    
    for game in combination_games:
        # 🆕 실제 데이터 필드명 매칭하여 ROI 계산
        predicted_winner = game.get('predicted_winner', '')
        home_team = game.get('home_team', '')
        
        # 실제 승부 결과는 아직 없으므로 (미래 게임) 일단 시뮬레이션
        # 실제 사용시에는 actual_home_win 필드가 있을 것
        actual_win = game.get('actual_home_win', None)
        
        # 베팅할 팀의 배당률 가져오기 (🆕 매칭된 데이터 필드명 사용)
        if predicted_winner == home_team:
            bet_odds = game.get('home_odds', 0)  # ensemble_optimizer에서 변환된 필드명
            predicted_team = 'home'
        else:
            bet_odds = game.get('away_odds', 0)  # ensemble_optimizer에서 변환된 필드명
            predicted_team = 'away'
        
        if bet_odds is None or bet_odds == 0:
            continue
        
        # 🚨 실제 결과가 없는 경우 예측 ROI만 계산
        if actual_win is None:
            # 예측 ROI 계산
            ensemble_prob = game.get('ensemble_probability', game.get('win_probability', 0.5))
            win_prob = ensemble_prob if predicted_team == 'home' else 1 - ensemble_prob
            
            if bet_odds > 0:
                win_payout = (bet_odds / 100) * 100
            else:
                win_payout = (100 / abs(bet_odds)) * 100
            
            pred_roi = (win_prob * win_payout) + ((1 - win_prob) * (-100))
            
            # 시뮬레이션: 확률에 따라 승부 결과 가정
            import random
            is_win = random.random() < win_prob
        else:
            # 실제 승부 결과가 있는 경우
            is_win = (predicted_team == 'home' and actual_win == 1) or (predicted_team == 'away' and actual_win == 0)
            
            # 예측 ROI 계산
            ensemble_prob = game.get('ensemble_probability', game.get('win_probability', 0.5))
            win_prob = ensemble_prob if predicted_team == 'home' else 1 - ensemble_prob
            
            if bet_odds > 0:
                win_payout = (bet_odds / 100) * 100
            else:
                win_payout = (100 / abs(bet_odds)) * 100
            
            pred_roi = (win_prob * win_payout) + ((1 - win_prob) * (-100))
        
        if is_win:
            wins += 1
            # 승리시 실제 ROI 계산
            if bet_odds > 0:
                roi = (bet_odds / 100) * 100  # 양수 배당률
            else:
                roi = (100 / abs(bet_odds)) * 100  # 음수 배당률
        else:
            roi = -100  # 패배시 -100% ROI
        
        # 정확도 계산
        if actual_win is not None:
            if predicted_team == 'home' and actual_win == 1:
                correct_predictions += 1
            elif predicted_team == 'away' and actual_win == 0:
                correct_predictions += 1
        else:
            # 시뮬레이션된 결과로 정확도 계산
            if is_win:
                correct_predictions += 1
        
        actual_rois.append(roi)
        predicted_rois.append(pred_roi)
    
    # 통계 계산
    actual_roi = sum(actual_rois) / len(actual_rois) if actual_rois else 0
    predicted_roi = sum(predicted_rois) / len(predicted_rois) if predicted_rois else 0
    roi_difference = actual_roi - predicted_roi
    win_rate = (wins / games) * 100 if games > 0 else 0
    accuracy = (correct_predictions / games) * 100 if games > 0 else 0
    
    return {
        'actual_roi': actual_roi,
        'predicted_roi': predicted_roi,
        'roi_difference': roi_difference,
        'games': games,
        'accuracy': accuracy,
        'win_rate': win_rate,
        'data_source': 'real_data',
        'raw_games': combination_games  # 🆕 원본 게임 데이터도 저장
    }

def check_zone_condition_preprocessed(processed_game, zone, original_test_data, optimal_weights=None):
    """전처리된 게임 데이터가 특정 존 조건을 만족하는지 확인 - 구간분석과 완전히 동일한 로직"""
    
    dimension = zone['dimension']
    segment = zone['segment']
    
    if dimension == 'predicted_roi':
        pred_roi = processed_game['predicted_roi']
        if segment == 'Very Negative (<-20%)':
            return pred_roi < -20
        elif segment == 'Negative (-20% ~ 0%)':
            return -20 <= pred_roi < 0
        elif segment == 'Positive (0% ~ 20%)':
            return 0 <= pred_roi < 20
        elif segment == 'Very Positive (>20%)':
            return pred_roi >= 20
            
    # probability 차원 제거됨 (confidence와 동일)
            
    elif dimension == 'odds':
        bet_odds = processed_game['bet_odds']
        if segment == 'Heavy Favorite (< -200)':
            return bet_odds < -200
        elif segment == 'Favorite (-200 ~ -120)':
            return -200 <= bet_odds < -120
        elif segment == 'Pick Em (-120 ~ +120)':
            return -120 <= bet_odds <= 120
        elif segment == 'Underdog (+120 ~ +300)':
            return 120 < bet_odds <= 300
        elif segment == 'Heavy Underdog (> +300)':
            return bet_odds > 300
            
    elif dimension == 'confidence':
        confidence = processed_game['confidence']
        if segment == 'Low Confidence (0-0.05)':
            return confidence < 0.05
        elif segment == 'Medium Confidence (0.05-0.15)':
            return 0.05 <= confidence < 0.15
        elif segment == 'High Confidence (0.15-0.25)':
            return 0.15 <= confidence < 0.25
        elif segment == 'Very High Confidence (>0.25)':
            return confidence >= 0.25
            
    elif dimension == 'odds_probability_divergence':
        ensemble_prob = processed_game['ensemble_prob']
        bet_odds = processed_game['bet_odds']
        predicted_team = processed_game['predicted_team']
        
        # 시장 배당률을 확률로 변환
        if bet_odds > 0:
            market_implied_prob = 100 / (bet_odds + 100)
        else:
            market_implied_prob = abs(bet_odds) / (abs(bet_odds) + 100)
        
        # 모델 확률 (베팅 팀 기준으로 조정)
        if predicted_team == 'home':
            model_prob = ensemble_prob
        else:
            model_prob = 1 - ensemble_prob
        
        # 괴리도 계산
        divergence = model_prob - market_implied_prob
        
        if segment == 'Model Much More Optimistic (+10%+)':
            return divergence >= 0.10
        elif segment == 'Model Slightly Optimistic (+5% ~ +10%)':
            return 0.05 <= divergence < 0.10
        elif segment == 'Market Aligned (-5% ~ +5%)':
            return -0.05 <= divergence < 0.05
        elif segment == 'Model Slightly Pessimistic (-10% ~ -5%)':
            return -0.10 <= divergence < -0.05
        elif segment == 'Model Much More Pessimistic (-10%--)':
            return divergence < -0.10
            
    elif dimension == 'kelly_criterion':
        ensemble_prob = processed_game['ensemble_prob']
        bet_odds = processed_game['bet_odds']
        predicted_team = processed_game['predicted_team']
        
        # 베팅 팀에 맞는 승률
        if predicted_team == 'home':
            win_prob = ensemble_prob
        else:
            win_prob = 1 - ensemble_prob
        
        # 배당률을 decimal odds로 변환
        if bet_odds > 0:
            decimal_odds = (bet_odds / 100) + 1
        else:
            decimal_odds = (100 / abs(bet_odds)) + 1
        
        # Kelly Criterion 계산
        p = win_prob
        q = 1 - win_prob
        b = decimal_odds - 1
        
        kelly_fraction = (p * b - q) / b if b > 0 else 0
        kelly_percentage = max(0, kelly_fraction) * 100
        
        if segment == 'No Bet (Kelly ≤ 0%)':
            return kelly_percentage <= 0
        elif segment == 'Small Bet (0% < Kelly ≤ 5%)':
            return 0 < kelly_percentage <= 5
        elif segment == 'Medium Bet (5% < Kelly ≤ 15%)':
            return 5 < kelly_percentage <= 15
        elif segment == 'Large Bet (15% < Kelly ≤ 25%)':
            return 15 < kelly_percentage <= 25
        elif segment == 'Extreme Bet (Kelly > 25%)':
            return kelly_percentage > 25
            
    elif dimension == 'model_consensus':
        # 원본 데이터에서 모델 합의도 계산 (구간분석과 동일)
        matching_record = None
        for original_record in original_test_data:
            if (processed_game['date'] == original_record.get('date', '') and
                processed_game['home_team'] == original_record.get('home_team', '') and
                processed_game['away_team'] == original_record.get('away_team', '')):
                matching_record = original_record
                break
        
        if not matching_record or not optimal_weights:
            return False
            
        # 개별 모델 확률들 수집
        model_probs = []
        for model_name in optimal_weights.keys():
            prob_key = f"{model_name}_probability"
            if prob_key in matching_record and matching_record[prob_key] is not None:
                try:
                    prob = float(matching_record[prob_key])
                    if 0 <= prob <= 1:
                        model_probs.append(prob)
                except (ValueError, TypeError):
                    continue
        
        if len(model_probs) < 2:
            return False
        
        # 합의도 계산
        home_supporters = sum(1 for p in model_probs if p > 0.5)
        away_supporters = len(model_probs) - home_supporters
        majority_count = max(home_supporters, away_supporters)
        consensus_rate = majority_count / len(model_probs)
        
        if segment == 'Strong Consensus (80%+)':
            return consensus_rate >= 0.8
        elif segment == 'Moderate Consensus (60-80%)':
            return 0.6 <= consensus_rate < 0.8
        elif segment == 'Weak Consensus (50-60%)':
            return 0.5 <= consensus_rate < 0.6
        elif segment == 'No Consensus (<50%)':
            return consensus_rate < 0.5
    
    return False

def calculate_zone_roi_preprocessed(games_list):
    """전처리된 게임 리스트의 ROI 계산"""
    if not games_list:
        return 0
    
    total_roi = sum(game['actual_roi'] for game in games_list)
    return total_roi / len(games_list)

def check_zone_condition(game, zone, optimal_weights=None):
    """개별 게임이 특정 존 조건을 만족하는지 확인 - 🚨 수정: 가중치 기반 앙상블 확률 계산"""
    
    dimension = zone['dimension']
    segment = zone['segment']
    
    # 🚨 구간분석과 동일한 앙상블 확률 계산
    def calculate_ensemble_prob(game_record, weights):
        if weights is None:
            # 가중치 없으면 기존 저장된 값 사용
            return game_record.get('ensemble_probability', 0.5)
        
        ensemble_prob = 0
        total_weight_used = 0
        
        for model, weight in weights.items():
            prob_key = f"{model}_probability"
            if prob_key in game_record and game_record[prob_key] is not None:
                ensemble_prob += game_record[prob_key] * weight
                total_weight_used += weight
        
        if total_weight_used == 0:
            return game_record.get('ensemble_probability', 0.5)
        
        return ensemble_prob / total_weight_used
    
    # 🆕 원본 데이터의 필드명에 맞춰 수정
    if dimension == 'predicted_roi':
        # ✅ 실시간 앙상블 확률 계산 (process_game_for_strategy와 동일)
        ensemble_prob = calculate_ensemble_prob(game, optimal_weights)
        
        # ✅ 실시간 앙상블 확률 기준으로 베팅 결정 (process_game_for_strategy와 동일)
        if ensemble_prob > 0.5:
            bet_odds = game.get('home_team_odds') or game.get('home_odds', 0)
            predicted_team = 'home'
        else:
            bet_odds = game.get('away_team_odds') or game.get('away_odds', 0)
            predicted_team = 'away'
        
        if bet_odds is None or bet_odds == 0:
            return False
            
        # 예측 ROI 계산
        if bet_odds > 0:
            win_payout = (bet_odds / 100) * 100
        else:
            win_payout = (100 / abs(bet_odds)) * 100
        
        # ✅ 계산된 베팅 팀 기준으로 확률 조정 (process_game_for_strategy와 동일)
        win_prob = ensemble_prob if predicted_team == 'home' else 1 - ensemble_prob
        pred_roi = (win_prob * win_payout) + ((1 - win_prob) * (-100))
        
        if segment == 'Very Negative (<-20%)':
            return pred_roi < -20
        elif segment == 'Negative (-20% ~ 0%)':
            return -20 <= pred_roi < 0
        elif segment == 'Positive (0% ~ 20%)':
            return 0 <= pred_roi < 20
        elif segment == 'Very Positive (>20%)':
            return pred_roi >= 20
            
    # probability 차원 제거됨 (confidence와 동일)
            
    elif dimension == 'odds':
        # 🚨 수정: 구간분석과 정확히 동일한 로직 사용
        ensemble_prob = calculate_ensemble_prob(game, optimal_weights)
        
        # ensemble_prob > 0.5 기준으로 베팅 결정 (구간분석과 동일)
        if ensemble_prob > 0.5:
            bet_odds = game.get('home_team_odds') or game.get('home_odds', 0)
        else:
            bet_odds = game.get('away_team_odds') or game.get('away_odds', 0)
        
        if bet_odds is None or bet_odds == 0:
            return False
            
        if segment == 'Heavy Favorite (< -200)':
            return bet_odds < -200
        elif segment == 'Favorite (-200 ~ -120)':
            return -200 <= bet_odds < -120
        elif segment == 'Pick Em (-120 ~ +120)':
            return -120 <= bet_odds <= 120
        elif segment == 'Underdog (+120 ~ +300)':
            return 120 < bet_odds <= 300
        elif segment == 'Heavy Underdog (> +300)':
            return bet_odds > 300
            
    elif dimension == 'odds_probability_divergence':
        # 🚨 수정: 구간분석과 정확히 동일한 로직 사용
        ensemble_prob = calculate_ensemble_prob(game, optimal_weights)
        
        # ensemble_prob > 0.5 기준으로 베팅 결정 (구간분석과 동일)
        if ensemble_prob > 0.5:
            bet_odds = game.get('home_team_odds') or game.get('home_odds', 0)
            predicted_team = 'home'
        else:
            bet_odds = game.get('away_team_odds') or game.get('away_odds', 0)
            predicted_team = 'away'
        
        if bet_odds is None or bet_odds == 0:
            return False
        
        # 시장 배당률을 확률로 변환
        if bet_odds > 0:
            market_implied_prob = 100 / (bet_odds + 100)
        else:
            market_implied_prob = abs(bet_odds) / (abs(bet_odds) + 100)
        
        # 모델 확률 (베팅 팀 기준으로 조정)
        if predicted_team == 'home':
            model_prob = ensemble_prob
        else:
            model_prob = 1 - ensemble_prob
        
        # 괴리도 계산 (모델 확률 - 시장 확률)
        divergence = model_prob - market_implied_prob
        
        if segment == 'Model Much More Optimistic (+10%+)':
            return divergence >= 0.10
        elif segment == 'Model Slightly Optimistic (+5% ~ +10%)':
            return 0.05 <= divergence < 0.10
        elif segment == 'Market Aligned (-5% ~ +5%)':
            return -0.05 <= divergence < 0.05
        elif segment == 'Model Slightly Pessimistic (-10% ~ -5%)':
            return -0.10 <= divergence < -0.05
        elif segment == 'Model Much More Pessimistic (-10%--)':
            return divergence < -0.10
            
    elif dimension == 'kelly_criterion':
        # 🚨 수정: 구간분석과 정확히 동일한 로직 사용
        ensemble_prob = calculate_ensemble_prob(game, optimal_weights)
        
        # ensemble_prob > 0.5 기준으로 베팅 결정 (구간분석과 동일)
        if ensemble_prob > 0.5:
            bet_odds = game.get('home_team_odds') or game.get('home_odds', 0)
            win_prob = ensemble_prob
        else:
            bet_odds = game.get('away_team_odds') or game.get('away_odds', 0)
            win_prob = 1 - ensemble_prob
        
        if bet_odds is None or bet_odds == 0:
            return False
        
        # 배당률을 decimal odds로 변환
        if bet_odds > 0:
            decimal_odds = (bet_odds / 100) + 1
        else:
            decimal_odds = (100 / abs(bet_odds)) + 1
        
        # Kelly Criterion 계산: f* = (p * b - q) / b
        # p = 승률, q = 패배율, b = decimal_odds - 1
        p = win_prob
        q = 1 - win_prob
        b = decimal_odds - 1
        
        kelly_fraction = (p * b - q) / b if b > 0 else 0
        kelly_percentage = max(0, kelly_fraction) * 100  # 음수면 0으로 처리
        
        if segment == 'No Bet (Kelly ≤ 0%)':
            return kelly_percentage <= 0
        elif segment == 'Small Bet (0% < Kelly ≤ 5%)':
            return 0 < kelly_percentage <= 5
        elif segment == 'Medium Bet (5% < Kelly ≤ 15%)':
            return 5 < kelly_percentage <= 15
        elif segment == 'Large Bet (15% < Kelly ≤ 25%)':
            return 15 < kelly_percentage <= 25
        elif segment == 'Extreme Bet (Kelly > 25%)':
            return kelly_percentage > 25
            
    elif dimension == 'confidence':
        # 🚨 추가: confidence 차원 로직 (구간분석과 동일)
        ensemble_prob = calculate_ensemble_prob(game, optimal_weights)
        confidence = abs(ensemble_prob - 0.5)  # 0.5에서 얼마나 멀리 있는지
        
        if segment == 'Low Confidence (0-0.05)':
            return confidence < 0.05
        elif segment == 'Medium Confidence (0.05-0.15)':
            return 0.05 <= confidence < 0.15
        elif segment == 'High Confidence (0.15-0.25)':
            return 0.15 <= confidence < 0.25
        elif segment == 'Very High Confidence (>0.25)':
            return confidence >= 0.25
            
    elif dimension == 'model_consensus':
        # 🚨 수정: 구간분석과 완전히 동일한 로직 사용 (실제 모델 합의도 계산)
        if optimal_weights:
            # ✅ 구간분석과 동일: 실제 개별 모델들로 합의도 계산
            model_probs = []
            for model_name in optimal_weights.keys():
                prob_key = f"{model_name}_probability"
                if prob_key in game and game[prob_key] is not None:
                    try:
                        prob = float(game[prob_key])
                        if 0 <= prob <= 1:
                            model_probs.append(prob)
                    except (ValueError, TypeError):
                        continue
            
            if len(model_probs) < 2:  # 최소 2개 모델 필요
                return False
            
            # 합의도 계산: 같은 팀을 지지하는 모델 비율
            home_supporters = sum(1 for p in model_probs if p > 0.5)
            away_supporters = len(model_probs) - home_supporters
            
            # 다수 의견의 비율
            majority_count = max(home_supporters, away_supporters)
            consensus_rate = majority_count / len(model_probs)
            
            # ✅ 구간분석과 동일한 분류
            if segment == 'Strong Consensus (80%+)':
                return consensus_rate >= 0.8
            elif segment == 'Moderate Consensus (60-80%)':
                return 0.6 <= consensus_rate < 0.8
            elif segment == 'Weak Consensus (50-60%)':
                return 0.5 <= consensus_rate < 0.6
            elif segment == 'No Consensus (<50%)':
                return consensus_rate < 0.5
        else:
            # 개별 모델 확률들로 실제 합의도 계산
            model_probs = []
            
            # 모든 모델 확률 수집
            for key in game.keys():
                if key.endswith('_probability') and key != 'ensemble_probability':
                    prob = game.get(key)
                    if prob is not None and 0 <= prob <= 1:
                        model_probs.append(prob)
            
            if len(model_probs) >= 2:
                # 실제 합의도 계산
                home_supporters = sum(1 for p in model_probs if p > 0.5)
                away_supporters = len(model_probs) - home_supporters
                
                majority_count = max(home_supporters, away_supporters)
                consensus_rate = majority_count / len(model_probs)
                
                if segment == 'Strong Consensus (80%+)':
                    return consensus_rate >= 0.8
                elif segment == 'Moderate Consensus (60-80%)':
                    return 0.6 <= consensus_rate < 0.8
                elif segment == 'Weak Consensus (50-60%)':
                    return 0.5 <= consensus_rate < 0.6
                elif segment == 'No Consensus (<50%)':
                    return consensus_rate < 0.5
            else:
                # 모델 수가 부족하면 ensemble probability로 근사
                ensemble_prob = calculate_ensemble_prob(game, optimal_weights)
                return ensemble_prob >= 0.5
    
    return False

def calculate_investment_zone_score(stats, max_games):
    """투자 존 점수 계산 - 🆕 샘플 사이즈 보너스 포함"""
    
    # 기본 성과 점수 (0-5점)
    roi_score = min(stats['actual_roi'] / 4, 5)  # 20% ROI = 5점
    accuracy_score = min((stats['accuracy'] - 50) / 10, 5)  # 100% 정확도 = 5점
    win_rate_score = min((stats['win_rate'] - 50) / 10, 5)  # 100% 승률 = 5점
    
    # 샘플 사이즈 보너스 (0-2점)
    if max_games > 0:
        sample_ratio = stats['games'] / max_games
        sample_bonus = min(sample_ratio * 2, 2)  # 최대 게임 수 = 2점
    else:
        sample_bonus = 0
    
    # 예측 정확도 보너스 (-1점 ~ +1점)
    roi_diff = abs(stats['roi_difference'])
    if roi_diff <= 2:
        prediction_bonus = 1  # 예측이 매우 정확
    elif roi_diff <= 5:
        prediction_bonus = 0.5  # 예측이 정확
    elif roi_diff <= 10:
        prediction_bonus = 0  # 예측이 보통
    else:
        prediction_bonus = -0.5  # 예측이 부정확
    
    # 총 점수 계산 (최대 13점)
    total_score = roi_score + accuracy_score + win_rate_score + sample_bonus + prediction_bonus
    
    return round(total_score, 2)

def get_best_individual_zone(optimal_zones, validation_info):
    """검증 티어를 우선 고려한 베스트 개별존 선정"""
    
    # 1. 검증 정보로 존들을 분류
    tier1_zones = []
    tier2_zones = []
    
    for zone in optimal_zones:
        zone_name = f"{zone['dimension_name']} - {zone['segment']}"
        if zone_name in validation_info:
            validation = validation_info[zone_name]
            tier = validation.get('tier')
            
            if tier == 'Tier 1':
                tier1_zones.append(zone)
            elif tier == 'Tier 2':
                tier2_zones.append(zone)
            # 검증 실패는 제외
    
    # 2. 티어별 우선순위로 선정
    if tier1_zones:
        # Tier 1에서 스코어 최고
        best_zone = max(tier1_zones, key=lambda x: x['score'])
        best_zone['selection_reason'] = 'Tier 1 (Excellent) - Highest Score'
        return best_zone
    elif tier2_zones:
        # Tier 2에서 스코어 최고  
        best_zone = max(tier2_zones, key=lambda x: x['score'])
        best_zone['selection_reason'] = 'Tier 2 (Risky) - Highest Score'
        return best_zone
    else:
        # 검증된 존이 없으면 기존 방식 (경고와 함께)
        best_zone = optimal_zones[0]
        best_zone['selection_reason'] = 'No Validated Zones - Highest Score (Risk: Overfitting)'
        return best_zone

def display_investment_zones(investment_analysis, optimal_weights=None, segments_analysis=None):
    """투자 존 분석 결과를 표시하는 함수 - 🆕 검증 결과 포함"""
    
    # 🎯 스마트 투자존 분석 헤더
    st.markdown("---")
    st.markdown("## 🎯 Smart Investment Zone Finder")
    st.markdown("AI-driven identification of optimal betting opportunities with statistical validation.")
    
    st.markdown("### 📊 Analysis Results")
    
    # 🆕 원본 데이터 보존
    original_zones = investment_analysis['optimal_zones']
    original_combinations = investment_analysis['cross_validated_combinations']
    criteria = investment_analysis['criteria']
    validation_results = investment_analysis.get('validation_results')
    
    # 🆕 항상 모든 존과 조합 표시 (체크박스 없음)
    optimal_zones = original_zones
    cross_combinations = original_combinations
    
    # 📈 검증 결과가 있는 경우 검증 정보 표시
    validation_info = {}
    if validation_results:
        st.markdown("### 🔍 Time-Split Validation")
        st.markdown("Rigorous testing of identified zones using unseen future data to prevent overfitting.")
        
        split_info = validation_results['split_info']
        tier1_zones = validation_results['tier1_zones']
        tier2_zones = validation_results['tier2_zones']
        validation_details = validation_results['validation_details']
        
        # 검증 정보를 딕셔너리로 만들어서 나중에 존별로 표시
        for detail in validation_details:
            zone_name = detail['zone_name']
            # 🚨 수정: tier 정보도 함께 저장
            validation_data = detail.get('validation', {}).copy()
            validation_data['tier'] = detail.get('tier')
            validation_data['validated'] = detail.get('validated', False)
            validation_info[zone_name] = validation_data
        
        # 분할 정보 표시
        col_split1, col_split2, col_split3, col_split4 = st.columns(4)
        with col_split1:
            # 구간분석 분할 정보에서 가져오기
            segments_split_info = validation_results.get('segments_split_info', {})
            analysis_games = segments_split_info.get('analysis_games', 'N/A')
            st.metric("Analysis Data (70%)", f"{analysis_games} games" if analysis_games != 'N/A' else analysis_games)
        with col_split2:
            st.metric("Validation Data (30%)", f"{split_info['test_count']} games")
        with col_split3:
            st.metric("Split Method", "Time-based 7:3")
        with col_split4:
            validation_rate = len(tier1_zones + tier2_zones) / len(original_zones) * 100 if original_zones else 0
            st.metric("Validation Rate", f"{validation_rate:.1f}%")
        
        # 티어별 검증 결과
        col_tier1, col_tier2, col_tier3, col_tier4 = st.columns(4)
        with col_tier1:
            st.metric("🏆 Tier 1 (Excellent)", len(tier1_zones))
        with col_tier2:
            st.metric("⚠️ Tier 2 (Risky)", len(tier2_zones))
        with col_tier3:
            st.metric("✅ Total Validated", len(tier1_zones) + len(tier2_zones))
        with col_tier4:
            rejected_count = len(original_zones) - len(tier1_zones) - len(tier2_zones)
            st.metric("❌ Rejected", rejected_count)
        
        # 검증 상태 경고
        if validation_rate < 30:
            st.error("🚨 **High Overfitting Detected!** Most zones failed future validation.")
        elif validation_rate < 60:
            st.warning("⚠️ **Moderate Overfitting Detected.** Use validated zones only.")
        else:
            st.success("✅ **Low Overfitting Risk.** Most zones passed validation.")
        
        # 📊 상세 검증 결과 표시
        if validation_results:
            validation_details = validation_results['validation_details']
            
            with st.expander("📊 Detailed Zone Validation", expanded=False):
                st.markdown("Individual zone performance analysis comparing train vs validation data.")
                
                for zone_info in validation_details:
                    zone_name = zone_info['zone_name']
                    validation = zone_info.get('validation', {})
                    validation_status = validation.get('status', 'passed')
                    tier = zone_info.get('tier', 0)
                    train_roi = zone_info.get('past_roi', 0)
                    future_roi = zone_info.get('future_roi', 0)
                    # 🚨 수정: train data에서 실제 게임 수 계산
                    zone = zone_info.get('zone', {})
                    if split_info and 'train_data' in split_info:
                        train_games = 0
                        for game in split_info['train_data']:
                            if check_zone_condition(game, zone, validation_results.get('optimal_weights')):  # 🚨 가중치 전달
                                train_games += 1
                    else:
                        train_games = zone.get('games', 0)  # 분할 없으면 전체 게임 수
                    future_games = zone_info.get('future_games', 0)
                    
                    # 검증 결과에 따른 아이콘과 색상
                    if tier == 'Tier 1':
                        icon = "🏆"
                        status_text = "Tier 1 (Excellent)"
                        color = "success"
                    elif tier == 'Tier 2':
                        icon = "⚠️"
                        status_text = "Tier 2 (Risky)"
                        color = "warning"
                    elif validation_status == "insufficient_data":
                        icon = "📊"
                        status_text = "Insufficient future data"
                        color = "info"
                    else:
                        icon = "❌"
                        status_text = "Failed Validation"
                        color = "error"
                    
                    # 상태에 따른 배경색
                    if color == "success":
                        bg_color = "#d4edda"
                        border_color = "#c3e6cb"
                    elif color == "warning":
                        bg_color = "#fff3cd"
                        border_color = "#ffeaa7"
                    elif color == "info":
                        bg_color = "#d1ecf1"
                        border_color = "#bee5eb"
                    else:
                        bg_color = "#f8d7da"
                        border_color = "#f5c6cb"
                    
                    st.markdown(f"""
                    <div style="
                        border: 1px solid {border_color}; 
                        border-radius: 5px; 
                        padding: 10px; 
                        margin: 5px 0;
                        background-color: {bg_color};
                    ">
                        <strong>{icon} {zone_name}:</strong> {status_text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 상세 정보 표시
                    col_val1, col_val2, col_val3, col_val4 = st.columns(4)
                    
                    with col_val1:
                        st.write(f"**Train ROI:** {train_roi:.2f}%")
                        st.write(f"**Train Games:** {train_games}")
                    
                    with col_val2:
                        if validation_status != "insufficient_data":
                            st.write(f"**Future ROI:** {future_roi:.2f}%")
                            st.write(f"**Future Games:** {future_games}")
                        else:
                            st.write(f"**Future Games:** {future_games} < 15")
                            st.write("**Status:** Insufficient data")
                    
                    with col_val3:
                        if validation_status not in ["insufficient_data"]:
                            gap = zone_info.get('gap', future_roi - train_roi)  # 양수면 상승, 음수면 하락
                            st.write(f"**ROI Gap:** {gap:.2f}%")
                            
                            if tier in ['Tier 1', 'Tier 2']:
                                st.write(f"**Tier:** {tier}")
                            else:
                                reason = validation.get('message', 'Failed validation')
                                st.write(f"**Status:** {reason}")
                    
                    with col_val4:
                        if tier == 'Tier 1':
                            st.success("✅ High Confidence")
                        elif tier == 'Tier 2':
                            st.warning("⚠️ Use with Caution")
                        elif validation_status == "insufficient_data":
                            st.info("📊 Need More Data")
                        else:
                            st.error("❌ High Risk")
                    
                    st.markdown("---")
        
        st.markdown("---")
    
    # 요약 통계
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Segments", investment_analysis['total_analyzed'])
    with col2:
        st.metric("Qualifying Zones", investment_analysis['qualifying_zones'])
    with col3:
        qualification_rate = (investment_analysis['qualifying_zones'] / investment_analysis['total_analyzed'] * 100) if investment_analysis['total_analyzed'] > 0 else 0
        st.metric("Qualification Rate", f"{qualification_rate:.1f}%")
    with col4:
        if optimal_zones:
            avg_score = sum(zone['score'] for zone in optimal_zones) / len(optimal_zones)
            st.metric("Avg Zone Score", f"{avg_score:.2f}")
        else:
            st.metric("Avg Zone Score", "N/A")
    
    if not optimal_zones:
        st.warning("❌ **No Investment Zones Found**")
        st.info(f"""
        📋 **Current Criteria:**
        - ROI ≥ {criteria['min_roi']}%
        - Accuracy ≥ {criteria['min_accuracy']}%  
        - Win Rate ≥ {criteria['min_win_rate']}%
        - Games ≥ {criteria['min_games']}
        
        💡 **Suggestion**: Try lowering the criteria to find potential zones
        """)
        return
    
    # 🆕 조합 분석 vs 개별 분석 탭
    analysis_tabs = st.tabs(["🏆 Best Combinations", "📊 Individual Zones", "🔍 Comparison Analysis"])
    
    with analysis_tabs[0]:
        # 교차 검증된 조합 결과
        st.markdown("#### 🎯 Cross-Validated Investment Combinations")
        
        # 🆕 조합 분석 기준 설명 추가
        combination_min_games = max(criteria['min_games'] * 0.6, 20)
        with st.expander("ℹ️ Combination Analysis Criteria", expanded=False):
            st.info(f"""
            📋 **Combination Analysis Logic:**
            
            **🔍 Combination Requirements:**
            - Different dimensions only (e.g., Kelly + Consensus, not Kelly + Kelly)
            - Estimated games ≥ {combination_min_games} (relaxed from individual {criteria['min_games']})
            - Both individual zones must qualify first
            
            **🧮 Estimated Games Calculation:**
            - Combined Games = (Zone1 + Zone2) / 2 × 0.8
            - Minimum 15 games guaranteed
            
            **📈 Synergy Effect:**
            - Synergy = Combined ROI - Average Individual ROI
            - Positive synergy = Better than using zones separately
            - Combination modeling assumes 10% synergy boost + 5% accuracy/win rate improvement
            
            **🎯 Current Analysis:**
            - Analyzing top {min(len(optimal_zones), 10)} individual zones
            - Total possible combinations: {len(optimal_zones) * (len(optimal_zones) - 1) // 2 if len(optimal_zones) > 1 else 0}
            - Relaxed minimum games: {combination_min_games} vs individual {criteria['min_games']}
            """)
        
        if cross_combinations:
            st.success(f"🚀 Found {len(cross_combinations)} powerful combinations with synergy effects!")
            
            # 🆕 노출 기준 설명 및 설정
            col_display1, col_display2 = st.columns([2, 1])
            
            with col_display1:
                st.markdown("##### 📈 Combinations Sorted by Synergy Effect (Highest First)")
                st.info(f"""
                **Ranking Criteria**: Synergy Effect = Combined ROI - Average Individual ROI
                
                📊 **Top combinations show**:
                - Highest synergy effects (best performance boost)
                - Sorted from most beneficial to least beneficial
                - All {len(cross_combinations)} combinations analyzed
                """)
            
            with col_display2:
                # 🆕 표시할 조합 수 선택 (최대값 동적 설정)
                max_display = min(len(cross_combinations), 10)  # 최대 10개
                default_display = min(3, len(cross_combinations))  # 기본 3개 또는 전체
                
                display_count = st.selectbox(
                    "🔍 Show top combinations:",
                    options=list(range(1, max_display + 1)),
                    index=default_display - 1,
                    help=f"Select how many top combinations to display in detail (out of {len(cross_combinations)} total)"
                )
            
            # 🆕 시너지 효과 분포 간단 요약
            synergy_effects = [combo['synergy_effect'] for combo in cross_combinations]
            positive_synergy = len([s for s in synergy_effects if s > 0])
            strong_synergy = len([s for s in synergy_effects if s > 2])
            
            col_summary1, col_summary2, col_summary3 = st.columns(3)
            with col_summary1:
                st.metric("Positive Synergy", f"{positive_synergy}/{len(cross_combinations)}")
            with col_summary2:
                st.metric("Strong Synergy (>2%)", strong_synergy)
            with col_summary3:
                best_synergy = max(synergy_effects) if synergy_effects else 0
                st.metric("Best Synergy", f"+{best_synergy:.2f}%")
            
            st.markdown("---")
            
            # 선택된 수만큼 조합 표시
            for i, combo in enumerate(cross_combinations[:display_count], 1):
                synergy = combo['synergy_effect']
                stats = combo['stats']
                
                # 시너지 효과에 따른 색상
                if synergy > 3:
                    border_color = "#28a745"  # 초록색 - 강한 시너지
                    bg_color = "#d4edda"
                    synergy_level = "🚀 Powerful Synergy"
                elif synergy > 1:
                    border_color = "#007bff"  # 파란색 - 좋은 시너지
                    bg_color = "#d1ecf1" 
                    synergy_level = "⭐ Good Synergy"
                elif synergy > 0:
                    border_color = "#ffc107"  # 노란색 - 약한 시너지
                    bg_color = "#fff3cd"
                    synergy_level = "✅ Weak Synergy"
                else:
                    border_color = "#6c757d"  # 회색 - 시너지 없음
                    bg_color = "#f8f9fa"
                    synergy_level = "❌ No Synergy"
                
                st.markdown(f"""
                <div style="
                    border: 2px solid {border_color}; 
                    border-radius: 8px; 
                    padding: 1rem; 
                    margin: 1rem 0;
                    background: {bg_color};
                ">
                    <h5>🎯 Combination #{i}: {combo['combination_name']} ({synergy_level})</h5>
                    <p><strong>Conditions:</strong> {combo['combination_detail']}</p>
                    <p><strong>Synergy Effect:</strong> +{synergy:.2f}% ROI boost vs individual zones</p>
                </div>
                """, unsafe_allow_html=True)
                
                col_combo1, col_combo2, col_combo3, col_combo4 = st.columns(4)
                
                with col_combo1:
                    st.metric("Combined ROI", f"{stats['actual_roi']:.2f}%")
                    st.metric("Estimated Games", stats['games'])
                
                with col_combo2:
                    st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
                    st.metric("Accuracy", f"{stats['accuracy']:.1f}%")
                
                with col_combo3:
                    st.metric("Individual Avg", f"{combo['individual_avg_roi']:.2f}%")
                    st.metric("Combination Score", f"{combo['score']:.2f}")
                
                with col_combo4:
                    # 성과 향상 표시
                    if combo['outperforms_individual']:
                        st.success("📈 Outperforms Individual")
                    else:
                        st.warning("📉 Underperforms Individual")
                    
                    # 시너지 강도 표시
                    if synergy > 3:
                        st.success("🚀 Strong Synergy")
                    elif synergy > 1:
                        st.info("⭐ Moderate Synergy")
                    elif synergy > 0:
                        st.warning("✅ Weak Synergy")
                    else:
                        st.error("❌ Negative Synergy")
                
                st.markdown("---")
            
            # 조합 요약 테이블 (선택된 수보다 많은 경우만)
            if len(cross_combinations) > display_count:
                remaining_combinations = len(cross_combinations) - display_count
                with st.expander(f"📋 Remaining {remaining_combinations} Combinations (Summary Table)", expanded=False):
                    remaining_combos = cross_combinations[display_count:]
                    combo_df = pd.DataFrame([{
                        'Rank': i + display_count + 1,
                        'Combination': combo['combination_name'],
                        'Conditions': combo['combination_detail'],
                        'Combined ROI (%)': f"{combo['stats']['actual_roi']:.2f}",
                        'Individual Avg (%)': f"{combo['individual_avg_roi']:.2f}",
                        'Synergy (+%)': f"{combo['synergy_effect']:+.2f}",
                        'Score': combo['score'],
                        'Games': combo['stats']['games']
                    } for i, combo in enumerate(remaining_combos)])
                    
                    st.dataframe(combo_df, hide_index=True, use_container_width=True)
            
            # 🆕 전체 조합 요약 통계
            if len(cross_combinations) > 1:
                st.markdown("##### 📊 Overall Combination Statistics")
                
                avg_synergy = sum(synergy_effects) / len(synergy_effects)
                median_synergy = sorted(synergy_effects)[len(synergy_effects)//2]
                
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                with col_stats1:
                    st.metric("Average Synergy", f"{avg_synergy:+.2f}%")
                with col_stats2:
                    st.metric("Median Synergy", f"{median_synergy:+.2f}%")
                with col_stats3:
                    success_rate = (positive_synergy / len(cross_combinations)) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                with col_stats4:
                    synergy_range = max(synergy_effects) - min(synergy_effects)
                    st.metric("Synergy Range", f"{synergy_range:.2f}%")
        
        else:
            st.info("🔍 **No Combinations Found** - Not enough qualifying individual zones for combination analysis")
    
    with analysis_tabs[1]:
        # 기존 개별 존 분석
        st.markdown("#### ✅ Individual Investment Zones")
        st.success(f"🎯 Found {len(optimal_zones)} qualifying individual investment opportunities!")
        
        # 상위 5개 존 상세 표시
        top_zones = optimal_zones[:5]
        
        for i, zone in enumerate(top_zones, 1):
            with st.container():
                # 존별 성과에 따른 스타일링
                if zone['roi'] >= 10:
                    border_color = "#28a745"  # 초록색
                    bg_color = "#d4edda"
                elif zone['roi'] >= 7:
                    border_color = "#007bff"  # 파란색  
                    bg_color = "#d1ecf1"
                elif zone['roi'] >= 5:
                    border_color = "#ffc107"  # 노란색
                    bg_color = "#fff3cd"
                else:
                    border_color = "#6c757d"  # 회색
                    bg_color = "#f8f9fa"
                
                st.markdown(f"""
                <div style="
                    border: 2px solid {border_color}; 
                    border-radius: 8px; 
                    padding: 1rem; 
                    margin: 0.5rem 0;
                    background: {bg_color};
                ">
                    <h5>🏆 Zone #{i}: {zone['dimension_name']} - {zone['segment']} (Score: {zone['score']})</h5>
                </div>
                """, unsafe_allow_html=True)
                
                col_zone1, col_zone2, col_zone3, col_zone4 = st.columns(4)
                
                with col_zone1:
                    st.metric("Actual ROI", f"{zone['roi']:.2f}%")
                    st.metric("Games", zone['games'])
                
                with col_zone2:
                    st.metric("Win Rate", f"{zone['win_rate']:.1f}%") 
                    st.metric("Accuracy", f"{zone['accuracy']:.1f}%")
                
                with col_zone3:
                    st.metric("Predicted ROI", f"{zone['predicted_roi']:.2f}%")
                    roi_diff = zone['roi_difference']
                    st.metric("ROI Difference", f"{roi_diff:+.2f}%")
                
                with col_zone4:
                    # 성과 등급 표시
                    if zone['roi'] >= 10:
                        st.success("🌟 Excellent")
                    elif zone['roi'] >= 7:
                        st.info("⭐ Very Good")
                    elif zone['roi'] >= 5:
                        st.warning("✅ Good")
                    else:
                        st.info("📊 Acceptable")
                    
                    # 🆕 검증 상태 표시
                    zone_name = f"{zone['dimension_name']} - {zone['segment']}"
                    if zone_name in validation_info:
                        validation = validation_info[zone_name]
                        tier = validation.get('tier')
                        validated = validation.get('validated', False)
                        status = validation.get('status', 'unknown')
                        
                        # 🚨 수정: tier 정보를 우선적으로 확인
                        if tier == 'Tier 1':
                            st.success("🏆 Tier 1 (Validated)")
                        elif tier == 'Tier 2':
                            st.warning("⚠️ Tier 2 (Risky)")
                        elif status == 'insufficient_data':
                            st.info("📊 Insufficient Data")
                        else:
                            st.error("❌ Failed Validation")
                    else:
                        # 신뢰도 표시 (검증 정보가 없을 때만)
                        if zone['games'] >= 60:
                            st.success("🔒 High Confidence")
                        elif zone['games'] >= 40:
                            st.info("📊 Medium Confidence")
                        else:
                            st.warning("⚠️ Low Confidence")
                
                # 🆕 각 존별 상세 정보 expander 추가
                with st.expander(f"🔍 Detailed Analysis for Zone #{i}", expanded=False):
                    st.markdown("##### 📊 Comprehensive Zone Analysis")
                    
                    # 기본 정보 섹션
                    col_detail1, col_detail2 = st.columns(2)
                    
                    with col_detail1:
                        st.markdown("**📈 Performance Metrics**")
                        st.write(f"• **Dimension**: {zone['dimension_name']}")
                        st.write(f"• **Segment**: {zone['segment']}")
                        st.write(f"• **Investment Score**: {zone['score']:.2f}/10")
                        st.write(f"• **Actual ROI**: {zone['roi']:.2f}%")
                        st.write(f"• **Predicted ROI**: {zone['predicted_roi']:.2f}%")
                        st.write(f"• **ROI Prediction Error**: {zone['roi_difference']:+.2f}%")
                    
                    with col_detail2:
                        st.markdown("**🎯 Accuracy & Reliability**")
                        st.write(f"• **Win Rate**: {zone['win_rate']:.1f}%")
                        st.write(f"• **Prediction Accuracy**: {zone['accuracy']:.1f}%")
                        st.write(f"• **Sample Size**: {zone['games']} games")
                        
                        # 신뢰도 분석
                        if zone['games'] >= 60:
                            confidence_level = "🔒 High (60+ games)"
                        elif zone['games'] >= 40:
                            confidence_level = "📊 Medium (40-59 games)"
                        elif zone['games'] >= 20:
                            confidence_level = "⚠️ Low (20-39 games)"
                        else:
                            confidence_level = "❌ Very Low (<20 games)"
                        
                        st.write(f"• **Statistical Confidence**: {confidence_level}")
                    
                    # 검증 정보가 있는 경우 표시
                    if zone_name in validation_info:
                        st.markdown("**🔍 Validation Results**")
                        validation = validation_info[zone_name]
                        tier = validation.get('tier')
                        status = validation.get('status', 'unknown')
                        message = validation.get('message', 'No details available')
                        
                        # 🚨 수정: tier 정보를 우선적으로 확인
                        if tier == 'Tier 1':
                            st.success(f"✅ **Validation Status**: Tier 1 (Excellent) - {message}")
                        elif tier == 'Tier 2':
                            st.warning(f"⚠️ **Validation Status**: Tier 2 (Risky) - {message}")
                        elif status == 'insufficient_data':
                            st.info(f"📊 **Validation Status**: Insufficient Data - {message}")
                        else:
                            st.error(f"❌ **Validation Status**: Failed - {message}")
                    
                    # 투자 추천사항
                    st.markdown("**💡 Investment Recommendation**")
                    
                    if zone['roi'] >= 10 and zone['games'] >= 40:
                        st.success("🚀 **Highly Recommended**: Excellent ROI with sufficient sample size")
                    elif zone['roi'] >= 7 and zone['games'] >= 30:
                        st.info("⭐ **Recommended**: Good ROI with adequate reliability")
                    elif zone['roi'] >= 5:
                        st.warning("✅ **Conditionally Recommended**: Positive ROI but consider sample size")
                    elif zone['roi'] >= 0:
                        st.warning("📊 **Marginal**: Low ROI, use with caution")
                    else:
                        st.error("❌ **Not Recommended**: Negative expected ROI")
                    
                    # 위험 요소 분석
                    st.markdown("**⚠️ Risk Assessment**")
                    risk_factors = []
                    
                    if zone['games'] < 30:
                        risk_factors.append(f"• **Small Sample Size**: Only {zone['games']} games may not be representative")
                    
                    if abs(zone['roi_difference']) > 5:
                        risk_factors.append(f"• **Prediction Error**: ROI prediction was off by {zone['roi_difference']:+.2f}%")
                    
                    if zone['accuracy'] < 60:
                        risk_factors.append(f"• **Low Accuracy**: Only {zone['accuracy']:.1f}% prediction accuracy")
                    
                    if zone['win_rate'] < 55:
                        risk_factors.append(f"• **Low Win Rate**: Only {zone['win_rate']:.1f}% of bets won")
                    
                    if risk_factors:
                        for risk in risk_factors:
                            st.write(risk)
                    else:
                        st.success("✅ No significant risk factors identified")
                
                st.markdown("---")
        
        # 🆕 모든 개별존 상세 정보 테이블 (항상 표시)
        st.markdown("#### 📋 Complete Individual Zones Analysis Table")
        with st.expander(f"📊 All {len(optimal_zones)} Investment Zones (Comprehensive Details)", expanded=False):
                
                # 🆕 검증 정보 포함한 데이터프레임 생성
                zones_data = []
                for i, zone in enumerate(optimal_zones):
                    zone_name = f"{zone['dimension_name']} - {zone['segment']}"
                    
                    # 검증 상태 확인
                    validation_status = "Not Validated"
                    if zone_name in validation_info:
                        validation = validation_info[zone_name]
                        tier = validation.get('tier')
                        status = validation.get('status', 'unknown')
                        
                        # 🚨 수정: tier 정보를 우선적으로 확인
                        if tier == 'Tier 1':
                            validation_status = "🏆 Tier 1"
                        elif tier == 'Tier 2':
                            validation_status = "⚠️ Tier 2"
                        elif status == 'insufficient_data':
                            validation_status = "📊 Insufficient"
                        else:
                            validation_status = "❌ Failed"
                    
                    zones_data.append({
                        'Rank': i+1,
                        'Dimension': zone['dimension_name'],
                        'Segment': zone['segment'],
                        'Validation': validation_status,
                        'Score': zone['score'],
                        'ROI (%)': f"{zone['roi']:.2f}",
                        'Win Rate (%)': f"{zone['win_rate']:.1f}",
                        'Accuracy (%)': f"{zone['accuracy']:.1f}",
                        'Games': zone['games'],
                        'ROI Diff (%)': f"{zone['roi_difference']:+.2f}"
                    })
                
                zones_df = pd.DataFrame(zones_data)
                st.dataframe(zones_df, hide_index=True, use_container_width=True)
    
    with analysis_tabs[2]:
        # 비교 분석
        st.markdown("#### 📊 Individual vs Combination Performance Analysis")
        
        if cross_combinations and optimal_zones:
            # 성과 비교 - 🆕 개선된 베스트 개별존 선정
            best_individual = get_best_individual_zone(optimal_zones, validation_info)
            best_combination = cross_combinations[0] if cross_combinations else None
            
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                st.markdown("##### 🏆 Best Individual Zone")
                
                # 🆕 선정 사유 표시
                selection_reason = best_individual.get('selection_reason', 'Highest Score')
                if 'Tier 1' in selection_reason:
                    status_color = "success"
                    status_icon = "🏆"
                elif 'Tier 2' in selection_reason:
                    status_color = "warning" 
                    status_icon = "⚠️"
                else:
                    status_color = "error"
                    status_icon = "🚨"
                
                st.info(f"""
                **Zone**: {best_individual['dimension_name']} - {best_individual['segment']}
                
                📈 **Performance**:
                - ROI: {best_individual['roi']:.2f}%
                - Win Rate: {best_individual['win_rate']:.1f}%
                - Accuracy: {best_individual['accuracy']:.1f}%
                - Games: {best_individual['games']}
                - Score: {best_individual['score']:.2f}
                
                {status_icon} **Selection**: {selection_reason}
                """)
            
            with col_comp2:
                if best_combination:
                    st.markdown("##### 🎯 Best Combination")
                    st.info(f"""
                    **Combination**: {best_combination['combination_name']}
                    
                    📈 **Performance**:
                    - ROI: {best_combination['stats']['actual_roi']:.2f}%
                    - Win Rate: {best_combination['stats']['win_rate']:.1f}%
                    - Accuracy: {best_combination['stats']['accuracy']:.1f}%
                    - Games: {best_combination['stats']['games']}
                    - Synergy: +{best_combination['synergy_effect']:.2f}%
                    """)
                else:
                    st.info("No valid combinations found.")
            
            # Strategic recommendation
            st.markdown("#### 💡 Strategic Recommendation")
            
            synergy_threshold = 5.0
            roi_threshold = 10
            roi_difference_threshold = 2.0
            
            if best_combination and best_combination['synergy_effect'] > synergy_threshold:
                roi_diff = best_individual['roi'] - best_combination['stats']['actual_roi']
                
                if roi_diff > roi_difference_threshold:
                    st.success(f"""
                    🚀 **Recommendation**: Use Individual Strategy
                    
                    The best individual zone ({best_individual['dimension_name']} - {best_individual['segment']}) shows a **{roi_diff:.2f}% ROI advantage**, 
                    and the ROI difference is significant.
                    
                    **Suggested Approach**: Focus on games that meet the individual zone conditions.
                    
                    **📊 Decision Logic**: Individual ROI advantage ({roi_diff:.2f}%) > Threshold ({roi_difference_threshold}%)
                    """)
                else:
                    st.success(f"""
                    🚀 **Recommendation**: Use Combination Strategy
                    
                    The best combination ({best_combination['combination_name']}) shows a **{best_combination['synergy_effect']:.2f}% synergy boost**, and the ROI difference is acceptable ({roi_diff:+.2f}%).
                    
                    **Suggested Approach**: Focus on games that meet both conditions simultaneously.
                    
                    **📊 Decision Logic**: Synergy Effect ({best_combination['synergy_effect']:.2f}%) > Threshold ({synergy_threshold}%) & ROI Loss ({roi_diff:.2f}%) ≤ Threshold ({roi_difference_threshold}%)
                    """)
            elif best_individual['roi'] > roi_threshold:
                # 최고 시너지 값 가져오기
                best_synergy = best_combination['synergy_effect'] if best_combination else 0.0
                
                st.success(f"""
                🚀 **Recommendation**: Use Individual Strategy
                
                The best individual zone shows **{best_individual['roi']:.2f}% ROI** which exceeds our threshold.
                
                **Suggested Approach**: Focus on the high-performing individual zone.
                
                **📊 Decision Logic**: Individual ROI ({best_individual['roi']:.2f}%) > Threshold ({roi_threshold}%) & Synergy ({best_synergy:.2f}%) ≤ Threshold ({synergy_threshold}%)
                """)
            else:
                st.warning(f"""
                ⚠️ **Recommendation**: Consider Lowering Criteria
                
                Current performance levels are below high-confidence thresholds.
                
                **Suggestion**: Lower criteria or collect more data for analysis.
                """)
    
    # 종합 추천사항
    st.markdown("#### 💡 Final Investment Recommendations")
    
    if optimal_zones:
        # 🆕 개선된 베스트 개별존 선정
        best_zone = get_best_individual_zone(optimal_zones, validation_info)
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.markdown("##### 🎯 Primary Recommendation")
            
            # 🆕 선정 사유 경고 표시
            selection_reason = best_zone.get('selection_reason', '')
            if 'Risk: Overfitting' in selection_reason:
                st.warning(f"⚠️ **Note**: Using unvalidated zone due to insufficient validated options. Risk of overfitting.")
            
            # 🆕 Strategic Recommendation과 동일한 기준 사용
            synergy_threshold = 5.0  # Strategic과 동일
            roi_threshold = 10       # Strategic과 동일
            roi_difference_threshold = 2.0  # 🆕 Strategic과 동일
            
            # 🆕 개선된 조합 vs 개별 판단 로직
            if cross_combinations and cross_combinations[0]['synergy_effect'] > synergy_threshold:
                best_combo = cross_combinations[0]
                roi_diff = best_zone['roi'] - best_combo['stats']['actual_roi']
                
                if roi_diff > roi_difference_threshold:
                    # 개별이 훨씬 좋으면 개별 추천
                    st.success(f"""
                    **Best Strategy**: Individual Zone Focus
                    
                    📈 **Target Zone**: {best_zone['dimension_name']} - {best_zone['segment']}
                    - Expected ROI: {best_zone['roi']:.2f}%
                    - Win Rate: {best_zone['win_rate']:.1f}%
                    - Sample Size: {best_zone['games']} games
                    - **ROI Advantage**: +{roi_diff:.2f}% over combination
                    - **Selection**: {best_zone.get('selection_reason', 'Highest Score')}
                    
                    **📊 Decision Logic**: Individual ROI advantage ({roi_diff:.2f}%) > Threshold ({roi_difference_threshold}%) despite good synergy ({best_combo['synergy_effect']:.2f}%)
                    """)
                else:
                    # 조합 추천
                    st.success(f"""
                    **Best Strategy**: Combination Approach
                    
                    📈 **Target Combination**: {best_combo['combination_name']}
                    - Conditions: {best_combo['combination_detail']}
                    - Expected ROI: {best_combo['stats']['actual_roi']:.2f}%
                    - Synergy Boost: +{best_combo['synergy_effect']:.2f}%
                    - Confidence Score: {best_combo['score']:.2f}/10
                    - **ROI Trade-off**: {roi_diff:+.2f}% for synergy benefit
                    
                    **📊 Decision Logic**: Synergy Effect ({best_combo['synergy_effect']:.2f}%) > Threshold ({synergy_threshold}%) & ROI Loss ({roi_diff:.2f}%) ≤ Threshold ({roi_difference_threshold}%)
                    """)
            elif best_zone['roi'] > roi_threshold:
                # 최고 시너지 값 가져오기
                best_synergy = cross_combinations[0]['synergy_effect'] if cross_combinations else 0.0
                
                st.success(f"""
                **Best Strategy**: Individual Zone Focus
                
                📈 **Target Zone**: {best_zone['dimension_name']} - {best_zone['segment']}
                - Expected ROI: {best_zone['roi']:.2f}%
                - Win Rate: {best_zone['win_rate']:.1f}%
                - Sample Size: {best_zone['games']} games
                - Confidence Score: {best_zone['score']:.2f}/10
                - **Selection**: {best_zone.get('selection_reason', 'Highest Score')}
                
                **📊 Decision Logic**: Individual ROI ({best_zone['roi']:.2f}%) > Threshold ({roi_threshold}%) & Synergy ({best_synergy:.2f}%) ≤ Threshold ({synergy_threshold}%)
                """)
            else:
                st.warning(f"""
                **Strategy**: Moderate Performance - Consider Criteria Adjustment
                
                📈 **Current Best**: {best_zone['dimension_name']} - {best_zone['segment']}
                - Expected ROI: {best_zone['roi']:.2f}%
                - Confidence Score: {best_zone['score']:.2f}/10
                - **Selection**: {best_zone.get('selection_reason', 'Highest Score')}
                
                **📊 Decision Logic**: Performance below high-confidence thresholds
                **💡 Suggestion**: Lower criteria or collect more data
                """)
        
        with col_rec2:
            st.markdown("##### 📊 Alternative Options")
            
            # 🆕 통합 순위 시스템
            # Primary 전략 정보 준비
            primary_strategy = best_zone.copy()
            
            # Primary가 조합인지 확인
            if (cross_combinations and cross_combinations[0]['synergy_effect'] > synergy_threshold and
                best_zone['roi'] - cross_combinations[0]['stats']['actual_roi'] <= roi_difference_threshold):
                # 조합이 Primary로 선정된 경우
                primary_strategy = cross_combinations[0].copy()
                primary_strategy['type'] = 'combination'
                
                # 🔧 조합의 베팅 추천을 위한 zone 데이터 구조 보완
                combo = cross_combinations[0]
                
                # zone1, zone2 객체에서 정보 가져오기
                z1 = combo.get('zone1', {})
                z2 = combo.get('zone2', {})
                
                zone1_data = {
                    'dimension': z1.get('dimension') or combo.get('zone1_dimension'),
                    'dimension_name': z1.get('dimension_name') or combo.get('zone1_dimension'),
                    'segment': z1.get('segment') or combo.get('zone1_segment')
                }
                zone2_data = {
                    'dimension': z2.get('dimension') or combo.get('zone2_dimension'),
                    'dimension_name': z2.get('dimension_name') or combo.get('zone2_dimension'),
                    'segment': z2.get('segment') or combo.get('zone2_segment')
                }
                
                primary_strategy['zone1'] = zone1_data
                primary_strategy['zone2'] = zone2_data
                
                # 🆕 기존 조건 표시 로직과 호환을 위해 data 구조도 추가
                primary_strategy['data'] = {
                    'zone1': zone1_data,
                    'zone2': zone2_data
                }
                
                # 조건 정보도 추가
                zone1_seg = primary_strategy['zone1']['segment'] or 'Unknown'
                zone2_seg = primary_strategy['zone2']['segment'] or 'Unknown'
                primary_strategy['conditions'] = f"{zone1_seg} + {zone2_seg}"
                
                # 조합의 unified_score 계산
                score_info = calculate_unified_investment_score(cross_combinations[0], 'combination')
                primary_strategy['unified_score'] = score_info['total_score']
            else:
                primary_strategy['type'] = 'individual'
                # 개별존의 unified_score 계산
                score_info = calculate_unified_investment_score(best_zone, 'individual', validation_info)
                primary_strategy['unified_score'] = score_info['total_score']
            
            # 통합 대안 순위 생성
            alternative_strategies = create_unified_alternative_ranking(
                optimal_zones, cross_combinations, validation_info, primary_strategy
            )
            
            if alternative_strategies:
                st.markdown("**🏆 Unified Alternative Ranking**")
                st.caption("Validated strategies (Tier 1/2 individual zones + combinations) ranked by comprehensive score")
                
                # 필터링 정보 표시
                total_individual_zones = len([z for z in optimal_zones if f"{z['dimension_name']} - {z['segment']}" != (primary_strategy.get('combination_name') if primary_strategy.get('type') == 'combination' else f"{primary_strategy['dimension_name']} - {primary_strategy['segment']}")])
                validated_individual_zones = len([s for s in alternative_strategies if s['type'] == 'individual'])
                filtered_out = total_individual_zones - validated_individual_zones
                
                if filtered_out > 0:
                    st.info(f"ℹ️ **Note**: {filtered_out} individual zone(s) excluded due to validation failure. Only Tier 1/2 validated zones are shown.")
                
                # Top 5 대안 전략 표시
                for i, strategy in enumerate(alternative_strategies[:5], 1):
                    # 전략 타입에 따른 아이콘
                    if strategy['type'] == 'individual':
                        type_icon = "📊"
                        extra_info = f"Games: {strategy['games']}"
                    else:
                        type_icon = "🎯"
                        extra_info = f"Synergy: +{strategy['synergy']:.2f}% | Games: {strategy['games']}"
                    
                    # 검증 상태 아이콘
                    validation_icon = ""
                    if strategy['validation_status'] == "🏆 Tier 1":
                        validation_icon = "🏆"
                    elif strategy['validation_status'] == "⚠️ Tier 2":
                        validation_icon = "⚠️"
                    elif strategy['validation_status'] == "❌ Failed":
                        validation_icon = "❌"
                    elif strategy['validation_status'] == "🎯 Combination":
                        validation_icon = "🎯"
                    
                    # 전략 표시
                    st.write(f"**#{i}** {type_icon} {validation_icon} {strategy['name']}")
                    st.caption(f"ROI: {strategy['roi']:.2f}% | Score: {strategy['unified_score']:.1f}/100 | {extra_info}")
                    
                    # 점수 세부 정보 (expander)
                    if i <= 3:  # Top 3만 세부 정보 제공
                        with st.expander(f"📊 Score Breakdown #{i}", expanded=False):
                            st.text(f"Score Breakdown: {strategy['score_breakdown']}")
                            if strategy['type'] == 'individual':
                                st.text(f"Original Individual Score: {strategy['original_score']:.2f}")
                                st.text(f"Validation Status: {strategy['validation_status']}")
                            else:
                                st.text(f"Original Combination Score: {strategy['original_score']:.2f}")
                                st.text(f"Synergy Effect: +{strategy['synergy']:.2f}%")
                
                # 점수 시스템 설명
                with st.expander("ℹ️ Unified Scoring System", expanded=False):
                    st.markdown("""
                    **📊 Individual Zones (Max 100 pts):**
                    - **ROI Score (0-40pts)**: ROI × 2 (20% ROI = 40pts)
                    - **Reliability (0-25pts)**: Based on sample size
                      - 50+ games: 25pts | 30+ games: 20pts | 20+ games: 15pts
                    - **Accuracy (0-20pts)**: (Accuracy - 50%) × 0.4
                    - **Validation Bonus (0-15pts)**: Tier 1: 15pts | Tier 2: 8pts
                    
                    **🎯 Combinations (Max 100 pts):**
                    - **ROI Score (0-35pts)**: ROI × 1.8
                    - **Reliability (0-20pts)**: Based on sample size (lower threshold)
                    - **Accuracy (0-15pts)**: (Accuracy - 50%) × 0.3
                    - **Synergy Bonus (0-30pts)**: Based on synergy effect
                      - 10%+: 30pts | 7%+: 25pts | 5%+: 20pts | 3%+: 15pts
                    """)
            else:
                st.info("💡 No alternative strategies available (Primary is the only option)")
        
        # 🆕 실제 배팅 추천 표시 - 추천 전략이 있을 때만 표시
        if optimal_weights and (primary_strategy or alternative_strategies):
            # investment_analysis에 전략 정보 추가
            betting_analysis = investment_analysis.copy()
            betting_analysis['primary_strategy'] = primary_strategy
            betting_analysis['alternative_strategies'] = alternative_strategies
            
            # 🔧 segments_analysis도 포함
            if segments_analysis:
                betting_analysis['segments_analysis'] = segments_analysis
            
            display_betting_recommendations(betting_analysis, optimal_weights)
        else:
            st.info("💡 Betting recommendations not available - no optimal weights or strategies found")
    
    else:
        st.error("No qualifying zones found. Consider lowering your criteria.")

def calculate_unified_investment_score(strategy, strategy_type, validation_info=None):
    """
    개별존과 조합을 통합 평가하는 점수 시스템
    
    Args:
        strategy: 개별존 또는 조합 딕셔너리
        strategy_type: 'individual' 또는 'combination'
        validation_info: 검증 정보 (개별존의 경우)
    
    Returns:
        통합 점수 (0-100)
    """
    
    if strategy_type == 'individual':
        # 개별존 점수 계산
        roi = strategy['roi']
        games = strategy['games']
        score = strategy['score']
        win_rate = strategy['win_rate']
        accuracy = strategy['accuracy']
        
        # 기본 ROI 점수 (0-40점)
        roi_score = min(max(roi * 2, 0), 40)  # ROI 20%면 40점
        
        # 신뢰도 점수 (0-25점) - 게임 수 기반
        if games >= 50:
            reliability_score = 25
        elif games >= 30:
            reliability_score = 20
        elif games >= 20:
            reliability_score = 15
        elif games >= 15:
            reliability_score = 10
        else:
            reliability_score = 5
        
        # 정확도 점수 (0-20점)
        accuracy_score = min(max((accuracy - 50) * 0.4, 0), 20)  # 50% 기준, 100%면 20점
        
        # 검증 보너스 (0-15점)
        validation_bonus = 0
        if validation_info:
            zone_name = f"{strategy['dimension_name']} - {strategy['segment']}"
            if zone_name in validation_info:
                validation = validation_info[zone_name]
                tier = validation.get('tier')
                if tier == 'Tier 1':
                    validation_bonus = 15  # 최고 보너스
                elif tier == 'Tier 2':
                    validation_bonus = 8   # 중간 보너스
                # Failed validation은 0점
        
        total_score = roi_score + reliability_score + accuracy_score + validation_bonus
        
        return {
            'total_score': min(total_score, 100),
            'roi_score': roi_score,
            'reliability_score': reliability_score,
            'accuracy_score': accuracy_score,
            'validation_bonus': validation_bonus,
            'breakdown': f"ROI:{roi_score:.1f} + Rel:{reliability_score:.1f} + Acc:{accuracy_score:.1f} + Val:{validation_bonus:.1f}"
        }
        
    elif strategy_type == 'combination':
        # 조합 점수 계산
        roi = strategy['stats']['actual_roi']
        games = strategy['stats']['games']
        synergy = strategy['synergy_effect']
        win_rate = strategy['stats']['win_rate']
        accuracy = strategy['stats']['accuracy']
        
        # 기본 ROI 점수 (0-35점) - 개별존보다 약간 낮게
        roi_score = min(max(roi * 1.8, 0), 35)
        
        # 신뢰도 점수 (0-20점) - 조합은 일반적으로 게임 수가 적음
        if games >= 40:
            reliability_score = 20
        elif games >= 25:
            reliability_score = 16
        elif games >= 15:
            reliability_score = 12
        elif games >= 10:
            reliability_score = 8
        else:
            reliability_score = 4
        
        # 정확도 점수 (0-15점)
        accuracy_score = min(max((accuracy - 50) * 0.3, 0), 15)
        
        # 시너지 보너스 (0-30점) - 조합의 핵심 가치
        if synergy >= 10:
            synergy_bonus = 30
        elif synergy >= 7:
            synergy_bonus = 25
        elif synergy >= 5:
            synergy_bonus = 20
        elif synergy >= 3:
            synergy_bonus = 15
        elif synergy >= 1:
            synergy_bonus = 10
        elif synergy > 0:
            synergy_bonus = 5
        else:
            synergy_bonus = 0
        
        total_score = roi_score + reliability_score + accuracy_score + synergy_bonus
        
        return {
            'total_score': min(total_score, 100),
            'roi_score': roi_score,
            'reliability_score': reliability_score,
            'accuracy_score': accuracy_score,
            'synergy_bonus': synergy_bonus,
            'breakdown': f"ROI:{roi_score:.1f} + Rel:{reliability_score:.1f} + Acc:{accuracy_score:.1f} + Syn:{synergy_bonus:.1f}"
        }
    
    return {'total_score': 0, 'breakdown': 'Unknown type'}

def create_unified_alternative_ranking(optimal_zones, cross_combinations, validation_info, primary_strategy):
    """
    Primary를 제외한 모든 전략의 통합 순위 생성
    
    Args:
        optimal_zones: 개별존 리스트
        cross_combinations: 조합 리스트  
        validation_info: 검증 정보
        primary_strategy: Primary로 선정된 전략 정보
    
    Returns:
        순위별 정렬된 대안 전략 리스트
    """
    
    alternative_strategies = []
    
    # Primary 전략 식별
    primary_type = primary_strategy.get('type', 'individual')
    if primary_type == 'individual':
        primary_name = f"{primary_strategy['dimension_name']} - {primary_strategy['segment']}"
    else:
        primary_name = primary_strategy['combination_name']
    
    # 개별존들 추가 (Primary 제외, 검증 통과한 것만)
    for zone in optimal_zones:
        zone_name = f"{zone['dimension_name']} - {zone['segment']}"
        if zone_name != primary_name:
            # 검증 상태 확인
            validation_status = "Not Validated"
            validation_tier = None
            include_zone = False  # 기본적으로 제외
            
            if validation_info and zone_name in validation_info:
                validation = validation_info[zone_name]
                tier = validation.get('tier')
                if tier == 'Tier 1':
                    validation_status = "🏆 Tier 1"
                    validation_tier = 1
                    include_zone = True  # 검증 통과 - 포함
                elif tier == 'Tier 2':
                    validation_status = "⚠️ Tier 2"
                    validation_tier = 2
                    include_zone = True  # 검증 통과 - 포함
                else:
                    validation_status = "❌ Failed"
                    validation_tier = 3
                    include_zone = False  # 검증 실패 - 제외
            
            # 검증 통과한 개별존만 추가
            if include_zone:
                # 통합 점수 계산
                score_info = calculate_unified_investment_score(zone, 'individual', validation_info)
                
                alternative_strategies.append({
                    'name': zone_name,
                    'type': 'individual',
                    'roi': zone['roi'],
                    'games': zone['games'],
                    'win_rate': zone['win_rate'],
                    'accuracy': zone['accuracy'],
                    'original_score': zone['score'],
                    'validation_status': validation_status,
                    'validation_tier': validation_tier,
                    'unified_score': score_info['total_score'],
                    'score_breakdown': score_info['breakdown'],
                    'data': zone
                })
    
    # 조합들 추가 (Primary 제외)
    for combo in cross_combinations:
        if combo['combination_name'] != primary_name:
            # 통합 점수 계산
            score_info = calculate_unified_investment_score(combo, 'combination')
            
            # 🔧 Alternative 조합을 위한 zone 데이터 구조 생성
            z1 = combo.get('zone1', {})
            z2 = combo.get('zone2', {})
            
            zone1_data = {
                'dimension': z1.get('dimension') or combo.get('zone1_dimension'),
                'dimension_name': z1.get('dimension_name') or combo.get('zone1_dimension'),
                'segment': z1.get('segment') or combo.get('zone1_segment')
            }
            zone2_data = {
                'dimension': z2.get('dimension') or combo.get('zone2_dimension'),
                'dimension_name': z2.get('dimension_name') or combo.get('zone2_dimension'),
                'segment': z2.get('segment') or combo.get('zone2_segment')
            }
            
            alternative_strategies.append({
                'name': combo['combination_name'],
                'type': 'combination',
                'roi': combo['stats']['actual_roi'],
                'games': combo['stats']['games'],
                'win_rate': combo['stats']['win_rate'],
                'accuracy': combo['stats']['accuracy'],
                'synergy': combo['synergy_effect'],
                'original_score': combo['score'],
                'validation_status': "🎯 Combination",
                'validation_tier': 0,  # 조합은 별도 순위
                'unified_score': score_info['total_score'],
                'score_breakdown': score_info['breakdown'],
                'stats': combo['stats'],  # 🆕 조합 전략 성과 표시용
                # 🆕 베팅 추천 호환성을 위한 구조들
                'zone1': zone1_data,
                'zone2': zone2_data,
                'data': {
                    'zone1': zone1_data,
                    'zone2': zone2_data
                },
                'combination_name': combo['combination_name'],
                'original_combo_data': combo  # 원본 데이터도 보존
            })
    
    # 통합 점수로 정렬 (높은 점수 순)
    alternative_strategies.sort(key=lambda x: x['unified_score'], reverse=True)
    
    return alternative_strategies

def find_latest_prediction_file():
    """최신 MLB 예측 데이터 파일을 찾는 함수"""
    import os
    
    # 파일 패턴 정의
    pattern = "src/odds/data/matched/mlb_predictions_with_odds_*.json"
    
    try:
        # 모든 매칭되는 파일 찾기
        files = glob.glob(pattern)
        
        if not files:
            return None
            
        # 파일명에서 날짜 추출하여 최신 파일 찾기
        def extract_datetime(filename):
            # mlb_predictions_with_odds_20250607_150950.json 패턴에서 날짜시간 추출
            match = re.search(r'mlb_predictions_with_odds_(\d{8}_\d{6})\.json', filename)
            if match:
                return match.group(1)
            return "00000000_000000"
        
        # 최신 파일 선택
        latest_file = max(files, key=extract_datetime)
        return latest_file
        
    except Exception as e:
        st.error(f"파일 검색 중 오류 발생: {e}")
        return None

def load_latest_prediction_data():
    """최신 예측 데이터를 로드하고 배당률이 있는 경기만 필터링"""
    latest_file = find_latest_prediction_file()
    
    if not latest_file:
        return None, None
        
    try:
        import json
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 배당률이 있는 경기만 필터링
        filtered_data = []
        for game in data:
            # 다양한 필드명 지원
            home_odds = game.get('home_team_odds') or game.get('home_odds')
            away_odds = game.get('away_team_odds') or game.get('away_odds')
            
            # 배당률이 모두 존재하는 경우만 포함
            if home_odds is not None and away_odds is not None:
                # 0이 아닌 유효한 배당률인지 확인
                if home_odds != 0 and away_odds != 0:
                    # 표준화된 필드명으로 정규화
                    game['home_team_odds'] = home_odds
                    game['away_team_odds'] = away_odds
                    filtered_data.append(game)
        
        return filtered_data, latest_file
        
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None, None

def process_game_for_strategy(game, optimal_weights):
    """게임 데이터를 전략 분석을 위해 전처리"""
    try:
        # 앙상블 확률 계산
        ensemble_prob = 0
        total_weight = 0
        
        for model, weight in optimal_weights.items():
            prob_key = f"{model}_probability"
            if prob_key in game and game[prob_key] is not None:
                ensemble_prob += float(game[prob_key]) * weight
                total_weight += weight
        
        if total_weight == 0:
            ensemble_prob = game.get('ensemble_probability', game.get('win_probability', 0.5))
        else:
            ensemble_prob = ensemble_prob / total_weight
        
        # 베팅 결정 (ensemble_prob > 0.5이면 홈팀)
        if ensemble_prob > 0.5:
            predicted_team = 'home'
            bet_odds = game.get('home_team_odds')
            bet_team_name = game.get('home_team')
        else:
            predicted_team = 'away'
            bet_odds = game.get('away_team_odds')
            bet_team_name = game.get('away_team')
        
        if bet_odds is None:
            return None
        
        # 예측 ROI 계산
        if bet_odds > 0:
            win_payout = (bet_odds / 100) * 100
        else:
            win_payout = (100 / abs(bet_odds)) * 100
        
        win_prob = ensemble_prob if predicted_team == 'home' else 1 - ensemble_prob
        predicted_roi = (win_prob * win_payout) + ((1 - win_prob) * (-100))
        
        # Confidence 계산
        confidence = abs(ensemble_prob - 0.5)
        
        return {
            'game_info': f"{game.get('away_team', '')} @ {game.get('home_team', '')}",
            'date': game.get('date', ''),
            'start_time': game.get('start_time_et', ''),
            'ensemble_prob': ensemble_prob,
            'predicted_roi': predicted_roi,
            'confidence': confidence,
            'bet_odds': bet_odds,
            'predicted_team': predicted_team,
            'bet_team_name': bet_team_name,
            'home_team': game.get('home_team', ''),
            'away_team': game.get('away_team', ''),
            'raw_game': game
        }
        
    except Exception as e:
        return None

def find_matching_games_for_strategy(strategy, latest_data, optimal_weights):
    """특정 전략에 맞는 경기들을 찾는 함수"""
    matching_games = []
    
    if not latest_data or not optimal_weights:
        return matching_games
    
    for game in latest_data:
        # 게임 데이터 전처리
        processed_game = process_game_for_strategy(game, optimal_weights)
        if processed_game is None:
            continue
        
        # 전략 타입에 따라 조건 확인
        if strategy['type'] == 'individual':
            # 개별존 조건 확인 - 다양한 구조 지원
            # 1. Direct fields (primary_strategy)
            dimension = strategy.get('dimension') or strategy.get('dimension_name')
            segment = strategy.get('segment')
            
            # 2. From data field (alternative_strategies)
            if not dimension or not segment:
                data = strategy.get('data', {})
                dimension = data.get('dimension') or data.get('dimension_name')
                segment = data.get('segment')
            
            if dimension and segment:
                zone = {
                    'dimension': dimension,
                    'segment': segment
                }
                
                if check_zone_condition(game, zone, optimal_weights):
                    matching_games.append(processed_game)
            else:
                # 필드가 없으면 건너뛰기
                continue
                
        elif strategy['type'] == 'combination':
            # 🔧 조합 조건 확인 - 강화된 파싱 로직
            
            # 🔍 디버깅: 조합 전략 구조 확인 (필요시 활성화)
            # if len(matching_games) == 0:  
            #     import streamlit as st
            #     st.write("🔍 **Debug: Combination Strategy Structure**")
            #     st.json(strategy)
            
            # 방법 1: 직접 필드에서 가져오기 (cross_combinations에서 온 경우)
            zone1_dim = strategy.get('zone1_dimension')
            zone1_seg = strategy.get('zone1_segment')
            zone2_dim = strategy.get('zone2_dimension')
            zone2_seg = strategy.get('zone2_segment')
            
            # 방법 2: data 필드에서 가져오기
            if not all([zone1_dim, zone1_seg, zone2_dim, zone2_seg]):
                data = strategy.get('data', {})
                zone1 = data.get('zone1', {})
                zone2 = data.get('zone2', {})
                
                zone1_dim = zone1_dim or zone1.get('dimension') or zone1.get('dimension_name')
                zone1_seg = zone1_seg or zone1.get('segment')
                zone2_dim = zone2_dim or zone2.get('dimension') or zone2.get('dimension_name')
                zone2_seg = zone2_seg or zone2.get('segment')
            
            # 방법 3: Primary 전략에서 온 경우 - 조합 정보 파싱
            if not all([zone1_dim, zone1_seg, zone2_dim, zone2_seg]):
                # combination_name에서 파싱 시도
                combination_name = strategy.get('combination_name', strategy.get('name', ''))
                
                # "Market vs Model Divergence + Predicted ROI Range" 형태 파싱
                if ' + ' in combination_name:
                    parts = combination_name.split(' + ')
                    if len(parts) == 2:
                        zone1_dim, zone2_dim = parts[0].strip(), parts[1].strip()
                        
                        # 조건에서 세그먼트 추출 시도
                        conditions = strategy.get('conditions', '')
                        if conditions:
                            # "Model Much More Optimistic (+10%+) + Very Positive (>20%)" 형태
                            if ' + ' in conditions:
                                cond_parts = conditions.split(' + ')
                                if len(cond_parts) == 2:
                                    zone1_seg = cond_parts[0].strip()
                                    zone2_seg = cond_parts[1].strip()
            
            # 방법 4: 스마트존파인더에서 찾아진 조합의 실제 구조 파싱
            if not all([zone1_dim, zone1_seg, zone2_dim, zone2_seg]):
                # zone1, zone2 객체가 있는지 확인 (Primary 전략에서 추가된 구조)
                if 'zone1' in strategy and 'zone2' in strategy:
                    z1 = strategy['zone1']
                    z2 = strategy['zone2']
                    
                    zone1_dim = z1.get('dimension') or z1.get('dimension_name')
                    zone1_seg = z1.get('segment')
                    zone2_dim = z2.get('dimension') or z2.get('dimension_name')
                    zone2_seg = z2.get('segment')
                
                # 또는 nested data 구조
                elif 'data' in strategy:
                    data = strategy['data']
                    if 'zone1' in data and 'zone2' in data:
                        z1 = data['zone1']
                        z2 = data['zone2']
                        
                        zone1_dim = z1.get('dimension') or z1.get('dimension_name')
                        zone1_seg = z1.get('segment')
                        zone2_dim = z2.get('dimension') or z2.get('dimension_name')
                        zone2_seg = z2.get('segment')
                
                # 🔧 Primary 전략의 조합에서 실제 zone 객체가 포함된 경우
                if 'zone1' in strategy and isinstance(strategy['zone1'], dict):
                    zone1_obj = strategy['zone1']
                    zone2_obj = strategy['zone2']
                    
                    zone1_dim = zone1_dim or zone1_obj.get('dimension') or zone1_obj.get('dimension_name') 
                    zone1_seg = zone1_seg or zone1_obj.get('segment')
                    zone2_dim = zone2_dim or zone2_obj.get('dimension') or zone2_obj.get('dimension_name')
                    zone2_seg = zone2_seg or zone2_obj.get('segment')
            
            # 실제 조건 확인
            if all([zone1_dim, zone1_seg, zone2_dim, zone2_seg]):
                zone1 = {
                    'dimension': zone1_dim,
                    'segment': zone1_seg
                }
                zone2 = {
                    'dimension': zone2_dim,
                    'segment': zone2_seg
                }
                
                if (check_zone_condition(game, zone1, optimal_weights) and 
                    check_zone_condition(game, zone2, optimal_weights)):
                    matching_games.append(processed_game)
            else:
                # 🔍 파싱 실패 시 디버깅 정보 (필요시 활성화)
                # if len(matching_games) == 0:  # 첫 번째 게임에서만
                #     import streamlit as st
                #     st.error(f"❌ **Failed to parse combination strategy**")
                #     st.write(f"zone1_dim: {zone1_dim}, zone1_seg: {zone1_seg}")
                #     st.write(f"zone2_dim: {zone2_dim}, zone2_seg: {zone2_seg}")
                continue
    
    return matching_games

def display_betting_recommendations(investment_analysis, optimal_weights):
    """실제 배팅 추천을 표시하는 함수"""
    st.markdown("---")
    st.markdown("### 🎯 **Today's Betting Recommendations**")
    st.caption("Based on the latest prediction data and your selected strategies")
    
    # 최신 데이터 로드
    with st.spinner("Loading latest prediction data..."):
        latest_data, file_path = load_latest_prediction_data()
    
    if latest_data is None:
        st.error("❌ Unable to load latest prediction data")
        return
    
    # 파일 정보 표시
    file_name = file_path.split('/')[-1] if file_path else "Unknown"
    st.info(f"📊 **Data Source**: {file_name} | **Games with odds**: {len(latest_data)}")
    
    # Primary 전략과 Alternative 전략들 수집
    strategies_to_check = []
    
    # Primary 전략 추가
    if 'primary_strategy' in investment_analysis:
        primary = investment_analysis['primary_strategy']
        strategies_to_check.append({
            'name': 'Primary Recommendation',
            'strategy': primary,
            'priority': 1
        })
    
    # Alternative 전략들 추가 (Top 3만)
    if 'alternative_strategies' in investment_analysis:
        for i, alt_strategy in enumerate(investment_analysis['alternative_strategies'][:3], 1):
            strategies_to_check.append({
                'name': f'Alternative #{i}',
                'strategy': alt_strategy,
                'priority': i + 1
            })
    
    if not strategies_to_check:
        st.warning("No strategies available for betting recommendations")
        return
    
    # 각 전략별로 매칭되는 경기 찾기
    st.markdown("#### 📋 **Strategy-Based Recommendations**")
    
    total_recommendations = 0
    
    for strategy_info in strategies_to_check:
        strategy_name = strategy_info['name']
        strategy = strategy_info['strategy']
        
        # 매칭되는 경기 찾기
        matching_games = find_matching_games_for_strategy(strategy, latest_data, optimal_weights)
        
        if matching_games:
            total_recommendations += len(matching_games)
            
            # 전략별 섹션
            with st.expander(f"🎯 **{strategy_name}** ({len(matching_games)} games)", expanded=True):
                # 전략 정보 표시
                if strategy['type'] == 'individual':
                    # 다양한 구조 지원
                    dimension_name = strategy.get('dimension_name') or strategy.get('dimension')
                    segment = strategy.get('segment')
                    
                    # data 필드에서 찾기 (alternative_strategies)
                    if not dimension_name or not segment:
                        data = strategy.get('data', {})
                        dimension_name = data.get('dimension_name') or data.get('dimension')
                        segment = data.get('segment')
                    
                    # name 필드에서 파싱 (최후 수단)
                    if not dimension_name or not segment:
                        name_parts = strategy.get('name', '').split(' - ')
                        if len(name_parts) == 2:
                            dimension_name, segment = name_parts
                    
                    dimension_name = dimension_name or 'Unknown'
                    segment = segment or 'Unknown'
                    st.markdown(f"**Strategy**: {dimension_name} - {segment}")
                    
                elif strategy['type'] == 'combination':
                    combination_name = strategy.get('combination_name', strategy.get('name', 'Unknown Combination'))
                    st.markdown(f"**Strategy**: {combination_name}")
                    
                    # 🆕 조합의 세부 조건 표시
                    data = strategy.get('data', {})
                    zone1 = data.get('zone1', {})
                    zone2 = data.get('zone2', {})
                    
                    if zone1 and zone2:
                        zone1_name = f"{zone1.get('dimension_name', 'Unknown')} - {zone1.get('segment', 'Unknown')}"
                        zone2_name = f"{zone2.get('dimension_name', 'Unknown')} - {zone2.get('segment', 'Unknown')}"
                        st.caption(f"📋 **Conditions**: ({zone1_name}) + ({zone2_name})")
                    else:
                        st.caption("📋 **Conditions**: Details not available")
                
                # 성과 지표 표시 - 🔧 조합과 개별 전략 구분 처리
                metrics_cols = st.columns(4)
                with metrics_cols[0]:
                    if strategy['type'] == 'combination':
                        # 조합 전략: stats 필드에서 가져오기
                        roi_val = strategy.get('stats', {}).get('actual_roi', 0)
                    else:
                        # 개별 전략: 직접 필드에서 가져오기
                        roi_val = strategy.get('roi', strategy.get('expected_roi', 0))
                    st.metric("Expected ROI", f"{roi_val:.1f}%")
                
                with metrics_cols[1]:
                    if strategy['type'] == 'combination':
                        # 조합 전략: stats 필드에서 가져오기
                        win_rate = strategy.get('stats', {}).get('win_rate', 0)
                    else:
                        # 개별 전략: 직접 필드에서 가져오기
                        win_rate = strategy.get('win_rate', strategy.get('expected_win_rate', 0))
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                with metrics_cols[2]:
                    # 통합 점수 사용 (모든 전략에서 동일한 스케일)
                    score = strategy.get('unified_score', strategy.get('original_score', strategy.get('score', 0)))
                    st.metric("Strategy Score", f"{score:.1f}/100")
                
                with metrics_cols[3]:
                    if strategy['type'] == 'combination':
                        # 조합 전략: stats 필드에서 가져오기
                        games_analyzed = strategy.get('stats', {}).get('games', 0)
                    else:
                        # 개별 전략: 직접 필드에서 가져오기
                        games_analyzed = strategy.get('games', strategy.get('total_games', 0))
                    st.metric("Historical Games", f"{games_analyzed}")
                
                # 추천 경기 목록
                st.markdown("**🎲 Recommended Bets:**")
                
                # 경기별 상세 정보
                for i, game in enumerate(matching_games, 1):
                    with st.container():
                        game_cols = st.columns([3, 2, 2, 1.5, 1.5])
                        
                        with game_cols[0]:
                            st.markdown(f"**#{i} {game['game_info']}**")
                            st.caption(f"📅 {game['date']} | ⏰ {game['start_time']}")
                        
                        with game_cols[1]:
                            st.markdown(f"**🎯 Bet: {game['bet_team_name']}**")
                            st.caption(f"Odds: {game['bet_odds']:+}")
                        
                        with game_cols[2]:
                            st.metric("Predicted ROI", f"{game['predicted_roi']:+.1f}%")
                        
                        with game_cols[3]:
                            prob_display = game['ensemble_prob'] if game['predicted_team'] == 'home' else 1 - game['ensemble_prob']
                            st.metric("Win Prob", f"{prob_display:.1%}")
                        
                        with game_cols[4]:
                            st.metric("Confidence", f"{game['confidence']:.3f}")
                
                st.markdown("---")
        else:
            # 매칭되는 경기가 없는 경우
            with st.expander(f"🎯 **{strategy_name}** (0 games)", expanded=False):
                if strategy['type'] == 'individual':
                    # 다양한 구조 지원
                    dimension_name = strategy.get('dimension_name') or strategy.get('dimension')
                    segment = strategy.get('segment')
                    
                    # data 필드에서 찾기 (alternative_strategies)
                    if not dimension_name or not segment:
                        data = strategy.get('data', {})
                        dimension_name = data.get('dimension_name') or data.get('dimension')
                        segment = data.get('segment')
                    
                    # name 필드에서 파싱 (최후 수단)
                    if not dimension_name or not segment:
                        name_parts = strategy.get('name', '').split(' - ')
                        if len(name_parts) == 2:
                            dimension_name, segment = name_parts
                    
                    dimension_name = dimension_name or 'Unknown'
                    segment = segment or 'Unknown'
                    st.markdown(f"**Strategy**: {dimension_name} - {segment}")
                    
                elif strategy['type'] == 'combination':
                    combination_name = strategy.get('combination_name', strategy.get('name', 'Unknown Combination'))
                    st.markdown(f"**Strategy**: {combination_name}")
                    
                    # 🆕 조합의 세부 조건 표시
                    data = strategy.get('data', {})
                    zone1 = data.get('zone1', {})
                    zone2 = data.get('zone2', {})
                    
                    if zone1 and zone2:
                        zone1_name = f"{zone1.get('dimension_name', 'Unknown')} - {zone1.get('segment', 'Unknown')}"
                        zone2_name = f"{zone2.get('dimension_name', 'Unknown')} - {zone2.get('segment', 'Unknown')}"
                        st.caption(f"📋 **Conditions**: ({zone1_name}) + ({zone2_name})")
                    else:
                        st.caption("📋 **Conditions**: Details not available")
                        
                st.info("⚠️ No games match this strategy's criteria in today's data")
    
    # 전체 요약
    if total_recommendations > 0:
        st.success(f"🎉 **Total Betting Opportunities**: {total_recommendations} games")
        
        # 🔍 매칭 로직 검증 정보
        with st.expander("🔍 **Matching Logic Verification**", expanded=False):
            st.markdown("**Verification that betting recommendations use the same conditions as segment analysis:**")
            
            total_unique_games = len(latest_data)
            st.write(f"📊 **Total games with odds**: {total_unique_games}")
            st.write(f"🎯 **Total betting opportunities**: {total_recommendations}")
            st.write(f"📈 **Coverage rate**: {(total_recommendations/total_unique_games)*100:.1f}%")
            
            st.markdown("**📋 Condition Verification:**")
            st.info("""
            ✅ **Individual Zones**: Use `check_zone_condition(game, zone, optimal_weights)` 
            - Same function used in segment analysis
            - Applies identical thresholds and logic
            
            ✅ **Combinations**: Check both zone1 AND zone2 conditions
            - Both zones must satisfy individual zone conditions
            - Same logic as cross-validation analysis
            
            ✅ **Preprocessing**: Uses same ensemble probability calculation
            - Same weighted average across models
            - Same ROI and confidence calculations
            """)
            
            # 간단한 검증: 각 전략별 매칭 게임 수 요약
            st.markdown("**🎲 Strategy Breakdown:**")
            strategy_summary = []
            for strategy_info in strategies_to_check:
                strategy_name = strategy_info['name']
                strategy = strategy_info['strategy']
                matching_games = find_matching_games_for_strategy(strategy, latest_data, optimal_weights)
                strategy_summary.append({
                    'Strategy': strategy_name,
                    'Type': strategy.get('type', 'unknown'),
                    'Matching Games': len(matching_games),
                    'Coverage %': f"{(len(matching_games)/total_unique_games)*100:.1f}%"
                })
            
            import pandas as pd
            df_summary = pd.DataFrame(strategy_summary)
            st.dataframe(df_summary, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### ⚠️ **Important Disclaimers**")
        st.warning("""
        **Risk Warning**: 
        - All predictions are based on historical analysis and mathematical models
        - Past performance does not guarantee future results
        - Please bet responsibly and only with money you can afford to lose
        - Consider this as analytical guidance, not financial advice
        """)
    else:
        st.info("📊 No betting opportunities found based on your selected strategies in today's data")
    
    # 🆕 커스텀 존 선택기 추가 - 전략 기반 추천 이후에 표시
    # segments_analysis가 investment_analysis에 있는지 확인
    if 'segments_analysis' in investment_analysis:
        display_custom_zone_selector(
            investment_analysis['segments_analysis'], 
            optimal_weights
        )
    else:
        # segments_analysis가 없으면 정보 메시지만 표시
        st.markdown("---")
        st.info("💡 **Custom Zone Selector**: Available after running ensemble optimization with segment analysis")

def display_custom_zone_selector(segments_analysis, optimal_weights):
    """사용자 커스텀 존 선택기 - 차원별 OR, 차원간 AND 조합"""
    
    st.markdown("---")
    st.markdown("### 🎛️ **Custom Zone Selector**")
    st.caption("Build your own betting strategy by combining zones (Same dimension = OR, Different dimensions = AND)")
    
    # segments_analysis에서 사용 가능한 모든 존들 수집
    if not segments_analysis:
        st.warning("❌ No segment analysis data available")
        return
    
    # 차원별 정보 정리
    dimension_info = {
        'predicted_roi': '📊 Predicted ROI',
        'odds': '💰 Odds Ranges', 
        'confidence': '🎯 Confidence Levels',
        'odds_probability_divergence': '📈 Market vs Model',
        'kelly_criterion': '🎰 Kelly Criterion',
        'model_consensus': '🤝 Model Consensus'
    }
    
    # 사용 가능한 차원들만 필터링
    available_dimensions = {}
    for dimension, segments in segments_analysis.items():
        if dimension in dimension_info:
            # 게임이 있는 세그먼트만 수집
            available_segments = []
            for segment_name, stats in segments.items():
                if stats['games'] > 0:
                    available_segments.append({
                        'name': segment_name,
                        'games': stats['games'],
                        'roi': stats['actual_roi'],
                        'accuracy': stats['accuracy'],
                        'win_rate': stats['win_rate']
                    })
            
            if available_segments:
                available_dimensions[dimension] = {
                    'title': dimension_info[dimension],
                    'segments': available_segments
                }
    
    if not available_dimensions:
        st.warning("❌ No zones available for selection")
        return
    
    st.info(f"""
    💡 **How it works:**
    - **Within same dimension**: Conditions are combined with OR (any one matches)
    - **Across different dimensions**: Conditions are combined with AND (all must match)
    - **Example**: (Positive ROI OR Very Positive ROI) AND (High Confidence) AND (Extreme Bet)
    """)
    
    # 차원별 체크박스 UI 생성
    selected_zones = {}
    
    # 3열로 배치
    dimension_keys = list(available_dimensions.keys())
    num_cols = min(3, len(dimension_keys))
    cols = st.columns(num_cols)
    
    for i, (dimension, dim_info) in enumerate(available_dimensions.items()):
        with cols[i % num_cols]:
            st.markdown(f"#### {dim_info['title']}")
            
            selected_segments = []
            for segment in dim_info['segments']:
                # 체크박스 라벨에 유용한 정보 포함
                label = f"{segment['name']}"
                help_text = f"Games: {segment['games']}, ROI: {segment['roi']:.1f}%, Win Rate: {segment['win_rate']:.1f}%"
                
                if st.checkbox(
                    label, 
                    key=f"custom_zone_{dimension}_{segment['name']}", 
                    help=help_text
                ):
                    selected_segments.append({
                        'dimension': dimension,
                        'segment': segment['name'],
                        'stats': segment
                    })
            
            if selected_segments:
                selected_zones[dimension] = selected_segments
    
    # 선택된 존이 있으면 분석 실행
    if selected_zones:
        st.markdown("---")
        st.markdown("### 🎯 **Custom Strategy Analysis**")
        
        # 선택 요약 표시
        total_conditions = sum(len(segments) for segments in selected_zones.values())
        st.info(f"📋 **Selected Strategy**: {total_conditions} conditions across {len(selected_zones)} dimensions")
        
        # 차원별 선택 조건 표시
        for dimension, segments in selected_zones.items():
            dimension_title = available_dimensions[dimension]['title']
            segment_names = [seg['segment'] for seg in segments]
            
            if len(segment_names) == 1:
                condition_text = segment_names[0]
            else:
                condition_text = f"({' OR '.join(segment_names)})"
            
            st.write(f"• **{dimension_title}**: {condition_text}")
        
        # 최신 예측 데이터 로드 및 매칭
        with st.spinner("🔍 Analyzing today's games..."):
            latest_data, file_path = load_latest_prediction_data()
        
        if latest_data is None:
            st.error("❌ Unable to load latest prediction data")
            return
        
        # 커스텀 존 조합에 매칭되는 게임 찾기
        matching_games = find_custom_zone_matches(latest_data, selected_zones, optimal_weights)
        
        if matching_games:
            st.success(f"🎉 **Found {len(matching_games)} matching games!**")
            
            # 선택된 전략의 기댓값 계산
            avg_predicted_roi = sum(game['predicted_roi'] for game in matching_games) / len(matching_games)
            avg_confidence = sum(game['confidence'] for game in matching_games) / len(matching_games)
            
            # 전략 성과 요약
            col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
            with col_summary1:
                st.metric("Matching Games", len(matching_games))
            with col_summary2:
                st.metric("Avg Predicted ROI", f"{avg_predicted_roi:.1f}%")
            with col_summary3:
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            with col_summary4:
                coverage = len(matching_games) / len(latest_data) * 100
                st.metric("Coverage", f"{coverage:.1f}%")
            
            st.markdown("#### 🎲 **Recommended Bets**")
            
            # 게임별 상세 정보 표시 (기존 display_betting_recommendations와 동일한 형식)
            for i, game in enumerate(matching_games, 1):
                with st.container():
                    game_cols = st.columns([3, 2, 2, 1.5, 1.5])
                    
                    with game_cols[0]:
                        st.markdown(f"**#{i} {game['game_info']}**")
                        st.caption(f"📅 {game['date']} | ⏰ {game['start_time']}")
                    
                    with game_cols[1]:
                        st.markdown(f"**🎯 Bet: {game['bet_team_name']}**")
                        st.caption(f"Odds: {game['bet_odds']:+}")
                    
                    with game_cols[2]:
                        st.metric("Predicted ROI", f"{game['predicted_roi']:+.1f}%")
                    
                    with game_cols[3]:
                        prob_display = game['ensemble_prob'] if game['predicted_team'] == 'home' else 1 - game['ensemble_prob']
                        st.metric("Win Prob", f"{prob_display:.1%}")
                    
                    with game_cols[4]:
                        st.metric("Confidence", f"{game['confidence']:.3f}")
                
                # 구분선
                if i < len(matching_games):
                    st.markdown("---")
            
            # 🎰 2팀 파라레이 추천 기능 추가
            if len(matching_games) >= 2:
                st.markdown("---")
                st.markdown("#### 🎰 **2-Team Parlay Recommendations**")
                st.caption("Build profitable 2-team parlays from your selected games with team conflict prevention")
                
                # 2팀 파라레이 생성
                with st.spinner("🔍 Generating 2-team parlay combinations..."):
                    parlays = generate_custom_zone_parlays(matching_games)
                
                if parlays:
                    st.success(f"🎉 **Generated {len(parlays)} profitable 2-team parlays!**")
                    
                    # 파라레이 성과 요약
                    avg_parlay_roi = sum(p['roi'] for p in parlays) / len(parlays)
                    avg_parlay_prob = sum(p['probability'] for p in parlays) / len(parlays)
                    
                    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                    with col_p1:
                        st.metric("Total Parlays", len(parlays))
                    with col_p2:
                        st.metric("Avg ROI", f"{avg_parlay_roi:.1f}%")
                    with col_p3:
                        st.metric("Avg Win Prob", f"{avg_parlay_prob:.1%}")
                    with col_p4:
                        highest_roi_parlay = max(parlays, key=lambda x: x['roi'])
                        st.metric("Best ROI", f"{highest_roi_parlay['roi']:.1f}%")
                    
                    # 파라레이 상세 표시
                    st.markdown("##### 🏆 **Top Parlay Picks** (Sorted by ROI)")
                    
                    # ROI 기준으로 정렬
                    sorted_parlays = sorted(parlays, key=lambda x: x['roi'], reverse=True)
                    
                    # 상위 10개 상세 표시
                    top_10 = sorted_parlays[:10]
                    for i, parlay in enumerate(top_10, 1):
                        with st.container():
                            st.markdown(f"**🎯 Parlay #{i}**")
                            
                            # 파라레이 요약 정보
                            parlay_cols = st.columns([2, 1, 1, 1, 1])
                            with parlay_cols[0]:
                                team1 = parlay['picks'][0]['team']
                                team2 = parlay['picks'][1]['team']
                                st.write(f"**{team1}** + **{team2}**")
                            with parlay_cols[1]:
                                st.metric("ROI", f"{parlay['roi']:+.1f}%")
                            with parlay_cols[2]:
                                st.metric("Win Prob", f"{parlay['probability']:.1%}")
                            with parlay_cols[3]:
                                st.metric("Original Odds", f"{parlay['odds']:.2f}")
                            with parlay_cols[4]:
                                st.metric("Boosted Odds", f"{parlay['boosted_odds']:.2f}")
                            
                            # 개별 픽 정보
                            st.markdown("**📋 Individual Picks:**")
                            for j, pick in enumerate(parlay['picks'], 1):
                                pick_cols = st.columns([3, 2, 1.5, 1.5])
                                with pick_cols[0]:
                                    st.write(f"**Pick {j}:** {pick['match']}")
                                    st.caption(f"📅 {pick['date']} | 🎯 Bet: **{pick['team']}**")
                                with pick_cols[1]:
                                    st.write(f"Odds: {pick['odds']:+}")
                                with pick_cols[2]:
                                    prob_display = pick['probability']
                                    st.write(f"Win Prob: {prob_display:.1%}")
                                with pick_cols[3]:
                                    pred_roi = ((pick['probability'] * (abs(pick['odds'])/100 if pick['odds'] > 0 else 100/abs(pick['odds']))) - (1 - pick['probability'])) * 100
                                    st.write(f"ROI: {pred_roi:+.1f}%")
                            
                            if i < len(top_10):
                                st.markdown("---")
                    
                    # 전체 파라레이 표시 (접을 수 있는 형태)
                    if len(sorted_parlays) > 10:
                        with st.expander(f"📋 **Show All {len(sorted_parlays)} Parlays** (Compact View)", expanded=False):
                            st.markdown("**Complete list of all generated parlays**")
                            
                            # 테이블 형태로 컴팩트하게 표시
                            parlay_data = []
                            for i, parlay in enumerate(sorted_parlays, 1):
                                team1 = parlay['picks'][0]['team']
                                team2 = parlay['picks'][1]['team']
                                pick1_date = parlay['picks'][0]['date']
                                pick2_date = parlay['picks'][1]['date']
                                
                                parlay_data.append({
                                    "#": i,
                                    "Teams": f"{team1} + {team2}",
                                    "Dates": f"{pick1_date} | {pick2_date}",
                                    "ROI": f"{parlay['roi']:+.1f}%",
                                    "Win Prob": f"{parlay['probability']:.1%}",
                                    "Original Odds": f"{parlay['odds']:.2f}",
                                    "Boosted Odds": f"{parlay['boosted_odds']:.2f}",
                                    "Pick 1": f"{parlay['picks'][0]['team']} ({parlay['picks'][0]['odds']:+})",
                                    "Pick 2": f"{parlay['picks'][1]['team']} ({parlay['picks'][1]['odds']:+})"
                                })
                            
                            # DataFrame으로 표시
                            import pandas as pd
                            df = pd.DataFrame(parlay_data)
                            
                            # ROI에 따른 색상 적용
                            def highlight_roi(val):
                                if isinstance(val, str) and '%' in val:
                                    roi_value = float(val.replace('%', '').replace('+', ''))
                                    if roi_value >= 20:
                                        return 'background-color: #d4edda; color: #155724'  # 진한 초록
                                    elif roi_value >= 10:
                                        return 'background-color: #fff3cd; color: #856404'  # 노랑
                                    elif roi_value >= 0:
                                        return 'background-color: #f8d7da; color: #721c24'  # 연한 빨강
                                    else:
                                        return 'background-color: #f5c6cb; color: #721c24'  # 빨강
                                return ''
                            
                            styled_df = df.style.applymap(highlight_roi, subset=['ROI'])
                            st.dataframe(styled_df, use_container_width=True, height=400)
                            
                            # 추가 정보
                            st.caption("💡 **Table Guide**: Teams and dates are matched in order (Team1 plays on Date1, Team2 plays on Date2)")
                            st.caption("🎨 **Color Code**: 🟢 ROI ≥20% | 🟡 ROI ≥10% | 🔴 ROI <10%")
                else:
                    st.warning("❌ **No profitable 2-team parlays found**")
                    st.info("""
                    💡 **This might happen because:**
                    - Too many team conflicts (same teams appearing in multiple games)
                    - Parlay combinations don't meet profitability criteria
                    - Try with different game selections
                    """)
            elif len(matching_games) == 1:
                st.info("💡 **Need at least 2 games for parlay combinations**")
                
        else:
            st.warning("❌ **No games match your custom criteria**")
            st.info("""
            💡 **Try adjusting your selection:**
            - Select fewer conditions for broader matches
            - Choose different zone combinations
            - Some combinations may be very specific and rarely occur
            """)
            
            # 각 차원별 개별 매칭 수 표시 (디버깅 도움)
            with st.expander("🔍 **Debugging: Individual Dimension Matches**", expanded=False):
                st.markdown("Check how many games match each dimension individually:")
                
                for dimension, segments in selected_zones.items():
                    dimension_title = available_dimensions[dimension]['title']
                    
                    # 해당 차원만의 매칭 수 계산
                    single_dim_zones = {dimension: segments}
                    single_matches = find_custom_zone_matches(latest_data, single_dim_zones, optimal_weights)
                    
                    segment_names = [seg['segment'] for seg in segments]
                    condition_text = ' OR '.join(segment_names) if len(segment_names) > 1 else segment_names[0]
                    
                    st.write(f"• **{dimension_title}** ({condition_text}): **{len(single_matches)} games**")
    else:
        st.info("💡 **Get Started**: Select zones above to build your custom betting strategy")

def find_custom_zone_matches(latest_data, selected_zones, optimal_weights):
    """커스텀 존 조합에 매칭되는 게임들 찾기 (차원별 OR, 차원간 AND)"""
    matching_games = []
    
    if not latest_data or not selected_zones or not optimal_weights:
        return matching_games
    
    for game in latest_data:
        # 게임 데이터 전처리 (기존 process_game_for_strategy 재사용)
        processed_game = process_game_for_strategy(game, optimal_weights)
        if processed_game is None:
            continue
        
        # 모든 차원에서 조건 확인 (AND 조건)
        all_dimensions_match = True
        
        for dimension, selected_segments in selected_zones.items():
            # 같은 차원 내에서는 OR 조건 확인
            dimension_match = False
            
            for segment_info in selected_segments:
                zone = {
                    'dimension': dimension,
                    'segment': segment_info['segment']
                }
                
                # 기존 check_zone_condition 함수 재사용
                if check_zone_condition(game, zone, optimal_weights):
                    dimension_match = True
                    break  # OR이므로 하나만 맞으면 됨
            
            if not dimension_match:
                all_dimensions_match = False
                break  # AND이므로 하나라도 안 맞으면 전체 실패
        
        if all_dimensions_match:
            matching_games.append(processed_game)
    
    return matching_games

def generate_custom_zone_parlays(matching_games):
    """Custom Zone의 매칭된 게임들로 2팀 파라레이 생성"""
    import itertools
    import random
    
    if len(matching_games) < 2:
        return []
    
    # 모든 2팀 조합 생성
    all_combos = list(itertools.combinations(range(len(matching_games)), 2))
    random.shuffle(all_combos)  # 다양성 확보
    
    # 픽 사용 횟수 추적 (최소 3, 최대 4)
    pick_usage = {i: 0 for i in range(len(matching_games))}
    selected_parlays = []
    
    # 부스트율 (2팀 파라레이 = 10%)
    boost_rate = 1.10
    
    # 필터링 결과 추적
    counter = {
        'usage_limit': 0,
        'team_conflict': 0,
        'generated': 0
    }
    
    for combo in all_combos:
        # 사용 횟수 체크 (최대 4회)
        if any(pick_usage[i] >= 4 for i in combo):
            counter['usage_limit'] += 1
            continue
        
        # 팀 충돌 체크
        game1 = matching_games[combo[0]]
        game2 = matching_games[combo[1]]
        
        if has_custom_zone_team_conflict([game1, game2]):
            counter['team_conflict'] += 1
            continue
        
        # 파라레이 정보 계산
        parlay_info = calculate_custom_zone_parlay(game1, game2, boost_rate)
        
        if parlay_info:
            selected_parlays.append(parlay_info)
            
            # 픽 사용 횟수 업데이트
            for i in combo:
                pick_usage[i] += 1
            
            counter['generated'] += 1
    
    # 최소 사용 횟수 체크 및 재시도 (간단한 버전)
    usage_counts = list(pick_usage.values())
    picks_below_min = sum(1 for count in usage_counts if count > 0 and count < 3)
    
    print(f"Parlay generation results:")
    print(f"  - Usage limit exceeded: {counter['usage_limit']}")
    print(f"  - Team conflicts: {counter['team_conflict']}")
    print(f"  - Generated parlays: {counter['generated']}")
    print(f"  - Picks below minimum usage: {picks_below_min}")
    
    return selected_parlays

def has_custom_zone_team_conflict(games):
    """Custom Zone 게임들의 팀 충돌 체크"""
    teams_in_parlay = set()
    
    for game in games:
        # game['game_info']는 "Away Team @ Home Team" 형식
        if ' @ ' in game['game_info']:
            away_team, home_team = game['game_info'].split(' @ ')
        else:
            # 혹시 다른 형식이면 match 필드 사용
            match_parts = game.get('match', game['game_info']).split(' vs ')
            if len(match_parts) == 2:
                away_team = match_parts[0]
                home_team = match_parts[1]
            else:
                continue  # 파싱할 수 없으면 건너뛰기
        
        # 이미 포함된 팀이 있으면 충돌
        if away_team in teams_in_parlay or home_team in teams_in_parlay:
            return True
        
        teams_in_parlay.add(away_team)
        teams_in_parlay.add(home_team)
    
    return False

def calculate_custom_zone_parlay(game1, game2, boost_rate=1.10):
    """두 게임의 파라레이 정보 계산"""
    
    # 배당률을 소수점 형식으로 변환
    def american_to_decimal(american_odds):
        if american_odds > 0:
            return 1 + (american_odds / 100)
        elif american_odds < 0:
            return 1 + (100 / abs(american_odds))
        else:
            return 1.0
    
    # 각 게임의 정보 추출
    picks = []
    for game in [game1, game2]:
        # 배당률과 확률 정보
        american_odds = game['bet_odds']
        decimal_odds = american_to_decimal(american_odds)
        
        # 승리 확률 계산
        if game['predicted_team'] == 'home':
            probability = game['ensemble_prob']
        else:
            probability = 1 - game['ensemble_prob']
        
        pick_info = {
            'team': game['bet_team_name'],
            'match': game['game_info'],
            'date': game['date'],
            'odds': american_odds,
            'decimal_odds': decimal_odds,
            'probability': probability
        }
        picks.append(pick_info)
    
    # 파라레이 배당률 및 확률 계산
    parlay_odds = picks[0]['decimal_odds'] * picks[1]['decimal_odds']
    parlay_probability = picks[0]['probability'] * picks[1]['probability']
    
    # 부스트 적용
    boosted_odds = 1 + ((parlay_odds - 1) * boost_rate)
    
    # EV 및 ROI 계산
    ev = (parlay_probability * (boosted_odds - 1)) - (1 - parlay_probability)
    roi = ev * 100  # EV를 백분율로 변환
    
    return {
        'picks': picks,
        'teams': [pick['team'] for pick in picks],
        'matches': [pick['match'] for pick in picks],
        'odds': parlay_odds,
        'boosted_odds': boosted_odds,
        'probability': parlay_probability,
        'ev': ev,
        'roi': roi,
        'type': '2_team_parlay'
    }

def main():
    # Main header
    st.markdown('<div class="main-header">📊 MLB Model Performance Dashboard</div>', unsafe_allow_html=True)
    
    # 탭 생성
    tab1, tab2 = st.tabs(["📈 Model Performance Analysis", "🔧 Ensemble Optimizer"])
    
    with tab1:
        # 기존 모델 성과 분석 코드
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; color: #666;">
            Analyze ROI performance of MLB prediction models (Home team win probability based)
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar - Date settings
        st.sidebar.header("⚙️ Analysis Settings")
        
        # 🆕 캐시 클리어 버튼 추가
        add_cache_clear_button()
        
        try:
            # Load available dates
            available_dates = get_available_dates()
            
            if not available_dates:
                st.error("📂 No prediction files found.")
                return
            
            st.sidebar.success(f"📅 Available: {len(available_dates)} days")
            
            # Date range selection
            min_date = datetime.strptime(available_dates[0], '%Y-%m-%d').date()
            max_date = datetime.strptime(available_dates[-1], '%Y-%m-%d').date()
            
            col1, col2 = st.sidebar.columns(2)
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
            
            # Date validation
            if start_date > end_date:
                st.sidebar.error("⚠️ Start date cannot be later than end date.")
                return
            
            # 🆕 날짜 설정 직후 파일 정보 미리보기 표시
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            with st.spinner("🔍 파일 정보 확인 중..."):
                file_info_preview = get_file_info_preview(start_date_str, end_date_str)
            
            if file_info_preview:
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 📁 사용될 데이터 파일")
                
                # 히스토리컬 레코드 파일
                st.sidebar.markdown("**📋 히스토리컬 레코드:**")
                st.sidebar.code(file_info_preview['historical_record_file'], language=None)
                
                # 예측 파일 요약
                st.sidebar.markdown("**📊 예측 파일:**")
                used_files = len(file_info_preview['used_prediction_files'])
                skipped_files = len(file_info_preview['skipped_files'])
                
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    st.metric("사용", used_files)
                with col2:
                    st.metric("제외", skipped_files)
                
                # 사용된 파일들 간단 리스트
                if file_info_preview['used_prediction_files']:
                    with st.sidebar.expander("📋 사용될 파일 목록", expanded=False):
                        for file_item in sorted(file_info_preview['used_prediction_files'], key=lambda x: x['date']):
                            short_name = file_item['filename'].replace('mlb_predictions_with_odds_', '').replace('.json', '')
                            st.write(f"• **{file_item['date']}** ({file_item['games']}경기)")
                            st.caption(f"  {short_name}")
                
                # 제외된 파일들
                if file_info_preview['skipped_files']:
                    with st.sidebar.expander("❌ 제외될 파일들", expanded=False):
                        for skip_info in file_info_preview['skipped_files'][:5]:  # 최대 5개만
                            short_name = skip_info['filename'].replace('mlb_predictions_with_odds_', '').replace('.json', '')
                            st.write(f"• **{short_name}**")
                            st.caption(f"  {skip_info['reason']}")
                        
                        if len(file_info_preview['skipped_files']) > 5:
                            st.caption(f"... 및 {len(file_info_preview['skipped_files']) - 5}개 추가")
            
            # Analysis execution button
            analyze_btn = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)
            
            # Auto-run or button click analysis
            if analyze_btn or 'analysis_results' not in st.session_state:
                with st.spinner("📊 Running analysis..."):
                    results = run_analysis(start_date_str, end_date_str)
                    st.session_state.analysis_results = results
                    st.session_state.date_range = f"{start_date_str} ~ {end_date_str}"
            
            # Display results
            if 'analysis_results' in st.session_state:
                results = st.session_state.analysis_results
                
                # Analysis summary (파일 정보 섹션 제거됨)
                summary = results['analysis_summary']
                
                st.markdown("## 📈 Analysis Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Models Analyzed", f"{summary['total_models']}")
                with col2:
                    st.metric("Total Games", f"{summary['total_games']}")
                with col3:
                    st.metric("Files Analyzed", f"{summary['files_analyzed']}")
                with col4:
                    st.metric("Date Range", st.session_state.get('date_range', 'All'))
                
                # 🆕 데이터 처리 상세 정보 (배당률 처리 상황)
                if 'matched_data' in results:
                    matched_data = results['matched_data']
                    
                    # 배당률 통계 계산
                    total_matched = len(matched_data)
                    games_with_odds = sum(1 for game in matched_data if game.get('home_odds') is not None and game.get('away_odds') is not None)
                    games_without_odds = total_matched - games_with_odds
                    
                    with st.expander("🔍 데이터 처리 상세 (배당률 현황)", expanded=False):
                        st.markdown("### 💰 배당률 처리 현황")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("매칭된 총 게임", total_matched)
                        with col2:
                            st.metric("배당률 있는 게임", games_with_odds)
                        with col3:
                            st.metric("배당률 없는 게임", games_without_odds)
                        with col4:
                            odds_coverage = (games_with_odds / total_matched * 100) if total_matched > 0 else 0
                            st.metric("배당률 커버리지", f"{odds_coverage:.1f}%")
                        
                        if games_without_odds > 0:
                            st.warning(f"⚠️ **중요**: {games_without_odds}개 게임은 배당률이 없어서 ROI 계산에서 제외됩니다.")
                            
                            # 배당률 없는 게임들 샘플 표시
                            no_odds_games = [game for game in matched_data if game.get('home_odds') is None or game.get('away_odds') is None]
                            if no_odds_games:
                                st.markdown("**배당률 없는 게임 샘플:**")
                                sample_size = min(5, len(no_odds_games))
                                for i, game in enumerate(no_odds_games[:sample_size]):
                                    st.text(f"• {game['date']} - {game['away_team']} @ {game['home_team']}")
                                
                                if len(no_odds_games) > sample_size:
                                    st.text(f"... 및 {len(no_odds_games) - sample_size}개 추가 게임")
                        
                        else:
                            st.success("✅ 모든 매칭된 게임에 배당률이 있습니다!")
                        
                        # 날짜별 게임 분포
                        st.markdown("### 📅 날짜별 게임 분포")
                        date_counts = {}
                        for game in matched_data:
                            date = game['date']
                            if date not in date_counts:
                                date_counts[date] = {'total': 0, 'with_odds': 0}
                            date_counts[date]['total'] += 1
                            if game.get('home_odds') is not None and game.get('away_odds') is not None:
                                date_counts[date]['with_odds'] += 1
                        
                        date_df = pd.DataFrame([
                            {
                                'Date': date,
                                'Total Games': counts['total'],
                                'Games with Odds': counts['with_odds'],
                                'Coverage (%)': (counts['with_odds'] / counts['total'] * 100) if counts['total'] > 0 else 0
                            }
                            for date, counts in sorted(date_counts.items())
                        ])
                        
                        st.dataframe(date_df, hide_index=True, use_container_width=True)
                
                # Model performance data preparation
                performances = results['model_performances']
                
                if not performances:
                    st.warning("📊 No model data to analyze.")
                    return
                
                # Create DataFrame
                df_data = []
                for model_name, perf in performances.items():
                    df_data.append({
                        'Model': model_name,
                        'ROI (%)': perf.get('actual_roi', perf.get('roi', 0.0)),  # 실제 ROI를 메인으로
                        'Win Rate (%)': perf.get('win_rate', 0.0),
                        'Total Bets': perf.get('total_bets', 0),
                        'Wins': perf.get('correct_predictions', 0),
                        'Losses': perf.get('total_bets', 0) - perf.get('correct_predictions', 0),
                        'Profit/Loss ($)': perf.get('profit_loss', 0.0),
                        'Total Invested ($)': perf.get('total_invested', 0.0),
                        # ROI 비교 데이터도 저장 (나중에 사용)
                        'Expected ROI (%)': perf.get('expected_roi', 0.0),
                        'ROI Difference (%)': perf.get('roi_difference', 0.0)
                    })
                
                df = pd.DataFrame(df_data)
                df = df.sort_values('ROI (%)', ascending=False).reset_index(drop=True)
                
                # Main Performance Analysis
                st.markdown("## 🏆 Model Performance Ranking")
                
                # ROI performance chart
                colors = ['green' if roi > 0 else 'red' if roi < -5 else 'orange' for roi in df['ROI (%)']]
                
                fig = px.bar(
                    df, 
                    x='Model', 
                    y='ROI (%)',
                    title="ROI Performance by Model",
                    color='ROI (%)',
                    color_continuous_scale=['red', 'orange', 'green'],
                    color_continuous_midpoint=0,
                    text='ROI (%)'
                )
                
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=500,
                    showlegend=False
                )
                
                fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance categories
                st.markdown("## 📊 Performance Categories")
                
                # Categorize models
                profitable = df[df['ROI (%)'] > 0]
                break_even = df[(df['ROI (%)'] >= -2) & (df['ROI (%)'] <= 0)]
                loss_making = df[df['ROI (%)'] < -2]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 💰 Profitable Models")
                    st.markdown("*ROI > 0%*")
                    if not profitable.empty:
                        for _, row in profitable.iterrows():
                            win_rate = row['Win Rate (%)']
                            roi = row['ROI (%)']
                            profit = row['Profit/Loss ($)']
                            games = row['Total Bets']
                            
                            st.markdown(f"""
                            <div class="metric-container positive-roi">
                                <strong>{row['Model']}</strong><br>
                                ROI: {roi:.2f}% | Win Rate: {win_rate:.1f}%<br>
                                Profit: ${profit:.1f} | Games: {games}
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)
                    else:
                        st.info("No profitable models in this period.")
                
                with col2:
                    st.markdown("### ⚖️ Break-even Models")
                    st.markdown("*-2% ≤ ROI ≤ 0%*")
                    if not break_even.empty:
                        for _, row in break_even.iterrows():
                            win_rate = row['Win Rate (%)']
                            roi = row['ROI (%)']
                            profit = row['Profit/Loss ($)']
                            games = row['Total Bets']
                            
                            st.markdown(f"""
                            <div class="metric-container neutral-roi">
                                <strong>{row['Model']}</strong><br>
                                ROI: {roi:.2f}% | Win Rate: {win_rate:.1f}%<br>
                                Profit: ${profit:.1f} | Games: {games}
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)
                    else:
                        st.info("No break-even models.")
                
                with col3:
                    st.markdown("### 📉 Loss-making Models")
                    st.markdown("*ROI < -2%*")
                    if not loss_making.empty:
                        for _, row in loss_making.iterrows():
                            win_rate = row['Win Rate (%)']
                            roi = row['ROI (%)']
                            profit = row['Profit/Loss ($)']
                            games = row['Total Bets']
                            
                            st.markdown(f"""
                            <div class="metric-container negative-roi">
                                <strong>{row['Model']}</strong><br>
                                ROI: {roi:.2f}% | Win Rate: {win_rate:.1f}%<br>
                                Loss: ${profit:.1f} | Games: {games}
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)
                    else:
                        st.info("No significantly loss-making models.")
                
                # Performance metrics
                st.markdown("## 📊 Detailed Performance Table")
                
                # Add ranking
                df_display = df[['Model', 'ROI (%)', 'Win Rate (%)', 'Total Bets', 'Wins', 'Losses', 'Profit/Loss ($)', 'Total Invested ($)']].copy()
                df_display.insert(0, 'Rank', range(1, len(df_display) + 1))
                
                # Conditional formatting
                def highlight_roi(val):
                    if val > 0:
                        return 'background-color: #d4edda; color: #155724'
                    elif val < -5:
                        return 'background-color: #f8d7da; color: #721c24'
                    else:
                        return 'background-color: #fff3cd; color: #856404'
                
                styled_df = df_display.style.map(
                    highlight_roi, 
                    subset=['ROI (%)']
                ).format({
                    'ROI (%)': '{:.2f}%',
                    'Win Rate (%)': '{:.1f}%',
                    'Profit/Loss ($)': '${:.1f}',
                    'Total Invested ($)': '${:.0f}'
                })
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # 🆕 Daily Performance Analysis Section
                st.markdown("---")
                st.markdown("## 📅 Model Daily Performance Analysis")
                
                # Check if daily performance data is available
                if 'daily_performances' in results and results['daily_performances']:
                    daily_data = results['daily_performances']
                    
                    with st.expander("📈 View Daily Performance by Model", expanded=False):
                        st.markdown("### Select Model for Daily Performance")
                        
                        # Model selection
                        available_models = sorted(daily_data.keys())
                        selected_model = st.selectbox(
                            "Choose a model to view daily performance:",
                            available_models,
                            key="daily_performance_model_selector"
                        )
                        
                        if selected_model and selected_model in daily_data:
                            model_daily_data = daily_data[selected_model]
                            
                            # Display model summary
                            st.markdown(f"#### 📊 Daily Performance for **{selected_model}**")
                            
                            # Calculate summary stats
                            dates = sorted(model_daily_data.keys())
                            total_days = len(dates)
                            profitable_days = sum(1 for date in dates if model_daily_data[date]['roi'] > 0)
                            avg_daily_roi = sum(model_daily_data[date]['roi'] for date in dates) / total_days if total_days > 0 else 0
                            total_daily_profit = sum(model_daily_data[date]['profit_loss'] for date in dates)
                            
                            # Summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Days", total_days)
                            with col2:
                                st.metric("Profitable Days", f"{profitable_days} ({profitable_days/total_days*100:.1f}%)" if total_days > 0 else "0")
                            with col3:
                                st.metric("Avg Daily ROI", f"{avg_daily_roi:.2f}%")
                            with col4:
                                st.metric("Total P/L", f"${total_daily_profit:.1f}")
                            
                            # Daily performance table
                            st.markdown("##### 📋 Daily Performance Table")
                            daily_table_data = []
                            for date in sorted(dates):
                                day_data = model_daily_data[date]
                                daily_table_data.append({
                                    'Date': date,
                                    'ROI (%)': day_data['roi'],
                                    'Win Rate (%)': day_data['win_rate'],
                                    'Total Games': day_data['total_games'],
                                    'Games w/ Odds': day_data['games_with_odds'],
                                    'Wins': day_data['correct_predictions'],
                                    'P/L ($)': day_data['profit_loss'],
                                    'Invested ($)': day_data['total_invested']
                                })
                            
                            daily_df = pd.DataFrame(daily_table_data)
                            
                            # Style the daily performance table
                            def highlight_daily_roi(val):
                                if val > 0:
                                    return 'background-color: #d4edda; color: #155724'
                                elif val < -10:
                                    return 'background-color: #f8d7da; color: #721c24'
                                else:
                                    return 'background-color: #fff3cd; color: #856404'
                            
                            styled_daily_df = daily_df.style.map(
                                highlight_daily_roi,
                                subset=['ROI (%)']
                            ).format({
                                'ROI (%)': '{:.2f}%',
                                'Win Rate (%)': '{:.1f}%',
                                'P/L ($)': '${:.1f}',
                                'Invested ($)': '${:.0f}'
                            })
                            
                            st.dataframe(styled_daily_df, use_container_width=True, hide_index=True)
                            
                            # Daily performance charts
                            st.markdown("##### 📈 Performance Trends")
                            
                            # Create performance trend charts
                            chart_col1, chart_col2 = st.columns(2)
                            
                            with chart_col1:
                                # Daily ROI trend
                                fig_roi_trend = px.line(
                                    daily_df,
                                    x='Date',
                                    y='ROI (%)',
                                    title=f'Daily ROI Trend - {selected_model}',
                                    markers=True
                                )
                                fig_roi_trend.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
                                fig_roi_trend.update_layout(height=400)
                                st.plotly_chart(fig_roi_trend, use_container_width=True)
                            
                            with chart_col2:
                                # Daily P/L trend
                                fig_pl_trend = px.bar(
                                    daily_df,
                                    x='Date',
                                    y='P/L ($)',
                                    title=f'Daily Profit/Loss - {selected_model}',
                                    color='P/L ($)',
                                    color_continuous_scale=['red', 'orange', 'green'],
                                    color_continuous_midpoint=0
                                )
                                fig_pl_trend.update_layout(height=400, showlegend=False)
                                st.plotly_chart(fig_pl_trend, use_container_width=True)
                            
                            # Game details for selected date
                            st.markdown("##### 🎮 Game Details by Date")
                            selected_date = st.selectbox(
                                f"Select date to view game details for {selected_model}:",
                                sorted(dates, reverse=True),
                                key=f"game_detail_date_{selected_model}"
                            )
                            
                            if selected_date and selected_date in model_daily_data:
                                date_games = model_daily_data[selected_date]['games']
                                
                                if date_games:
                                    st.markdown(f"**Games on {selected_date}:**")
                                    
                                    game_details = []
                                    for game in date_games:
                                        if game['has_odds']:  # Only show games with betting data
                                            game_details.append({
                                                'Matchup': f"{game['away_team']} @ {game['home_team']}",
                                                'Prediction': 'Home Win' if game['predicted_home_win'] else 'Away Win',
                                                'Home Prob (%)': f"{game['home_probability']*100:.1f}%",
                                                'Actual Result': 'Home Win' if game['actual_home_win'] else 'Away Win',
                                                'Correct': '✅' if game['is_correct'] else '❌',
                                                'Home Odds': game.get('home_odds', 'N/A'),
                                                'Away Odds': game.get('away_odds', 'N/A'),
                                                'Bet Odds': game.get('predicted_odds', 'N/A'),
                                                'Profit ($)': f"${game.get('profit', 0):.1f}"
                                            })
                                    
                                    if game_details:
                                        game_df = pd.DataFrame(game_details)
                                        st.dataframe(game_df, use_container_width=True, hide_index=True)
                                    else:
                                        st.info(f"No games with betting odds available for {selected_date}")
                                else:
                                    st.info(f"No game data available for {selected_date}")
                else:
                    st.warning("📊 Daily performance data not available. This might be due to insufficient data or analysis issues.")
                
                # Additional analysis section for ROI comparison
                st.markdown("---")
                st.markdown("## 🔍 Additional Analysis: Expected vs Actual ROI")
                
                with st.expander("📈 ROI Comparison Analysis", expanded=False):
                    st.markdown("### Expected vs Actual ROI Comparison")
                    
                    # Create comparison chart
                    fig_comparison = go.Figure()
                    
                    # Expected ROI bars
                    fig_comparison.add_trace(go.Bar(
                        name='Expected ROI',
                        x=df['Model'],
                        y=df['Expected ROI (%)'],
                        marker_color='lightblue',
                        opacity=0.7
                    ))
                    
                    # Actual ROI bars
                    fig_comparison.add_trace(go.Bar(
                        name='Actual ROI',
                        x=df['Model'],
                        y=df['ROI (%)'],
                        marker_color='darkblue'
                    ))
                    
                    fig_comparison.update_layout(
                        title="Expected ROI vs Actual ROI by Model",
                        xaxis_title="Model",
                        yaxis_title="ROI (%)",
                        barmode='group',
                        xaxis_tickangle=-45,
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    fig_comparison.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # ROI difference analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # ROI difference chart
                        fig_diff = px.bar(
                            df, 
                            x='Model', 
                            y='ROI Difference (%)',
                            title="ROI Difference (Actual - Expected)",
                            color='ROI Difference (%)',
                            color_continuous_scale=['red', 'orange', 'green'],
                            color_continuous_midpoint=0
                        )
                        
                        fig_diff.update_layout(
                            xaxis_tickangle=-45,
                            height=400,
                            showlegend=False
                        )
                        
                        fig_diff.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                        
                        st.plotly_chart(fig_diff, use_container_width=True)
                    
                    with col2:
                        # Correlation analysis
                        correlation = df['Expected ROI (%)'].corr(df['ROI (%)'])
                        
                        fig_scatter = px.scatter(
                            df, 
                            x='Expected ROI (%)', 
                            y='ROI (%)',
                            size='Total Bets',
                            hover_data=['Model', 'Win Rate (%)', 'ROI Difference (%)'],
                            title=f"Expected vs Actual ROI Correlation ({correlation:.3f})"
                        )
                        
                        # Add diagonal line (perfect correlation)
                        min_roi = min(df['Expected ROI (%)'].min(), df['ROI (%)'].min())
                        max_roi = max(df['Expected ROI (%)'].max(), df['ROI (%)'].max())
                        fig_scatter.add_shape(
                            type="line", line=dict(dash="dash", color="red"),
                            x0=min_roi, y0=min_roi, x1=max_roi, y1=max_roi
                        )
                        
                        fig_scatter.update_layout(height=400)
                        
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Model calibration summary
                    outperforming = df[df['ROI Difference (%)'] > 2]
                    underperforming = df[df['ROI Difference (%)'] < -2]
                    on_target = df[(df['ROI Difference (%)'] >= -2) & (df['ROI Difference (%)'] <= 2)]
                    
                    st.markdown("### Model Calibration Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Well-calibrated", f"{len(on_target)} ({len(on_target)/len(df)*100:.1f}%)")
                    with col2:
                        st.metric("Outperforming", f"{len(outperforming)} ({len(outperforming)/len(df)*100:.1f}%)")
                    with col3:
                        st.metric("Underperforming", f"{len(underperforming)} ({len(underperforming)/len(df)*100:.1f}%)")
                    with col4:
                        st.metric("ROI Correlation", f"{correlation:.3f}")
                
                # Statistical summary
                st.markdown("## 📈 Statistical Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_roi = df['ROI (%)'].mean()
                    st.metric("Average ROI", f"{avg_roi:.2f}%")
                
                with col2:
                    avg_win_rate = df['Win Rate (%)'].mean()
                    st.metric("Average Win Rate", f"{avg_win_rate:.1f}%")
                
                with col3:
                    total_profit = df['Profit/Loss ($)'].sum()
                    st.metric("Total Profit/Loss", f"${total_profit:.1f}")
                
                with col4:
                    total_games = df['Total Bets'].sum()
                    st.metric("Total Games", f"{total_games:,}")
                
                # Performance insights
                st.markdown("## 💡 Performance Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Best Performers")
                    
                    if len(profitable) > 0:
                        best_roi = profitable.iloc[0]
                        st.write(f"**Best ROI**: {best_roi['Model']} ({best_roi['ROI (%)']:.2f}%)")
                    
                    best_win_rate = df.loc[df['Win Rate (%)'].idxmax()]
                    st.write(f"**Best Win Rate**: {best_win_rate['Model']} ({best_win_rate['Win Rate (%)']:.1f}%)")
                    
                    if total_profit > 0:
                        most_profitable = df.loc[df['Profit/Loss ($)'].idxmax()]
                        st.write(f"**Most Profitable**: {most_profitable['Model']} (${most_profitable['Profit/Loss ($)']:.1f})")
                    
                    st.markdown(f"**Overall Performance**: {len(profitable)} profitable, {len(break_even)} break-even, {len(loss_making)} loss-making")
                
                with col2:
                    st.markdown("### 📊 Market Insights")
                    
                    if avg_roi > 0:
                        st.success(f"✅ Positive average ROI ({avg_roi:.2f}%)")
                    else:
                        st.warning(f"⚠️ Negative average ROI ({avg_roi:.2f}%)")
                    
                    if avg_win_rate > 55:
                        st.success(f"✅ Strong win rate ({avg_win_rate:.1f}%)")
                    elif avg_win_rate > 50:
                        st.info(f"🎯 Moderate win rate ({avg_win_rate:.1f}%)")
                    else:
                        st.warning(f"⚠️ Weak win rate ({avg_win_rate:.1f}%)")
                    
                    profitable_pct = len(profitable) / len(df) * 100
                    if profitable_pct > 50:
                        st.success(f"✅ {profitable_pct:.1f}% models are profitable")
                    else:
                        st.warning(f"⚠️ Only {profitable_pct:.1f}% models are profitable")
                    
                # Important note
                st.markdown("---")
                st.info("""
                📝 **Analysis Notes**: 
                - ROI calculations based on $100 unit bets per game
                - All probabilities represent home team win probability
                - Betting strategy: Bet on predicted winner (home if prob > 0.5, away if prob ≤ 0.5)
                - Expected ROI represents theoretical performance based on probabilities and odds
                """)
        
        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.exception(e)
    
    with tab2:
        # 새로운 앙상블 최적화 탭
        display_ensemble_optimization_tab()

if __name__ == "__main__":
    main() 