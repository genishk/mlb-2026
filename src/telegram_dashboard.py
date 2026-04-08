import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path
import numpy as np
import glob
import re
import json
from typing import Dict, List, Any

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.simple_model_analyzer import SimpleModelAnalyzer
from src.ensemble_optimizer import StableEnsembleOptimizer

# 페이지 설정
st.set_page_config(
    page_title="📱 MLB Telegram Analytics Hub",
    page_icon="📱", 
    layout="wide"
)

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
    
    .performance-container {
        background: linear-gradient(90deg, #f0f8ff, #ffffff);
        border-left: 4px solid #007bff;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .telegram-message {
        background: linear-gradient(90deg, #e8f4fd, #ffffff);
        border-left: 4px solid #0088cc;
        padding: 1.5rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        margin: 1rem 0;
        font-size: 14px;
        line-height: 1.4;
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
        color: #ffffff;
        font-size: 1.1rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .analysis-result {
        color: #00ff88;
        font-weight: bold;
        font-size: 1.1rem;
        text-shadow: 0 0 15px rgba(0, 255, 136, 0.5);
        background: rgba(0, 255, 136, 0.1);
        padding: 4px 8px;
        border-radius: 6px;
        border-left: 3px solid #00ff88;
        margin: 4px 0;
    }
    
    .model-weight {
        background: #fff3cd;
        border-radius: 6px;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-left: 3px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_analyzer(data_prefix="None"):
    """Load analyzer (cached)"""
    return SimpleModelAnalyzer(data_prefix=data_prefix)

@st.cache_data
def run_analysis(start_date, end_date, data_prefix="None"):
    """Run analysis (cached) - 정확히 Simple Dashboard와 동일"""
    analyzer = load_analyzer(data_prefix)
    return analyzer.analyze(start_date, end_date)

class TelegramAnalyticsHub:
    """텔레그램 분석 허브"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.predictions_dir = self.project_root / "src" / "odds" / "data" / "matched"
        self.records_dir = self.project_root / "data" / "records"
        self.picks_dir = self.project_root / "data" / "picks"  # 픽 저장 디렉토리 추가
        
        # 모든 모델 목록 (Simple Dashboard와 동일)
        self.all_models = [
            'model1', 'model2', 'model3', 'model4', 'model5', 'model6', 
            'model7', 'model8', 'model9', 'model_rf', 'model_nn', 'model_svm',
            'model_advanced_catboost_basic', 'model_advanced_catboost',
            'model_advanced_lgbm_basic', 'model_advanced_lgbm',
            'model_advanced_nn', 'model_advanced_rf', 'model_advanced_svm',
            'model_advanced_xgboost_basic', 'model_advanced_xgboost',
            'model1_extended_lgbm', 'model2_extended_catboost', 'model3_extended_xgboost'
        ]
        
        # Zone 옵션 - predicted ROI 세분화
        self.zone_options = {
            'predicted_roi': [
                'Very Negative (<-20%)', 'Negative (-20% ~ 0%)', 
                'Positive (0% ~ 20%)', 'Very Positive A (20% ~ 40%)', 'Very Positive B (40% ~ 60%)', 
                'Extremely Positive A (60% ~ 100%)', 'Extremely Positive B (>100%)'
            ],
            'odds': [
                'Heavy Favorite (< -200)', 'Favorite (-200 ~ -120)', 
                'Pick Em (-120 ~ +120)', 'Underdog (+120 ~ +150)', 'Strong Underdog (+150 ~ +300)',
                'Heavy Underdog (> +300)'
            ],
            'confidence': [
                'Low Confidence (0-0.05)', 'Medium Confidence (0.05-0.15)', 
                'High Confidence (0.15-0.25)', 'Very High Confidence (>0.25)'
            ],
            'odds_probability_divergence': [
                'Model Much More Pessimistic (<-10%)', 'Model Slightly Pessimistic (-10% ~ -5%)',
                'Market Aligned (-5% ~ +5%)', 'Model Slightly Optimistic (+5% ~ +10%)', 
                'Model Much More Optimistic (+10%+)'
            ],
            'kelly_criterion': [
                'No Selection (Kelly ≤ 0%)', 'Low Confidence (0% < Kelly ≤ 5%)', 
                'Medium Confidence (5% < Kelly ≤ 15%)', 'High Confidence (15% < Kelly ≤ 25%)', 
                'Very High Confidence (25% ~ 60%)', 'Extremely High Confidence (Kelly > 60%)'
            ],
            'model_consensus': [
                'Strong Consensus (80%+)', 'Moderate Consensus (60-80%)',
                'Weak Consensus (50-60%)', 'No Consensus (<50%)'
            ]
        }
    
    def get_available_prefixes(self):
        """사용 가능한 구분자들을 탐지"""
        try:
            all_files = list(self.predictions_dir.glob("*mlb_predictions_with_odds_*.json"))
            prefixes = set()
            
            for file_path in all_files:
                filename = file_path.name
                if filename.startswith("mlb_predictions_with_odds_"):
                    # 구분자 없는 파일
                    prefixes.add("None")
                else:
                    # 구분자 있는 파일 - 첫 번째 _까지가 구분자
                    parts = filename.split("_")
                    if len(parts) >= 7 and parts[-1].endswith(".json"):
                        # prefix_mlb_predictions_with_odds_date_time.json 형태
                        # 예: 55_105_mlb_predictions_with_odds_20250722_091712.json
                        prefix_parts = parts[:2]  # [55, 105]
                        if all(part.isdigit() for part in prefix_parts):
                            prefix = "_".join(prefix_parts) + "_"  # 55_105_
                            prefixes.add(prefix)
            
            # 정렬해서 반환 (None을 맨 앞에)
            sorted_prefixes = sorted([p for p in prefixes if p != "None"])
            return ["None"] + sorted_prefixes
            
        except Exception as e:
            st.error(f"구분자 탐지 중 오류: {e}")
            return ["None"]
    
    def get_available_dates_for_prefix(self, prefix):
        """특정 구분자에 대한 사용 가능한 날짜들을 반환"""
        try:
            if prefix == "None":
                pattern = "mlb_predictions_with_odds_*.json"
            else:
                pattern = f"{prefix}mlb_predictions_with_odds_*.json"
            
            files = list(self.predictions_dir.glob(pattern))
            dates = []
            
            for file_path in files:
                filename = file_path.name
                # 날짜 추출 로직
                if prefix == "None":
                    # mlb_predictions_with_odds_20250722_091712.json
                    parts = filename.replace(".json", "").split("_")
                    if len(parts) >= 5:
                        date_part = parts[4]  # YYYYMMDD
                        if len(date_part) == 8 and date_part.isdigit():
                            formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                            dates.append(formatted_date)
                else:
                    # 55_105_mlb_predictions_with_odds_20250722_091712.json
                    parts = filename.replace(".json", "").split("_")
                    if len(parts) >= 7:
                        date_part = parts[6]  # YYYYMMDD
                        if len(date_part) == 8 and date_part.isdigit():
                            formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                            dates.append(formatted_date)
            
            return sorted(list(set(dates)))
            
        except Exception as e:
            st.error(f"날짜 탐지 중 오류: {e}")
            return []
    
    def find_latest_prediction_file(self):
        """최신 예측 파일 찾기 (Simple Dashboard와 동일 로직)"""
        pattern = str(self.predictions_dir / "mlb_predictions_with_odds_*.json")
        
        try:
            files = glob.glob(pattern)
            if not files:
                return None
                
            def extract_datetime(filename):
                # mlb_predictions_with_odds_20250607_150950.json 패턴에서 날짜시간 추출
                match = re.search(r'mlb_predictions_with_odds_(\d{8}_\d{6})\.json', filename)
                if match:
                    return match.group(1)
                return "00000000_000000"
            
            latest_file = max(files, key=extract_datetime)
            return latest_file
            
        except Exception as e:
            st.error(f"파일 검색 중 오류: {e}")
            return None
    
    def load_latest_prediction_data(self):
        """최신 예측 데이터 로드 (Simple Dashboard와 동일)"""
        latest_file = self.find_latest_prediction_file()
        
        if not latest_file:
            return None, None
            
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 배당률 필터링 (Simple Dashboard 로직)
            filtered_data = []
            for game in data:
                home_odds = game.get('home_team_odds')
                away_odds = game.get('away_team_odds')
                
                if home_odds is not None and away_odds is not None:
                    game['home_odds'] = home_odds  # 필드명 통일
                    game['away_odds'] = away_odds
                    filtered_data.append(game)
            
            return filtered_data, latest_file
            
        except Exception as e:
            st.error(f"데이터 로드 오류: {e}")
            return None, None
    
    def calculate_ensemble_prob(self, game, weights):
        """앙상블 확률 계산 (Simple Dashboard와 동일)"""
        ensemble_prob = 0
        total_weight = 0
        
        for model, weight in weights.items():
            prob_key = f"{model}_probability"
            if prob_key in game and game[prob_key] is not None and weight > 0:
                ensemble_prob += float(game[prob_key]) * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return ensemble_prob / total_weight
    
    def process_game_for_strategy(self, game, weights):
        """게임 데이터 전처리 (Simple Dashboard 로직 사용)"""
        try:
            ensemble_prob = self.calculate_ensemble_prob(game, weights)
            
            # 예측 결정
            if ensemble_prob > 0.5:
                predicted_team = 'home'
                selection_odds = game.get('home_odds')
                selection_team_name = game.get('home_team')
            else:
                predicted_team = 'away'
                selection_odds = game.get('away_odds')
                selection_team_name = game.get('away_team')
            
            if selection_odds is None:
                return None
            
            selection_odds = float(selection_odds)
            
            # 예측 ROI 계산
            win_prob = ensemble_prob if predicted_team == 'home' else 1 - ensemble_prob
            
            if selection_odds > 0:
                win_payout = (selection_odds / 100) * 100
            else:
                win_payout = (100 / abs(selection_odds)) * 100
            
            predicted_roi = (win_prob * win_payout) + ((1 - win_prob) * (-100))
            
            # 추가 메트릭
            confidence = abs(ensemble_prob - 0.5)
            
            # 시장 확률 vs 모델 확률 차이
            market_prob = self._convert_odds_to_probability(selection_odds)
            model_prob = win_prob
            odds_probability_divergence = (model_prob - market_prob) * 100
            
            # Kelly Criterion
            kelly_criterion = self._calculate_kelly_criterion(win_prob, selection_odds)
            
            return {
                'game_info': f"{game.get('away_team', 'Unknown')} @ {game.get('home_team', 'Unknown')}",
                'predicted_team': predicted_team,
                'selection_team_name': selection_team_name,
                'selection_odds': selection_odds,
                'ensemble_prob': ensemble_prob,
                'win_prob': win_prob,
                'predicted_roi': predicted_roi,
                'confidence': confidence,
                'odds_probability_divergence': odds_probability_divergence,
                'kelly_criterion': kelly_criterion,
                'game_date': game.get('date', 'Unknown'),
                'start_time': game.get('start_time_et', 'Unknown'),
                'raw_game': game
            }
            
        except Exception as e:
            return None
    
    def _convert_odds_to_probability(self, odds):
        """배당률을 확률로 변환"""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    def _calculate_kelly_criterion(self, win_prob, odds):
        """Kelly Criterion 계산"""
        try:
            if odds > 0:
                decimal_odds = (odds / 100) + 1
            else:
                decimal_odds = (100 / abs(odds)) + 1
            
            kelly = (win_prob * decimal_odds - 1) / (decimal_odds - 1)
            return max(0, kelly * 100)
            
        except:
            return 0
    
    def check_zone_condition(self, game, zone, weights):
        """Zone 조건 확인 (Simple Dashboard와 동일 로직)"""
        dimension = zone['dimension']
        segment = zone['segment']
        
        processed_game = self.process_game_for_strategy(game, weights)
        if processed_game is None:
            return False
        
        if dimension == 'predicted_roi':
            pred_roi = processed_game['predicted_roi']
            if segment == 'Very Negative (<-20%)':
                return pred_roi < -20
            elif segment == 'Negative (-20% ~ 0%)':
                return -20 <= pred_roi < 0
            elif segment == 'Positive (0% ~ 20%)':
                return 0 <= pred_roi < 20
            elif segment == 'Very Positive A (20% ~ 40%)':
                return 20 <= pred_roi < 40
            elif segment == 'Very Positive B (40% ~ 60%)':
                return 40 <= pred_roi < 60
            elif segment == 'Extremely Positive A (60% ~ 100%)':
                return 60 <= pred_roi < 100
            elif segment == 'Extremely Positive B (>100%)':
                return pred_roi >= 100
                
        elif dimension == 'odds':
            selection_odds = processed_game['selection_odds']
            if segment == 'Heavy Favorite (< -200)':
                return selection_odds < -200
            elif segment == 'Favorite (-200 ~ -120)':
                return -200 <= selection_odds < -120
            elif segment == 'Pick Em (-120 ~ +120)':
                return -120 <= selection_odds <= 120
            elif segment == 'Underdog (+120 ~ +150)':
                return 120 < selection_odds <= 150
            elif segment == 'Strong Underdog (+150 ~ +300)':
                return 150 < selection_odds <= 300
            elif segment == 'Heavy Underdog (> +300)':
                return selection_odds > 300
                
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
            divergence = processed_game['odds_probability_divergence']
            if segment == 'Model Much More Pessimistic (<-10%)':
                return divergence < -10
            elif segment == 'Model Slightly Pessimistic (-10% ~ -5%)':
                return -10 <= divergence < -5
            elif segment == 'Market Aligned (-5% ~ +5%)':
                return -5 <= divergence < 5
            elif segment == 'Model Slightly Optimistic (+5% ~ +10%)':
                return 5 <= divergence < 10
            elif segment == 'Model Much More Optimistic (+10%+)':
                return divergence >= 10
                
        elif dimension == 'kelly_criterion':
            kelly = processed_game['kelly_criterion']
            if segment == 'No Selection (Kelly ≤ 0%)':
                return kelly <= 0
            elif segment == 'Low Confidence (0% < Kelly ≤ 5%)':
                return 0 < kelly <= 5
            elif segment == 'Medium Confidence (5% < Kelly ≤ 15%)':
                return 5 < kelly <= 15
            elif segment == 'High Confidence (15% < Kelly ≤ 25%)':
                return 15 < kelly <= 25
            elif segment == 'Very High Confidence (25% ~ 60%)':
                return 25 < kelly <= 60
            elif segment == 'Extremely High Confidence (Kelly > 60%)':
                return kelly > 60
        
        return False
    
    def find_custom_zone_matches(self, latest_data, selected_zones, weights):
        """커스텀 존 매칭 게임 찾기 (Simple Dashboard와 동일)"""
        matching_games = []
        
        if not latest_data or not selected_zones or not weights:
            return matching_games
        
        for game in latest_data:
            all_dimensions_match = True
            
            for dimension, selected_segments in selected_zones.items():
                dimension_match = False
                
                for segment in selected_segments:
                    zone = {'dimension': dimension, 'segment': segment}
                    
                    if self.check_zone_condition(game, zone, weights):
                        dimension_match = True
                        break
                
                if not dimension_match:
                    all_dimensions_match = False
                    break
            
            if all_dimensions_match:
                processed_game = self.process_game_for_strategy(game, weights)
                if processed_game:
                    matching_games.append(processed_game)
        
        return matching_games
    
    def generate_telegram_message(self, matching_games, weights, selected_zones):
        """텔레그램 메시지 생성 - 정보 제공 중심"""
        if not matching_games:
            return "📊 No analytical data found for today's selections"
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 메시지 시작
        message = f"📊 **MLB Statistical Insights - {today}**\n\n"
        
        # 분석 설정 정보
        active_models = [k.replace('model_', '').replace('_', ' ').title() for k, v in weights.items() if v > 0]
        message += f"🤖 **Statistical Models**: {', '.join(active_models)}\n"
        
        # 선택된 조건들
        total_dimensions = len([dim for dim, segments in selected_zones.items() if segments])
        message += f"🎯 **Selection Filters**: {total_dimensions} dimensions applied\n"
        for dim, segments in selected_zones.items():
            if segments:  # 선택된 조건이 있는 경우만
                dim_name = dim.replace('_', ' ').title()
                # 구체적인 필터 조건들 표시
                conditions_text = ", ".join(segments)
                message += f"   • {dim_name}: {conditions_text}\n"
        
        message += f"\n📈 **Filtered Results**: {len(matching_games)} preferred matches identified\n\n"
        
        # 각 게임 정보
        for i, game in enumerate(matching_games, 1):
            message += f"⚾ **Match #{i}**\n"
            message += f"🏟️ {game['game_info']}\n"
            message += f"📊 Statistical Lean: **{game['selection_team_name']} ({game['selection_odds']:+})**\n"
            message += f"📈 Model Projection: {game['predicted_roi']:+.1f}%\n"
            message += f"🎲 Win Probability: {game['win_prob']:.1%}\n"
            message += f"💡 Confidence Level: {game['confidence']:.3f}\n"
            if game['start_time'] != 'Unknown':
                message += f"⏰ Scheduled Time: {game['start_time']}\n"
            message += "\n"
        
        # 통계 요약
        avg_roi = sum(game['predicted_roi'] for game in matching_games) / len(matching_games)
        avg_prob = sum(game['win_prob'] for game in matching_games) / len(matching_games)
        
        message += f"📊 **Statistical Summary**\n"
        message += f"Average Model Projection: {avg_roi:+.1f}%\n"
        message += f"Average Win Probability: {avg_prob:.1%}\n\n"
        
        # 면책 조항
        message += "⚠️ **IMPORTANT LEGAL DISCLAIMER**\n"
        message += "📊 **Educational/Analytical Content Only** - NOT betting advice\n"
        message += "🚫 **No Guarantees** - Past performance ≠ future results\n"
        message += "💰 **High Risk Warning** - Gambling can cause significant financial loss\n"
        message += "⚖️ **Your Responsibility** - Comply with local laws, make your own decisions\n"
        message += "🔍 **Independent Verification Required** - Data may contain errors\n"
        message += "📋 **By using this info, you accept full responsibility and risk**"
        
        return message
    
    def generate_model_performance_report(self, model_performances, daily_performances, best_model_name, start_date, end_date):
        """최고 성과 모델의 기간별 성과 리포트 생성"""
        
        # 최고 모델의 전체 기간 성과
        best_model_data = model_performances[best_model_name]
        
        # 최고 모델의 일별 성과 데이터
        model_daily_data = daily_performances.get(best_model_name, {})
        
        # 기간 내 일별 데이터 필터링
        daily_stats = []
        for date_str, day_data in model_daily_data.items():
            if start_date <= date_str <= end_date:
                daily_stats.append({
                    'date': date_str,
                    'roi': day_data['roi'],
                    'win_rate': day_data['win_rate'],
                    'games_with_odds': day_data['games_with_odds'],
                    'profit_loss': day_data['profit_loss']
                })
        
        daily_stats.sort(key=lambda x: x['date'])
        
        if not daily_stats:
            return f"❌ No daily data available for {best_model_name} in period {start_date} to {end_date}"
        
        # 통계 계산
        profitable_days = len([d for d in daily_stats if d['roi'] > 0])
        best_day = max(daily_stats, key=lambda x: x['roi'])
        worst_day = min(daily_stats, key=lambda x: x['roi'])
        
        # 연속 수익/손실 계산
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        temp_streak = 0
        last_positive = None
        
        for day in daily_stats:
            is_positive = day['roi'] > 0
            if last_positive == is_positive:
                temp_streak += 1
            else:
                if last_positive is True:
                    max_win_streak = max(max_win_streak, temp_streak)
                elif last_positive is False:
                    max_loss_streak = max(max_loss_streak, temp_streak)
                temp_streak = 1
                last_positive = is_positive
        
        # 마지막 스트릭 처리
        if last_positive is True:
            max_win_streak = max(max_win_streak, temp_streak)
            current_streak = temp_streak if daily_stats[-1]['roi'] > 0 else 0
        elif last_positive is False:
            max_loss_streak = max(max_loss_streak, temp_streak)
            current_streak = -temp_streak if daily_stats[-1]['roi'] <= 0 else 0
        
        # 리포트 생성
        model_display_name = best_model_name.replace('model_', '').replace('_', ' ').title()
        
        report = f"""🏆 **{model_display_name} Performance Report**
📅 **Analysis Period**: {start_date} to {end_date}

📋 **Important Notes**
• Following statistics reflect comprehensive analysis of all available matches
• Statistical insights recommendations utilize zone-based filtering to identify optimal conditions
• Zone analysis targets specific market segments with historically superior performance
• Filtered selections represent refined subset of total analyzed matches

💰 **Statistical Performance**
Model ROI: {best_model_data.get('actual_roi', 0):+.2f}%
Prediction Accuracy: {best_model_data.get('win_rate', 0):.1f}%
Total Analyzed: {best_model_data.get('total_bets', 0)} games
Correct Predictions: {best_model_data.get('correct_predictions', 0)}
Projected Value: ${1000 * best_model_data.get('actual_roi', 0) / 100:+.1f}
Simulation Base: $1,000

📊 **Daily Performance Summary**
Total Days: {len(daily_stats)}
Positive Days: {profitable_days}/{len(daily_stats)} ({profitable_days/len(daily_stats)*100:.1f}%)
Current Streak: {"🟢" if current_streak > 0 else "🔴" if current_streak < 0 else "⚪"} {abs(current_streak)} days

🎯 **Key Statistics**
Best Day: {best_day['date']} ({best_day['roi']:+.2f}%)
Worst Day: {worst_day['date']} ({worst_day['roi']:+.2f}%)
Max Positive Streak: {max_win_streak} days
Max Negative Streak: {max_loss_streak} days

📈 **Recent Performance** (Last 5 Days)"""

        # 최근 5일 성과
        recent_days = daily_stats[-5:] if len(daily_stats) >= 5 else daily_stats
        for day in recent_days:
            status = "🟢" if day['roi'] > 0 else "🔴" if day['roi'] < -5 else "🟡"
            projected_value = 1000 * day['roi'] / 100
            report += f"\n{day['date']}: {status} {day['roi']:+.2f}% | {day['win_rate']:.1f}% | {day['games_with_odds']} analyzed | ${projected_value:+.1f} projected"
        
        # 성과 하이라이트
        if best_model_data.get('actual_roi', 0) > 10:
            performance_note = "🔥 Exceptional Performance"
        elif best_model_data.get('actual_roi', 0) > 5:
            performance_note = "✅ Strong Performance"
        elif best_model_data.get('actual_roi', 0) > 0:
            performance_note = "📈 Positive Performance"
        else:
            performance_note = "📉 Below Expectations"
        
        report += f"""

🎯 **Performance Rating**: {performance_note}
💡 **Model Status**: Currently our top-performing model
🔄 **Consistency**: {profitable_days/len(daily_stats)*100:.1f}% positive days

ℹ️ **Track Record for {start_date} to {end_date}**
📊 Performance data for transparency and evaluation
⚠️ Past performance does not guarantee future results"""
        
        return report
    
    def save_picks_to_json(self, matching_games, weights, selected_zones):
        """픽 정보를 JSON 파일로 저장"""
        try:
            # picks 디렉토리 생성 (없으면)
            self.picks_dir.mkdir(exist_ok=True)
            
            # 오늘 날짜로 파일명 생성
            today = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"daily_picks_{today}.json"
            filepath = self.picks_dir / filename
            
            # 저장할 데이터 구조
            picks_data = {
                'date': datetime.now().strftime("%Y-%m-%d"),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'configuration': {
                    'model_weights': {k: v for k, v in weights.items() if v > 0},
                    'selected_zones': selected_zones
                },
                'picks': []
            }
            
            # 각 픽 정보 저장
            for i, game in enumerate(matching_games, 1):
                pick_info = {
                    'pick_number': i,
                    'game_info': game['game_info'],
                    'predicted_team': game['predicted_team'],
                    'selection_team_name': game['selection_team_name'],
                    'selection_odds': game['selection_odds'],
                    'ensemble_prob': game['ensemble_prob'],
                    'win_prob': game['win_prob'],
                    'predicted_roi': game['predicted_roi'],
                    'confidence': game['confidence'],
                    'odds_probability_divergence': game['odds_probability_divergence'],
                    'kelly_criterion': game['kelly_criterion'],
                    'game_date': game['game_date'],
                    'start_time': game['start_time'],
                    # 나중에 결과 추적을 위한 필드들
                    'actual_result': None,  # 실제 결과 (나중에 업데이트)
                    'actual_roi': None,     # 실제 ROI (나중에 업데이트)
                    'is_correct': None      # 예측 정확도 (나중에 업데이트)
                }
                picks_data['picks'].append(pick_info)
            
            # JSON 파일로 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(picks_data, f, ensure_ascii=False, indent=2)
            
            return filepath, len(matching_games)
            
        except Exception as e:
            return None, str(e)

def main():
    # 헤더
    st.markdown('<div class="main-header">📱 MLB Telegram Analytics Hub</div>', unsafe_allow_html=True)
    
    # 면책조항 (상단에 명확히 표시)
    st.error("""
    ⚠️ **LEGAL DISCLAIMER**: This service provides **statistical analysis for educational purposes only**. 
    NOT betting advice. High risk of financial loss. Past performance ≠ future results. 
    You are **solely responsible** for your decisions and must comply with local laws.
    """)
    
    # 생성기 초기화
    hub = TelegramAnalyticsHub()
    
    # 탭 생성
    tab1, tab2 = st.tabs(["📈 Performance Tracker", "🎯 Statistical Insights"])
    
    # === 탭 1: 성과 트래커 ===
    with tab1:
        st.header("📈 Model Performance Tracker")
        st.info("Historical performance analysis of all MLB prediction models")
        
        # 구분자 선택
        st.markdown("### 📊 Data Source Selection")
        available_prefixes = hub.get_available_prefixes()
        
        # 구분자 설명
        prefix_display_names = {}
        for prefix in available_prefixes:
            if prefix == "None":
                prefix_display_names[prefix] = "None (기본 - 구분자 없는 파일)"
            else:
                prefix_display_names[prefix] = f"{prefix.rstrip('_')} (구분자 파일)"
        
        selected_prefix = st.selectbox(
            "📂 Select Data Source:",
            available_prefixes,
            format_func=lambda x: prefix_display_names[x],
            help="구분자 없는 파일은 기본 분석, 구분자 파일은 특별 테스트 분석입니다."
        )
        
        st.markdown("---")
        
        # 날짜 선택
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            try:
                # 선택된 구분자에 따른 사용 가능한 날짜 가져오기
                available_dates = hub.get_available_dates_for_prefix(selected_prefix)
                
                if available_dates:
                    min_date = datetime.strptime(available_dates[0], '%Y-%m-%d').date()
                    max_date = datetime.strptime(available_dates[-1], '%Y-%m-%d').date()
                    
                    start_date = st.date_input(
                        "Start Date",
                        value=min_date,  # 가장 첫 파일 기준
                        min_value=min_date,
                        max_value=max_date,
                        key=f"start_date_{selected_prefix}"
                    )
                else:
                    st.error(f"선택한 구분자 '{selected_prefix}'에 대한 데이터가 없습니다.")
                    return
                    
            except Exception as e:
                st.error(f"날짜 로드 오류: {e}")
                return
        
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key=f"end_date_{selected_prefix}"
            )
        
        # 자동 분석 실행
        if start_date <= end_date:
            with st.spinner("📊 Analyzing model performance..."):
                try:
                    start_date_str = start_date.strftime('%Y-%m-%d')
                    end_date_str = end_date.strftime('%Y-%m-%d')
                    
                    results = run_analysis(start_date_str, end_date_str, selected_prefix)
                    
                    # 기본 통계 (SimpleModelAnalyzer 구조에 맞게)
                    st.markdown("## 📊 Model Performance Results")
                    
                    # 분석 정보 표시
                    if selected_prefix != "None":
                        st.info(f"📂 **분석 데이터**: {selected_prefix.rstrip('_')} 구분자 파일들 | 📅 **기간**: {start_date_str} ~ {end_date_str}")
                    else:
                        st.info(f"📂 **분석 데이터**: 기본 파일들 (구분자 없음) | 📅 **기간**: {start_date_str} ~ {end_date_str}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # 올바른 키 사용
                    total_games = results.get('analysis_summary', {}).get('total_games', 0)
                    analysis_period = (end_date - start_date).days
                    models_count = len(results.get('model_performances', {}))
                    
                    # 최고 ROI 계산 (올바른 구조)
                    if results.get('model_performances'):
                        best_roi = max([model.get('actual_roi', 0) for model in results['model_performances'].values()], default=0)
                    else:
                        best_roi = 0
                    
                    with col1:
                        st.metric("Total Games", total_games)
                    with col2:
                        st.metric("Analysis Period", f"{analysis_period} days")
                    with col3:
                        st.metric("Models Analyzed", models_count)
                    with col4:
                        st.metric("Best Model ROI", f"{best_roi:.1f}%")
                    
                    # 모델별 성과 테이블 (SimpleModelAnalyzer 구조에 맞게)
                    if results.get('model_performances'):
                        st.markdown("### 🏆 Model Performance Rankings")
                        
                        # 데이터 정리 (SimpleModelAnalyzer 방식)
                        model_data = []
                        for model_name, model_info in results['model_performances'].items():
                            model_data.append({
                                'Model': model_name,
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
                        
                        # 🏆 최고 성과 모델 하이라이트
                        best_model = df.iloc[0]  # ROI 순으로 정렬되어 있으므로 첫 번째가 최고
                        best_model_data = df[df['Model'] == best_model['Model']]
                        other_data = df[df['Model'] != best_model['Model']]
                        
                        if not best_model_data.empty:
                            st.markdown(f"#### 🏆 **{best_model['Model']} Performance** (Best in Period)")
                            
                            # 최고 모델 스타일링
                            def highlight_best_roi(val):
                                if val > 20:
                                    return 'background-color: #28a745; color: white; font-weight: bold'
                                elif val > 10:
                                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                                elif val > 0:
                                    return 'background-color: #d4edda; color: #155724'
                                else:
                                    return 'background-color: #fff3cd; color: #856404'
                            
                            styled_best_df = best_model_data.style.map(
                                highlight_best_roi, 
                                subset=['ROI (%)']
                            ).format({
                                'ROI (%)': '{:.2f}%',
                                'Win Rate (%)': '{:.1f}%',
                                'Profit/Loss ($)': '${:.1f}',
                                'Total Invested ($)': '${:.0f}'
                            })
                            
                            st.dataframe(styled_best_df, hide_index=True, use_container_width=True)
                        
                        # 다른 모델들 (축소 표시)
                        if not other_data.empty:
                            with st.expander("📊 Other Models Performance", expanded=False):
                                # 일반 모델 스타일링
                                def highlight_roi(val):
                                    if val > 0:
                                        return 'background-color: #d4edda; color: #155724'
                                    elif val < -5:
                                        return 'background-color: #f8d7da; color: #721c24'
                                    else:
                                        return 'background-color: #fff3cd; color: #856404'
                                
                                styled_df = other_data.style.map(
                                    highlight_roi, 
                                    subset=['ROI (%)']
                                ).format({
                                    'ROI (%)': '{:.2f}%',
                                    'Win Rate (%)': '{:.1f}%',
                                    'Profit/Loss ($)': '${:.1f}',
                                    'Total Invested ($)': '${:.0f}'
                                })
                                
                                st.dataframe(styled_df, hide_index=True, use_container_width=True)
                        
                        # 🆕 Daily Performance Analysis Section (Simple Dashboard와 동일)
                        st.markdown("---")
                        st.markdown("## 📅 Model Daily Performance Analysis")
                        
                        # Check if daily performance data is available
                        if 'daily_performances' in results and results['daily_performances']:
                            daily_data = results['daily_performances']
                            
                            # 모델 선택 (가나다 순)
                            available_models = sorted(daily_data.keys())
                            model_options = available_models
                            
                            selected_model = st.selectbox(
                                "Choose a model to view daily performance:",
                                model_options,
                                index=0,
                                key="daily_performance_model_selector"
                            )
                            
                            if selected_model and selected_model in daily_data:
                                model_daily_data = daily_data[selected_model]
                                
                                # 모델 이름 표시
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
                                    profitable_pct = profitable_days/total_days*100 if total_days > 0 else 0
                                    st.metric("Profitable Days", f"{profitable_days} ({profitable_pct:.1f}%)")
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
                                    if val > 20:
                                        return 'background-color: #28a745; color: white; font-weight: bold'
                                    elif val > 0:
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
                        else:
                            st.info("Daily performance data not available for this period.")
                        
                        # 🆕 Segment Performance Analysis
                        st.markdown("---")
                        st.markdown("## 📊 Segment Performance Analysis")
                        st.info("Detailed analysis of model performance across different prediction confidence levels and market scenarios.")
                        
                        # 구간 분석 관련 함수들 추가
                        def analyze_model_segments_telegram(matched_data, model_name):
                            """개별 모델의 구간별 성과 분석 (public_performance_tracker와 동일)"""
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
                                unit_amount = 100  # 분석 단위
                                
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
                                
                                confidence = abs(prob - 0.5)  # 신뢰도 계산
                                
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
                            
                            # 구간별 분석 수행
                            segments = {}
                            
                            # 1. 예측 ROI 구간별 분석
                            segments['predicted_roi'] = analyze_predicted_roi_segments_telegram(analysis_data)
                            
                            # 2. 배당률 구간별 분석
                            segments['odds'] = analyze_odds_segments_telegram(analysis_data)
                            
                            # 3. 신뢰도 구간별 분석
                            segments['confidence'] = analyze_confidence_segments_telegram(analysis_data)
                            
                            # 4. Kelly Criterion 분석
                            segments['kelly'] = analyze_kelly_segments_telegram(analysis_data)
                            
                            # 5. 시장 vs 모델 괴리도 분석
                            segments['market_divergence'] = analyze_market_divergence_segments_telegram(analysis_data)
                            
                            return segments
                        
                        def analyze_predicted_roi_segments_telegram(analysis_data):
                            """예측 ROI 구간별 분석 - 세분화된 구간"""
                            segments = {
                                'Very Negative (<-20%)': [],
                                'Negative (-20% ~ 0%)': [],
                                'Positive (0% ~ 20%)': [],
                                'Very Positive A (20% ~ 40%)': [],
                                'Very Positive B (40% ~ 60%)': [],
                                'Extremely Positive A (60% ~ 100%)': [],
                                'Extremely Positive B (>100%)': []
                            }
                            
                            for data in analysis_data:
                                pred_roi = data['predicted_roi']
                                if pred_roi < -20:
                                    segments['Very Negative (<-20%)'].append(data)
                                elif pred_roi < 0:
                                    segments['Negative (-20% ~ 0%)'].append(data)
                                elif pred_roi < 20:
                                    segments['Positive (0% ~ 20%)'].append(data)
                                elif pred_roi < 40:
                                    segments['Very Positive A (20% ~ 40%)'].append(data)
                                elif pred_roi < 60:
                                    segments['Very Positive B (40% ~ 60%)'].append(data)
                                elif pred_roi < 100:
                                    segments['Extremely Positive A (60% ~ 100%)'].append(data)
                                else:
                                    segments['Extremely Positive B (>100%)'].append(data)
                            
                            return calculate_segment_performance_telegram(segments)
                        
                        def analyze_confidence_segments_telegram(analysis_data):
                            """신뢰도 구간별 분석"""
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
                            
                            return calculate_segment_performance_telegram(segments)
                        
                        def analyze_odds_segments_telegram(analysis_data):
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
                            
                            return calculate_segment_performance_telegram(segments)
                        
                        def analyze_market_divergence_segments_telegram(analysis_data):
                            """시장 vs 모델 괴리도 분석"""
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
                            
                            return calculate_segment_performance_telegram(segments)
                        
                        def analyze_kelly_segments_telegram(analysis_data):
                            """Kelly Criterion 분석"""
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
                            
                            return calculate_segment_performance_telegram(segments)
                        
                        def calculate_segment_performance_telegram(segments):
                            """구간별 성과 계산"""
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
                        
                        def display_segment_performance_table_telegram(segment_data, segment_type):
                            """구간별 성과 테이블 표시"""
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
                            
                            # 테이블 스타일링
                            styled_df = df.style.map(
                                highlight_segment_roi, 
                                subset=['Actual ROI (%)']
                            ).format({
                                'Predicted ROI (%)': '{:.2f}%',
                                'Actual ROI (%)': '{:.2f}%',
                                'ROI Difference (%)': '{:.2f}%',
                                'Win Rate (%)': '{:.1f}%',
                                'Accuracy (%)': '{:.1f}%'
                            })
                            
                            st.dataframe(styled_df, use_container_width=True, hide_index=True)
                        
                        # Segment Analysis UI
                        if results.get('model_performances') and results.get('daily_performances'):
                            # 모델 선택 (구간 분석용)
                            segment_models = sorted(results['model_performances'].keys())
                            selected_segment_model = st.selectbox(
                                "🎯 Select model for segment analysis:",
                                segment_models,
                                key="segment_analysis_model"
                            )
                            
                            if selected_segment_model:
                                with st.spinner(f"🔍 Analyzing segments for {selected_segment_model}..."):
                                    # 매칭된 데이터 가져오기 (SimpleModelAnalyzer와 동일한 방식으로)
                                    try:
                                        # SimpleModelAnalyzer를 통해 데이터 로드 및 매칭
                                        analyzer = load_analyzer(selected_prefix)
                                        data = analyzer.load_data(start_date_str, end_date_str)
                                        matched_data = analyzer.match_predictions_with_results(
                                            data['predictions'], 
                                            data['historical_records']
                                        )
                                        
                                        if matched_data:
                                            # 구간 분석 수행
                                            segment_results = analyze_model_segments_telegram(matched_data, selected_segment_model)
                                            
                                            if segment_results:
                                                # 구간별 결과 표시
                                                segment_tabs = st.tabs([
                                                    "📊 Predicted ROI",
                                                    "💰 Odds Ranges", 
                                                    "🎯 Confidence Levels",
                                                    "📈 Market vs Model",
                                                    "🎰 Kelly Criterion"
                                                ])
                                                
                                                with segment_tabs[0]:
                                                    st.markdown("### 📊 Performance by Predicted ROI Ranges")
                                                    st.caption("Analysis of how well the model performs across different expected return levels.")
                                                    display_segment_performance_table_telegram(
                                                        segment_results['predicted_roi'], 
                                                        "Predicted ROI Range"
                                                    )
                                                
                                                with segment_tabs[1]:
                                                    st.markdown("### 💰 Performance by Odds Ranges")
                                                    st.caption("Analysis of model performance across different betting odds scenarios.")
                                                    display_segment_performance_table_telegram(
                                                        segment_results['odds'], 
                                                        "Odds Range"
                                                    )
                                                
                                                with segment_tabs[2]:
                                                    st.markdown("### 🎯 Performance by Confidence Levels")
                                                    st.caption("Analysis of model performance based on prediction confidence levels.")
                                                    display_segment_performance_table_telegram(
                                                        segment_results['confidence'], 
                                                        "Confidence Level"
                                                    )
                                                
                                                with segment_tabs[3]:
                                                    st.markdown("### 📈 Performance by Market vs Model Divergence")
                                                    st.caption("Analysis of performance when model probability differs from market expectations.")
                                                    display_segment_performance_table_telegram(
                                                        segment_results['market_divergence'], 
                                                        "Market Divergence"
                                                    )
                                                
                                                with segment_tabs[4]:
                                                    st.markdown("### 🎰 Performance by Kelly Criterion")
                                                    st.caption("Analysis of performance across different Kelly Criterion bet sizing recommendations.")
                                                    display_segment_performance_table_telegram(
                                                        segment_results['kelly'], 
                                                        "Kelly Criterion Range"
                                                    )
                                            else:
                                                st.warning(f"No segment analysis data available for {selected_segment_model}")
                                        else:
                                            st.warning("No matched data available for segment analysis")
                                            
                                    except Exception as e:
                                        st.error(f"Error in segment analysis: {str(e)}")
                        else:
                            st.info("Model performance data required for segment analysis")
                        
                        # 🆕 Home/Away Segment Performance Analysis
                        st.markdown("---")
                        st.markdown("## 🏠🛫 Home/Away Segment Performance Analysis")
                        st.info("Detailed analysis of model performance by home/away team selection across different prediction confidence levels and market scenarios.")
                        
                        # 홈/어웨이 구간 분석 관련 함수들 추가
                        def analyze_model_segments_home_away_telegram(matched_data, model_name):
                            """개별 모델의 홈/어웨이 구간별 성과 분석"""
                            prob_key = f'{model_name}_probability'
                            
                            # 해당 모델의 데이터만 필터링
                            model_data = [r for r in matched_data if r.get(prob_key) is not None and r.get(prob_key) > 0]
                            
                            if not model_data:
                                return None, None
                            
                            # 홈팀/어웨이팀 분석 데이터 준비
                            home_analysis_data = []
                            away_analysis_data = []
                            
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
                                unit_amount = 100  # 분석 단위
                                
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
                                
                                confidence = abs(prob - 0.5)  # 신뢰도 계산
                                
                                analysis_item = {
                                    'ensemble_prob': prob,
                                    'predicted_roi': predicted_roi,
                                    'actual_roi': actual_roi,
                                    'confidence': confidence,
                                    'selection_odds': selection_odds,
                                    'predicted_team': predicted_team,
                                    'actual_win': actual_home_win,
                                    'home_odds': home_odds,
                                    'away_odds': away_odds
                                }
                                
                                # 홈/어웨이에 따라 분리
                                if predicted_team == 'home':
                                    home_analysis_data.append(analysis_item)
                                else:
                                    away_analysis_data.append(analysis_item)
                            
                            # 각각에 대해 구간별 분석 수행
                            home_segments = None
                            away_segments = None
                            
                            if home_analysis_data:
                                home_segments = {
                                    'predicted_roi': analyze_predicted_roi_segments_telegram(home_analysis_data),
                                    'odds': analyze_odds_segments_telegram(home_analysis_data),
                                    'confidence': analyze_confidence_segments_telegram(home_analysis_data),
                                    'kelly': analyze_kelly_segments_telegram(home_analysis_data),
                                    'market_divergence': analyze_market_divergence_segments_telegram(home_analysis_data)
                                }
                            
                            if away_analysis_data:
                                away_segments = {
                                    'predicted_roi': analyze_predicted_roi_segments_telegram(away_analysis_data),
                                    'odds': analyze_odds_segments_telegram(away_analysis_data),
                                    'confidence': analyze_confidence_segments_telegram(away_analysis_data),
                                    'kelly': analyze_kelly_segments_telegram(away_analysis_data),
                                    'market_divergence': analyze_market_divergence_segments_telegram(away_analysis_data)
                                }
                            
                            return home_segments, away_segments
                        
                        def display_home_away_segment_performance_table_telegram(home_segment_data, away_segment_data, segment_type):
                            """홈/어웨이 구간별 성과 테이블 표시"""
                            
                            # 홈팀 데이터 표시
                            st.markdown(f"#### 🏠 Home Team Picks - {segment_type}")
                            if home_segment_data and any(data['games'] > 0 for data in home_segment_data.values()):
                                # 홈팀 데이터 정리
                                home_table_data = []
                                for segment_name, data in home_segment_data.items():
                                    if data['games'] > 0:
                                        home_table_data.append({
                                            segment_type: segment_name,
                                            'Games': data['games'],
                                            'Predicted ROI (%)': data['predicted_roi'],
                                            'Actual ROI (%)': data['actual_roi'],
                                            'ROI Difference (%)': data['roi_difference'],
                                            'Win Rate (%)': data['win_rate'],
                                            'Accuracy (%)': data['accuracy']
                                        })
                                
                                if home_table_data:
                                    home_df = pd.DataFrame(home_table_data)
                                    
                                    # ROI에 따른 색상 함수
                                    def highlight_segment_roi(val):
                                        if val > 5:
                                            return 'background-color: #d4edda; color: #155724'
                                        elif val < -5:
                                            return 'background-color: #f8d7da; color: #721c24'
                                        else:
                                            return 'background-color: #fff3cd; color: #856404'
                                    
                                    # 홈팀 테이블 스타일링
                                    styled_home_df = home_df.style.map(
                                        highlight_segment_roi, 
                                        subset=['Actual ROI (%)']
                                    ).format({
                                        'Predicted ROI (%)': '{:.2f}%',
                                        'Actual ROI (%)': '{:.2f}%',
                                        'ROI Difference (%)': '{:.2f}%',
                                        'Win Rate (%)': '{:.1f}%',
                                        'Accuracy (%)': '{:.1f}%'
                                    })
                                    
                                    st.dataframe(styled_home_df, use_container_width=True, hide_index=True)
                                else:
                                    st.warning(f"No home team {segment_type} data with games available.")
                            else:
                                st.warning(f"No home team {segment_type} data available.")
                            
                            # 어웨이팀 데이터 표시
                            st.markdown(f"#### 🛫 Away Team Picks - {segment_type}")
                            if away_segment_data and any(data['games'] > 0 for data in away_segment_data.values()):
                                # 어웨이팀 데이터 정리
                                away_table_data = []
                                for segment_name, data in away_segment_data.items():
                                    if data['games'] > 0:
                                        away_table_data.append({
                                            segment_type: segment_name,
                                            'Games': data['games'],
                                            'Predicted ROI (%)': data['predicted_roi'],
                                            'Actual ROI (%)': data['actual_roi'],
                                            'ROI Difference (%)': data['roi_difference'],
                                            'Win Rate (%)': data['win_rate'],
                                            'Accuracy (%)': data['accuracy']
                                        })
                                
                                if away_table_data:
                                    away_df = pd.DataFrame(away_table_data)
                                    
                                    # 어웨이팀 테이블 스타일링
                                    styled_away_df = away_df.style.map(
                                        highlight_segment_roi, 
                                        subset=['Actual ROI (%)']
                                    ).format({
                                        'Predicted ROI (%)': '{:.2f}%',
                                        'Actual ROI (%)': '{:.2f}%',
                                        'ROI Difference (%)': '{:.2f}%',
                                        'Win Rate (%)': '{:.1f}%',
                                        'Accuracy (%)': '{:.1f}%'
                                    })
                                    
                                    st.dataframe(styled_away_df, use_container_width=True, hide_index=True)
                                else:
                                    st.warning(f"No away team {segment_type} data with games available.")
                            else:
                                st.warning(f"No away team {segment_type} data available.")
                        
                        # Home/Away Segment Analysis UI
                        if results.get('model_performances') and results.get('daily_performances'):
                            # 모델 선택 (홈/어웨이 구간 분석용)
                            home_away_segment_models = sorted(results['model_performances'].keys())
                            selected_home_away_segment_model = st.selectbox(
                                "🎯 Select model for home/away segment analysis:",
                                home_away_segment_models,
                                key="home_away_segment_analysis_model"
                            )
                            
                            if selected_home_away_segment_model:
                                with st.spinner(f"🔍 Analyzing home/away segments for {selected_home_away_segment_model}..."):
                                    try:
                                        # SimpleModelAnalyzer를 통해 데이터 로드 및 매칭
                                        analyzer = load_analyzer(selected_prefix)
                                        data = analyzer.load_data(start_date_str, end_date_str)
                                        matched_data = analyzer.match_predictions_with_results(
                                            data['predictions'], 
                                            data['historical_records']
                                        )
                                        
                                        if matched_data:
                                            # 홈/어웨이 구간 분석 수행
                                            home_segment_results, away_segment_results = analyze_model_segments_home_away_telegram(matched_data, selected_home_away_segment_model)
                                            
                                            if home_segment_results or away_segment_results:
                                                # 홈/어웨이 구간별 결과 표시
                                                home_away_segment_tabs = st.tabs([
                                                    "📊 Predicted ROI",
                                                    "💰 Odds Ranges", 
                                                    "🎯 Confidence Levels",
                                                    "📈 Market vs Model",
                                                    "🎰 Kelly Criterion"
                                                ])
                                                
                                                with home_away_segment_tabs[0]:
                                                    st.markdown("### 📊 Home/Away Performance by Predicted ROI Ranges")
                                                    st.caption("Comparison of home vs away team picks across different expected return levels.")
                                                    display_home_away_segment_performance_table_telegram(
                                                        home_segment_results['predicted_roi'] if home_segment_results else None,
                                                        away_segment_results['predicted_roi'] if away_segment_results else None,
                                                        "Predicted ROI Range"
                                                    )
                                                
                                                with home_away_segment_tabs[1]:
                                                    st.markdown("### 💰 Home/Away Performance by Odds Ranges")
                                                    st.caption("Comparison of home vs away team picks across different betting odds scenarios.")
                                                    display_home_away_segment_performance_table_telegram(
                                                        home_segment_results['odds'] if home_segment_results else None,
                                                        away_segment_results['odds'] if away_segment_results else None,
                                                        "Odds Range"
                                                    )
                                                
                                                with home_away_segment_tabs[2]:
                                                    st.markdown("### 🎯 Home/Away Performance by Confidence Levels")
                                                    st.caption("Comparison of home vs away team picks based on prediction confidence levels.")
                                                    display_home_away_segment_performance_table_telegram(
                                                        home_segment_results['confidence'] if home_segment_results else None,
                                                        away_segment_results['confidence'] if away_segment_results else None,
                                                        "Confidence Level"
                                                    )
                                                
                                                with home_away_segment_tabs[3]:
                                                    st.markdown("### 📈 Home/Away Performance by Market vs Model Divergence")
                                                    st.caption("Comparison when model probability differs from market expectations for home vs away picks.")
                                                    display_home_away_segment_performance_table_telegram(
                                                        home_segment_results['market_divergence'] if home_segment_results else None,
                                                        away_segment_results['market_divergence'] if away_segment_results else None,
                                                        "Market Divergence"
                                                    )
                                                
                                                with home_away_segment_tabs[4]:
                                                    st.markdown("### 🎰 Home/Away Performance by Kelly Criterion")
                                                    st.caption("Comparison across different Kelly Criterion bet sizing recommendations for home vs away picks.")
                                                    display_home_away_segment_performance_table_telegram(
                                                        home_segment_results['kelly'] if home_segment_results else None,
                                                        away_segment_results['kelly'] if away_segment_results else None,
                                                        "Kelly Criterion Range"
                                                    )
                                            else:
                                                st.warning(f"No home/away segment analysis data available for {selected_home_away_segment_model}")
                                        else:
                                            st.warning("No matched data available for home/away segment analysis")
                                            
                                    except Exception as e:
                                        st.error(f"Error in home/away segment analysis: {str(e)}")
                        else:
                            st.info("Model performance data required for home/away segment analysis")
                        
                        # 🆕 Detailed Odds Segment Performance Analysis
                        st.markdown("---")
                        st.markdown("## 💰📊 Detailed Odds Segment Performance Analysis")
                        st.info("Fine-grained analysis of model performance across more detailed betting odds ranges for precise market segment insights.")
                        
                        # 세분화된 배당률 구간 분석 관련 함수들 추가
                        def analyze_detailed_odds_segments_telegram(analysis_data):
                            """더 세분화된 배당률별 성과 분석 - Pure Home/Away와 동일한 구간"""
                            segments = {
                                'Heavy Favorite (< -300)': [],
                                'Strong Favorite (-300 ~ -200)': [],
                                'Moderate Favorite (-200 ~ -150)': [],
                                'Light Favorite (-150 ~ -120)': [],
                                'Near Favorite (-120 ~ Even)': [],
                                'Near Underdog (Even ~ +120)': [],
                                'Light Underdog A (+120 ~ +140)': [],
                                'Light Underdog B (+140 ~ +160)': [],
                                'Moderate Underdog A (+160 ~ +180)': [],
                                'Moderate Underdog B (+180 ~ +200)': [],
                                'Moderate Underdog C (+200 ~ +220)': [],
                                'Strong Underdog (+220 ~ +300)': [],
                                'Heavy Underdog (> +300)': []
                            }
                            
                            for data in analysis_data:
                                odds = data['selection_odds']
                                
                                if odds < -300:
                                    segments['Heavy Favorite (< -300)'].append(data)
                                elif odds < -200:
                                    segments['Strong Favorite (-300 ~ -200)'].append(data)
                                elif odds < -150:
                                    segments['Moderate Favorite (-200 ~ -150)'].append(data)
                                elif odds < -120:
                                    segments['Light Favorite (-150 ~ -120)'].append(data)
                                elif odds < 100:  # -120 ~ Even
                                    segments['Near Favorite (-120 ~ Even)'].append(data)
                                elif odds <= 120:  # Even ~ +120
                                    segments['Near Underdog (Even ~ +120)'].append(data)
                                elif odds <= 140:  # +120 ~ +140
                                    segments['Light Underdog A (+120 ~ +140)'].append(data)
                                elif odds <= 160:  # +140 ~ +160
                                    segments['Light Underdog B (+140 ~ +160)'].append(data)
                                elif odds <= 180:  # +160 ~ +180
                                    segments['Moderate Underdog A (+160 ~ +180)'].append(data)
                                elif odds <= 200:  # +180 ~ +200
                                    segments['Moderate Underdog B (+180 ~ +200)'].append(data)
                                elif odds <= 220:  # +200 ~ +220
                                    segments['Moderate Underdog C (+200 ~ +220)'].append(data)
                                elif odds <= 300:
                                    segments['Strong Underdog (+220 ~ +300)'].append(data)
                                else:
                                    segments['Heavy Underdog (> +300)'].append(data)
                            
                            return calculate_segment_performance_telegram(segments)
                        
                        def analyze_model_detailed_odds_segments_telegram(matched_data, model_name):
                            """개별 모델의 세분화된 배당률 구간별 성과 분석 - 홈/어웨이 분리"""
                            prob_key = f'{model_name}_probability'
                            
                            # 해당 모델의 데이터만 필터링
                            model_data = [r for r in matched_data if r.get(prob_key) is not None and r.get(prob_key) > 0]
                            
                            if not model_data:
                                return None, None
                            
                            # 홈팀/어웨이팀 분석 데이터 준비
                            home_analysis_data = []
                            away_analysis_data = []
                            
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
                                unit_amount = 100  # 분석 단위
                                
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
                                
                                confidence = abs(prob - 0.5)  # 신뢰도 계산
                                
                                analysis_item = {
                                    'ensemble_prob': prob,
                                    'predicted_roi': predicted_roi,
                                    'actual_roi': actual_roi,
                                    'confidence': confidence,
                                    'selection_odds': selection_odds,
                                    'predicted_team': predicted_team,
                                    'actual_win': actual_home_win,
                                    'home_odds': home_odds,
                                    'away_odds': away_odds
                                }
                                
                                # 홈/어웨이에 따라 분리
                                if predicted_team == 'home':
                                    home_analysis_data.append(analysis_item)
                                else:
                                    away_analysis_data.append(analysis_item)
                            
                            # 각각에 대해 구간별 분석 수행
                            home_segments = None
                            away_segments = None
                            
                            if home_analysis_data:
                                home_segments = {
                                    'detailed_odds': analyze_detailed_odds_segments_telegram(home_analysis_data)
                                }
                            
                            if away_analysis_data:
                                away_segments = {
                                    'detailed_odds': analyze_detailed_odds_segments_telegram(away_analysis_data)
                                }
                            
                            return home_segments, away_segments
                        
                        def display_detailed_odds_segment_performance_table_telegram(segment_data, segment_type):
                            """세분화된 배당률 구간별 성과 테이블 표시"""
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
                            
                            # ROI에 따른 색상 함수 (더 세분화된 색상)
                            def highlight_detailed_odds_roi(val):
                                if val > 15:
                                    return 'background-color: #28a745; color: white; font-weight: bold'  # 진한 초록
                                elif val > 5:
                                    return 'background-color: #d4edda; color: #155724; font-weight: bold'  # 연한 초록
                                elif val > 0:
                                    return 'background-color: #d4edda; color: #155724'  # 매우 연한 초록
                                elif val > -10:
                                    return 'background-color: #fff3cd; color: #856404'  # 노랑
                                else:
                                    return 'background-color: #f8d7da; color: #721c24'  # 빨강
                            
                            # 테이블 스타일링
                            styled_df = df.style.map(
                                highlight_detailed_odds_roi, 
                                subset=['Actual ROI (%)']
                            ).format({
                                'Predicted ROI (%)': '{:.2f}%',
                                'Actual ROI (%)': '{:.2f}%',
                                'ROI Difference (%)': '{:.2f}%',
                                'Win Rate (%)': '{:.1f}%',
                                'Accuracy (%)': '{:.1f}%'
                            })
                            
                            st.dataframe(styled_df, use_container_width=True, hide_index=True)
                            
                            # 추가 인사이트 표시
                            if table_data:
                                best_segment = max(table_data, key=lambda x: x['Actual ROI (%)'])
                                worst_segment = min(table_data, key=lambda x: x['Actual ROI (%)'])
                                most_games = max(table_data, key=lambda x: x['Games'])
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(
                                        "🏆 Best Segment", 
                                        best_segment[segment_type].split('(')[0].strip(),
                                        f"{best_segment['Actual ROI (%)']:.1f}%"
                                    )
                                with col2:
                                    st.metric(
                                        "📉 Worst Segment", 
                                        worst_segment[segment_type].split('(')[0].strip(),
                                        f"{worst_segment['Actual ROI (%)']:.1f}%"
                                    )
                                with col3:
                                    st.metric(
                                        "📊 Most Active", 
                                        most_games[segment_type].split('(')[0].strip(),
                                        f"{most_games['Games']} games"
                                    )
                        
                        # Detailed Odds Segment Analysis UI
                        if results.get('model_performances') and results.get('daily_performances'):
                            # 모델 선택 (세분화된 배당률 구간 분석용)
                            detailed_odds_segment_models = sorted(results['model_performances'].keys())
                            selected_detailed_odds_segment_model = st.selectbox(
                                "🎯 Select model for detailed odds segment analysis:",
                                detailed_odds_segment_models,
                                key="detailed_odds_segment_analysis_model"
                            )
                            
                            if selected_detailed_odds_segment_model:
                                with st.spinner(f"🔍 Analyzing detailed odds segments for {selected_detailed_odds_segment_model}..."):
                                    try:
                                        # SimpleModelAnalyzer를 통해 데이터 로드 및 매칭
                                        analyzer = load_analyzer(selected_prefix)
                                        data = analyzer.load_data(start_date_str, end_date_str)
                                        matched_data = analyzer.match_predictions_with_results(
                                            data['predictions'], 
                                            data['historical_records']
                                        )
                                        
                                        if matched_data:
                                            # 세분화된 배당률 구간 분석 수행
                                            home_detailed_odds_results, away_detailed_odds_results = analyze_model_detailed_odds_segments_telegram(matched_data, selected_detailed_odds_segment_model)
                                            
                                            if home_detailed_odds_results or away_detailed_odds_results:
                                                st.markdown("### 💰 Home/Away Detailed Odds Range Performance Analysis")
                                                st.caption("Comparison of home vs away team picks across 13 detailed betting odds segments for precision market insights.")
                                                
                                                # 홈팀 데이터 표시
                                                st.markdown("#### 🏠 Home Team Picks - Detailed Odds Range")
                                                if home_detailed_odds_results and home_detailed_odds_results['detailed_odds']:
                                                    display_detailed_odds_segment_performance_table_telegram(
                                                        home_detailed_odds_results['detailed_odds'], 
                                                        "Detailed Odds Range"
                                                    )
                                                else:
                                                    st.warning("No home team detailed odds data available.")
                                                
                                                # 어웨이팀 데이터 표시
                                                st.markdown("#### 🛫 Away Team Picks - Detailed Odds Range")
                                                if away_detailed_odds_results and away_detailed_odds_results['detailed_odds']:
                                                    display_detailed_odds_segment_performance_table_telegram(
                                                        away_detailed_odds_results['detailed_odds'], 
                                                        "Detailed Odds Range"
                                                    )
                                                else:
                                                    st.warning("No away team detailed odds data available.")
                                            else:
                                                st.warning(f"No detailed odds segment analysis data available for {selected_detailed_odds_segment_model}")
                                        else:
                                            st.warning("No matched data available for detailed odds segment analysis")
                                            
                                    except Exception as e:
                                        st.error(f"Error in detailed odds segment analysis: {str(e)}")
                        else:
                            st.info("Model performance data required for detailed odds segment analysis")
                        
                        # 🆕 Pure Home/Away Odds Performance Analysis
                        st.markdown("---")
                        st.markdown("## 🏠🛫💰 Pure Home/Away Odds Performance Analysis")
                        st.info("Model-independent analysis: Performance of betting on home/away teams purely based on their odds ranges, regardless of model predictions.")
                        
                        # 순수 홈/어웨이 배당률 분석 관련 함수들 추가
                        def analyze_pure_home_away_odds_segments_telegram(matched_data):
                            """모델 예측과 관계없이 순수한 홈/어웨이 + 배당률 구간별 성과 분석"""
                            
                            if not matched_data:
                                return None, None
                            
                            # 홈팀 분석 데이터 (모든 경기에서 홈팀에 베팅)
                            home_analysis_data = []
                            # 어웨이팀 분석 데이터 (모든 경기에서 어웨이팀에 베팅)
                            away_analysis_data = []
                            
                            for record in matched_data:
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
                                
                                unit_amount = 100  # 분석 단위
                                
                                # === 홈팀 베팅 분석 ===
                                # 홈팀 배당률로 ROI 계산
                                if home_odds > 0:
                                    home_decimal_odds = (home_odds / 100) + 1
                                else:
                                    home_decimal_odds = (100 / abs(home_odds)) + 1
                                
                                # 홈팀 베팅 결과
                                home_is_correct = actual_home_win == 1
                                if home_is_correct:
                                    if home_odds > 0:
                                        home_profit = unit_amount * (home_odds / 100)
                                    else:
                                        home_profit = unit_amount * (100 / abs(home_odds))
                                    home_actual_roi = (home_profit / unit_amount) * 100
                                else:
                                    home_actual_roi = -100  # 전액 손실
                                
                                home_analysis_data.append({
                                    'selection_odds': home_odds,
                                    'actual_roi': home_actual_roi,
                                    'is_correct': home_is_correct,
                                    'team_type': 'home'
                                })
                                
                                # === 어웨이팀 베팅 분석 ===
                                # 어웨이팀 배당률로 ROI 계산
                                if away_odds > 0:
                                    away_decimal_odds = (away_odds / 100) + 1
                                else:
                                    away_decimal_odds = (100 / abs(away_odds)) + 1
                                
                                # 어웨이팀 베팅 결과
                                away_is_correct = actual_home_win == 0
                                if away_is_correct:
                                    if away_odds > 0:
                                        away_profit = unit_amount * (away_odds / 100)
                                    else:
                                        away_profit = unit_amount * (100 / abs(away_odds))
                                    away_actual_roi = (away_profit / unit_amount) * 100
                                else:
                                    away_actual_roi = -100  # 전액 손실
                                
                                away_analysis_data.append({
                                    'selection_odds': away_odds,
                                    'actual_roi': away_actual_roi,
                                    'is_correct': away_is_correct,
                                    'team_type': 'away'
                                })
                            
                            # 각각에 대해 배당률 구간별 분석 수행
                            home_segments = None
                            away_segments = None
                            
                            if home_analysis_data:
                                home_segments = analyze_pure_odds_segments_telegram(home_analysis_data)
                            
                            if away_analysis_data:
                                away_segments = analyze_pure_odds_segments_telegram(away_analysis_data)
                            
                            return home_segments, away_segments
                        
                        def analyze_pure_odds_segments_telegram(analysis_data):
                            """순수 배당률 구간별 성과 분석"""
                            segments = {
                                'Heavy Favorite (< -300)': [],
                                'Strong Favorite (-300 ~ -200)': [],
                                'Moderate Favorite (-200 ~ -150)': [],
                                'Light Favorite (-150 ~ -120)': [],
                                'Near Favorite (-120 ~ Even)': [],
                                'Near Underdog (Even ~ +120)': [],
                                'Light Underdog A (+120 ~ +140)': [],
                                'Light Underdog B (+140 ~ +160)': [],
                                'Moderate Underdog A (+160 ~ +180)': [],
                                'Moderate Underdog B (+180 ~ +200)': [],
                                'Moderate Underdog C (+200 ~ +220)': [],
                                'Strong Underdog (+220 ~ +300)': [],
                                'Heavy Underdog (> +300)': []
                            }
                            
                            for data in analysis_data:
                                odds = data['selection_odds']
                                
                                if odds < -300:
                                    segments['Heavy Favorite (< -300)'].append(data)
                                elif odds < -200:
                                    segments['Strong Favorite (-300 ~ -200)'].append(data)
                                elif odds < -150:
                                    segments['Moderate Favorite (-200 ~ -150)'].append(data)
                                elif odds < -120:
                                    segments['Light Favorite (-150 ~ -120)'].append(data)
                                elif odds < 100:  # -120 ~ Even
                                    segments['Near Favorite (-120 ~ Even)'].append(data)
                                elif odds <= 120:  # Even ~ +120
                                    segments['Near Underdog (Even ~ +120)'].append(data)
                                elif odds <= 140:  # +120 ~ +140
                                    segments['Light Underdog A (+120 ~ +140)'].append(data)
                                elif odds <= 160:  # +140 ~ +160
                                    segments['Light Underdog B (+140 ~ +160)'].append(data)
                                elif odds <= 180:  # +160 ~ +180
                                    segments['Moderate Underdog A (+160 ~ +180)'].append(data)
                                elif odds <= 200:  # +180 ~ +200
                                    segments['Moderate Underdog B (+180 ~ +200)'].append(data)
                                elif odds <= 220:  # +200 ~ +220
                                    segments['Moderate Underdog C (+200 ~ +220)'].append(data)
                                elif odds <= 300:
                                    segments['Strong Underdog (+220 ~ +300)'].append(data)
                                else:
                                    segments['Heavy Underdog (> +300)'].append(data)
                            
                            return calculate_pure_segment_performance_telegram(segments)
                        
                        def calculate_pure_segment_performance_telegram(segments):
                            """순수 구간별 성과 계산"""
                            results = {}
                            
                            for segment_name, segment_data in segments.items():
                                if not segment_data:
                                    results[segment_name] = {
                                        'games': 0,
                                        'actual_roi': 0,
                                        'win_rate': 0,
                                        'accuracy': 0
                                    }
                                    continue
                                
                                games = len(segment_data)
                                actual_roi = sum(d['actual_roi'] for d in segment_data) / games
                                
                                # 승률 계산 (실제 수익이 난 경우)
                                wins = sum(1 for d in segment_data if d['actual_roi'] > 0)
                                win_rate = (wins / games) * 100 if games > 0 else 0
                                
                                # 정확도 계산 (베팅한 팀이 실제로 이긴 경우)
                                correct_predictions = sum(1 for d in segment_data if d['is_correct'])
                                accuracy = (correct_predictions / games) * 100 if games > 0 else 0
                                
                                results[segment_name] = {
                                    'games': games,
                                    'actual_roi': actual_roi,
                                    'win_rate': win_rate,
                                    'accuracy': accuracy
                                }
                            
                            return results
                        
                        def display_pure_home_away_odds_performance_table_telegram(home_segment_data, away_segment_data, segment_type):
                            """순수 홈/어웨이 배당률 구간별 성과 테이블 표시"""
                            
                            # 홈팀 데이터 표시
                            st.markdown(f"#### 🏠 Home Team Performance by {segment_type}")
                            st.caption("Performance of betting on home teams based purely on their odds ranges")
                            if home_segment_data and any(data['games'] > 0 for data in home_segment_data.values()):
                                # 홈팀 데이터 정리
                                home_table_data = []
                                for segment_name, data in home_segment_data.items():
                                    if data['games'] > 0:
                                        home_table_data.append({
                                            segment_type: segment_name,
                                            'Games': data['games'],
                                            'Actual ROI (%)': data['actual_roi'],
                                            'Win Rate (%)': data['win_rate'],
                                            'Accuracy (%)': data['accuracy']
                                        })
                                
                                if home_table_data:
                                    home_df = pd.DataFrame(home_table_data)
                                    
                                    # ROI에 따른 색상 함수
                                    def highlight_pure_roi(val):
                                        if val > 10:
                                            return 'background-color: #28a745; color: white; font-weight: bold'
                                        elif val > 0:
                                            return 'background-color: #d4edda; color: #155724'
                                        elif val > -15:
                                            return 'background-color: #fff3cd; color: #856404'
                                        else:
                                            return 'background-color: #f8d7da; color: #721c24'
                                    
                                    # 홈팀 테이블 스타일링
                                    styled_home_df = home_df.style.map(
                                        highlight_pure_roi, 
                                        subset=['Actual ROI (%)']
                                    ).format({
                                        'Actual ROI (%)': '{:.2f}%',
                                        'Win Rate (%)': '{:.1f}%',
                                        'Accuracy (%)': '{:.1f}%'
                                    })
                                    
                                    st.dataframe(styled_home_df, use_container_width=True, hide_index=True)
                                else:
                                    st.warning(f"No home team {segment_type} data with games available.")
                            else:
                                st.warning(f"No home team {segment_type} data available.")
                            
                            # 어웨이팀 데이터 표시
                            st.markdown(f"#### 🛫 Away Team Performance by {segment_type}")
                            st.caption("Performance of betting on away teams based purely on their odds ranges")
                            if away_segment_data and any(data['games'] > 0 for data in away_segment_data.values()):
                                # 어웨이팀 데이터 정리
                                away_table_data = []
                                for segment_name, data in away_segment_data.items():
                                    if data['games'] > 0:
                                        away_table_data.append({
                                            segment_type: segment_name,
                                            'Games': data['games'],
                                            'Actual ROI (%)': data['actual_roi'],
                                            'Win Rate (%)': data['win_rate'],
                                            'Accuracy (%)': data['accuracy']
                                        })
                                
                                if away_table_data:
                                    away_df = pd.DataFrame(away_table_data)
                                    
                                    # 어웨이팀 테이블 스타일링
                                    styled_away_df = away_df.style.map(
                                        highlight_pure_roi, 
                                        subset=['Actual ROI (%)']
                                    ).format({
                                        'Actual ROI (%)': '{:.2f}%',
                                        'Win Rate (%)': '{:.1f}%',
                                        'Accuracy (%)': '{:.1f}%'
                                    })
                                    
                                    st.dataframe(styled_away_df, use_container_width=True, hide_index=True)
                                    
                                    # 추가 인사이트: 홈 vs 어웨이 비교
                                    if home_table_data:
                                        st.markdown("#### 📊 Home vs Away Comparison Insights")
                                        
                                        # 같은 구간별로 홈/어웨이 비교
                                        comparison_data = []
                                        for home_row in home_table_data:
                                            segment = home_row[segment_type]
                                            home_roi = home_row['Actual ROI (%)']
                                            home_games = home_row['Games']
                                            
                                            # 같은 구간의 어웨이 데이터 찾기
                                            away_row = next((row for row in away_table_data if row[segment_type] == segment), None)
                                            if away_row:
                                                away_roi = away_row['Actual ROI (%)']
                                                away_games = away_row['Games']
                                                roi_diff = home_roi - away_roi
                                                
                                                comparison_data.append({
                                                    'Odds Range': segment.replace(' (', '\n('),
                                                    'Home ROI (%)': home_roi,
                                                    'Away ROI (%)': away_roi,
                                                    'Home Advantage': roi_diff,
                                                    'Total Games': home_games + away_games
                                                })
                                        
                                        if comparison_data:
                                            # 홈 어드밴티지 차트
                                            comp_df = pd.DataFrame(comparison_data)
                                            
                                            fig_comparison = px.bar(
                                                comp_df,
                                                x='Odds Range',
                                                y='Home Advantage',
                                                color='Home Advantage',
                                                title='Home Field Advantage by Odds Range',
                                                color_continuous_scale=['red', 'white', 'green'],
                                                color_continuous_midpoint=0,
                                                text='Total Games'
                                            )
                                            
                                            fig_comparison.update_traces(texttemplate='%{text} games', textposition='outside')
                                            fig_comparison.update_layout(
                                                height=400,
                                                xaxis_tickangle=-45,
                                                showlegend=False,
                                                xaxis_title="Odds Range",
                                                yaxis_title="Home Advantage (Home ROI - Away ROI)"
                                            )
                                            fig_comparison.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                                            
                                            st.plotly_chart(fig_comparison, use_container_width=True)
                                else:
                                    st.warning(f"No away team {segment_type} data with games available.")
                            else:
                                st.warning(f"No away team {segment_type} data available.")
                        
                        # Pure Home/Away Odds Analysis UI
                        if results.get('model_performances') and results.get('daily_performances'):
                            st.markdown("### 🔍 Pure Odds-Based Analysis")
                            st.caption("This analysis is completely independent of model predictions. It shows what would happen if you bet on all home teams or all away teams in specific odds ranges.")
                            
                            with st.spinner("🔍 Analyzing pure home/away odds performance..."):
                                try:
                                    # SimpleModelAnalyzer를 통해 데이터 로드 및 매칭
                                    analyzer = load_analyzer(selected_prefix)
                                    data = analyzer.load_data(start_date_str, end_date_str)
                                    matched_data = analyzer.match_predictions_with_results(
                                        data['predictions'], 
                                        data['historical_records']
                                    )
                                    
                                    if matched_data:
                                        # 순수 홈/어웨이 배당률 분석 수행
                                        pure_home_segments, pure_away_segments = analyze_pure_home_away_odds_segments_telegram(matched_data)
                                        
                                        if pure_home_segments or pure_away_segments:
                                            st.markdown("### 💰 Pure Home/Away Odds Performance Analysis")
                                            st.caption("Model-independent performance analysis of betting purely on home/away teams by odds ranges. Light Underdog and Moderate Underdog ranges are subdivided into 20-point segments for precision analysis.")
                                            
                                            # 순수 홈/어웨이 배당률 구간 결과 표시
                                            display_pure_home_away_odds_performance_table_telegram(
                                                pure_home_segments,
                                                pure_away_segments,
                                                "Pure Odds Range"
                                            )
                                        else:
                                            st.warning("No pure home/away odds analysis data available")
                                    else:
                                        st.warning("No matched data available for pure home/away odds analysis")
                                        
                                except Exception as e:
                                    st.error(f"Error in pure home/away odds analysis: {str(e)}")
                        else:
                            st.info("Model performance data required for pure home/away odds analysis")
                        
                        # 🆕 Performance Report Generator
                        st.markdown("---")
                        st.markdown("## 📱 Model Performance Report Generator")
                        st.info("Generate telegram-ready performance report for the best model in selected period")
                        
                        col1, col2, col3 = st.columns([1, 1, 2])
                        
                        with col1:
                            st.info(f"Analysis Period: {start_date_str} to {end_date_str}")
                            st.info("Will generate report for the best performing model in this period")
                        
                        with col2:
                            if st.button("📱 Generate Performance Report", type="primary"):
                                if results.get('model_performances') and results.get('daily_performances'):
                                    # 전체 기간 최고 모델 찾기
                                    best_model_name = max(
                                        results['model_performances'].items(),
                                        key=lambda x: x[1].get('actual_roi', 0)
                                    )[0]
                                    
                                    performance_report = hub.generate_model_performance_report(
                                        results['model_performances'],
                                        results['daily_performances'],
                                        best_model_name,
                                        start_date_str,
                                        end_date_str
                                    )
                                    st.session_state.performance_report = performance_report
                                else:
                                    st.error("No performance data available")
                        
                        # 생성된 리포트 표시
                        if 'performance_report' in st.session_state:
                            st.markdown("### 📋 Generated Performance Report")
                            st.code(st.session_state.performance_report, language=None)
                            
                            # 복사 버튼 안내
                            st.success("✅ Report generated! Copy the text above and paste it into Telegram.")
                    else:
                        st.warning("No model data available for the selected period.")
                        
                except Exception as e:
                    st.error(f"Error analyzing performance: {str(e)}")
                    st.info("Please check if the data files are available for the selected date range.")
        else:
            st.error("Start date must be before end date")
    
    # === 탭 2: 픽 분석 ===
    with tab2:
        st.header("🎯 Statistical Insights Generator")
        st.info("Generate statistical insights and telegram messages for today's preferred matches")
        
        # 최신 데이터 로드
        with st.spinner("🔍 Loading latest prediction data..."):
            latest_data, file_path = hub.load_latest_prediction_data()
        
        if latest_data is None:
            st.error("❌ Unable to load prediction data")
            return
        
        st.success(f"✅ Loaded {len(latest_data)} games from {Path(file_path).name}")
        
        # 2열 레이아웃: 설정 | 결과
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### ⚙️ Statistical Configuration")
            
            # 모델 가중치 설정
            st.markdown("#### 🤖 Model Weights")
            weights = {}
            
            # 주요 모델들 (상단 표시)
            main_models = {
                'SVM': 'model_svm',
                'Neural Network': 'model_nn', 
                'Random Forest': 'model_rf',
                'Advanced LightGBM': 'model_advanced_lgbm',
                'Advanced CatBoost': 'model_advanced_catboost',
                'Advanced XGBoost': 'model_advanced_xgboost'
            }
            
            for display_name, model_name in main_models.items():
                default_val = 1.0 if model_name == 'model_svm' else 0.0  # SVM 기본값만 유지
                weight = st.slider(
                    display_name,
                    0.0, 1.0, default_val, 0.05,
                    key=f"weight_{model_name}"
                )
                weights[model_name] = weight
            
            # 기타 모델들 (축소 표시)
            with st.expander("🔧 Additional Models", expanded=False):
                other_models = [m for m in hub.all_models if m not in main_models.values()]
                for model in other_models:
                    display_name = model.replace('model_', '').replace('_', ' ').title()
                    weight = st.slider(
                        display_name,
                        0.0, 1.0, 0.0, 0.05,
                        key=f"weight_{model}"
                    )
                    weights[model] = weight
            
            # 가중치 정규화
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            # 활성 모델 표시
            active_weights = {k: v for k, v in weights.items() if v > 0}
            if active_weights:
                st.markdown("**📊 Active Models:**")
                for model, weight in active_weights.items():
                    display_name = model.replace('model_', '').replace('_', ' ').title()
                    st.markdown(f'<div class="model-weight">• {display_name}: {weight:.2f}</div>', 
                              unsafe_allow_html=True)
            
            # Zone 선택
            st.markdown("#### 🎯 Selection Zones")
            st.caption("Same dimension = OR, Different dimensions = AND")
            
            selected_zones = {}
            
            zone_titles = {
                'predicted_roi': '📊 Predicted ROI',
                'odds': '💰 Odds Ranges',
                'confidence': '🎯 Confidence Levels',
                'odds_probability_divergence': '📈 Market vs Model',
                'kelly_criterion': '🎰 Kelly Criterion',
                'model_consensus': '🤝 Model Consensus'
            }
            
            for dimension, title in zone_titles.items():
                with st.expander(title, expanded=False):
                    selected_segments = []
                    
                    for option in hub.zone_options[dimension]:
                        if st.checkbox(option, key=f"zone_{dimension}_{option}"):
                            selected_segments.append(option)
                    
                    if selected_segments:
                        selected_zones[dimension] = selected_segments
        
        with col2:
            st.markdown("### 📊 Analysis Results")
            
            if selected_zones and active_weights:
                # 매칭 게임 찾기
                with st.spinner("🔍 Finding matching games..."):
                    matching_games = hub.find_custom_zone_matches(latest_data, selected_zones, weights)
                
                if matching_games:
                    # 결과 요약
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Matching Games", len(matching_games))
                    with col_b:
                        avg_roi = sum(g['predicted_roi'] for g in matching_games) / len(matching_games)
                        st.metric("Avg Expected ROI", f"{avg_roi:+.1f}%")
                    with col_c:
                        avg_conf = sum(g['confidence'] for g in matching_games) / len(matching_games)
                        st.metric("Avg Confidence", f"{avg_conf:.3f}")
                    
                    # 텔레그램 메시지 생성
                    telegram_message = hub.generate_telegram_message(matching_games, weights, selected_zones)
                    
                    st.markdown("### 📱 Telegram Message")
                    st.markdown(f'<div class="telegram-message">{telegram_message}</div>', 
                              unsafe_allow_html=True)
                    
                    # 복사용 텍스트 영역
                    st.text_area("📋 Copy for Telegram:", telegram_message, height=200)
                    
                    # 상세 게임 정보
                    st.markdown("### 📋 Detailed Analysis")
                    
                    # 픽 필터링 기능 추가
                    with st.expander("🔧 Filter Picks (Optional)", expanded=False):
                        st.markdown("**Remove specific teams from the analysis:**")
                        exclude_teams = st.text_input(
                            "Enter team names to exclude (comma-separated):",
                            placeholder="e.g., Twins, Yankees, Dodgers",
                            help="Type team names you want to exclude from the picks. Use partial names (e.g., 'Twins' for 'Minnesota Twins')"
                        )
                        
                        # 제외할 팀 리스트 처리
                        excluded_team_list = []
                        if exclude_teams.strip():
                            excluded_team_list = [team.strip().lower() for team in exclude_teams.split(',') if team.strip()]
                    
                    # 픽 필터링 적용
                    filtered_matching_games = []
                    excluded_count = 0
                    
                    for game in matching_games:
                        should_exclude = False
                        game_info_lower = game['game_info'].lower()
                        selection_team_lower = game['selection_team_name'].lower()
                        
                        # 제외할 팀 체크
                        for excluded_team in excluded_team_list:
                            if excluded_team in game_info_lower or excluded_team in selection_team_lower:
                                should_exclude = True
                                excluded_count += 1
                                break
                        
                        if not should_exclude:
                            filtered_matching_games.append(game)
                    
                    # 필터링 결과 표시
                    if excluded_count > 0:
                        st.info(f"🔧 Filtered out {excluded_count} picks. Showing {len(filtered_matching_games)} remaining picks.")
                    
                    # 픽 저장 버튼
                    col_save1, col_save2, col_save3 = st.columns([1, 1, 2])
                    
                    with col_save1:
                        if st.button("💾 Save Picks to JSON", type="primary", key="save_picks_btn"):
                            # 필터링된 게임들로 저장
                            filepath, result = hub.save_picks_to_json(filtered_matching_games, weights, selected_zones)
                            
                            if filepath:
                                st.success(f"✅ Saved {result} picks to: {filepath.name}")
                                st.info(f"📁 Full path: {filepath}")
                            else:
                                st.error(f"❌ Error saving picks: {result}")
                    
                    with col_save2:
                        st.info(f"📊 {len(filtered_matching_games)} picks ready to save")
                    
                    # 픽 카드들 표시 (필터링된 게임들)
                    for i, game in enumerate(filtered_matching_games, 1):
                        st.markdown(f"""
                        <div class="pick-detail">
                            <strong>#{i} {game['game_info']}</strong><br>
                            <div class="analysis-result">
                                🎯 Analysis Result: {game['selection_team_name']} ({game['selection_odds']:+})
                            </div>
                            🎲 Statistical Probability: {game['win_prob']:.1%} | 📈 ROI Projection: {game['predicted_roi']:+.1f}% | 
                            💡 Confidence Level: {game['confidence']:.3f}<br>
                            ⏰ {game['start_time']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("❌ No games match your selected criteria")
                    st.info("💡 Try adjusting your zone selections or model weights")
            else:
                st.info("👆 Please configure model weights and select zones to generate analysis")
    
    # 하단 면책조항 (재차 강조)
    st.markdown("---")
    st.warning("""
    📋 **Final Reminder**: This analysis is for **educational purposes only**. 
    Any use of this information is **at your own risk**. Always **verify independently** 
    and **consult professionals** before making financial decisions.
    """)

if __name__ == "__main__":
    main() 