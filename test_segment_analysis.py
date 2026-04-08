#!/usr/bin/env python3
"""구간분석 로직 검증 스크립트"""

import json
import sys
sys.path.append('src')

def test_segment_analysis_logic():
    """구간분석 로직 검증"""
    
    print("=== 구간분석 로직 검증 ===\n")
    
    # 샘플 데이터 로드
    with open('src/odds/data/matched/25_mlb_predictions_with_odds_20250602_200543.json', 'r') as f:
        sample_data = json.load(f)
    
    print(f"📊 총 {len(sample_data)}개 게임 데이터 로드됨\n")
    
    # 1. 확률과 예측팀 일치성 검증
    print("=== 1. 확률과 예측팀 일치성 검증 ===")
    
    mismatch_count = 0
    total_checked = 0
    
    for i, game in enumerate(sample_data[:10]):  # 처음 10개 확인
        ensemble_prob = game.get('ensemble_probability', 0)
        predicted_winner = game.get('predicted_winner', '')
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        
        # 로직에 따른 예측팀
        should_be_winner = home_team if ensemble_prob > 0.5 else away_team
        is_consistent = predicted_winner == should_be_winner
        
        if not is_consistent:
            mismatch_count += 1
            print(f"❌ 게임 {i+1}: {home_team} vs {away_team}")
            print(f"   홈팀 승률: {ensemble_prob:.3f}")
            print(f"   예측 승자: {predicted_winner}")
            print(f"   로직상 승자: {should_be_winner}")
        
        total_checked += 1
    
    print(f"\n일치성 검증 결과: {total_checked - mismatch_count}/{total_checked} 일치 ({'✅' if mismatch_count == 0 else '❌'})\n")
    
    # 2. 배당률 null 처리 검증
    print("=== 2. 배당률 null 처리 검증 ===")
    
    null_odds_count = 0
    valid_odds_count = 0
    
    for game in sample_data:
        home_odds = game.get('home_team_odds')
        away_odds = game.get('away_team_odds')
        
        if home_odds is None or away_odds is None:
            null_odds_count += 1
        else:
            valid_odds_count += 1
    
    print(f"배당률 있는 게임: {valid_odds_count}개")
    print(f"배당률 없는 게임: {null_odds_count}개")
    print(f"배당률 비율: {valid_odds_count/(valid_odds_count+null_odds_count)*100:.1f}%\n")
    
    # 3. 미국식 배당률 구간 분석 개선 검증
    print("=== 3. 미국식 배당률 구간 분석 개선 검증 ===")
    
    # 미국식 배당률 구간 분석 시뮬레이션
    odds_segments = {
        'Heavy Favorite (< -200)': [],
        'Favorite (-200 ~ -120)': [],
        'Pick Em (-120 ~ +120)': [],
        'Underdog (+120 ~ +300)': [],
        'Heavy Underdog (> +300)': []
    }
    
    for i, game in enumerate(sample_data[:10]):
        ensemble_prob = game.get('ensemble_probability', 0)
        home_odds = game.get('home_team_odds')
        away_odds = game.get('away_team_odds')
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        
        if home_odds is None or away_odds is None:
            print(f"게임 {i+1}: 배당률 없음 - 제외됨")
            continue
            
        # 베팅 결정
        if ensemble_prob > 0.5:
            bet_odds = home_odds
            bet_team = "홈팀"
        else:
            bet_odds = away_odds
            bet_team = "원정팀"
            
        # 구간 분류 (수정된 로직)
        if bet_odds < -200:
            segment = 'Heavy Favorite (< -200)'
        elif bet_odds < -120:
            segment = 'Favorite (-200 ~ -120)'
        elif -120 <= bet_odds <= 120:
            segment = 'Pick Em (-120 ~ +120)'
        elif bet_odds <= 300:
            segment = 'Underdog (+120 ~ +300)'
        else:
            segment = 'Heavy Underdog (> +300)'
            
        print(f"게임 {i+1}: {home_team} vs {away_team}")
        print(f"  베팅: {bet_team}, 배당: {bet_odds}")
        print(f"  구간: {segment}")
        
        odds_segments[segment].append({
            'game': f"{home_team} vs {away_team}",
            'bet_odds': bet_odds,
            'bet_team': bet_team
        })
    
    print(f"\n=== 배당률 구간별 분포 요약 ===")
    for segment, games in odds_segments.items():
        if games:
            odds_values = [g['bet_odds'] for g in games]
            print(f"{segment}: {len(games)}개 (배당률 범위: {min(odds_values)} ~ {max(odds_values)})")
        else:
            print(f"{segment}: 0개")
    
    # 4. 새로운 분석: 배당률 vs 확률 괴리도 테스트
    print(f"\n=== 4. 배당률 vs 확률 괴리도 분석 테스트 ===")
    
    divergence_segments = {
        'Model Much More Optimistic (+10%+)': [],
        'Model Slightly Optimistic (+5% ~ +10%)': [],
        'Market Aligned (-5% ~ +5%)': [],
        'Model Slightly Pessimistic (-10% ~ -5%)': [],
        'Model Much More Pessimistic (-10%--)': []
    }
    
    for i, game in enumerate(sample_data[:5]):  # 5개만 테스트
        ensemble_prob = game.get('ensemble_probability', 0)
        home_odds = game.get('home_team_odds')
        away_odds = game.get('away_team_odds')
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        
        if home_odds is None or away_odds is None:
            print(f"게임 {i+1}: 배당률 없음 - 제외됨")
            continue
        
        # 베팅 결정
        if ensemble_prob > 0.5:
            bet_odds = home_odds
            bet_team = "홈팀"
            predicted_team = "home"
        else:
            bet_odds = away_odds
            bet_team = "원정팀"
            predicted_team = "away"
        
        # 시장 배당률을 확률로 변환
        if bet_odds > 0:
            market_implied_prob = 100 / (bet_odds + 100)
        else:
            market_implied_prob = abs(bet_odds) / (abs(bet_odds) + 100)
        
        # 모델 확률 (베팅 팀 기준)
        if predicted_team == 'home':
            model_prob = ensemble_prob
        else:
            model_prob = 1 - ensemble_prob
        
        # 괴리도 계산
        divergence = model_prob - market_implied_prob
        
        # 구간 분류
        if divergence >= 0.10:
            segment = 'Model Much More Optimistic (+10%+)'
        elif divergence >= 0.05:
            segment = 'Model Slightly Optimistic (+5% ~ +10%)'
        elif -0.05 <= divergence < 0.05:
            segment = 'Market Aligned (-5% ~ +5%)'
        elif divergence >= -0.10:
            segment = 'Model Slightly Pessimistic (-10% ~ -5%)'
        else:
            segment = 'Model Much More Pessimistic (-10%--)'
        
        print(f"게임 {i+1}: {home_team} vs {away_team}")
        print(f"  베팅: {bet_team} (배당: {bet_odds})")
        print(f"  모델 확률: {model_prob:.3f} vs 시장 확률: {market_implied_prob:.3f}")
        print(f"  괴리도: {divergence:+.3f} ({divergence*100:+.1f}%)")
        print(f"  구간: {segment}")
        
        divergence_segments[segment].append({
            'game': f"{home_team} vs {away_team}",
            'divergence': divergence,
            'model_prob': model_prob,
            'market_prob': market_implied_prob
        })
    
    print(f"\n=== 괴리도 구간별 분포 요약 ===")
    for segment, games in divergence_segments.items():
        if games:
            divergences = [g['divergence'] for g in games]
            print(f"{segment}: {len(games)}개 (괴리도 범위: {min(divergences):+.3f} ~ {max(divergences):+.3f})")
        else:
            print(f"{segment}: 0개")
    
    # 5. Kelly Criterion 분석 테스트
    print(f"\n=== 5. Kelly Criterion 분석 테스트 ===")
    
    kelly_segments = {
        'No Bet (Kelly ≤ 0%)': [],
        'Small Bet (0% < Kelly ≤ 5%)': [],
        'Medium Bet (5% < Kelly ≤ 15%)': [],
        'Large Bet (15% < Kelly ≤ 25%)': [],
        'Extreme Bet (Kelly > 25%)': []
    }
    
    for i, game in enumerate(sample_data[:5]):  # 5개만 테스트
        ensemble_prob = game.get('ensemble_probability', 0)
        home_odds = game.get('home_team_odds')
        away_odds = game.get('away_team_odds')
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        
        if home_odds is None or away_odds is None:
            print(f"게임 {i+1}: 배당률 없음 - 제외됨")
            continue
        
        # 베팅 결정
        if ensemble_prob > 0.5:
            bet_odds = home_odds
            bet_team = "홈팀"
            predicted_team = "home"
            win_prob = ensemble_prob
        else:
            bet_odds = away_odds
            bet_team = "원정팀"
            predicted_team = "away"
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
        
        # 구간 분류
        if kelly_percentage <= 0:
            segment = 'No Bet (Kelly ≤ 0%)'
        elif kelly_percentage <= 5:
            segment = 'Small Bet (0% < Kelly ≤ 5%)'
        elif kelly_percentage <= 15:
            segment = 'Medium Bet (5% < Kelly ≤ 15%)'
        elif kelly_percentage <= 25:
            segment = 'Large Bet (15% < Kelly ≤ 25%)'
        else:
            segment = 'Extreme Bet (Kelly > 25%)'
        
        print(f"게임 {i+1}: {home_team} vs {away_team}")
        print(f"  베팅: {bet_team} (배당: {bet_odds})")
        print(f"  승률: {win_prob:.3f}, Decimal Odds: {decimal_odds:.3f}")
        print(f"  Kelly %: {kelly_percentage:.1f}%")
        print(f"  구간: {segment}")
        
        kelly_segments[segment].append({
            'game': f"{home_team} vs {away_team}",
            'kelly_percentage': kelly_percentage,
            'win_prob': win_prob,
            'decimal_odds': decimal_odds
        })
    
    print(f"\n=== Kelly Criterion 구간별 분포 요약 ===")
    for segment, games in kelly_segments.items():
        if games:
            kelly_values = [g['kelly_percentage'] for g in games]
            print(f"{segment}: {len(games)}개 (Kelly 범위: {min(kelly_values):.1f}% ~ {max(kelly_values):.1f}%)")
        else:
            print(f"{segment}: 0개")

if __name__ == "__main__":
    test_segment_analysis_logic() 