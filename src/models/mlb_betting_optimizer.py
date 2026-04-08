import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json
from datetime import datetime
import itertools

class MLBBettingOptimizer:
    def __init__(self):
        self.predictions_dir = Path(__file__).parent.parent / "odds" / "data" / "matched"
        self.analysis_dir = Path(__file__).parent.parent / "data" / "analysis"
        self.analysis_dir.mkdir(exist_ok=True, parents=True)
        
        # 기본 설정
        self.min_bet_amount = 5  # 최소 베팅액 $5
        self.max_parlay_size = 5  # 최대 5팀 파라레이
        self.min_roi_threshold = 5.0  # 파라레이 구성에 사용할 최소 ROI 기준 (%)
        self.boost_rates = {
            2: 1.10,  # 10% boost for 2-team parlay
            3: 1.20,  # 20% boost for 3-team parlay
            4: 1.30,  # 30% boost for 4-team parlay
            5: 1.40,  # 40% boost for 5-team parlay
            6: 1.50,  # 50% boost for 6-team parlay
            7: 1.60   # 60% boost for 7-team parlay
        }
        
        # 팀 매치 충돌 체크를 위한 데이터 구조
        self.team_matches = {}  # 팀별 모든 매치 정보 저장
        
    def load_predictions_with_odds(self) -> List[Dict]:
        """최신 예측 결과와 배당률 로드"""
        data_dir = Path(__file__).parent.parent / "odds" / "data" / "matched"
        pred_files = list(data_dir.glob("mlb_predictions_with_odds_*.json"))
        if not pred_files:
            raise FileNotFoundError("예측 결과 파일을 찾을 수 없습니다.")
        
        latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading predictions and odds from: {latest_file}")
        
        with open(latest_file, 'r') as f:
            predictions = json.load(f)
        return predictions
    
    def american_to_decimal_odds(self, american_odds: float) -> float:
        """미국식 배당률을 소수 배당률로 변환"""
        if american_odds is None:
            return 0.0
            
        if american_odds > 0:
            return 1 + (american_odds / 100)
        elif american_odds < 0:
            return 1 + (100 / abs(american_odds))
        else:
            return 1.0  # 0인 경우
    
    def calculate_ev(self, probability: float, decimal_odds: float) -> float:
        """Expected Value 계산 (소수 배당률 사용)"""
        if decimal_odds <= 1.0:  # 유효하지 않은 배당률
            return -1.0
            
        return (probability * (decimal_odds - 1)) - (1 - probability)
    
    def calculate_parlay_odds(self, individual_odds: List[float]) -> float:
        """파라레이 배당률 계산 (소수 배당률)"""
        return np.prod(individual_odds)
    
    def calculate_parlay_probability(self, individual_probs: List[float]) -> float:
        """파라레이 승리 확률 계산"""
        return np.prod(individual_probs)
    
    def add_match(self, date: str, home_team: str, away_team: str):
        """매치 정보 추가 및 팀 충돌 체크를 위한 데이터 구조 업데이트"""
        match_info = (date, home_team, away_team)
        
        # 홈팀의 매치 기록
        if home_team not in self.team_matches:
            self.team_matches[home_team] = []
        self.team_matches[home_team].append(match_info)
        
        # 원정팀의 매치 기록
        if away_team not in self.team_matches:
            self.team_matches[away_team] = []
        self.team_matches[away_team].append(match_info)
    
    def get_all_related_teams(self, team: str) -> Set[str]:
        """특정 팀이 포함된 모든 매치의 모든 팀 반환"""
        related_teams = set()
        for match in self.team_matches.get(team, []):
            _, home, away = match
            related_teams.add(home)
            related_teams.add(away)
        return related_teams
    
    def has_team_match_conflict(self, picks: List[Dict]) -> bool:
        """
        파라레이 내 팀 매치 충돌 검사
        - 같은 팀이 포함된 다른 경기들은 같은 파라레이에 포함될 수 없음
        - 예: '메츠 vs 신시내티'와 '메츠 vs 다저스'는 메츠가 겹치므로 같은 파라레이에 포함 불가
        """
        # 파라레이에 포함된 팀들 추적
        teams_in_parlay = set()
        
        for pick in picks:
            match_parts = pick['match'].split(' vs ')
            away_team = match_parts[0]  # 'Team A vs Team B' 형식에서 Team A가 원정팀
            home_team = match_parts[1]  # 'Team A vs Team B' 형식에서 Team B가 홈팀
            
            # 이미 이 팀들이 파라레이에 포함되어 있는지 확인
            if away_team in teams_in_parlay or home_team in teams_in_parlay:
                return True  # 충돌 발생
            
            # 이 팀들을 파라레이에 추가
            teams_in_parlay.add(away_team)
            teams_in_parlay.add(home_team)
        
        return False  # 충돌 없음

    def generate_parlays(self, single_bets: List[Dict], parlay_size: int) -> List[Dict]:
        """
        주어진 크기의 파라레이 조합 생성
        - 팀 매치 충돌 체크
        - 각 픽의 최대 사용 횟수 제한
        - 각 픽의 최소 사용 횟수 설정 (추가된 기능)
        """
        # EV > 0인 베팅만 선택
        confident_bets = [bet for bet in single_bets if bet['ev'] > 0]
        
        print(f"Total bets with EV > 0: {len(confident_bets)}")
        if len(confident_bets) < parlay_size:
            print(f"Not enough confident bets for {parlay_size}-team parlay")
            return [], []
        
        # 모든 가능한 조합 생성
        all_combos = list(itertools.combinations(range(len(confident_bets)), parlay_size))
        print(f"Total possible {parlay_size}-team combinations: {len(all_combos)}")
        
        # 랜덤하게 섞기
        import random
        random.shuffle(all_combos)
        
        # 각 픽의 사용 횟수 추적 (팔레이 크기별로 차등 설정)
        if parlay_size == 3:
            max_pick_usage = 4 if len(confident_bets) >= 6 else 3
            min_pick_usage = 3 if len(confident_bets) >= 6 else 2
        elif parlay_size == 4:
            max_pick_usage = 4 if len(confident_bets) >= 6 else 3
            min_pick_usage = 3 if len(confident_bets) >= 6 else 2
        elif parlay_size >= 5:
            max_pick_usage = 5 if len(confident_bets) >= 6 else 4
            min_pick_usage = 4 if len(confident_bets) >= 6 else 3
        else:
            max_pick_usage = 4  # 기본값
            min_pick_usage = 3  # 기본값
            
        print(f"Max pick usage for {parlay_size}-team parlays: {max_pick_usage}")
        print(f"Min pick usage for {parlay_size}-team parlays: {min_pick_usage}")
        
        # 최대 반복 횟수 설정
        max_iterations = 10000
        final_parlays = []
        final_usage_data = []
        
        # 모든 픽이 최소 사용 횟수를 충족할 때까지 반복
        for iteration in range(1, max_iterations + 1):
            # 각 반복마다 사용 횟수 초기화
            pick_usage = {i: 0 for i in range(len(confident_bets))}
            
            # 결과 저장용 리스트
            selected_parlays = []
            # 각 픽별 사용 횟수 추적을 위한 변수
            pick_usage_details = {i: {'team': confident_bets[i]['team'], 
                                    'match': confident_bets[i]['match'], 
                                    'date': confident_bets[i]['date'] if 'date' in confident_bets[i] else 'N/A',
                                    'count': 0} for i in range(len(confident_bets))}
            
            # 필터링 결과 추적용 카운터
            counter = {
                'usage_limit': 0,
                'team_conflict': 0,
                'negative_ev': 0,
                'passed': 0
            }
            
            # 제한된 수의 조합만 처리 (디버깅용)
            debug_limit = min(100000, len(all_combos))
            print(f"Processing first {debug_limit} combinations for debugging")
            
            # 조합 처리
            for i, combo in enumerate(all_combos[:debug_limit]):
                # 진행 상황 출력 (100개마다)
                if i % 100 == 0:
                    print(f"Processing combination {i}/{debug_limit}...")
                
                # 사용 제한 확인
                if any(pick_usage[pick_id] >= max_pick_usage for pick_id in combo):
                    counter['usage_limit'] += 1
                    continue
                
                # 파라레이 픽 구성
                parlay_picks = [confident_bets[i] for i in combo]
                
                # 팀 매치 충돌 검사
                if self.has_team_match_conflict(parlay_picks):
                    counter['team_conflict'] += 1
                    continue
                
                # 배당률 및 확률 계산 (소수 배당률 사용)
                odds = self.calculate_parlay_odds([pick['decimal_odds'] for pick in parlay_picks])
                probability = self.calculate_parlay_probability([pick['probability'] for pick in parlay_picks])
                
                # 부스트 적용
                boosted_odds = 1 + ((odds - 1) * self.boost_rates[parlay_size])
                
                # EV 계산
                ev = self.calculate_ev(probability, boosted_odds)
                
                # 디버깅: EV 출력
                if i < 5:  # 처음 5개 조합에 대해서만 상세 정보 출력
                    print(f"Combo {i}: odds={odds:.2f}, boosted={boosted_odds:.2f}, prob={probability:.4f}, ev={ev:.4f}")
                
                # 결과에 추가 (EV > 0인 경우만)
                if ev > 0:
                    roi = (ev * boosted_odds) * 100  # 간단한 ROI 계산
                    selected_parlays.append({
                        'picks': parlay_picks,
                        'teams': [pick['team'] for pick in parlay_picks],
                        'matches': [pick['match'] for pick in parlay_picks],
                        'odds': odds,
                        'boosted_odds': boosted_odds,
                        'probability': probability,
                        'ev': ev,
                        'roi': roi,
                        'type': f'{parlay_size}_team_parlay',
                        'date': parlay_picks[0]['date'] if 'date' in parlay_picks[0] else None
                    })
                    
                    # 픽 사용 횟수 업데이트
                    for pick_id in combo:
                        pick_usage[pick_id] += 1
                        pick_usage_details[pick_id]['count'] += 1
                    
                    counter['passed'] += 1
                else:
                    counter['negative_ev'] += 1
            
            # 필터링 결과 출력
            print(f"Filtering results for {parlay_size}-team parlays (Iteration {iteration}/{max_iterations}):")
            print(f"  - Usage limit exceeded: {counter['usage_limit']}")
            print(f"  - Team conflicts: {counter['team_conflict']}")
            print(f"  - Negative EV: {counter['negative_ev']}")
            print(f"  - Passed all filters: {counter['passed']}")
            
            # 픽 사용 횟수 정보 확인
            usage_data = [{'team': info['team'], 'match': info['match'], 'date': info['date'], 'count': info['count']} 
                        for pick_id, info in pick_usage_details.items() if info['count'] > 0]
            
            # 최소 사용 횟수를 충족하지 못한 픽 확인
            min_usage_not_met = [data for data in usage_data if data['count'] < min_pick_usage]
            
            # 결과 저장
            final_parlays = selected_parlays
            final_usage_data = usage_data
            
            # 모든 픽이 최소 사용 횟수를 충족하면 종료
            if not min_usage_not_met:
                print(f"All picks meet the minimum usage requirement of {min_pick_usage} in iteration {iteration}.")
                break
            else:
                num_not_met = len(min_usage_not_met)
                total_picks = len([d for d in usage_data if d['count'] > 0])
                print(f"{num_not_met} out of {total_picks} picks do not meet minimum usage ({min_pick_usage}).")
                
                # 마지막 반복인 경우 경고 메시지 출력
                if iteration == max_iterations:
                    print(f"WARNING: Reached maximum iterations ({max_iterations}). Using best result found.")
                    
                    # 최소 사용 횟수를 충족하지 못한 픽 목록 출력
                    if min_usage_not_met:
                        print("Picks that did not meet minimum usage requirement:")
                        for data in min_usage_not_met:
                            print(f"  - {data['team']} ({data['date']}): used {data['count']} times (min: {min_pick_usage})")
                
                # 다음 반복을 위해 all_combos 다시 섞기
                random.shuffle(all_combos)
        
        return final_parlays, final_usage_data
    
    def calculate_optimal_bet(self, probability: float, decimal_odds: float, bankroll: float) -> float:
        """Kelly Criterion을 사용한 최적 베팅액 계산 (보수적 접근)"""
        if decimal_odds <= 1.0:  # 유효하지 않은 배당률
            return 0.0
            
        b = decimal_odds - 1  # 순이익률
        q = 1 - probability  # 패배 확률
        kelly = (probability * b - q) / b
        
        # 보수적 접근: Kelly의 1/4을 사용하고 최대 베팅액 제한
        kelly = max(0, min(0.05, kelly / 4))  # 최대 5%로 제한
        
        bet_amount = bankroll * kelly
        return max(self.min_bet_amount, min(bet_amount, bankroll * 0.05))
    
    def analyze_and_save(self, team_odds: Dict[str, str], bankroll: float) -> Dict:
        """
        베팅 포트폴리오 분석 및 최적화
        - 단일 베팅
        - 3-5팀 파라레이
        """
        # 예측 및 배당률 데이터 로드
        predictions_data = self.load_predictions_with_odds()
        
        # 단일 베팅 데이터 구성
        single_bets = []
        # 경기별 EV 추적을 위한 딕셔너리
        game_ev_tracker = {}
        
        for game in predictions_data:
            home_team = game['home_team']
            away_team = game['away_team']
            game_date = game['date'].split('T')[0] if 'T' in game['date'] else game['date']
            
            # 배당률이 없는 경기는 건너뛰기
            if game['home_team_odds'] is None or game['away_team_odds'] is None:
                continue
                
            # 매치 정보 추가 (충돌 체크용)
            self.add_match(game_date, home_team, away_team)
            
            # 홈팀과 원정팀 모두에 대한 승률 평가
            predicted_winner = game['predicted_winner']
            win_probability = game['win_probability']
            
            # win_probability는 항상 홈팀의 승리 확률
            home_win_prob = win_probability  # 홈팀 승리 확률
            away_win_prob = 1 - win_probability  # 원정팀 승리 확률
            
            # 경기 식별자 (날짜_홈팀_원정팀 형식)
            game_id = f"{game_date}_{home_team}_{away_team}"
            game_ev_tracker[game_id] = {"home": None, "away": None, "best_team": None}
            
            # 홈팀 승률이 49.5% 이상인 경우 계산
            if home_win_prob >= 0.495:
                american_odds = game['home_team_odds']
                decimal_odds = self.american_to_decimal_odds(american_odds)
                ev = self.calculate_ev(home_win_prob, decimal_odds)
                
                bet_info = {
                    'team': home_team,
                    'match': f"{away_team} vs {home_team}",
                    'date': game_date,
                    'probability': home_win_prob,
                    'american_odds': american_odds,
                    'decimal_odds': decimal_odds,
                    'ev': ev,
                    'game_id': game_id
                }
                
                # EV 추적 업데이트
                game_ev_tracker[game_id]["home"] = ev
                
                # EV가 양수인 경우만 추가 (일단 모든 양수 EV 픽 수집)
                if ev > 0:
                    single_bets.append(bet_info)
            
            # 원정팀 승률이 49.5% 이상인 경우 계산
            if away_win_prob >= 0.495:
                american_odds = game['away_team_odds']
                decimal_odds = self.american_to_decimal_odds(american_odds)
                ev = self.calculate_ev(away_win_prob, decimal_odds)
                
                bet_info = {
                    'team': away_team,
                    'match': f"{away_team} vs {home_team}",
                    'date': game_date,
                    'probability': away_win_prob,
                    'american_odds': american_odds,
                    'decimal_odds': decimal_odds,
                    'ev': ev,
                    'game_id': game_id
                }
                
                # EV 추적 업데이트
                game_ev_tracker[game_id]["away"] = ev
                
                # EV가 양수인 경우만 추가 (일단 모든 양수 EV 픽 수집)
                if ev > 0:
                    single_bets.append(bet_info)
            
            # 각 경기에서 EV가 더 높은 팀 결정
            home_ev = game_ev_tracker[game_id]["home"] or -999
            away_ev = game_ev_tracker[game_id]["away"] or -999
            
            if home_ev > away_ev and home_ev > 0:
                game_ev_tracker[game_id]["best_team"] = home_team
            elif away_ev > home_ev and away_ev > 0:
                game_ev_tracker[game_id]["best_team"] = away_team
        
        # 같은 경기에서 EV가 더 높은 팀만 선택
        filtered_bets = []
        for bet in single_bets:
            game_id = bet['game_id']
            if game_ev_tracker[game_id]["best_team"] == bet['team']:
                # 필터링 후 필요없는 game_id 제거
                bet_copy = bet.copy()
                del bet_copy['game_id']
                filtered_bets.append(bet_copy)
        
        # 포트폴리오 초기화
        portfolio = {
            'singles': [],
            'parlays': [],
            'total_investment': 0,
            'expected_profit': 0,
            'max_loss': 0,
            'pick_usage': {
                '3_team': [],
                '4_team': [],
                '5_team': []
            }
        }
        
        # 단일 베팅 처리 (EV > 0인 경우만)
        singles_with_roi = []
        all_single_bets = []  # 모든 싱글 베팅 저장 (ROI 필터링 전)
        
        for bet in filtered_bets:
            amount = self.calculate_optimal_bet(bet['probability'], bet['decimal_odds'], bankroll)
            potential_profit = amount * (bet['decimal_odds'] - 1)
            expected_profit = amount * bet['ev']
            roi = (expected_profit / amount) * 100 if amount > 0 else 0
            
            # 모든 EV > 0 베팅 정보 저장 (표시용)
            bet_info = {
                'team': bet['team'],
                'match': bet['match'],
                'date': bet['date'],
                'probability': bet['probability'],
                'odds': bet['american_odds'],  # 미국식 배당률 표시
                'amount': amount,
                'potential_profit': potential_profit,
                'expected_profit': expected_profit,
                'roi': roi
            }
            all_single_bets.append(bet_info)
            
            # ROI >= min_roi_threshold 인 베팅만 파라레이 구성에 사용하고 포트폴리오에 추가
            if roi >= self.min_roi_threshold:
                bet_copy = bet.copy()
                bet_copy['roi'] = roi
                singles_with_roi.append(bet_copy)
                
                # ROI가 기준 이상인 베팅만 포트폴리오에 포함
                portfolio['singles'].append(bet_info)
                portfolio['total_investment'] += amount
                portfolio['expected_profit'] += expected_profit
                portfolio['max_loss'] += amount
        
        # 필터링 결과를 포트폴리오에 저장 (분석 결과 표시용)
        portfolio['all_single_bets'] = all_single_bets  # 모든 싱글 베팅 (ROI 필터링 전)
        
        # ROI >= min_roi_threshold 인 베팅만 사용하여 파라레이 생성
        print(f"\nUsing {len(singles_with_roi)} bets with ROI >= {self.min_roi_threshold}% for parlay generation")
        
        # 각 팔레이 크기별 최소 사용량 충족 여부 추적
        min_usage_met = {
            3: True,  # 기본값은 True (충족했다고 가정)
            4: True,
            5: True
        }
        
        # 파라레이 생성 및 처리 (3-5팀 중점)
        # 크기별로 파라레이 생성하고 포트폴리오에 저장
        parlays_by_size = {3: [], 4: [], 5: []}  # 크기별 파라레이 저장용
        
        for size in range(3, 6):
            print(f"\n===== Generating {size}-team parlays =====")
            parlays, usage_data = self.generate_parlays(singles_with_roi, size)
            print(f"Generated {len(parlays)} profitable {size}-team parlays")
            
            # 팔레이 크기별 픽 사용 횟수 정보 저장
            portfolio['pick_usage'][f'{size}_team'] = sorted(usage_data, key=lambda x: x['count'], reverse=True)
            
            # 최소 사용량 충족 여부 확인
            min_pick_usage = 3 if size == 3 else (3 if size == 4 else 4)
            picks_below_min = [item for item in usage_data if item['count'] < min_pick_usage]
            
            if picks_below_min:
                min_usage_met[size] = False
                print(f"WARNING: {len(picks_below_min)} picks do not meet minimum usage requirement ({min_pick_usage}) for {size}-team parlays")
            
            # 이 크기의 파라레이들 처리하여 저장
            for parlay in parlays:
                amount = self.calculate_optimal_bet(parlay['probability'], parlay['boosted_odds'], bankroll)
                potential_profit = amount * (parlay['boosted_odds'] - 1)
                expected_profit = amount * parlay['ev']
                roi = (expected_profit / amount) * 100 if amount > 0 else 0
                
                print(f"Parlay ROI: {roi:.2f}%")
                
                # 파라레이 정보 생성 (adjusted_amount는 나중에 계산)
                parlay_info = {
                    'teams': parlay['teams'],
                    'picks': [{
                        'team': pick['team'],
                        'match': pick['match'],
                        'probability': pick['probability'],
                        'odds': pick['american_odds'],  # 미국식 배당률 표시
                        'date': pick['date'] if 'date' in pick else None  # date 정보 추가
                    } for pick in parlay['picks']],
                    'type': parlay['type'],
                    'probability': parlay['probability'],
                    'odds': parlay['odds'],
                    'boosted_odds': parlay['boosted_odds'],
                    'amount': amount,
                    'potential_profit': potential_profit,
                    'expected_profit': expected_profit,
                    'size': size  # 크기 정보 임시 저장
                }
                
                # 크기별로 분류해서 저장
                parlays_by_size[size].append(parlay_info)
                
                portfolio['total_investment'] += amount
                portfolio['expected_profit'] += expected_profit
                portfolio['max_loss'] += amount
        
        # 크기별 adjusted_amount 계산 (대시보드와 동일한 방식)
        for size in range(3, 6):
            size_parlays = parlays_by_size[size]
            if size_parlays:
                # 이 크기의 모든 파라레이 amount 합계를 개수로 나눔
                total_amount = sum(parlay['amount'] for parlay in size_parlays)
                adjusted_amount = total_amount / len(size_parlays)
                
                # 각 파라레이에 adjusted_amount 추가하고 포트폴리오에 추가
                for parlay_info in size_parlays:
                    parlay_info['adjusted_amount'] = adjusted_amount
                    # 임시 저장한 size 정보 제거
                    del parlay_info['size']
                    portfolio['parlays'].append(parlay_info)
        
        # 모든 팔레이 크기에서 최소 사용량 조건이 충족되었는지 확인
        all_min_usage_met = all(min_usage_met.values())
        
        # 분석 결과 저장 (최소 사용량 조건이 충족된 경우에만)
        if all_min_usage_met:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.analysis_dir / f"mlb_betting_analysis_{timestamp}.json"
            
            with open(output_file, 'w') as f:
                json.dump(portfolio, f, indent=2)
            
            print(f"Analysis saved to: {output_file}")
        else:
            print("WARNING: Analysis was NOT saved because minimum pick usage requirements were not met for all parlay sizes.")
            # 충족되지 않은 팔레이 크기 출력
            for size, met in min_usage_met.items():
                if not met:
                    print(f"  - {size}-team parlays: minimum usage requirement not met")
        
        return portfolio

# 직접 실행 테스트용 코드
if __name__ == "__main__":
    optimizer = MLBBettingOptimizer()
    portfolio = optimizer.analyze_and_save({}, 500)
    
    # 결과 요약 출력
    print("\n===== 베팅 분석 결과 요약 =====")
    print(f"총 단일 베팅 수: {len(portfolio['singles'])}")
    print(f"총 파라레이 베팅 수: {len(portfolio['parlays'])}")
    print(f"총 투자 금액: ${portfolio['total_investment']:.2f}")
    print(f"예상 수익: ${portfolio['expected_profit']:.2f}")
    
    # 파라레이 종류별 집계
    parlay_counts = {}
    for parlay in portfolio['parlays']:
        parlay_type = parlay['type']
        if parlay_type not in parlay_counts:
            parlay_counts[parlay_type] = 0
        parlay_counts[parlay_type] += 1
    
    print("\n--- 파라레이 종류별 집계 ---")
    for parlay_type, count in parlay_counts.items():
        print(f"{parlay_type}: {count}개") 