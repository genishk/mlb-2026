import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class HistoricalRecordsDateFilter:
    """
    MLB Historical Records 날짜 필터링 도구
    
    기능:
    1. 최신 mlb_historical_records 파일 로드
    2. 날짜 범위로 필터링
    3. 필터링된 데이터를 동일한 이름으로 저장
    """
    
    def __init__(self):
        """초기화"""
        # 프로젝트 루트 설정
        self.project_root = Path(__file__).parent
        self.records_dir = self.project_root / "data" / "records"
        
        print(f"📁 Records 디렉토리: {self.records_dir}")
        
        # 디렉토리 존재 확인
        if not self.records_dir.exists():
            raise FileNotFoundError(f"Records 디렉토리를 찾을 수 없습니다: {self.records_dir}")
    
    def find_latest_records_file(self) -> Path:
        """가장 최신 mlb_historical_records 파일 찾기"""
        pattern = "mlb_historical_records_*.json"
        files = list(self.records_dir.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"'{pattern}' 패턴의 파일을 찾을 수 없습니다.")
        
        # 파일 수정 시간 기준으로 최신 파일 선택
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        
        print(f"📄 최신 파일 발견: {latest_file.name}")
        print(f"   파일 크기: {latest_file.stat().st_size / (1024*1024):.1f} MB")
        
        return latest_file
    
    def load_records(self, file_path: Path) -> List[Dict[str, Any]]:
        """레코드 파일 로드"""
        print(f"\n🔄 파일 로딩 중: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            print(f"✅ 로드 완료: {len(records):,}개 레코드")
            
            # 날짜 형식 확인 (처음 몇 개 레코드로)
            if records:
                sample_dates = [r.get('date', 'No date') for r in records[:5]]
                print(f"📅 날짜 형식 샘플: {sample_dates}")
            
            return records
            
        except Exception as e:
            print(f"❌ 파일 로드 실패: {e}")
            raise
    
    def filter_by_date_range(self, records: List[Dict[str, Any]], 
                           start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """날짜 범위로 레코드 필터링"""
        print(f"\n🔍 날짜 필터링 시작:")
        print(f"   시작 날짜: {start_date}")
        print(f"   종료 날짜: {end_date}")
        print(f"   원본 레코드 수: {len(records):,}개")
        
        filtered_records = []
        no_date_count = 0
        invalid_date_count = 0
        
        for record in records:
            record_date = record.get('date')
            
            if not record_date:
                no_date_count += 1
                continue
            
            try:
                # 날짜 형식이 이미 YYYY-MM-DD인지 확인
                if isinstance(record_date, str) and len(record_date) == 10:
                    # YYYY-MM-DD 형식으로 가정
                    if start_date <= record_date <= end_date:
                        filtered_records.append(record)
                else:
                    invalid_date_count += 1
                    
            except Exception as e:
                invalid_date_count += 1
                continue
        
        print(f"\n📊 필터링 결과:")
        print(f"   필터링된 레코드: {len(filtered_records):,}개")
        print(f"   날짜 없는 레코드: {no_date_count:,}개")
        print(f"   잘못된 날짜 형식: {invalid_date_count:,}개")
        print(f"   필터링 비율: {len(filtered_records)/len(records)*100:.1f}%")
        
        return filtered_records
    
    def save_filtered_records(self, filtered_records: List[Dict[str, Any]], 
                             original_file_path: Path) -> Path:
        """필터링된 레코드를 동일한 이름으로 저장"""
        # 백업 파일명 생성 (원본 보존)
        backup_file_path = original_file_path.with_suffix('.backup.json')
        
        # 원본 파일 백업
        if not backup_file_path.exists():
            print(f"💾 원본 파일 백업: {backup_file_path.name}")
            with open(original_file_path, 'r', encoding='utf-8') as src:
                with open(backup_file_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
        else:
            print(f"ℹ️  백업 파일이 이미 존재합니다: {backup_file_path.name}")
        
        # 필터링된 데이터를 원본 파일명으로 저장
        print(f"\n💾 필터링된 데이터 저장 중: {original_file_path.name}")
        
        try:
            with open(original_file_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_records, f, ensure_ascii=False, indent=2)
            
            new_size = original_file_path.stat().st_size
            print(f"✅ 저장 완료!")
            print(f"   파일 경로: {original_file_path}")
            print(f"   새 파일 크기: {new_size / (1024*1024):.1f} MB")
            print(f"   레코드 수: {len(filtered_records):,}개")
            
            return original_file_path
            
        except Exception as e:
            print(f"❌ 저장 실패: {e}")
            raise
    
    def show_date_range_in_records(self, records: List[Dict[str, Any]]) -> None:
        """레코드의 날짜 범위 표시"""
        dates = []
        for record in records:
            date = record.get('date')
            if date and isinstance(date, str):
                dates.append(date)
        
        if dates:
            dates.sort()
            print(f"\n📅 레코드 날짜 범위:")
            print(f"   최초 날짜: {dates[0]}")
            print(f"   최종 날짜: {dates[-1]}")
            print(f"   날짜가 있는 레코드: {len(dates):,}개")
        else:
            print("\n❌ 유효한 날짜를 가진 레코드를 찾을 수 없습니다.")
    
    def process_date_filtering(self, start_date: str, end_date: str) -> str:
        """전체 날짜 필터링 프로세스 실행"""
        print("="*60)
        print("🗓️  MLB Historical Records 날짜 필터링 도구")
        print("="*60)
        
        try:
            # 1. 최신 파일 찾기
            latest_file = self.find_latest_records_file()
            
            # 2. 레코드 로드
            records = self.load_records(latest_file)
            
            # 3. 현재 날짜 범위 확인
            self.show_date_range_in_records(records)
            
            # 4. 날짜 필터링
            filtered_records = self.filter_by_date_range(records, start_date, end_date)
            
            if not filtered_records:
                return f"❌ 지정한 날짜 범위 ({start_date} ~ {end_date})에 해당하는 레코드가 없습니다."
            
            # 5. 필터링된 데이터 저장
            output_file = self.save_filtered_records(filtered_records, latest_file)
            
            return f"✅ 성공적으로 완료!\n   출력 파일: {output_file}\n   필터링된 레코드: {len(filtered_records):,}개"
            
        except Exception as e:
            error_msg = f"❌ 처리 중 오류 발생: {e}"
            print(error_msg)
            return error_msg


def main():
    """메인 실행 함수"""
    print("🚀 MLB Historical Records 날짜 필터링 도구 시작")
    
    # 사용자 입력 받기
    print("\n📅 필터링할 날짜 범위를 입력해주세요 (YYYY-MM-DD 형식):")
    
    try:
        start_date = input("시작 날짜 (예: 2025-07-01): ").strip()
        end_date = input("종료 날짜 (예: 2025-08-06): ").strip()
        
        # 날짜 형식 검증
        if len(start_date) != 10 or len(end_date) != 10:
            print("❌ 날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요.")
            return
        
        if start_date > end_date:
            print("❌ 시작 날짜가 종료 날짜보다 늦습니다.")
            return
        
        # 필터링 실행
        filter_tool = HistoricalRecordsDateFilter()
        result = filter_tool.process_date_filtering(start_date, end_date)
        
        print(f"\n{result}")
        
    except KeyboardInterrupt:
        print("\n\n👋 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")


if __name__ == "__main__":
    main()
