import sys
import os
from dotenv import load_dotenv # .env 로딩 추가
from datetime import date # 현재 날짜 사용을 위해 추가

# 현재 파일의 상위 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# .env 파일 로드 (가장 먼저 실행)
load_dotenv()

from config import LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, RETRIEVER_K, MACHINE_INFO_CSV, FAISS_INDEX_PATH, FAISS_INDEX_NAME
from agents.order_parser import OrderParserAgent
from agents.retriever import RetrieverAgent
from agents.planner import PlannerAgent
from agents.gt_generator import GTGeneratorAgent # gt_generator 디렉토리/모듈 변경 고려
from agents.evaluator import EvaluatorAgent
from utils.data_loader import load_machine_data_from_csv, initialize_faiss_vectorstore_from_dataframe
from typing import Dict, Any, List

class MultiAgentRAGSystem:
    def __init__(self):
        # 1. 데이터 로드 및 FAISS 초기화
        # machine_data_df는 DataFrame 형태가 될 것임.
        self.machine_data_df = load_machine_data_from_csv(MACHINE_INFO_CSV)
        
        # FAISS 벡터스토어는 이제 item-machine-capacity 관계를 포함하도록 생성
        self.vectorstore = initialize_faiss_vectorstore_from_dataframe(
            self.machine_data_df, 
            EMBEDDING_MODEL_NAME,
            faiss_index_path=FAISS_INDEX_PATH,
            faiss_index_name=FAISS_INDEX_NAME
        )

        # 2. 에이전트 인스턴스화
        self.order_parser = OrderParserAgent(llm_model=LLM_MODEL_NAME)
        self.retriever = RetrieverAgent(vector_store=self.vectorstore, k=RETRIEVER_K)
        self.planner = PlannerAgent(llm_model=LLM_MODEL_NAME)
        
        # GTGeneratorAgent는 DataFrame 형태의 기계 데이터와 새로운 주문 필드를 받도록 조정
        self.gt_generator = GTGeneratorAgent(machine_item_capacity_data=self.machine_data_df)
        self.evaluator = EvaluatorAgent(llm_model=LLM_MODEL_NAME)

    # 하루에 한 번 호출될 스케줄링 함수
    def run_daily_scheduling(self, order_text: str):
        print(f"\n--- 일일 스케줄링 시작 (기준 날짜: {date.today()}) ---")

        # 1. OrderParserAgent: 자연어 주문 파싱 (item, qty, due_date, cost, urgent)
        # parsed_order는 이제 cost와 urgent 필드를 포함
        parsed_order = self.order_parser.parse_order(order_text)
        print(f"\n[OrderParserAgent] 파싱된 주문: {parsed_order}")
        if not parsed_order or parsed_order.get("item") == "UNKNOWN":
            print("[System] 주문 파싱에 실패하여 파이프라인을 종료합니다.")
            return

        # 2. RetrieverAgent: 품목 기반 기계-생산량 정보 검색
        # retrieved_machine_item_capacity_info는 "machine_name (item) - capacity" 형태의 텍스트 리스트
        retrieved_machine_item_capacity_info = self.retriever.retrieve_machine_item_capacity_info(parsed_order["item"])
        print(f"\n[RetrieverAgent] 검색된 기계-품목-생산량 정보:\n{'- ' + '\\n- '.join(retrieved_machine_item_capacity_info)}")
        if not retrieved_machine_item_capacity_info:
            print("[System] 관련 기계 정보를 찾을 수 없습니다. 파이프라인을 종료합니다.")
            return

        # 3. PlannerAgent: LLM 기반 최적 스케줄 생성
        llm_generated_schedule = self.planner.generate_schedule(
            order_info=parsed_order, # 모든 주문 정보를 한 번에 전달
            machine_item_capacity_info=retrieved_machine_item_capacity_info,
            current_date=date.today().strftime('%Y-%m-%d') # 현재 날짜 전달
        )
        print(f"\n[PlannerAgent] AI 생성 스케줄:\n{llm_generated_schedule}")

        # 4. GTGeneratorAgent: GA 기반 정답(GT) 스케줄 생성
        # GTGeneratorAgent도 모든 주문 정보와 현재 날짜를 받음
        gt_schedule = self.gt_generator.generate_gt_schedule(
            order_info=parsed_order,
            current_date=date.today().strftime('%Y-%m-%d')
        )
        print(f"\n[GTGeneratorAgent] GT 스케줄 (GA 기반):\n{gt_schedule}")

        # 5. EvaluatorAgent: 생성된 스케줄과 GT 비교 평가
        evaluation_results = self.evaluator.evaluate_schedules(
            llm_schedule=llm_generated_schedule,
            gt_schedule=gt_schedule,
            order_info=parsed_order # 모든 주문 정보 전달
        )
        print(f"\n--- [EvaluatorAgent] 스케줄 평가 결과 ---\n{evaluation_results}")
        print("\n--- 일일 스케줄링 종료 ---")

# --- 시스템 실행 ---
if __name__ == "__main__":
    system = MultiAgentRAGSystem()
    print("시스템 준비 완료. 일일 스케줄링을 시작합니다.")

    # 예시 주문 실행
    # 실제 사용 시에는 주문 정보가 외부 시스템으로부터 주입될 수 있습니다.
    test_order_text = "금속 부품 100개, 납기는 2025년 7월 15일까지, 개당 가격 5000원, 긴급으로 처리해줘"
    system.run_daily_scheduling(test_order_text)

    # 다른 주문 테스트
    # system.run_daily_scheduling("플라스틱 케이스 1000개, 8월 10일까지, 개당 1200원, 일반 주문")