import json
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class EvaluatorAgent:
    def __init__(self, llm_model: str):
        self.llm = ChatGoogleGenerativeAI(model=lll_model, temperature=0.2)
        self.prompt_template = PromptTemplate(
            input_variables=["llm_schedule", "gt_schedule", "item", "qty", "due_date", "cost_per_item", "is_urgent"],
            template="""
            당신은 AI 스케줄 평가 전문가입니다. AI가 생성한 스케줄과 유전 알고리즘(GT)으로 생성된 정답 스케줄을 비교하여 객관적으로 평가해주세요.

            **주문 정보:**
            - 품목: {item}
            - 수량: {qty}개
            - 납기: {due_date}
            - 개당 가격: {cost_per_item}원
            - 긴급 여부: {is_urgent}

            **AI 생성 스케줄:**
            ```json
            {llm_schedule}
            ```

            **GT 정답 스케줄 (최적이라고 가정):**
            ```
            {gt_schedule}
            ```

            **평가 항목 및 점수 (각 항목 10점 만점):**
            1.  **납기 준수 가능성:** AI 스케줄이 납기일을 얼마나 잘 지킬 것으로 예상되는가? 특히 긴급 주문의 경우 가산점/감점 요인. (10점 만점)
            2.  **총 비용 효율성:** AI 스케줄이 제시하는 총 생산 비용이 합리적이고 효율적인가? GT 대비 비용 차이. (10점 만점)
            3.  **자원 효율성:** 기계 활용, 시간 등의 자원 사용이 얼마나 효율적인가. (10점 만점)
            4.  **현실성 및 실용성:** 스케줄이 실제 공정에서 구현 가능한 수준인가 (예: 시작일이 현재 날짜 이후인가, 기계의 제약사항을 고려했는가). (10점 만점)
            5.  **GT 스케줄과의 일치도:** AI 스케줄이 GT 스케줄과 얼마나 유사한가 (예: 추천 기계, 시간, 비용). (10점 만점)

            **평가 결과는 다음 JSON 형식으로만 출력해주세요:**
            {{
              "item": "{item}",
              "qty": {qty},
              "due_date": "{due_date}",
              "cost_per_item": {cost_per_item},
              "is_urgent": {is_urgent},
              "evaluation_summary": "AI 스케줄에 대한 전반적인 평가 요약.",
              "scores": {{
                "due_date_adherence": <점수>,
                "total_cost_efficiency": <점수>,
                "resource_efficiency": <점수>,
                "realism_practicality": <점수>,
                "gt_consistency": <점수>
              }},
              "details": {{
                "due_date_adherence_reason": "납기 준수 가능성에 대한 상세 이유.",
                "total_cost_efficiency_reason": "총 비용 효율성에 대한 상세 이유.",
                "resource_efficiency_reason": "자원 효율성에 대한 상세 이유.",
                "realism_practicality_reason": "현실성 및 실용성에 대한 상세 이유.",
                "gt_consistency_reason": "GT 스케줄과의 일치도에 대한 상세 이유."
              }},
              "overall_recommendation": "개선 방향 또는 최종 의견."
            }}
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def evaluate_schedules(self, llm_schedule: str, gt_schedule: str, order_info: Dict[str, Any]) -> str:
        print(f"  EvaluatorAgent: 스케줄 평가 요청 - 품목: {order_info.get('item')}, 긴급: {order_info.get('urgent')}...")
        try:
            evaluation_result = self.chain.run(
                llm_schedule=llm_schedule,
                gt_schedule=gt_schedule,
                item=order_info.get("item", "N/A"),
                qty=order_info.get("qty", "N/A"),
                due_date=order_info.get("time", "N/A"),
                cost_per_item=order_info.get("cost", "N/A"),
                is_urgent=order_info.get("urgent", "N/A")
            )
            import re
            import json
            match = re.search(r'\{.*\}', evaluation_result, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    parsed_evaluation = json.loads(json_str)
                    return json.dumps(parsed_evaluation, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    print(f"  EvaluatorAgent 경고: LLM이 유효하지 않은 JSON을 반환했습니다. 원시 출력: {evaluation_result}")
                    return evaluation_result
            else:
                print(f"  EvaluatorAgent 경고: LLM 출력에서 JSON 형식을 찾을 수 없습니다. 원시 출력: {evaluation_result}")
                return evaluation_result
        except Exception as e:
            print(f"  EvaluatorAgent 오류: 스케줄 평가 중 예외 발생 - {e}")
            return "스케줄 평가 실패"