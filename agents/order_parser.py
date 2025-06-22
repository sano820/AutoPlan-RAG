import json
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
from datetime import datetime # 날짜 처리 추가

class OrderParserAgent:
    def __init__(self, llm_model: str):
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.1)
        self.prompt_template = PromptTemplate(
            input_variables=["order_text"],
            template="""
            당신은 주문 파서입니다. 다음 자연어 주문에서 **품목(item)**, **수량(qty)**, **납기(time)**, **개당 가격(cost)**, **긴급 여부(urgent)**를 정확히 파싱하여 JSON 형태로만 출력해주세요.
            납기일(time)은 'YYYY-MM-DD' 형식이어야 합니다. 년도가 명시되지 않으면 현재 년도 (2025년)로 가정합니다.
            긴급 여부(urgent)는 '긴급'이라는 단어가 있으면 true, 없으면 false로 설정해주세요.
            개당 가격(cost)은 숫자로 변환하고, '원'과 같은 단위는 제거해주세요.

            JSON 출력은 반드시 다음 키를 포함해야 합니다: 'item', 'qty', 'time', 'cost', 'urgent'.
            수량(qty)은 숫자로, 납기(time)는 'YYYY-MM-DD' 문자열로 변환해주세요.

            예시 1:
            주문: "금속 부품 100개, 7월 15일까지, 개당 가격 5000원, 긴급으로 처리해줘"
            결과: {{"item": "금속 부품", "qty": 100, "time": "2025-07-15", "cost": 5000, "urgent": true}}

            예시 2:
            주문: "시제품 5개 만들어줘, 7월 20일까지 필요해, 개당 1200원"
            결과: {{"item": "시제품", "qty": 5, "time": "2025-07-20", "cost": 1200, "urgent": false}}

            주문: "{order_text}"
            결과:
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def parse_order(self, order_text: str) -> Dict[str, Any]:
        print(f"  OrderParserAgent: 주문 '{order_text}' 파싱 요청...")
        try:
            raw_output = self.chain.run(order_text=order_text)
            match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if match:
                json_str = match.group(0)
                parsed_data = json.loads(json_str)
                
                # 데이터 유효성 검사 및 기본값 설정
                parsed_data['item'] = parsed_data.get('item', 'UNKNOWN').strip()
                parsed_data['qty'] = int(parsed_data.get('qty', 0))
                parsed_data['cost'] = int(parsed_data.get('cost', 0)) # cost 추가
                parsed_data['urgent'] = bool(parsed_data.get('urgent', False)) # urgent 추가
                
                # 납기일 날짜 포맷팅 검사 및 보정
                due_date_str = parsed_data.get('time', 'UNKNOWN')
                if not re.match(r'^\d{4}-\d{2}-\d{2}$', due_date_str):
                    # 현재 연도를 사용하여 날짜 보정 시도 (예: "7월 15일" -> "2025-07-15")
                    try:
                        # 간편한 예시, 실제로는 dateutil.parser 같은 라이브러리 사용 권장
                        # 7월 15일 -> YYYY-07-15
                        if re.match(r'^\d{1,2}월 \d{1,2}일$', due_date_str):
                            month, day = map(int, re.findall(r'\d+', due_date_str))
                            due_date_str = f"{datetime.now().year}-{month:02d}-{day:02d}"
                        elif re.match(r'^\d{2}-\d{2}$', due_date_str): # MM-DD
                             due_date_str = f"{datetime.now().year}-{due_date_str}"

                        if re.match(r'^\d{4}-\d{2}-\d{2}$', due_date_str): # 최종적으로 YYYY-MM-DD 형태가 되었는지 확인
                            datetime.strptime(due_date_str, '%Y-%m-%d') # 유효한 날짜인지 확인
                            parsed_data['time'] = due_date_str
                        else:
                            print(f"  OrderParserAgent 경고: 납기일 '{due_date_str}' 파싱 실패. 'UNKNOWN'으로 처리.")
                            parsed_data['time'] = 'UNKNOWN'
                    except Exception as date_parse_e:
                        print(f"  OrderParserAgent 경고: 납기일 '{due_date_str}' 파싱 중 오류 발생 - {date_parse_e}. 'UNKNOWN'으로 처리.")
                        parsed_data['time'] = 'UNKNOWN'
                
                return parsed_data
            else:
                print(f"  OrderParserAgent 오류: LLM 출력에서 JSON 형식을 찾을 수 없습니다. 원시 출력: {raw_output}")
                return {"item": "UNKNOWN", "qty": 0, "time": "UNKNOWN", "cost": 0, "urgent": False}
        except json.JSONDecodeError as e:
            print(f"  OrderParserAgent 오류: JSON 파싱 실패 - {e}. 원시 출력: {raw_output}")
            return {"item": "UNKNOWN", "qty": 0, "time": "UNKNOWN", "cost": 0, "urgent": False}
        except Exception as e:
            print(f"  OrderParserAgent 예외 발생: {e}. 원시 출력: {raw_output}")
            return {"item": "UNKNOWN", "qty": 0, "time": "UNKNOWN", "cost": 0, "urgent": False}