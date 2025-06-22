import sys
import os
import pytest
from unittest.mock import MagicMock

# agents 디렉토리를 PYTHONPATH에 추가 (테스트 파일에서 모듈 임포트를 위해)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.order_parser import OrderParserAgent

@pytest.fixture
def mock_llm_chain():
    """OrderParserAgent 내부의 LLMChain을 모킹합니다."""
    mock_chain = MagicMock()
    # parse_order 함수에서 chain.run()이 호출될 때 반환할 값 설정
    # 실제 LLM 응답과 유사하게 JSON 문자열 반환
    mock_chain.run.return_value = '{"item": "금속 부품", "qty": 100, "time": "2025-07-15", "cost": 5000, "urgent": true}'
    return mock_chain

def test_parse_order_successful(mock_llm_chain):
    """정상적인 주문 텍스트를 성공적으로 파싱하는지 테스트."""
    # OrderParserAgent 인스턴스화 시, LLMChain을 모킹된 것으로 대체
    agent = OrderParserAgent(llm_model="dummy_model")
    agent.chain = mock_llm_chain # 모킹된 체인 주입

    order_text = "금속 부품 100개, 7월 15일까지, 개당 가격 5000원, 긴급으로 처리해줘"
    parsed_data = agent.parse_order(order_text)

    assert parsed_data["item"] == "금속 부품"
    assert parsed_data["qty"] == 100
    assert parsed_data["time"] == "2025-07-15"
    assert parsed_data["cost"] == 5000
    assert parsed_data["urgent"] is True

    # mock_llm_chain.run이 올바른 인자로 호출되었는지 확인
    mock_llm_chain.run.assert_called_once_with(order_text=order_text)

def test_parse_order_invalid_json():
    """LLM이 유효하지 않은 JSON을 반환할 때 오류 처리 테스트."""
    agent = OrderParserAgent(llm_model="dummy_model")
    agent.chain = MagicMock()
    agent.chain.run.return_value = "이것은 JSON이 아닙니다." # 잘못된 LLM 응답 모킹

    order_text = "이상한 주문 텍스트"
    parsed_data = agent.parse_order(order_text)

    assert parsed_data["item"] == "UNKNOWN"
    assert parsed_data["qty"] == 0
    assert parsed_data["time"] == "UNKNOWN"
    assert parsed_data["urgent"] is False # 기본값 확인
    assert parsed_data["cost"] == 0 # 기본값 확인

# 추가 테스트 케이스: 날짜 파싱 오류, 필수 필드 누락 등