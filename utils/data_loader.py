import pandas as pd
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
import pickle # FAISS 인덱스 저장을 위해 추가

def load_machine_data_from_csv(file_path: str) -> pd.DataFrame:
    print(f"  Data Loader: '{file_path}'에서 기계 데이터 로드 중...")
    try:
        # CSV 파일 로드. 첫 행이 헤더이므로 header=0
        df = pd.read_csv(file_path)
        # 필요한 컬럼이 있는지 확인
        required_cols = ['item', 'machine', 'capacity']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV 파일에 필수 컬럼 {required_cols} 중 하나 이상이 누락되었습니다.")
        print(f"  Data Loader: '{file_path}'에서 기계 데이터 로드 완료.")
        return df
    except FileNotFoundError:
        print(f"  Data Loader 오류: 파일 '{file_path}'를 찾을 수 없습니다. 더미 데이터를 사용합니다.")
        # 파일이 없을 경우를 대비한 더미 데이터 (개발/테스트용)
        dummy_data = {
            'item': ['금속 부품', '금속 부품', '시제품', '아크릴 판', '플라스틱 케이스', '강철 프레임'],
            'machine': ['CNC 밀링 머신', '용접 로봇', '3D 프린터 (SLA)', '레이저 커팅 머신', '사출 성형기', '용접 로봇'],
            'capacity': [5, 3, 1, 10, 50, 8] # 단위 시간당 처리 가능량
        }
        return pd.DataFrame(dummy_data)
    except Exception as e:
        print(f"  Data Loader 오류: CSV 파일 로드 중 오류 발생 - {e}")
        return pd.DataFrame() # 빈 DataFrame 반환

def initialize_faiss_vectorstore_from_dataframe(
    machine_data_df: pd.DataFrame, 
    embedding_model_name: str,
    faiss_index_path: str = "data/faiss_index",
    faiss_index_name: str = "machines_items_capacity"
) -> FAISS:
    print(f"  Data Loader: FAISS 벡터스토어 초기화 및 '{embedding_model_name}' 임베딩 모델 로드 중...")
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    # FAISS 인덱스 파일 경로
    full_index_path = os.path.join(faiss_index_path, f"{faiss_index_name}.faiss")
    full_index_pkl_path = os.path.join(faiss_index_path, f"{faiss_index_name}.pkl") # LangChain FAISS는 .pkl 파일도 함께 생성

    # 기존 FAISS 인덱스 로드 시도
    if os.path.exists(full_index_path) and os.path.exists(full_index_pkl_path):
        print(f"  Data Loader: 기존 FAISS 인덱스 로드 중: {full_index_path}")
        try:
            vectorstore = FAISS.load_local(faiss_index_path, embeddings, index_name=faiss_index_name, allow_dangerous_deserialization=True)
            print("  Data Loader: FAISS 인덱스 로드 완료.")
            return vectorstore
        except Exception as e:
            print(f"  Data Loader 경고: FAISS 인덱스 로드 실패 ({e}). 새로 생성합니다.")
    
    # 인덱스 없거나 로드 실패 시 새로 생성
    docs = []
    # DataFrame의 각 행을 바탕으로 검색 가능한 텍스트 문서 생성
    for index, row in machine_data_df.iterrows():
        docs.append(f"기계: {row['machine']}, 생산 품목: {row['item']}, 시간당 생산량: {row['capacity']}개")
    
    if not docs:
        raise ValueError("임베딩할 문서가 없습니다. machine_data_df가 비어있거나 형식이 올바르지 않습니다.")

    vectorstore = FAISS.from_texts(docs, embeddings)
    
    # FAISS 인덱스 저장
    os.makedirs(faiss_index_path, exist_ok=True)
    vectorstore.save_local(faiss_index_path, index_name=faiss_index_name)
    print(f"  Data Loader: FAISS 인덱스 저장 완료: {faiss_index_path}/{faiss_index_name}.faiss")

    print("  Data Loader: FAISS 벡터스토어 초기화 완료.")
    return vectorstore