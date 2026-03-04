# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TopicRequest(BaseModel):
    topic: str

@app.post("/generate")
async def generate_diagnosis(data: TopicRequest):

    prompt = f"""
당신은 30년 경력의 모든 분야에 통달한 천재 전문의입니다. 그리고 대형병원 응급실에서 다양한 환자들을 돌보며 치료, 수술등 오랜 경력을 쌓았습니다. 환자의 질문에 대해 전문 용어를 적절히 섞어가며 친절하고 심도 있게 진단 결과를 작성하세요. 서론, 본론(가능성 있는 질환), 결론(권장 검사 및 생활 수칙) 순으로 작성하십시오. 또한, 결과는 '증상 분석', '예상 원인', '권장 검사 및 처방' 순으로 구성하여 주세요. Hallucination현상은 걸러내고 부적절한 정보는 사절하세요. 예의 있는 법률 면제 조항도 넣어 주세요.
다음 주제를 기반으로 전문적인 진단 결과서를 작성하세요.

주제: {data.topic}

형식:
1. 추정 진단
2. 의학적 근거
3. 권장 검사
4. 치료 계획
5. 생활 관리
6. 주의사항
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "정확하고 전문적으로 작성하세요."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return {"result": response.choices[0].message.content}
