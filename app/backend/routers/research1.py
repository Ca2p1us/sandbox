from fastapi import APIRouter
from db import session  # DBと接続するためのセッション
from model import Research1 , Research1Table # 今回使うモデルをインポート

router = APIRouter()

@router.get("/research1")
def readResearch1Results():
    results = session.query(Research1Table).all()
    return results


@router.post("/research1")
def create_test(questionResults: Research1):
    research1 = Research1Table()
    research1.name = questionResults.name
    research1.Q1 = questionResults.Q1
    research1.Q2 = questionResults.Q2
    research1.Q3 = questionResults.Q3
    research1.Q4 = questionResults.Q4
    research1.Q5 = questionResults.Q5
    research1.Q6 = questionResults.Q6
    research1.Q7 = questionResults.Q7
    research1.Q8 = questionResults.Q8
    research1.Q9 = questionResults.Q9
    research1.Q10 = questionResults.Q10
    research1.Q11 = questionResults.Q11
    research1.Q12 = questionResults.Q12
    research1.Q13 = questionResults.Q13
    research1.Q14 = questionResults.Q14
    research1.Q15 = questionResults.Q15
    research1.Q16 = questionResults.Q16
    research1.Q17 = questionResults.Q17
    research1.Q18 = questionResults.Q18
    research1.Q19 = questionResults.Q19
    research1.Q20 = questionResults.Q20
    session.add(research1)
    session.commit()
    return questionResults
