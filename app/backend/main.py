from fastapi import FastAPI, Response, Cookie
from starlette.middleware.cors import CORSMiddleware  # CORSを回避するために必要
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routers import manipulation
from routers import evaluation
import random
import string

app = FastAPI()
template_file_path = "./static/templates/index.html"
evaluation_research_file_path = "./static/templates/evaluation.html"
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORSを回避するために設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------APIの実装------------

# この記述により@app.get("/{giongo}")でfavicon.icoが処理されない


@app.get("/favicon.ico")
async def deal_favicon():
    return


# 生成実験で生成した音を評価する
@app.get("/evaluation/{giongo}")
async def evaluation_research(giongo: str):
    # 結果記録用ファイル名が被らないための文字列
    random_strings_list: list = [random.choice(string.ascii_letters + string.digits)
                                 for i in range(7)]
    random_strings = ''.join(random_strings_list)
    # HTMLファイルを返す
    response = FileResponse(evaluation_research_file_path)
    # 目的の擬音語の値
    response.set_cookie(key="giongo", value=giongo)
    # 記録ファイル名を被らせないためのランダム文字列
    response.set_cookie(key="random_strings", value=random_strings)
    return response


# 擬音語に沿った音を生成する
@app.get("/{giongo}")
async def main(giongo: str):
    # 結果記録用ファイル名が被らないための文字列
    random_strings_list: list = [random.choice(string.ascii_letters + string.digits)
                                 for i in range(7)]
    random_strings = ''.join(random_strings_list)
    # HTMLファイルを返す
    response = FileResponse(template_file_path)
    # 目的の擬音語の値
    response.set_cookie(key="giongo", value=giongo)
    # 記録ファイル名を被らせないためのランダム文字列
    response.set_cookie(key="random_strings", value=random_strings)
    return response


app.include_router(manipulation.router)
app.include_router(evaluation.router)