from typing import List
from fastapi import APIRouter
from db import session  # DBと接続するためのセッション
from model import UserTable, User, Research1Table  # 今回使うモデルをインポート

router = APIRouter()

@router.get("/users")
def read_users():
    users = session.query(UserTable).all()
    return users

# idにマッチするユーザ情報を取得 GET


@router.get("/users/{user_id}")
def read_user(user_id: int):
    user = session.query(UserTable).\
        filter(UserTable.id == user_id).first()
    return user

# ユーザ情報を登録 POST
@router.post("/user")
# クエリでnameとstrを受け取る
# /user?name="三郎"&age=10
async def create_user(name: str, age: int):
    user = UserTable()
    user.name = name
    user.age = age
    session.add(user)
    session.commit()

# 複数のユーザ情報を更新 PUT


@router.put("/users")
# modelで定義したUserモデルのリクエストbodyをリストに入れた形で受け取る
# users=[{"id": 1, "name": "一郎", "age": 16},{"id": 2, "name": "二郎", "age": 20}]
async def update_users(users: List[User]):
    for new_user in users:
        user = session.query(UserTable).filter(
            UserTable.id == new_user.id).first()
        user.name = new_user.name
        user.age = new_user.age
        session.commit()
