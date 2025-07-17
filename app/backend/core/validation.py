from fastapi import HTTPException

def validation(name: str,age: str, gender: str, hearing: str, chromosomes_params):
    # バリデーション
    # 名前が未入力の場合エラーを返す
    if name == "":
        raise HTTPException(status_code=422, detail="名前が入力されていません")
    # 年齢が未入力の場合エラーを返す
    if age == "":
        raise HTTPException(status_code=422, detail="年齢が入力されていません")
    # 性別が未入力の場合エラーを返す
    if gender == "":
        raise HTTPException(status_code=422, detail="性別が入力されていません")
    # 聞こえ方が未入力の場合エラーを返す
    if hearing == "":
        raise HTTPException(status_code=422, detail="聞こえ方について入力されていません")
    # 未入力の適応度がある場合エラーを返す
    for chromosome in chromosomes_params.values():
        if chromosome["fitness"] == "":
            raise HTTPException(status_code=422, detail="未入力の適応度があります")

    return