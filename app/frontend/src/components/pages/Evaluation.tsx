import React, { useState, useEffect, useCallback } from "react";
import TextField from "@material-ui/core/TextField";
import Select from "@mui/material/Select";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Button from "@material-ui/core/Button";
import { Description } from "../molecules/EvaluateDescription";
import { Option } from "../organisms/Option";
import { chromosomesParams } from "../../@types/chromosomesParams";

import axios from "axios";
import { getCookie } from "../../Common/getCookie";

export function Evaluation(): JSX.Element {
  // 被験者指名、ニックネーム
  const [name, setName] = useState("");
  // 被験者年齢
  const [age, setAge] = useState("");
  // 被験者性別
  const [gender, setGender] = useState("");
  // 健聴など（聞こえ方）
  const [hearing, setHearing] = useState("");
  // 生成する擬音語
  const [gion, setGion] = useState("");
  // 初回のみ実行するメソッド用
  const [count, _setCount] = useState(1);

  // １つの世代に存在する遺伝子の情報
  const [chromosomesParams, setChromosomesParams] =
    useState<chromosomesParams | null>(null);

  // 世代の適応度、および音生成パラメータを送信。次世代の音生成パラメータの格納
  const postResult = useCallback(async () => {
    // 被験者名とパラメータを結合する
    const answer = Object.assign(
      {},
      chromosomesParams,
      { name: name },
      { age: age },
      { gender: gender },
      { hearing: hearing }
    );
    // 送信先URL
    // 結果を送信する
    axios
      .post(`/evaluation/log`, answer)
      .then((_response) => {
        alert(
          "ご協力ありがとうございました。" +
            "\n" +
            "実験は終了です。ページを閉じていただいて構いません。"
        );
      })
      .catch((err) => {
        alert("データの送信に失敗しました" + "\n" + err.response.data.detail);
      });
  }, [chromosomesParams, name, age, gender, hearing]);

  {
    /* 1世代目はランダムにパラメータを生成 */
  }
  useEffect(() => {
    const url = `${location.pathname}/initialize`;
    console.log("location", location.pathname, url);

    if (count == 1) {
      axios
        .get(url)
        .then((response) => {
          const bestFitChromosomes: chromosomesParams = response.data;
          // 遺伝子を初期化
          setChromosomesParams(bestFitChromosomes);
          // 生成する目的の擬音語を設定
          setGion(getAimGiongo());
        })
        .catch(() => {
          alert("データの受信に失敗しました");
        });
    }
  }, []);

  // サーバーから受け取った遺伝子情報をフォーマットして格納
  const OptionsEL = () => {
    // 1つの遺伝子の評価をするためのコンポーネントのリスト
    const Options: JSX.Element[] = [];
    if (chromosomesParams) {
      for (const [chromosomeNum, chromosomeParam] of Object.entries(
        chromosomesParams
      )) {
        Options.push(
          <div>
            <Option
              key={chromosomeNum}
              chromosomesParams={chromosomesParams}
              setChromosomesParameters={setChromosomesParams}
              chromosomeNumber={chromosomeNum}
              algorithmNum={chromosomeParam.algorithmNum}
              chromosomesFitnessValue={chromosomeParam.fitness}
              soundParamsList={chromosomeParam.fmParamsList}
            ></Option>
            <audio id={chromosomeNum + "audio"} controls></audio>
          </div>
        );
      }
    }
    return Options;
  };

  return (
    <div>
      {/* 実験説明 */}
      <Description gion={gion}></Description>
      {/* 被験者名 */}
      <p>名前を入力して下さい(英小文字、英大文字)</p>
      <TextField
        id="name"
        label="名前"
        variant="outlined"
        onChange={(e) => {
          setName(e.target.value);
        }}
      />
      <p>年齢を入力して下さい</p>
      <TextField
        id="age"
        value={age}
        label="年齢"
        variant="outlined"
        onChange={(e) => {
          setAge(e.target.value);
        }}
      />
      <p>性別を選択して下さい</p>
      <InputLabel id="gender-label">性別</InputLabel>
      <Select
        labelId="gender-label"
        id="gender-select"
        value={gender}
        label="性別"
        onChange={(e) => {
          setGender(e.target.value);
        }}
      >
        <MenuItem value={""}>-</MenuItem>
        <MenuItem value={"male"}>男性</MenuItem>
        <MenuItem value={"female"}>女性</MenuItem>
      </Select>
      <p>聞こえにくい音や聞こえない音などはありますか</p>
      <Select
        labelId="hearing-label"
        id="hearing-select"
        value={hearing}
        label="聞こえ方"
        onChange={(e) => {
          setHearing(e.target.value);
        }}
      >
        <MenuItem value={""}>-</MenuItem>
        <MenuItem value={"0"}>ない</MenuItem>
        <MenuItem value={"1"}>聞こえにくい音がある</MenuItem>
      </Select>

      <div id="answer">
        {/* 各遺伝子コンポーネントの描画 */}
        {OptionsEL()}
        {/* 遺伝子の評価を送信 */}
        <Button
          variant="contained"
          color="primary"
          onClick={() => {
            postResult();
          }}
        >
          回答を送信
        </Button>
      </div>
    </div>
  );
}

function getAimGiongo() {
  const aimedGiongo = getCookie("giongo");
  if (aimedGiongo == "pi") {
    return "ピッ";
  }
  if (aimedGiongo == "pu") {
    return "プッ";
  }
  if (aimedGiongo == "bu-") {
    return "ブー";
  }
  if (aimedGiongo == "bi") {
    return "ビィ";
  }
  if (aimedGiongo == "ti-n") {
    return "チーン";
  }
  if (aimedGiongo == "fa-n") {
    return "ファーン";
  }
  if (aimedGiongo == "ga-n") {
    return "ガーン";
  }
  return "適切なURLを使用してください";
}
