// import React, { useState, useEffect, useCallback } from "react";
// import TextField from "@material-ui/core/TextField";
// import AlertDialog from "../organisms/AlertDialog";
// import Button from "@material-ui/core/Button";
// import { SoundButton } from "../atoms/SoundButton";
// import { postFire } from "../../Common/postFire";
// import axios from "axios";
// import { research1Data } from "../../data/research1Data";
// declare global {
//   interface Window {
//     webkitAudioContext: any;
//   }
// }

// export function Research1(): JSX.Element {
//   const [name, setName] = useState("");
//   const [questionNumber, setQuestionNumber] = useState(1);
//   const [questionResults, setquestionResults] = useState({});
//   const [dialogOpen, setDialogOpen] = useState(false);

//   const postResultToDB = useCallback(async () => {
//     // 結果を送信する
//     console.log(questionResults, "question");

//     axios
//       .post("/research1", { ...questionResults, ...{ name: name } })
//       .then((response) => {
//         alert(
//           "ご協力ありがとうございました。ブラウザを閉じていただいても大丈夫です"
//         );
//       })
//       .catch((err) => {
//         alert("データの送信に失敗しました");
//       });
//   }, [questionResults, questionNumber, name]);

//   useEffect(() => {
//     if (questionNumber == Object.keys(research1Data).length + 1) {
//       postResultToDB();
//     }
//   });
//   return (
//     <div>
//       <div>
//         <strong>実験手引き</strong>
//         <ul>
//           <li>実験の問題では2つのボタンが表示されます</li>
//           <li>
//             ボタンにマウスを重ねると音が鳴ります(実験はPCで行ってください)
//           </li>
//           <li>それぞれの問題に対して点数をつけてください</li>
//         </ul>
//         <strong>点数のつけ方</strong>
//         <ul>
//           <li>
//             左のボタンで行われる操作が
//             右のボタンで行われる操作に比べ危険だと感じれば 2 点{" "}
//           </li>
//           <li>
//             左のボタンで行われる操作の方がどちらかと言えば、危険だと感じれば 1
//             点
//           </li>
//           <li>危険さの違いを感じなければ 0点</li>
//           <li>
//             右のボタンで行われる操作の方がどちらかと言えば、危険だと感じれば -1
//             点
//           </li>
//           <li>
//             {" "}
//             右のボタンで行われる操作が
//             左のボタンで行われる操作に比べ危険だと感じれば -2 点{" "}
//           </li>
//         </ul>
//         <strong>実験準備</strong>
//         <ul>
//           <li>音量はうるさくないと感じる程度にあげてください。</li>
//         </ul>
//         <audio src="static/sounds/sawtooth880Hz.wav" id="test"></audio>
//         <button
//           onClick={() => {
//             let audioElement;
//             audioElement = document.getElementById("test") as HTMLAudioElement;
//             audioElement.currentTime = 0;
//             audioElement.play();
//           }}
//         >
//           音量チェック。音量に注意してください。
//         </button>
//         <br></br>
//         <strong>注意事項</strong>
//         <ul>
//           <li>推奨ブラウザは「Google Chrome」です</li>
//           <li>
//             マウスを何度も重ねると音が鳴らなくなる場合があります。
//             その時はしばらく時間をおいてからマウスを重ねると再び音が鳴ります。
//           </li>
//           <li>
//             すべての質問に答えたら「結果を送信する]ボタンを押してください(ページの最後にあります)
//           </li>
//           <li>音は何度聞いても大丈夫です。</li>
//         </ul>
//       </div>
//       <p>
//         名前を入力してください（ニックネームなどどのような名前でも結構です。今後の実験の際には
//         同じ名前を使用してください。）
//       </p>
//       <input
//         type="text"
//         value={name}
//         onChange={(e) => {
//           setName(e.target.value);
//         }}
//       ></input>
//       <SoundButton></SoundButton>
//       {questionNumber <= Object.keys(research1Data).length && (
//         <AlertDialog
//           dialogOpen={dialogOpen}
//           questionNumber={questionNumber}
//           setQuestionNumber={setQuestionNumber}
//           questionResults={questionResults}
//           setQuestionResults={setquestionResults}
//           postResultToDB={postResultToDB}
//         ></AlertDialog>
//       )}
//     </div>
//   );
// }
