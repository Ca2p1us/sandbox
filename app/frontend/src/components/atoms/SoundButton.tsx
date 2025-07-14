import React from "react";
import Button from "@material-ui/core/Button";
import { makeFMModel } from "../../Common/makeFMModel";
import { fmParamsList } from "../../@types/fmParams";

import { visualizeFFT } from "../../Common/visualizeFFT";
import { saveAudio } from "../../Common/saveAudio";

interface SoundButtonProps {
  algoNum: number;
  soundParamsList: fmParamsList;
  chromosomeNum: string;
}
export function SoundButton(props: SoundButtonProps) {
  return (
    <Button
      variant="contained"
      color="primary"
      onClick={() => {
        const soundData = makeFMModel(props.algoNum, props.soundParamsList);
        const operatorsInfo = soundData.operatorsInfo;
        const analyzerNodeForSpeaker = soundData.analyzerNodeForSpeaker;
        const streamDestinationNode = soundData.streamDestinationNode;

        // 再生音を記録する 実験時はコメントアウト
        const recorder = new MediaRecorder(streamDestinationNode.stream);
        recorder.addEventListener("dataavailable", function (evt) {
          saveAudio(evt.data, props.chromosomeNum);
        });

        // 音を再生
        Object.keys(operatorsInfo).forEach((key) => {
          operatorsInfo[key].oscillatorNode.start();
          const currentTime = operatorsInfo[key].audioContext.currentTime;
          const soundLength =
            props.soundParamsList[key].attack +
            props.soundParamsList[key].decay +
            props.soundParamsList[key].sustainTime +
            props.soundParamsList[key].release;
          operatorsInfo[key].oscillatorNode.stop(currentTime + soundLength + 1);

          // 音をaudio要素に記録 実験時にはコメントアウト
          if (!operatorsInfo[key].isModulator) {
            recorder.start();
            operatorsInfo[key].oscillatorNode.onended = () => {
              recorder.stop();
            };
          }
        });

        // 出力音の周波数スペクトルを描画
        visualizeFFT(
          new AudioContext(),
          analyzerNodeForSpeaker,
          props.chromosomeNum
        );
      }}
    >
      クリックすると音が鳴ります
    </Button>
  );
}
