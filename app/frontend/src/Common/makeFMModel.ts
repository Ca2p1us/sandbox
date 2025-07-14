import { fmParamsList } from "../@types/fmParams";
import { operatorParams } from "../@types/operatorParams";

export function makeFMModel(algorithmNum: number, fmParamsList: fmParamsList) {
  // アルゴリズムをセット
  const operatorsInfoWithGainNode = setOperatorsInfo(algorithmNum);
  const operatorsInfo = operatorsInfoWithGainNode["operatorsInfo"];
  const analyzerNodeForSpeaker = operatorsInfoWithGainNode["analyzerNode"];
  const streamDestinationNode =
    operatorsInfoWithGainNode["streamDestinationNode"];
  const audioContext = operatorsInfoWithGainNode["audioContext"];
  // モジュレータの周波数は基音の倍数（非整数倍を含む）
  let fundamentalFrequency: number;

  // 基音を設定;
  Object.keys(operatorsInfo).forEach((key) => {
    const operatorParams = operatorsInfo[key];
    const fmParams = fmParamsList[key];
    if (!operatorParams.isModulator) {
      fundamentalFrequency = fmParams.frequency;
    }
  });

  // オシレータの周波数、変調指数をセット
  Object.keys(operatorsInfo).forEach((key) => {
    const operatorParams = operatorsInfo[key];
    const fmParams = fmParamsList[key];

    if (operatorParams.isModulator) {
      // モジュレーターの周波数を設定
      operatorParams.oscillatorNode.frequency.value =
        // fundamentalFrequency * fmParams.ratioToFundamentalFrequency;
        // それぞれの周波数パラメータに倍率をかけた周波数をモジュレータ周波数とする
        fmParams.frequency * fmParams.ratioToFundamentalFrequency;
      //  モジュレータの変調指数を変更(振幅を変えることと同義)
      for (const gainNodeToDestinaion of Object.values(
        operatorParams.destination
      )) {
        gainNodeToDestinaion.gain.value = fmParams.modulationIndex;
      }
    } else {
      // キャリアの周波数を設定
      operatorParams.oscillatorNode.frequency.value = fmParams.frequency;
    }
  });

  // エンベロープをセット;
  Object.keys(operatorsInfo).forEach((key) => {
    Object.keys(operatorsInfo[key].destination).forEach(
      (gainNodeDestination) => {
        const fmParams = fmParamsList[key];
        const startTime = operatorsInfo[key].startTime;
        const gainNode = operatorsInfo[key].destination[gainNodeDestination];
        // キャリアにADSRを付与 if文をなくせばモジュレータにもADSRを付与
        if (gainNodeDestination == "gainNodeToSpeaker") {
          setEnvelop(startTime, gainNode, fmParams);
        }
      }
    );
  });

  return {
    operatorsInfo: operatorsInfo,
    analyzerNodeForSpeaker: analyzerNodeForSpeaker,
    streamDestinationNode: streamDestinationNode,
  };
}

function setOperatorsInfo(algorithmNum: number) {
  const audioContext: AudioContext = new AudioContext();
  // startTimeは０になる
  const startTime: number = audioContext.currentTime;
  // スピーカに接続するゲインノードを生成（複数のオシレーターからの入力を受け付ける）
  const gainNodeToSpeaker = new GainNode(audioContext);
  const streamDestinationNode = audioContext.createMediaStreamDestination();
  gainNodeToSpeaker.connect(streamDestinationNode);
  gainNodeToSpeaker.connect(audioContext.destination);
  // アナライザーノードに接続
  const analyzerNode = new AnalyserNode(audioContext);
  gainNodeToSpeaker.connect(analyzerNode);

  const operatorsInfo: {
    [key: string]: operatorParams;
  } = {};

  // オペレーターに関する情報をセット
  setAlgorithm(algorithmNum, startTime, audioContext, operatorsInfo);
  for (const operatorParams of Object.values(operatorsInfo)) {
    for (const [destinationNode, gainNode] of Object.entries(
      operatorParams.destination
    )) {
      operatorParams.oscillatorNode.connect(gainNode);
      if (!(destinationNode == "gainNodeToSpeaker")) {
        // 接続先オペレーターの周波数変調を行うために接続
        gainNode.connect(
          operatorsInfo[destinationNode].oscillatorNode.frequency
        );
      } else {
        // スピーカー出力用のゲインノードに接続
        gainNode.connect(gainNodeToSpeaker);
      }
    }
  }

  return {
    audioContext: audioContext,
    operatorsInfo: operatorsInfo,
    gainNodeToSpeaker: gainNodeToSpeaker,
    analyzerNode: analyzerNode,
    streamDestinationNode: streamDestinationNode,
  };
}

function setEnvelop(t0: number, gainNode: GainNode, envelopParams: any) {
  const attack = t0 + envelopParams.attack;
  const decay = envelopParams.decay;
  const sustain = envelopParams.sustain;
  const sustainTime = envelopParams.sustainTime;
  const release = envelopParams.release;
  const gainValue = gainNode.gain.value;

  // ゲインが0から始めるように初期化
  gainNode.gain.setValueAtTime(0, t0);
  // // ゲインの最大までゲインを線形的に増加
  // gainNode.gain.linearRampToValueAtTime(gainValue, t0 + attack);
  // // sustainまでゲインを線形的に減少
  // gainNode.gain.linearRampToValueAtTime(
  //   gainValue * sustain,
  //   t0 + attack + decay
  // );
  // // sustainTime中はsustainで値を固定
  // gainNode.gain.linearRampToValueAtTime(
  //   gainValue * sustain,
  //   t0 + attack + decay + sustainTime
  // );
  // // sustainからゲインを0まで減少
  // gainNode.gain.linearRampToValueAtTime(
  //   0,
  //   t0 + attack + decay + sustainTime + release
  // );

  // ゲインの最大までゲインを線形的に増加
  gainNode.gain.setTargetAtTime(gainValue, t0, attack);
  // sustainまでゲインを線形的に減少
  gainNode.gain.setTargetAtTime(sustain * gainValue, t0 + attack, decay);
  // sustainTime中はsustainで値を固定
  gainNode.gain.setTargetAtTime(
    sustain * gainValue,
    t0 + attack + decay,
    sustainTime
  );
  // sustainからゲインを0まで減少
  gainNode.gain.setTargetAtTime(0, t0 + attack + decay + sustainTime, release);
  console.log("gainNode.gain", gainNode.gain, "gainValue");
}

function setAlgorithm(
  algoNum: number,
  startTime: number,
  audioContext: AudioContext,
  operatorsInfo: { [key: string]: operatorParams | GainNode }
) {
  console.log("algoNum", algoNum, typeof algoNum);

  if (algoNum === 0) {
    //   1オペレータ、1キャリアのアルゴリズム
    operatorsInfo["operator1"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: {
        gainNodeToSpeaker: new GainNode(audioContext),
      },
      isModulator: false,
    };

    operatorsInfo["operator2"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: {
        operator1: new GainNode(audioContext),
      },
      isModulator: true,
    };
  }
  if (algoNum === 1) {
    // 1つのキャリアに対して並列に3つのモジュレータ
    operatorsInfo["operator1"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: { gainNodeToSpeaker: new GainNode(audioContext) },
      isModulator: false,
    };
    operatorsInfo["operator2"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: { operator1: new GainNode(audioContext) },
      isModulator: true,
    };
    operatorsInfo["operator3"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: { operator1: new GainNode(audioContext) },
      isModulator: true,
    };
    operatorsInfo["operator4"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: { operator1: new GainNode(audioContext) },
      isModulator: true,
    };
  }
  if (algoNum === 2) {
    operatorsInfo["operator1"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: { gainNodeToSpeaker: new GainNode(audioContext) },
      isModulator: false,
    };
    operatorsInfo["operator2"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: { operator1: new GainNode(audioContext) },
      isModulator: true,
    };
    operatorsInfo["operator3"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: { operator2: new GainNode(audioContext) },
      isModulator: true,
    };
    operatorsInfo["operator4"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: { operator2: new GainNode(audioContext) },
      isModulator: true,
    };
  }
  if (algoNum === 3) {
    operatorsInfo["operator1"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: { gainNodeToSpeaker: new GainNode(audioContext) },
      isModulator: false,
    };
    operatorsInfo["operator2"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: { operator1: new GainNode(audioContext) },
      isModulator: true,
    };
    operatorsInfo["operator3"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: { operator2: new GainNode(audioContext) },
      isModulator: true,
    };
    operatorsInfo["operator4"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: { operator1: new GainNode(audioContext) },
      isModulator: true,
    };
  }
  if (algoNum === 4) {
    // 1つのキャリアに対して、直列に三つのモジュレータ
    operatorsInfo["operator1"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: { gainNodeToSpeaker: new GainNode(audioContext) },
      isModulator: false,
    };
    operatorsInfo["operator2"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: { operator1: new GainNode(audioContext) },
      isModulator: true,
    };
    operatorsInfo["operator3"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: { operator2: new GainNode(audioContext) },
      isModulator: true,
    };
    operatorsInfo["operator4"] = {
      audioContext: audioContext,
      startTime: startTime,
      oscillatorNode: new OscillatorNode(audioContext),
      analyserNode: new AnalyserNode(audioContext),
      destination: { operator3: new GainNode(audioContext) },
      isModulator: true,
    };
  }
}
