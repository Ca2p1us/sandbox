// 各オペレータがもつ属性
export type operatorParams = {
  startTime: number;
  audioContext: AudioContext;
  oscillatorNode: OscillatorNode;
  analyserNode: AnalyserNode;
  destination: { [key: string]: GainNode };
  isModulator: boolean;
};
