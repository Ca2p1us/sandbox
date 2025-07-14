export type fmParamsList = {
  [key: string]: fmParamsType;
};

export type fmParamsType = {
  attack: number;
  decay: number;
  sustain: number;
  sustainTime: number;
  release: number;
  frequency: number;
  // キャリア周波数に対するモジュレータの周波数の比率
  ratioToFundamentalFrequency: number;
  //   変調指数
  modulationIndex: number;
};
