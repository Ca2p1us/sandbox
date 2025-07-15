import { fmParamsList } from "./fmParams";

export type chromosomesParams = {
  chromosome1: chromosomeParams;
  chromosome2: chromosomeParams;
  chromosome3: chromosomeParams;
  chromosome4: chromosomeParams;
  chromosome5: chromosomeParams;
  chromosome6: chromosomeParams;
  chromosome7: chromosomeParams;
  chromosome8: chromosomeParams;
  chromosome9: chromosomeParams;
  chromosome10: chromosomeParams;
};

type chromosomeParams = {
  fmParamsList: fmParamsList;
  fitness: string;
  algorithmNum: number;
  chromosomeId: number;
};
