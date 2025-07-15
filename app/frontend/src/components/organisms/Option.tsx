import React from "react";
import { SoundButton } from "../atoms/SoundButton";
// import { SelectFitness } from "../atoms/SelectAbsoluteFitness";
import { SelectFitness } from "../atoms/SelectFitness";
import { Canvas } from "../molecules/Canvas";
import { ParametersTable } from "../molecules/ParametersTable";
import { fmParamsList } from "../../@types/fmParams";
import { chromosomesParams } from "../../@types/chromosomesParams";

interface OptionProps {
  chromosomesParams: chromosomesParams;
  setChromosomesParameters: React.Dispatch<
    React.SetStateAction<chromosomesParams | null>
  >;
  chromosomeNumber: string;
  soundParamsList: fmParamsList;
  algorithmNum: number;
  chromosomesFitnessValue: string;
}

export function Option(props: OptionProps) {
  return (
    <div className="mt--md">
      <SoundButton
        algoNum={props.algorithmNum}
        soundParamsList={props.soundParamsList}
        chromosomeNum={props.chromosomeNumber}
      ></SoundButton>
      <SelectFitness
        chromosomeNumber={props.chromosomeNumber}
        chromosomesParams={props.chromosomesParams}
        setChromosomesParams={props.setChromosomesParameters}
        fitnessValue={props.chromosomesFitnessValue}
      ></SelectFitness>
      <Canvas chromosomeNumber={props.chromosomeNumber}></Canvas>
      <ParametersTable
        fmParamsList={props.soundParamsList}
        algorithmNum={props.algorithmNum}
      ></ParametersTable>
    </div>
  );
}
