import React from "react";
import { fmParamsList } from "../../@types/fmParams";

interface ParametersTableProps {
  fmParamsList: fmParamsList;
  algorithmNum: number;
}
export function ParametersTable(props: ParametersTableProps) {
  return (
    <table className="border text-center">
      <tbody>
        <tr>
          <th>operator</th>
          <th>attack</th>
          <th>decay</th>
          <th>sustain</th>
          <th>sustainTime</th>
          <th>release</th>
          <th>frequency</th>
          <th>ratioToFrequency</th>
          <th>modulationIndex</th>
          <th>algoNum</th>
        </tr>
        {Object.keys(props.fmParamsList).map((key: string, index: number) => {
          return (
            <tr
              key={
                props.fmParamsList[key].sustain +
                props.fmParamsList[key].modulationIndex +
                index
              }
            >
              <td>{key}</td>
              <td>{props.fmParamsList[key].attack}</td>
              <td>{props.fmParamsList[key].decay}</td>
              <td>{props.fmParamsList[key].sustain}</td>
              <td>{props.fmParamsList[key].sustainTime}</td>
              <td>{props.fmParamsList[key].release}</td>
              <td>{props.fmParamsList[key].frequency}</td>
              <td>{props.fmParamsList[key].ratioToFundamentalFrequency}</td>
              <td>{props.fmParamsList[key].modulationIndex}</td>
              <td>{props.algorithmNum}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
