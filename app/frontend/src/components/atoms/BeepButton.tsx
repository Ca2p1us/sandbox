import Button from "@material-ui/core/Button";
import { beepSound, changeSound } from "../../Common/beepSound";

type beepSoundType = "sine" | "sawtooth" | "triangle";

interface BeepButtonProps {
  questionNumber: any;
  setQuestionNumber: any;
  type: string;
  frequency: number;
  length: number;
  questionResults: any;
  setQuestionResults: any;
  postResultToDB: () => {};
}
export function BeepButton(props: BeepButtonProps) {
  const questionResults = props.questionResults;
  const questionNumber = props.questionNumber;
  const type = props.type;
  const frequency = props.frequency;
  const length = props.length;
  const soundConfig = type + "-" + frequency + "Hz" + "-" + length + "s";
  const QplusNumber = "Q" + props.questionNumber;
  const changeObj = { [QplusNumber]: soundConfig };
  return (
    <Button
      variant="contained"
      color="secondary"
      onMouseEnter={() => {
        // beepSound(type, frequency, length);
        changeSound("sine", 440, 5);
      }}
      onClick={() => {
        props.setQuestionResults({
          ...props.questionResults,
          ...changeObj,
        });

        props.setQuestionNumber(props.questionNumber + 1);
      }}
    >
      マウスを重ねると音がなります
    </Button>
  );
}
