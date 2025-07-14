import React from "react";
import ReactDOM from "react-dom";
import { ResearchIGA } from "./components/pages/ResearchIGA";
import { Evaluation } from "./components/pages/Evaluation";
import "./css/index.scss";

declare global {
  interface Window {
    webkitAudioContext: any;
  }
}

window.onload = () => {
  if (document.getElementById("research-iga")) {
    ReactDOM.render(<ResearchIGA />, document.getElementById("research-iga"));
  }
  if (document.getElementById("research-evaluation")) {
    ReactDOM.render(
      <Evaluation />,
      document.getElementById("research-evaluation")
    );
  }
  window.AudioContext = window.AudioContext || window.webkitAudioContext;
};
