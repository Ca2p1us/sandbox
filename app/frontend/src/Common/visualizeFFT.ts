export function visualizeFFT(
  audioContext: AudioContext,
  analyserNode: AnalyserNode,
  geneNumber: string
) {
  const canvasElement: HTMLCanvasElement = document.getElementById(
    geneNumber
  ) as HTMLCanvasElement;
  const canvasContext: CanvasRenderingContext2D = canvasElement?.getContext(
    "2d"
  ) as CanvasRenderingContext2D;

  const width = 1000;
  const height = 255;
  canvasElement.width = width;
  canvasElement.height = height;

  canvasContext.strokeStyle = "rgba(0, 0, 255, 1.0)";
  canvasContext.lineWidth = 2;
  canvasContext.lineCap = "round";
  canvasContext.lineJoin = "miter";

  analyserNode.minDecibels = -150; // Default -100 dB
  analyserNode.maxDecibels = 0; // Default  -30 dB

  // for drawing sound wave (spectrum)
  function drawWave() {
    const paddingTop = 20;
    const paddingBottom = 20;
    const paddingLeft = 30;
    const paddingRight = 30;

    const innerWidth = width - paddingLeft - paddingRight;
    const innerHeight = height - paddingTop - paddingBottom;

    const range = analyserNode.maxDecibels - analyserNode.minDecibels;

    // 周波数分解能
    // default サンプリングレート 48000Hz
    // default fftsize 2048
    const frequencyResulution: number =
      audioContext.sampleRate / analyserNode.fftSize;

    // This value is the number of samples during 500 Hz
    const n500Hz = Math.floor(500 / frequencyResulution);

    // Get data for drawing spectrum (dB)
    const spectrums = new Float32Array(analyserNode.frequencyBinCount / 4);
    analyserNode.getFloatFrequencyData(spectrums);

    // Clear previous data
    canvasContext.clearRect(0, 0, width, height);

    // Draw spectrum (dB)
    canvasContext.beginPath();

    for (let i = 0, len = spectrums.length; i < len; i++) {
      const x = Math.floor((i / len) * innerWidth) + paddingLeft;
      const y =
        Math.floor(
          -1 * ((spectrums[i] - analyserNode.maxDecibels) / range) * innerHeight
        ) + paddingTop;

      if (i === 0) {
        canvasContext.moveTo(x, y);
      } else {
        canvasContext.lineTo(x, y);
      }

      if (i % n500Hz === 0) {
        const text = 500 * (i / n500Hz) + " Hz"; // index -> frequency

        // Draw grid (X)
        canvasContext.fillStyle = "rgba(255, 0, 0, 1.0)";
        canvasContext.fillRect(x, paddingTop, 1, innerHeight);

        // Draw text (X)
        canvasContext.fillStyle = "rgba(0, 0, 0, 1.0)";
        canvasContext.font = '12px "Times New Roman"';
        canvasContext.fillText(
          text,
          x - canvasContext.measureText(text).width / 2,
          height - 3
        );
      }
    }

    canvasContext.stroke();

    // Draw grid and text (Y)
    for (
      let i = analyserNode.minDecibels;
      i <= analyserNode.maxDecibels;
      i += 10
    ) {
      const gy =
        Math.floor(
          -1 * ((i - analyserNode.maxDecibels) / range) * innerHeight
        ) + paddingTop;

      // Draw grid (Y)
      canvasContext.fillStyle = "rgba(255, 0, 0, 1.0)";
      canvasContext.fillRect(paddingLeft, gy, innerWidth, 1);

      // Draw text (Y)
      canvasContext.fillStyle = "rgba(0, 0, 0, 1.0)";
      canvasContext.font = '12px "Times New Roman"';
      canvasContext.fillText(i + " dB", 3, gy);
    }
  }
  window.setTimeout(drawWave, 500, analyserNode);
}
