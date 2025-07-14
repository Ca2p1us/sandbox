// const bufferSize = 1024;
// const audioData: any = [];

// const onAudioProcess = (e: AudioProcessingEvent) => {
//   var input = e.inputBuffer.getChannelData(0);
//   var bufferData = new Float32Array(bufferSize);
//   for (var i = 0; i < bufferSize; i++) {
//     bufferData[i] = input[i];
//   }
//   audioData.push(bufferData);
// };

// export const saveBuff = (ctx: AudioContext, gain: GainNode) => {
//   const scriptProcessor = ctx.createScriptProcessor(bufferSize, 1, 1);
//   gain.connect(scriptProcessor);
//   scriptProcessor.onaudioprocess = onAudioProcess;
//   //scriptProcessor.connect(ctx.destination);
//   setTimeout(() => {
//     const downloadLink = document.createElement("a");
//     downloadLink.href = exportWAV(audioData, ctx.sampleRate);
//     downloadLink.download = "test.wav";
//     downloadLink.click();
//   }, 10000);
// };

// /**
// Buffer -> URL
// */
// const exportWAV = (audioData: any, audio_sample_rate: number) => {
//   const encodeWAV = (samples: Float32Array, sampleRate: number) => {
//     const buffer = new ArrayBuffer(44 + samples.length * 2);
//     const view = new DataView(buffer);

//     const writeString = (view: any, offset: any, str: string) => {
//       for (let i = 0; i < str.length; i++) {
//         view.setUint8(offset + i, str.charCodeAt(i));
//       }
//     };

//     const floatTo16BitPCM = (output: any, offset: any, input: any) => {
//       for (let i = 0; i < input.length; i++, offset += 2) {
//         const s = Math.max(-1, Math.min(1, input[i]));
//         output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
//       }
//     };

//     writeString(view, 0, "RIFF"); // RIFFヘッダ
//     view.setUint32(4, 32 + samples.length * 2, true); // これ以降のファイルサイズ
//     writeString(view, 8, "WAVE"); // WAVEヘッダ
//     writeString(view, 12, "fmt "); // fmtチャンク
//     view.setUint32(16, 16, true); // fmtチャンクのバイト数
//     view.setUint16(20, 1, true); // フォーマットID
//     view.setUint16(22, 1, true); // チャンネル数
//     view.setUint32(24, sampleRate, true); // サンプリングレート
//     view.setUint32(28, sampleRate * 2, true); // データ速度
//     view.setUint16(32, 2, true); // ブロックサイズ
//     view.setUint16(34, 16, true); // サンプルあたりのビット数
//     writeString(view, 36, "data"); // dataチャンク
//     view.setUint32(40, samples.length * 2, true); // 波形データのバイト数
//     floatTo16BitPCM(view, 44, samples); // 波形データ

//     return view;
//   };

//   const mergeBuffers = (audioData: any) => {
//     const sl = audioData.reduce((a: number, c: any) => a + c.length, 0);
//     const samples = new Float32Array(sl);
//     let sampleIdx = 0;
//     for (let i = 0; i < audioData.length; i++) {
//       for (let j = 0; j < audioData[i].length; j++) {
//         samples[sampleIdx] = audioData[i][j];
//         sampleIdx++;
//       }
//     }
//     return samples;
//   };

//   const dataview = encodeWAV(mergeBuffers(audioData), audio_sample_rate);
//   const audioBlob = new Blob([dataview], { type: "audio/wav" });
//   const myURL = window.URL || window.webkitURL;
//   const url = myURL.createObjectURL(audioBlob);
//   return url;
// };
