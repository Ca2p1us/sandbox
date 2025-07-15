export function beepSound(
  soundType: string,
  frequency = 440,
  length: number
) {
  const ctx = new AudioContext();

  const gainNode = ctx.createGain();
  // 音量の初期値を0.5にする
  gainNode.gain.value = 0.05;
  const oscillator = ctx.createOscillator();
  oscillator.type = soundType as "sine" | "sawtooth" | "triangle";
  oscillator.frequency.setValueAtTime(frequency, 0);
  oscillator.connect(gainNode).connect(ctx.destination);
  oscillator.start();
  oscillator.stop(length);
}

export function changeSound(
  soundType: string,
  frequency = 440,
  length: number
) {
  const ctx = new AudioContext();

  const gainNode = ctx.createGain();
  // 音量の初期値を0.5にする
  gainNode.gain.value = 0.05;
  const oscillator = ctx.createOscillator();
  oscillator.type = soundType as "sine" | "sawtooth" | "triangle";
  oscillator.frequency.setValueAtTime(frequency, 0);
  oscillator.connect(gainNode).connect(ctx.destination);
  gainNode.gain.setValueAtTime(0, 0);
  gainNode.gain.linearRampToValueAtTime(1, 2);
  oscillator.start();
  oscillator.stop(length);
}

export function makeSound() {
  const ctx = new AudioContext();
  for (let k = 1; k < 3; k++) {
    const gainNode = ctx.createGain();
    gainNode.gain.value = 0.5;
    const oscillator = ctx.createOscillator();
    oscillator.type = "sine";
    oscillator.frequency.value = (k + 1) * 440;
    oscillator.connect(ctx.destination);
    oscillator.start();
    // oscillator.stop(length);
  }
  // 音量の初期値を0.5にする
  // oscillator.connect(gainNode).connect(ctx.destination);
  // gainNode.gain.setValueAtTime(0,0)
  // gainNode.gain.linearRampToValueAtTime(1,2)
}
