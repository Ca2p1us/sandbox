export function saveAudio(sound: Blob, elementId: string) {
  const audioElement = document.getElementById(
    elementId + "audio"
  ) as HTMLAudioElement;
  if (audioElement) {
    audioElement.src = URL.createObjectURL(sound);
  }
}
