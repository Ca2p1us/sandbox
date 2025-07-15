export function getCookie(cookieKey: string) {
  if (document.cookie != "") {
    const cookieValue = document.cookie
      .split("; ")
      .find((row) => row.startsWith(cookieKey))
      ?.split("=")[1];
    return cookieValue;
  }
}
