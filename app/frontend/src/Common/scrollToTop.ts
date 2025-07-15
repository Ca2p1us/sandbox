export function scrollToTop(position: HTMLElement) {
  const rect = position?.getBoundingClientRect();
  if (rect) {
    const elemtop = rect.top + window.pageYOffset;
    document.documentElement.scrollTop = elemtop;
  }
}
