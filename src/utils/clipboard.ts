export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* File URL or permission policy: use selection fallback. */
  }
  const previous = document.activeElement as HTMLElement | null;
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("aria-label", "Code to copy");
  area.style.cssText = "position:fixed;left:0;top:0;opacity:0";
  document.body.append(area);
  area.select();
  let copied = false;
  try {
    copied = document.execCommand("copy");
  } catch {
    /* Accurate failure is reported by the caller. */
  }
  area.remove();
  previous?.focus();
  return copied;
}
export function downloadPython(id: string, code: string) {
  const url = URL.createObjectURL(
    new Blob([code + "\n"], { type: "text/x-python;charset=utf-8" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = id + ".py";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
