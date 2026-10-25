const memory = new Map<string, string>();
export function readStored<T>(key: string, fallback: T): T {
  try {
    const value = localStorage.getItem(key) ?? memory.get(key);
    return value ? JSON.parse(value) : fallback;
  } catch {
    return memory.has(key) ? JSON.parse(memory.get(key)!) : fallback;
  }
}
export function writeStored(key: string, value: unknown) {
  const data = JSON.stringify(value);
  memory.set(key, data);
  try {
    localStorage.setItem(key, data);
  } catch {
    /* Private/file policies retain session memory. */
  }
}
