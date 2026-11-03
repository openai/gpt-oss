// 以 Promise、async/await 與 Result 型別處置並舉之算。
type Result<T, E> = { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };

async function measure(turns: number, buckets: number): Promise<Result<number, string>> {
  if (turns < 0 || buckets < 0) {
    return { ok: false, error: "負數不合" };
  }
  return { ok: true, value: turns * buckets };
}

async function total(readings: readonly (readonly [number, number])[]): Promise<number> {
  const settled = await Promise.all(
    readings.map(([turns, buckets]) => measure(turns, buckets)),
  );
  let sum = 0;
  for (const item of settled) {
    if (!item.ok) throw new Error(item.error);
    sum += item.value;
  }
  return sum;
}

async function verify(): Promise<void> {
  const value = await total([[2, 3], [4, 1]]);
  if (value !== 10) throw new Error("算驗不合。");
}

void verify();
