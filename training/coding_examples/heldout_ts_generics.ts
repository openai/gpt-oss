// 未見之題：以泛型映射與唯讀約束保管讀數。
function select<T, K extends keyof T>(items: readonly T[], key: K): T[K][] {
  return items.map((item) => item[key]);
}

type Reading = { readonly turns: number; readonly buckets: number };

const readings: readonly Reading[] = [
  { turns: 2, buckets: 3 },
  { turns: 4, buckets: 1 },
] as const;

const turns = select(readings, "turns");
if (turns.reduce((sum, value) => sum + value, 0) !== 6) {
  throw new Error("算驗不合。");
}
