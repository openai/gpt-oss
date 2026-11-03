// 以介面、聯集、泛型與型別守衛約束讀數。
type Power = "river" | "ox" | "wind";

interface Reading {
  readonly name: string;
  readonly turns: number;
  readonly buckets: number;
  readonly power: Power;
}

function isReading(value: unknown): value is Reading {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return typeof candidate.name === "string"
    && Number.isInteger(candidate.turns)
    && Number.isInteger(candidate.buckets)
    && (candidate.power === "river" || candidate.power === "ox" || candidate.power === "wind");
}

function lift<T extends Reading>(readings: readonly T[]): number {
  return readings.reduce((total, item) => {
    if (item.turns < 0 || item.buckets < 0) {
      throw new Error("轉數與筒數不可為負。");
    }
    return total + item.turns * item.buckets;
  }, 0);
}

const readings: readonly Reading[] = [
  { name: "堰流", turns: 3, buckets: 4, power: "river" },
  { name: "牛力", turns: 2, buckets: 2, power: "ox" },
] as const;

if (!isReading(readings[0]) || lift(readings) !== 16) {
  throw new Error("算驗不合。");
}
