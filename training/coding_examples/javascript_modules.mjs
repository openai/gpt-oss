// ES 模組、私欄、Map 與可選串接；不借外庫。
import assert from "node:assert/strict";

export class Ledger {
  #entries = new Map();

  add(name, turns, buckets) {
    if (typeof name !== "string" || turns < 0 || buckets < 0) {
      throw new TypeError("讀數不合。");
    }
    this.#entries.set(name, { turns, buckets });
    return this;
  }

  lifted(name) {
    const entry = this.#entries.get(name);
    return entry ? entry.turns * entry.buckets : 0;
  }

  names() {
    return [...this.#entries.keys()];
  }
}

const ledger = new Ledger().add("堰流", 3, 4).add("牛力", 2, 2);
assert.equal(ledger.lifted("堰流"), 12);
assert.equal(ledger.lifted("未錄"), 0);
assert.deepEqual(ledger.names(), ["堰流", "牛力"]);
