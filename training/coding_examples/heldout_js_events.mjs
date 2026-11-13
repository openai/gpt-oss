// 未見之題：以事件收發器記輪轉，不借外庫。
import assert from "node:assert/strict";

export class Signal {
  #listeners = new Set();

  on(listener) {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  emit(value) {
    for (const listener of this.#listeners) listener(value);
  }
}

const signal = new Signal();
let sum = 0;
const off = signal.on((value) => { sum += value; });
signal.emit(2);
signal.emit(3);
off();
signal.emit(8);
assert.equal(sum, 5);
