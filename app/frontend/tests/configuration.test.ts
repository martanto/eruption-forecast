import { describe, expect, it } from "vitest";
import { merge, updateAt, filename } from "../src/lib/configuration";

describe("configuration semantics", () => {
  it("preserves null and false without changing shared defaults", () => {
    const shared = {
      constructor: { overwrite: true },
      fit: { seeds: 10, resample_method: "auto" },
    };
    expect(
      merge(shared, {
        constructor: { overwrite: false },
        fit: { resample_method: null },
      }),
    ).toEqual({
      constructor: { overwrite: false },
      fit: { seeds: 10, resample_method: null },
    });
    expect(shared.fit.resample_method).toBe("auto");
  });
  it("removes overrides for inheritance", () => {
    expect(
      updateAt({ fit: { seeds: 2 } }, ["fit", "seeds"], undefined, true),
    ).toEqual({ fit: {} });
  });
  it("handles Windows artifact paths", () => {
    expect(filename("D:\\output\\training\\model.pkl")).toBe("model.pkl");
  });
});
