import "@testing-library/jest-dom/vitest";

// Reset localStorage between tests so auth helpers start clean.
import { afterEach } from "vitest";

afterEach(() => {
  if (typeof window !== "undefined" && window.localStorage) {
    window.localStorage.clear();
  }
});
