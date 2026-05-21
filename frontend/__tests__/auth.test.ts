import { describe, expect, it } from "vitest";
import {
  AUTH_STORAGE_KEY,
  clearApiKey,
  getApiKey,
  setApiKey,
} from "@/lib/auth";

describe("auth helpers", => {
  it("returns null when no key is stored", => {
    expect(getApiKey()).toBeNull();
  });

  it("round-trips a key through localStorage", => {
    setApiKey("ocr-secret-key");
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBe("ocr-secret-key");
    expect(getApiKey()).toBe("ocr-secret-key");
  });

  it("trims whitespace before storing", => {
    setApiKey("  ocr-trimmed  ");
    expect(getApiKey()).toBe("ocr-trimmed");
  });

  it("rejects empty keys", => {
    expect(() => setApiKey("   ")).toThrow(/empty/i);
  });

  it("treats blank stored values as missing", => {
    window.localStorage.setItem(AUTH_STORAGE_KEY, "   ");
    expect(getApiKey()).toBeNull();
  });

  it("clears the stored key", => {
    setApiKey("ocr-clear-me");
    clearApiKey();
    expect(getApiKey()).toBeNull();
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });
});
