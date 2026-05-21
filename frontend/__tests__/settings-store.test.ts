import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  SETTINGS_SCHEMA_VERSION,
  SETTINGS_STORAGE_KEY,
  applyTheme,
  buildApiKeyRedactedPreview,
  coerceSettings,
  getDefaultSettings,
  isValidApiBaseUrl,
  listSupportedTimezones,
  loadSettings,
  migrateSettings,
  resetSettings,
  saveSettings,
} from "@/lib/settings-store";

describe("settings-store", => {
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
    vi.restoreAllMocks();
    document.documentElement.classList.remove("theme-light", "theme-dark");
    delete document.documentElement.dataset.theme;
  });

  it("returns defaults when nothing is stored", => {
    const s = loadSettings();
    expect(s.schema_version).toBe(SETTINGS_SCHEMA_VERSION);
    expect(s.general.theme).toBe("system");
    expect(s.general.dateFormat).toBe("iso");
    expect(s.api.timeoutMs).toBe(30000);
    expect(s.display.pageSize).toBe(25);
    expect(s.display.autoRefreshSeconds).toBe(30);
    expect(s.display.compactMode).toBe(false);
    expect(s.notifications.onJobComplete).toBe(true);
    expect(s.notifications.onJobFailure).toBe(true);
  });

  it("falls back to defaults when stored JSON is corrupt", => {
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, "not-json{");
    const s = loadSettings();
    expect(s).toEqual(getDefaultSettings());
  });

  it("falls back to defaults when stored value is empty", => {
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, "");
    expect(loadSettings()).toEqual(getDefaultSettings());
  });

  it("migrates older shapes by filling in defaults", => {
    const partial = { general: { theme: "dark" } };
    const migrated = migrateSettings(partial);
    expect(migrated.schema_version).toBe(SETTINGS_SCHEMA_VERSION);
    expect(migrated.general.theme).toBe("dark");
    expect(migrated.general.dateFormat).toBe("iso");
    expect(migrated.api.timeoutMs).toBe(30000);
  });

  it("coerces invalid enum values to defaults", => {
    const raw = {
      general: { theme: "neon", dateFormat: "klingon" },
      display: { pageSize: 7 },
    };
    const out = coerceSettings(raw);
    expect(out.general.theme).toBe("system");
    expect(out.general.dateFormat).toBe("iso");
    expect(out.display.pageSize).toBe(25);
  });

  it("clamps autoRefreshSeconds into [5, 600]", => {
    expect(coerceSettings({ display: { autoRefreshSeconds: -10 } }).display.autoRefreshSeconds).toBe(5);
    expect(coerceSettings({ display: { autoRefreshSeconds: 100000 } }).display.autoRefreshSeconds).toBe(600);
    expect(coerceSettings({ display: { autoRefreshSeconds: 90 } }).display.autoRefreshSeconds).toBe(90);
  });

  it("clamps timeoutMs into [1000, 120000]", => {
    expect(coerceSettings({ api: { timeoutMs: 5 } }).api.timeoutMs).toBe(1000);
    expect(coerceSettings({ api: { timeoutMs: 9999999 } }).api.timeoutMs).toBe(120000);
    expect(coerceSettings({ api: { timeoutMs: 45000 } }).api.timeoutMs).toBe(45000);
  });

  it("round-trips a save through localStorage", => {
    const next = getDefaultSettings();
    next.general.theme = "dark";
    next.display.pageSize = 100;
    expect(saveSettings(next)).toBe(true);
    const back = loadSettings();
    expect(back.general.theme).toBe("dark");
    expect(back.display.pageSize).toBe(100);
  });

  it("returns false from saveSettings when localStorage throws", => {
    const spy = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("QuotaExceededError");
    });
    const ok = saveSettings(getDefaultSettings());
    expect(ok).toBe(false);
    spy.mockRestore();
  });

  it("resetSettings removes the persisted key", => {
    saveSettings(getDefaultSettings());
    expect(window.localStorage.getItem(SETTINGS_STORAGE_KEY)).not.toBeNull();
    resetSettings();
    expect(window.localStorage.getItem(SETTINGS_STORAGE_KEY)).toBeNull();
  });

  it("buildApiKeyRedactedPreview returns empty for short or missing keys", => {
    expect(buildApiKeyRedactedPreview(null)).toBe("");
    expect(buildApiKeyRedactedPreview("")).toBe("");
    expect(buildApiKeyRedactedPreview("12345678")).toBe("");
  });

  it("buildApiKeyRedactedPreview shows first 4 and last 4 of long keys", => {
    const out = buildApiKeyRedactedPreview("abcd1234efgh5678");
    expect(out.startsWith("abcd")).toBe(true);
    expect(out.endsWith("5678")).toBe(true);
    expect(out).not.toContain("1234efgh");
  });

  it("isValidApiBaseUrl accepts empty and absolute URLs", => {
    expect(isValidApiBaseUrl("")).toBe(true);
    expect(isValidApiBaseUrl("https://api.example.com")).toBe(true);
    expect(isValidApiBaseUrl("not a url")).toBe(false);
  });

  it("listSupportedTimezones returns a non-empty list of IANA zones", => {
    const list = listSupportedTimezones();
    expect(Array.isArray(list)).toBe(true);
    expect(list.length).toBeGreaterThan(0);
    // Sanity check: the list should contain at least one well-known IANA zone.
    // Different runtimes vary in whether "UTC" appears alone, so we look for
    // common geographic zones instead.
    expect(
      list.some((tz) => tz === "Europe/London" || tz === "America/New_York")
    ).toBe(true);
  });

  it("applyTheme writes the dataset attribute and class", => {
    applyTheme("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(document.documentElement.classList.contains("theme-dark")).toBe(true);

    applyTheme("light");
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(document.documentElement.classList.contains("theme-light")).toBe(true);
    expect(document.documentElement.classList.contains("theme-dark")).toBe(false);
  });

  it("applyTheme('system') resolves via prefers-color-scheme", => {
    const matchMediaSpy = vi.spyOn(window, "matchMedia").mockImplementation((query: string) => ({
      matches: query.includes("dark"),
      media: query,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
      onchange: null,
    }) as MediaQueryList);
    applyTheme("system");
    expect(document.documentElement.dataset.theme).toBe("dark");
    matchMediaSpy.mockRestore();
  });

  it("coerceSettings handles a non-object input", => {
    expect(coerceSettings(null)).toEqual(getDefaultSettings());
    expect(coerceSettings("hello")).toEqual(getDefaultSettings());
    expect(coerceSettings(42)).toEqual(getDefaultSettings());
  });

  it("coerceSettings tolerates unknown extra fields", => {
    const raw = {
      general: { theme: "light", extra: "ignored" },
      bogus: "section",
    };
    const out = coerceSettings(raw);
    expect(out.general.theme).toBe("light");
    expect((out as unknown as Record<string, unknown>).bogus).toBeUndefined();
  });
});
