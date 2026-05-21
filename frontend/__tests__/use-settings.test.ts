import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useSettings } from "@/lib/hooks";
import { SETTINGS_STORAGE_KEY, saveSettings } from "@/lib/settings-store";

describe("useSettings", => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.classList.remove("theme-light", "theme-dark");
    delete document.documentElement.dataset.theme;
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("hydrates from localStorage on mount", => {
    const seed = {
      schema_version: 1,
      general: { theme: "dark", timezone: "UTC", dateFormat: "us" },
      api: { baseUrl: "", timeoutMs: 45000, apiKeyRedactedPreview: "" },
      display: {
        autoRefreshSeconds: 60,
        pageSize: 50,
        compactMode: true,
        showAdvancedColumns: false,
      },
      notifications: {
        desktopEnabled: false,
        soundEnabled: false,
        onJobComplete: true,
        onJobFailure: true,
        onReviewItem: false,
      },
    };
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(seed));
    const { result } = renderHook(() => useSettings());
    expect(result.current.settings.general.theme).toBe("dark");
    expect(result.current.settings.api.timeoutMs).toBe(45000);
    expect(result.current.settings.display.pageSize).toBe(50);
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("update merges per-section without dropping other sections", => {
    const { result } = renderHook(() => useSettings());
    const initialTheme = result.current.settings.general.theme;
    act(() => {
      result.current.update({ display: { pageSize: 100 } });
    });
    expect(result.current.settings.display.pageSize).toBe(100);
    expect(result.current.settings.general.theme).toBe(initialTheme);
  });

  it("save persists current state and bumps lastSavedAt", => {
    const { result } = renderHook(() => useSettings());
    act(() => {
      result.current.update({ display: { pageSize: 10 } });
    });
    expect(result.current.lastSavedAt).toBeNull();
    let ok = false;
    act(() => {
      ok = result.current.save();
    });
    expect(ok).toBe(true);
    expect(result.current.lastSavedAt).not.toBeNull();
    const persisted = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    expect(persisted).not.toBeNull();
    expect(JSON.parse(persisted as string).display.pageSize).toBe(10);
  });

  it("save returns false when localStorage throws and does not bump lastSavedAt", => {
    const { result } = renderHook(() => useSettings());
    const spy = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("QuotaExceededError");
    });
    let ok = true;
    act(() => {
      ok = result.current.save();
    });
    expect(ok).toBe(false);
    expect(result.current.lastSavedAt).toBeNull();
    spy.mockRestore();
  });

  it("reset wipes localStorage and restores defaults", => {
    saveSettings({
      schema_version: 1,
      general: { theme: "dark", timezone: "UTC", dateFormat: "iso" },
      api: { baseUrl: "", timeoutMs: 30000, apiKeyRedactedPreview: "" },
      display: {
        autoRefreshSeconds: 30,
        pageSize: 100,
        compactMode: false,
        showAdvancedColumns: false,
      },
      notifications: {
        desktopEnabled: false,
        soundEnabled: false,
        onJobComplete: true,
        onJobFailure: true,
        onReviewItem: false,
      },
    });
    const { result } = renderHook(() => useSettings());
    expect(result.current.settings.display.pageSize).toBe(100);
    act(() => {
      result.current.reset();
    });
    expect(result.current.settings.display.pageSize).toBe(25);
    expect(result.current.settings.general.theme).toBe("system");
    expect(window.localStorage.getItem(SETTINGS_STORAGE_KEY)).toBeNull();
    expect(result.current.lastSavedAt).toBeNull();
  });

  it("update of theme applies the theme to the document immediately", => {
    const { result } = renderHook(() => useSettings());
    act(() => {
      result.current.update({ general: { theme: "light" } });
    });
    expect(document.documentElement.dataset.theme).toBe("light");
    act(() => {
      result.current.update({ general: { theme: "dark" } });
    });
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("multiple updates stack and a single save persists the merged value", => {
    const { result } = renderHook(() => useSettings());
    act(() => {
      result.current.update({ display: { compactMode: true } });
      result.current.update({ display: { showAdvancedColumns: true } });
      result.current.update({ general: { dateFormat: "eu" } });
      result.current.save();
    });
    const stored = JSON.parse(window.localStorage.getItem(SETTINGS_STORAGE_KEY) as string);
    expect(stored.display.compactMode).toBe(true);
    expect(stored.display.showAdvancedColumns).toBe(true);
    expect(stored.general.dateFormat).toBe("eu");
  });
});
