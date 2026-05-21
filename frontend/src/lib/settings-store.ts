"use client";

/**
 * D10 -- client-side, localStorage-backed settings store.
 *
 * Persisted shape lives at `ocr-local:settings`. The store never throws on a
 * read/write failure; it falls back to defaults and logs to the console. The
 * actual operator API key is owned by `auth.ts` (key: `ocr_local_api_key`)
 * and is never duplicated here -- only a redacted preview is kept.
 */

import type {
  Settings,
  SettingsDateFormat,
  SettingsDisplay,
  SettingsNotifications,
  SettingsPageSize,
  SettingsTheme,
} from "./types";

export const SETTINGS_STORAGE_KEY = "ocr-local:settings";
export const SETTINGS_SCHEMA_VERSION: 1 = 1;

const VALID_THEMES: readonly SettingsTheme[] = ["light", "dark", "system"];
const VALID_DATE_FORMATS: readonly SettingsDateFormat[] = ["iso", "us", "eu"];
const VALID_PAGE_SIZES: readonly SettingsPageSize[] = [10, 25, 50, 100];

const REFRESH_MIN = 5;
const REFRESH_MAX = 600;
const TIMEOUT_MIN = 1000;
const TIMEOUT_MAX = 120000;

/** Resolve the browser's IANA timezone, with a hard-coded fallback for SSR. */
function resolveBrowserTimezone(): string {
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    if (typeof tz === "string" && tz.length > 0) {
      return tz;
    }
  } catch {
    // Fall through.
  }
  return "UTC";
}

export function getDefaultSettings(): Settings {
  return {
    schema_version: SETTINGS_SCHEMA_VERSION,
    general: {
      theme: "system",
      timezone: resolveBrowserTimezone(),
      dateFormat: "iso",
    },
    api: {
      baseUrl: "",
      timeoutMs: 30000,
      apiKeyRedactedPreview: "",
    },
    display: {
      autoRefreshSeconds: 30,
      pageSize: 25,
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
  };
}

function clamp(value: number, lo: number, hi: number): number {
  if (!Number.isFinite(value)) return lo;
  return Math.min(hi, Math.max(lo, Math.floor(value)));
}

function pickEnum<T extends string>(
  value: unknown,
  allowed: readonly T[],
  fallback: T
): T {
  if (typeof value === "string" && (allowed as readonly string[]).includes(value)) {
    return value as T;
  }
  return fallback;
}

function pickPageSize(value: unknown, fallback: SettingsPageSize): SettingsPageSize {
  if (typeof value === "number" && (VALID_PAGE_SIZES as readonly number[]).includes(value)) {
    return value as SettingsPageSize;
  }
  return fallback;
}

function pickBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function pickString(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

/**
 * Coerce an arbitrary parsed object into a valid Settings shape, dropping
 * unknown fields and clamping numbers into range. Always returns a complete
 * object; never throws.
 */
export function coerceSettings(raw: unknown): Settings {
  const defaults = getDefaultSettings();
  if (!raw || typeof raw !== "object") {
    return defaults;
  }
  const r = raw as Record<string, unknown>;
  const general = (r.general && typeof r.general === "object"
    ? (r.general as Record<string, unknown>)
    : {}) as Record<string, unknown>;
  const api = (r.api && typeof r.api === "object"
    ? (r.api as Record<string, unknown>)
    : {}) as Record<string, unknown>;
  const display = (r.display && typeof r.display === "object"
    ? (r.display as Record<string, unknown>)
    : {}) as Record<string, unknown>;
  const notifications = (r.notifications && typeof r.notifications === "object"
    ? (r.notifications as Record<string, unknown>)
    : {}) as Record<string, unknown>;

  const dispOut: SettingsDisplay = {
    autoRefreshSeconds: clamp(
      typeof display.autoRefreshSeconds === "number"
        ? display.autoRefreshSeconds
        : defaults.display.autoRefreshSeconds,
      REFRESH_MIN,
      REFRESH_MAX
    ),
    pageSize: pickPageSize(display.pageSize, defaults.display.pageSize),
    compactMode: pickBoolean(display.compactMode, defaults.display.compactMode),
    showAdvancedColumns: pickBoolean(
      display.showAdvancedColumns,
      defaults.display.showAdvancedColumns
    ),
  };

  const notifOut: SettingsNotifications = {
    desktopEnabled: pickBoolean(
      notifications.desktopEnabled,
      defaults.notifications.desktopEnabled
    ),
    soundEnabled: pickBoolean(
      notifications.soundEnabled,
      defaults.notifications.soundEnabled
    ),
    onJobComplete: pickBoolean(
      notifications.onJobComplete,
      defaults.notifications.onJobComplete
    ),
    onJobFailure: pickBoolean(
      notifications.onJobFailure,
      defaults.notifications.onJobFailure
    ),
    onReviewItem: pickBoolean(
      notifications.onReviewItem,
      defaults.notifications.onReviewItem
    ),
  };

  return {
    schema_version: SETTINGS_SCHEMA_VERSION,
    general: {
      theme: pickEnum<SettingsTheme>(general.theme, VALID_THEMES, defaults.general.theme),
      timezone: pickString(general.timezone, defaults.general.timezone),
      dateFormat: pickEnum<SettingsDateFormat>(
        general.dateFormat,
        VALID_DATE_FORMATS,
        defaults.general.dateFormat
      ),
    },
    api: {
      baseUrl: pickString(api.baseUrl, defaults.api.baseUrl),
      timeoutMs: clamp(
        typeof api.timeoutMs === "number" ? api.timeoutMs : defaults.api.timeoutMs,
        TIMEOUT_MIN,
        TIMEOUT_MAX
      ),
      apiKeyRedactedPreview: pickString(
        api.apiKeyRedactedPreview,
        defaults.api.apiKeyRedactedPreview
      ),
    },
    display: dispOut,
    notifications: notifOut,
  };
}

/**
 * Migrate older persisted shapes to the current schema. Today there is only
 * v1, so this is functionally identical to `coerceSettings`, but it gives us
 * a single seam to bump the version in the future.
 */
export function migrateSettings(raw: unknown): Settings {
  // Shape with no schema_version, or any unrecognized version, is coerced to
  // the current shape using the defaults for missing fields.
  return coerceSettings(raw);
}

/** Read settings from localStorage, returning defaults if missing/corrupt. */
export function loadSettings(): Settings {
  if (typeof window === "undefined") {
    return getDefaultSettings();
  }
  try {
    const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (raw === null || raw === "") {
      return getDefaultSettings();
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      // Corrupt JSON -- recover by returning defaults. Do not delete the key
      // here so the operator can inspect it manually if needed.
      return getDefaultSettings();
    }
    return migrateSettings(parsed);
  } catch {
    return getDefaultSettings();
  }
}

/**
 * Persist settings. Returns true on success, false on quota / SecurityError /
 * any other write failure. Never throws so callers can render a UI message
 * without try/catch.
 */
export function saveSettings(value: Settings): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  try {
    const safe = coerceSettings(value);
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(safe));
    return true;
  } catch {
    return false;
  }
}

/** Wipe the persisted settings key. */
export function resetSettings(): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.removeItem(SETTINGS_STORAGE_KEY);
  } catch {
    // Ignore.
  }
}

/**
 * Build a redacted preview from a raw API key. Returns "" for short keys to
 * avoid leaking near-full keys (e.g. an 8-char key would otherwise round-trip
 * the entire value).
 */
export function buildApiKeyRedactedPreview(key: string | null | undefined): string {
  if (!key) return "";
  const trimmed = key.trim();
  if (trimmed.length <= 8) return "";
  return `${trimmed.slice(0, 4)}…${trimmed.slice(-4)}`;
}

/**
 * Apply the theme to the document root. Idempotent and SSR-safe.
 */
export function applyTheme(theme: SettingsTheme): void {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  let resolved: "light" | "dark";
  if (theme === "system") {
    const prefersDark =
      typeof window !== "undefined" &&
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches;
    resolved = prefersDark ? "dark" : "light";
  } else {
    resolved = theme;
  }
  root.classList.remove("theme-light", "theme-dark");
  root.classList.add(resolved === "dark" ? "theme-dark" : "theme-light");
  root.dataset.theme = resolved;
}

/**
 * Curated list of common IANA timezones, used as a fallback when
 * `Intl.supportedValuesOf("timeZone")` is not available in this engine.
 */
export const FALLBACK_TIMEZONES: readonly string[] = [
  "UTC",
  "Etc/GMT",
  "America/New_York",
  "America/Chicago",
  "America/Denver",
  "America/Los_Angeles",
  "America/Anchorage",
  "America/Phoenix",
  "America/Toronto",
  "America/Mexico_City",
  "America/Sao_Paulo",
  "America/Argentina/Buenos_Aires",
  "Europe/London",
  "Europe/Dublin",
  "Europe/Paris",
  "Europe/Berlin",
  "Europe/Madrid",
  "Europe/Rome",
  "Europe/Amsterdam",
  "Europe/Stockholm",
  "Europe/Helsinki",
  "Europe/Athens",
  "Europe/Moscow",
  "Africa/Cairo",
  "Africa/Johannesburg",
  "Asia/Jerusalem",
  "Asia/Dubai",
  "Asia/Kolkata",
  "Asia/Bangkok",
  "Asia/Shanghai",
  "Asia/Hong_Kong",
  "Asia/Singapore",
  "Asia/Tokyo",
  "Asia/Seoul",
  "Australia/Sydney",
  "Pacific/Auckland",
];

/**
 * Best-effort list of IANA timezones, preferring the runtime-provided list
 * when available.
 */
export function listSupportedTimezones(): readonly string[] {
  try {
    const supplier = (
      Intl as unknown as { supportedValuesOf?: (key: string) => string[] }
    ).supportedValuesOf;
    if (typeof supplier === "function") {
      const out = supplier("timeZone");
      if (Array.isArray(out) && out.length > 0) {
        return out;
      }
    }
  } catch {
    // Fall through.
  }
  return FALLBACK_TIMEZONES;
}

/** Validate a non-empty URL-shaped string, allowing "" to mean "current origin". */
export function isValidApiBaseUrl(value: string): boolean {
  if (value === "") return true;
  try {
    // eslint-disable-next-line no-new
    new URL(value);
    return true;
  } catch {
    return false;
  }
}
