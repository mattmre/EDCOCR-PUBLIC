"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, get } from "./api-client";
import {
  applyTheme,
  getDefaultSettings,
  loadSettings,
  resetSettings as resetStoredSettings,
  saveSettings as persistSettings,
} from "./settings-store";
import {
  listAlerts as fetchAlerts,
  listChannels as fetchChannels,
  listRules as fetchRules,
  getRule as fetchRule,
} from "./alerts-api";
import type {
  Alert as AdminAlert,
  AlertRule,
  GlossaryFilters,
  GlossaryListResponse,
  NotificationChannel,
  ReviewItem,
  ReviewQueueFilters,
  ReviewQueueResponse,
  Settings,
  TenantConfig,
} from "./types";

/**
 * Result returned by `useAutoRefresh`. The `refresh` callback runs the
 * fetcher immediately (e.g. on retry button click). `lastUpdated` is the
 * timestamp of the most recent successful fetch.
 */
export interface AutoRefreshResult<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
  lastUpdated: number | null;
  refresh: => void;
}

/**
 * Polls a fetcher on a fixed interval. Skips ticks while the document is
 * hidden (no point burning the API while the user is on another tab).
 *
 * - Cancels in-flight requests on unmount via AbortController.
 * - Cleans up the interval on unmount.
 * - Re-fetches when the tab becomes visible again.
 */
export function useAutoRefresh<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  intervalMs: number
): AutoRefreshResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);

  // Keep a stable ref to the latest fetcher so the polling loop can call
  // it without re-subscribing every render.
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  // Track the active controller so we can abort on unmount or refresh.
  const controllerRef = useRef<AbortController | null>(null);

  const runFetch = useCallback(async => {
    if (controllerRef.current) {
      controllerRef.current.abort();
    }
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    try {
      const result = await fetcherRef.current(controller.signal);
      if (!controller.signal.aborted) {
        setData(result);
        setError(null);
        setLastUpdated(Date.now());
      }
    } catch (err: unknown) {
      if ((err as { name?: string })?.name === "AbortError") {
        return;
      }
      if (!controller.signal.aborted) {
        setError(err instanceof Error ? err : new Error(String(err)));
      }
    } finally {
      if (!controller.signal.aborted) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    // Initial fetch.
    void runFetch();

    const tick = => {
      if (cancelled) return;
      // Skip the tick if the tab is hidden.
      if (typeof document !== "undefined" && document.hidden) {
        return;
      }
      void runFetch();
    };

    const id = setInterval(tick, intervalMs);

    const onVisibility = => {
      if (typeof document !== "undefined" && !document.hidden) {
        void runFetch();
      }
    };
    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", onVisibility);
    }

    return => {
      cancelled = true;
      clearInterval(id);
      if (typeof document !== "undefined") {
        document.removeEventListener("visibilitychange", onVisibility);
      }
      if (controllerRef.current) {
        controllerRef.current.abort();
        controllerRef.current = null;
      }
    };
  }, [intervalMs, runFetch]);

  return { data, error, loading, lastUpdated, refresh: => void runFetch() };
}

// ---------------------------------------------------------------------------
// D8: review queue hooks
// ---------------------------------------------------------------------------

const REVIEW_QUEUE_REFRESH_MS = 30_000;

export interface ReviewQueueQuery extends ReviewQueueFilters {
  limit: number;
  offset: number;
}

/**
 * Build the query string the backend expects. Only the first selected status
 * is forwarded -- the FastAPI handler accepts a single `status` filter. The
 * UI keeps the rest as a client-side display filter.
 */
export function reviewQueueToParams(query: ReviewQueueQuery): URLSearchParams {
  const params = new URLSearchParams();
  if (query.status.length === 1) {
    params.set("status", query.status[0]);
  }
  if (query.reason) {
    params.set("reason", query.reason);
  }
  params.set("limit", String(query.limit));
  params.set("offset", String(query.offset));
  return params;
}

/**
 * Polls `/api/v1/review/queue` every 30 s. Skips ticks while the tab is hidden.
 *
 * Client-side fallbacks layered on top of the server filters:
 *  - Multi-status selection narrows to the union after the fetch returns.
 *  - The `q` substring matches `job_id` (case-insensitive).
 *
 * When the query changes, refetches immediately so filter chips and pagination
 * feel responsive without waiting for the 30 s tick.
 */
export function useReviewQueue(query: ReviewQueueQuery): AutoRefreshResult<ReviewQueueResponse> {
  // Memoise stable derivations so the fetcher identity only changes when the
  // query actually changes.
  const queryKey = JSON.stringify(query);

  const fetcher = useCallback(
    async (signal: AbortSignal): Promise<ReviewQueueResponse> => {
      const params = reviewQueueToParams(query);
      const raw = await get<ReviewQueueResponse>(
        `/api/v1/review/queue?${params.toString()}`,
        { signal }
      );
      let items = raw.items ?? [];
      if (query.status.length > 1) {
        const statusSet = new Set<string>(query.status);
        items = items.filter((it) => statusSet.has(it.status));
      }
      const needle = query.q.trim().toLowerCase();
      if (needle) {
        items = items.filter((it) =>
          it.job_id.toLowerCase().includes(needle) ||
          it.review_id.toLowerCase().includes(needle)
        );
      }
      return { items, total: raw.total ?? items.length };
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [queryKey]
  );

  const result = useAutoRefresh<ReviewQueueResponse>(fetcher, REVIEW_QUEUE_REFRESH_MS);

  // useAutoRefresh keeps the fetcher in a ref and only re-fires on the polling
  // interval. To make filter changes feel snappy, manually trigger a refresh
  // whenever the query key changes.
  const lastKeyRef = useRef<string>(queryKey);
  useEffect(() => {
    if (lastKeyRef.current !== queryKey) {
      lastKeyRef.current = queryKey;
      result.refresh();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryKey]);

  return result;
}

export interface ReviewItemFetchResult {
  data: ReviewItem | null;
  error: Error | null;
  loading: boolean;
  refresh: => void;
}

// ---------------------------------------------------------------------------
// D9: Tenant management hooks
// ---------------------------------------------------------------------------

interface AsyncResource<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
  refresh: => void;
}

/**
 * Fetches a single review item by id. Does NOT auto-refresh -- the detail
 * page needs the caller to refresh explicitly after a decision so the
 * optimistic update can be committed.
 */
export function useReviewItem(reviewId: string | null | undefined): ReviewItemFetchResult {
  const [data, setData] = useState<ReviewItem | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const controllerRef = useRef<AbortController | null>(null);

  const refresh = useCallback(() => {
    if (!reviewId) {
      setData(null);
      setLoading(false);
      return;
    }
    if (controllerRef.current) {
      controllerRef.current.abort();
    }
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    get<ReviewItem>(`/api/v1/review/${encodeURIComponent(reviewId)}`, {
      signal: controller.signal,
    })
      .then((item) => {
        if (!controller.signal.aborted) {
          setData(item);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if ((err as { name?: string })?.name === "AbortError") return;
        if (!controller.signal.aborted) {
          setError(
            err instanceof ApiError
              ? new Error(`${err.status} ${err.message}`)
              : err instanceof Error
                ? err
                : new Error(String(err))
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      });
  }, [reviewId]);

  useEffect(() => {
    refresh();
    return => {
      if (controllerRef.current) {
        controllerRef.current.abort();
        controllerRef.current = null;
      }
    };
  }, [refresh]);

  return { data, error, loading, refresh };
}

/**
 * Lazy fetch + revalidate hook for a tenant translation config.
 *
 * The hook never throws; failure is exposed via `error`. A 404 is mapped to
 * `data = null` (no config row yet) so the UI can show a "create" affordance.
 */
export function useTenantConfig(tenantId: string | null | undefined): AsyncResource<TenantConfig> {
  const [data, setData] = useState<TenantConfig | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [tick, setTick] = useState<number>(0);

  useEffect(() => {
    if (!tenantId) {
      setData(null);
      setError(null);
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
    get<TenantConfig>(`/api/v1/translation/tenants/${encodeURIComponent(tenantId)}/config`, {
      signal: controller.signal,
    })
      .then((res) => {
        if (cancelled) return;
        setData(res);
        setError(null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if ((err as { name?: string })?.name === "AbortError") return;
        const status = (err as { status?: number })?.status;
        if (status === 404) {
          setData(null);
          setError(null);
          return;
        }
        setError(err instanceof Error ? err : new Error(String(err)));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return => {
      cancelled = true;
      controller.abort();
    };
  }, [tenantId, tick]);

  return { data, error, loading, refresh: => setTick((n) => n + 1) };
}

/**
 * Lazy fetch + revalidate hook for tenant glossary entries.
 *
 * Filters are serialized as query params; the hook re-fetches whenever any
 * filter primitive changes.
 */
export function useTenantGlossary(
  tenantId: string | null | undefined,
  filters?: GlossaryFilters
): AsyncResource<GlossaryListResponse> {
  const [data, setData] = useState<GlossaryListResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [tick, setTick] = useState<number>(0);

  const sourceLang = filters?.source_lang ?? "";
  const targetLang = filters?.target_lang ?? "";
  const page = filters?.page ?? 1;
  const pageSize = filters?.page_size ?? 100;

  useEffect(() => {
    if (!tenantId) {
      setData(null);
      setError(null);
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
    const params = new URLSearchParams();
    if (sourceLang) params.set("source_lang", sourceLang);
    if (targetLang) params.set("target_lang", targetLang);
    params.set("page", String(page));
    params.set("page_size", String(pageSize));
    get<GlossaryListResponse>(
      `/api/v1/translation/tenants/${encodeURIComponent(tenantId)}/glossary?${params.toString()}`,
      { signal: controller.signal }
    )
      .then((res) => {
        if (cancelled) return;
        setData(res);
        setError(null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if ((err as { name?: string })?.name === "AbortError") return;
        setError(err instanceof Error ? err : new Error(String(err)));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return => {
      cancelled = true;
      controller.abort();
    };
  }, [tenantId, sourceLang, targetLang, page, pageSize, tick]);

  return { data, error, loading, refresh: => setTick((n) => n + 1) };
}

// ---------------------------------------------------------------------------
// D10: Settings hook
//
// Reads/writes the localStorage-backed Settings shape via `settings-store.ts`.
// `update` performs a shallow-per-section merge; `save` returns true on
// success and false when persistence failed (e.g. quota exceeded).
// ---------------------------------------------------------------------------

/** Deep partial; only used internally by `update`. */
type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K];
};

export interface UseSettingsResult {
  settings: Settings;
  /** Merge a partial patch into `settings` (per-section shallow merge). */
  update: (patch: DeepPartial<Settings>) => void;
  /** Persist the current `settings` to localStorage. Returns success. */
  save: => boolean;
  /** Wipe persisted settings and reset state to defaults. */
  reset: => void;
  /** Timestamp (ms since epoch) of the most recent successful save, or null. */
  lastSavedAt: number | null;
}

function mergeSettings(base: Settings, patch: DeepPartial<Settings>): Settings {
  return {
    schema_version: base.schema_version,
    general: { ...base.general, ...(patch.general ?? {}) },
    api: { ...base.api, ...(patch.api ?? {}) },
    display: { ...base.display, ...(patch.display ?? {}) },
    notifications: { ...base.notifications, ...(patch.notifications ?? {}) },
  };
}

export function useSettings(): UseSettingsResult {
  // Initialize with defaults so SSR has a stable shape, then hydrate from
  // localStorage on mount.
  const [settings, setSettings] = useState<Settings>(() => getDefaultSettings());
  const [lastSavedAt, setLastSavedAt] = useState<number | null>(null);

  // Mirror `settings` into a ref so `save()` can read the latest value
  // synchronously (state setter callbacks aren't guaranteed to run before
  // the caller reads its return value).
  const settingsRef = useRef<Settings>(settings);
  settingsRef.current = settings;

  useEffect(() => {
    const stored = loadSettings();
    setSettings(stored);
    settingsRef.current = stored;
    applyTheme(stored.general.theme);
  }, []);

  const update = useCallback((patch: DeepPartial<Settings>) => {
    // Compute the next value from the ref so back-to-back synchronous calls
    // compose correctly (the React state setter is asynchronous, so reading
    // `settings` here would lose intermediate updates).
    const prev = settingsRef.current;
    const next = mergeSettings(prev, patch);
    if (patch.general?.theme && patch.general.theme !== prev.general.theme) {
      applyTheme(next.general.theme);
    }
    settingsRef.current = next;
    setSettings(next);
  }, []);

  const save = useCallback((): boolean => {
    const ok = persistSettings(settingsRef.current);
    if (ok) {
      setLastSavedAt(Date.now());
    }
    return ok;
  }, []);

  const reset = useCallback(() => {
    resetStoredSettings();
    const defaults = getDefaultSettings();
    setSettings(defaults);
    settingsRef.current = defaults;
    setLastSavedAt(null);
    applyTheme(defaults.general.theme);
  }, []);

  return { settings, update, save, reset, lastSavedAt };
}

// ---------------------------------------------------------------------------
// D11: Alert configuration hooks
// ---------------------------------------------------------------------------

const ACTIVE_ALERTS_REFRESH_MS = 15_000;
const ALERT_RULES_REFRESH_MS = 60_000;
const NOTIFICATION_CHANNELS_REFRESH_MS = 60_000;

/**
 * Polls /api/v1/admin/alerts for currently-firing alerts.
 *
 * Auto-refresh defaults to true (15 s tick) per the D11 spec; pass
 * `false` to disable polling on routes where it would be wasteful.
 */
export function useActiveAlerts(autoRefresh: boolean = true): AutoRefreshResult<AdminAlert[]> {
  const fetcher = useCallback(
    async (signal: AbortSignal): Promise<AdminAlert[]> => fetchAlerts(signal),
    []
  );
  // When polling is disabled, pass a very large interval so the timer never
  // fires within a normal session. Initial fetch still runs on mount.
  const interval = autoRefresh ? ACTIVE_ALERTS_REFRESH_MS : 24 * 60 * 60 * 1000;
  return useAutoRefresh<AdminAlert[]>(fetcher, interval);
}

/** Polls /api/v1/admin/alerts/rules every 60 s. */
export function useAlertRules(): AutoRefreshResult<AlertRule[]> {
  const fetcher = useCallback(
    async (signal: AbortSignal): Promise<AlertRule[]> => fetchRules(signal),
    []
  );
  return useAutoRefresh<AlertRule[]>(fetcher, ALERT_RULES_REFRESH_MS);
}

/**
 * Lazy fetch + revalidate hook for a single alert rule. Does NOT auto-poll;
 * the rule editor page calls `refresh()` after a successful PATCH.
 */
export function useAlertRule(ruleId: string | null | undefined): {
  data: AlertRule | null;
  error: Error | null;
  loading: boolean;
  refresh: => void;
} {
  const [data, setData] = useState<AlertRule | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [tick, setTick] = useState<number>(0);

  useEffect(() => {
    if (!ruleId) {
      setData(null);
      setError(null);
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
    fetchRule(ruleId, controller.signal)
      .then((rule) => {
        if (cancelled) return;
        setData(rule);
        setError(null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if ((err as { name?: string })?.name === "AbortError") return;
        setError(err instanceof Error ? err : new Error(String(err)));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return => {
      cancelled = true;
      controller.abort();
    };
  }, [ruleId, tick]);

  return { data, error, loading, refresh: => setTick((n) => n + 1) };
}

/** Polls /api/v1/admin/alert-channels every 60 s. */
export function useNotificationChannels(): AutoRefreshResult<NotificationChannel[]> {
  const fetcher = useCallback(
    async (signal: AbortSignal): Promise<NotificationChannel[]> => fetchChannels(signal),
    []
  );
  return useAutoRefresh<NotificationChannel[]>(fetcher, NOTIFICATION_CHANNELS_REFRESH_MS);
}

// ---------------------------------------------------------------------------
// D12: feature flag toggle UI hooks
// ---------------------------------------------------------------------------

const FEATURE_FLAGS_REFRESH_MS = 60_000;

/**
 * Polls the full flag list every 60 s. The 60 s cadence is intentionally
 * slower than D5's queue dashboard -- flag values are ops-grade slow-changing
 * state and the API can be relatively expensive on the coordinator side.
 */
export function useFeatureFlags() {
  // Local imports to keep the API surface explicit at the call site -- avoids
  // pulling feature-flags helpers into the hot path of unrelated hooks.
  const fetcher = useCallback(
    async (signal: AbortSignal) => {
      const { listFlags } = await import("./feature-flags-api");
      return listFlags(signal);
    },
    []
  );
  return useAutoRefresh(fetcher, FEATURE_FLAGS_REFRESH_MS);
}

/**
 * Fetch a single flag definition. Does NOT auto-refresh -- the detail page
 * refreshes manually after a change-request lands (analogous to
 * `useReviewItem`).
 */
export function useFeatureFlag(key: string | null | undefined) {
  const [data, setData] = useState<
    import("./types").FeatureFlag | null
  >(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const controllerRef = useRef<AbortController | null>(null);

  const refresh = useCallback(() => {
    if (!key) {
      setData(null);
      setLoading(false);
      return;
    }
    if (controllerRef.current) {
      controllerRef.current.abort();
    }
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    void (async => {
      try {
        const { getFlag } = await import("./feature-flags-api");
        const flag = await getFlag(key, controller.signal);
        if (!controller.signal.aborted) {
          setData(flag);
          setError(null);
        }
      } catch (err: unknown) {
        if ((err as { name?: string })?.name === "AbortError") return;
        if (!controller.signal.aborted) {
          setError(
            err instanceof ApiError
              ? new Error(`${err.status} ${err.message}`)
              : err instanceof Error
                ? err
                : new Error(String(err))
          );
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    })();
  }, [key]);

  useEffect(() => {
    refresh();
    return => {
      if (controllerRef.current) {
        controllerRef.current.abort();
        controllerRef.current = null;
      }
    };
  }, [refresh]);

  return { data, error, loading, refresh };
}

/**
 * Fetch the change history for a flag. Manual refresh only.
 */
export function useFlagHistory(key: string | null | undefined) {
  const [data, setData] = useState<
    import("./types").FeatureFlagHistoryEntry[] | null
  >(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [tick, setTick] = useState<number>(0);

  useEffect(() => {
    if (!key) {
      setData(null);
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
    void (async => {
      try {
        const { getFlagHistory } = await import("./feature-flags-api");
        const entries = await getFlagHistory(key, controller.signal);
        if (cancelled) return;
        setData(entries);
        setError(null);
      } catch (err: unknown) {
        if (cancelled) return;
        if ((err as { name?: string })?.name === "AbortError") return;
        // 404 -> treat as "no history yet".
        const status = (err as { status?: number })?.status;
        if (status === 404) {
          setData([]);
          setError(null);
          return;
        }
        setError(err instanceof Error ? err : new Error(String(err)));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return => {
      cancelled = true;
      controller.abort();
    };
  }, [key, tick]);

  return { data, error, loading, refresh: => setTick((n) => n + 1) };
}
