"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { getApiKey } from "@/lib/auth";
import type { JobLogLevel, JobLogRecord } from "@/lib/types";

const API_BASE_URL =
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_API_BASE_URL) ||
  "http://localhost:8000";

const POLL_INTERVAL_MS = 5_000;
const DEFAULT_LIMIT = 500;
const MAX_RETAINED_LINES = 5_000;

const LEVEL_COLORS: Record<string, string> = {
  DEBUG: "text-muted-foreground",
  INFO: "text-foreground",
  WARN: "text-amber-600 dark:text-amber-400",
  WARNING: "text-amber-600 dark:text-amber-400",
  ERROR: "text-red-600 dark:text-red-400",
};

const LEVEL_OPTIONS: Array<{ value: "" | JobLogLevel; label: string }> = [
  { value: "", label: "All levels" },
  { value: "DEBUG", label: "Debug" },
  { value: "INFO", label: "Info" },
  { value: "WARN", label: "Warn" },
  { value: "ERROR", label: "Error" },
];

export interface JobLogsProps {
  jobId: string;
  /** Stop polling automatically (used in tests). */
  pollingEnabled?: boolean;
  /** Override the polling interval (used in tests). */
  pollIntervalMs?: number;
}

interface FetchResult {
  records: JobLogRecord[];
  ok: boolean;
  status: number;
  notFound: boolean;
}

async function fetchLogPage(
  jobId: string,
  opts: { since?: string; level?: string; limit?: number; signal?: AbortSignal }
): Promise<FetchResult> {
  const params = new URLSearchParams();
  if (opts.since) params.set("since", opts.since);
  if (opts.level) params.set("level", opts.level);
  params.set("limit", String(opts.limit ?? DEFAULT_LIMIT));
  const headers = new Headers({ Accept: "application/x-ndjson" });
  const key = getApiKey();
  if (key) headers.set("X-API-Key", key);
  const url = `${API_BASE_URL.replace(/\/+$/, "")}/api/v1/jobs/${jobId}/logs?${params.toString()}`;
  const resp = await fetch(url, { method: "GET", headers, signal: opts.signal, cache: "no-store" });
  if (resp.status === 404) {
    return { records: [], ok: false, status: 404, notFound: true };
  }
  if (!resp.ok) {
    return { records: [], ok: false, status: resp.status, notFound: false };
  }
  const text = await resp.text();
  const records: JobLogRecord[] = [];
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      records.push(JSON.parse(trimmed) as JobLogRecord);
    } catch {
      // Skip malformed records rather than failing the whole stream.
    }
  }
  return { records, ok: true, status: resp.status, notFound: false };
}

export function JobLogs({
  jobId,
  pollingEnabled = true,
  pollIntervalMs = POLL_INTERVAL_MS,
}: JobLogsProps) {
  const [records, setRecords] = useState<JobLogRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [paused, setPaused] = useState<boolean>(false);
  const [levelFilter, setLevelFilter] = useState<"" | JobLogLevel>("");
  const [missing, setMissing] = useState<boolean>(false);

  const lastTsRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const tail = useCallback(
    async (initial: boolean) => {
      if (abortRef.current) abortRef.current.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      const result = await fetchLogPage(jobId, {
        since: lastTsRef.current ?? undefined,
        level: levelFilter || undefined,
        limit: DEFAULT_LIMIT,
        signal: ctrl.signal,
      });
      if (ctrl.signal.aborted) return;

      if (result.notFound) {
        setMissing(true);
        setLoading(false);
        return;
      }
      if (!result.ok) {
        setError(`Log fetch failed (HTTP ${result.status})`);
        setLoading(false);
        return;
      }
      setMissing(false);
      setError(null);

      if (result.records.length > 0) {
        setRecords((prev) => {
          const next = initial ? result.records : [...prev, ...result.records];
          // Cap retained line count so a long-running session doesn't blow up memory.
          return next.length > MAX_RETAINED_LINES
            ? next.slice(next.length - MAX_RETAINED_LINES)
            : next;
        });
        const lastTs = result.records[result.records.length - 1]?.ts;
        if (lastTs) lastTsRef.current = lastTs;
      }
      setLoading(false);
    },
    [jobId, levelFilter]
  );

  // Initial load.
  useEffect(() => {
    setRecords([]);
    setMissing(false);
    setError(null);
    setLoading(true);
    lastTsRef.current = null;
    void tail(true);
    return => {
      if (abortRef.current) abortRef.current.abort();
    };
  }, [jobId, levelFilter, tail]);

  // Auto-tail polling.
  useEffect(() => {
    if (!pollingEnabled || paused || missing) return;
    const id = setInterval(() => {
      void tail(false);
    }, pollIntervalMs);
    return => clearInterval(id);
  }, [pollingEnabled, paused, missing, pollIntervalMs, tail]);

  const visible = useMemo(() => records, [records]);

  if (missing) {
    return (
      <div
        className="rounded-md border border-border bg-background p-4 text-sm"
        data-testid="job-logs-missing"
      >
        <h3 className="text-sm font-semibold">Logs</h3>
        <p className="mt-2 text-muted-foreground">
          No per-job logs are available for {jobId} yet. Logs appear here once the
          pipeline emits its first event.
        </p>
      </div>
    );
  }

  return (
    <div
      className="space-y-3 rounded-md border border-border bg-background p-4 text-sm"
      data-testid="job-logs"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold">Logs</h3>
        <div className="flex items-center gap-2">
          <select
            data-testid="job-logs-level"
            value={levelFilter}
            onChange={(e) => setLevelFilter(e.target.value as "" | JobLogLevel)}
            className="h-8 rounded-md border border-input bg-background px-2 text-xs"
          >
            {LEVEL_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <Button
            variant="outline"
            size="sm"
            data-testid="job-logs-pause"
            onClick={() => setPaused((p) => !p)}
          >
            {paused ? "Resume" : "Pause"}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            data-testid="job-logs-clear"
            onClick={() => {
              setRecords([]);
              lastTsRef.current = null;
            }}
          >
            Clear
          </Button>
        </div>
      </div>

      {error ? (
        <div
          data-testid="job-logs-error"
          className="rounded-md border border-destructive/50 bg-destructive/10 p-2 text-xs text-destructive"
        >
          {error}
        </div>
      ) : null}

      {loading && visible.length === 0 ? (
        <p data-testid="job-logs-loading" className="text-xs text-muted-foreground">
          Loading logs…
        </p>
      ) : visible.length === 0 ? (
        <p data-testid="job-logs-empty" className="text-xs text-muted-foreground">
          No log records yet.
        </p>
      ) : (
        <pre
          data-testid="job-logs-pre"
          className="max-h-96 overflow-auto rounded-md border border-border bg-muted/30 p-3 font-mono text-xs leading-relaxed"
        >
          {visible.map((r, i) => {
            const colorCls = LEVEL_COLORS[String(r.level).toUpperCase()] ?? "text-foreground";
            const ts = r.ts ?? "";
            return (
              <div key={`${r.ts}-${i}`} className={colorCls}>
                <span className="text-muted-foreground">{ts}</span>{" "}
                <span className="font-semibold">[{r.level}]</span>{" "}
                <span className="text-muted-foreground">{r.code}</span>{" "}
                {r.message}
              </div>
            );
          })}
        </pre>
      )}
    </div>
  );
}
