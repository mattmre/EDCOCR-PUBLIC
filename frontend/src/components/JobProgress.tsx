"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cn";
import type { Job, JobWSMessage, WSStatus } from "@/lib/types";

export interface JobProgressProps {
  job: Job;
  wsStatus: WSStatus;
  lastMessage: JobWSMessage | null;
  onReconnect: => void;
}

interface DerivedProgress {
  pagesCompleted: number;
  totalPages: number;
  percent: number;
  stage: string;
}

function deriveProgress(job: Job, last: JobWSMessage | null): DerivedProgress {
  let pagesCompleted = job.progress?.pages_completed ?? 0;
  let totalPages = job.progress?.total_pages ?? 0;
  let percent = job.progress?.percent_complete ?? 0;
  let stage = job.progress?.current_stage ?? job.status;
  if (last && last.type === "progress") {
    if (typeof last.pages_completed === "number") pagesCompleted = last.pages_completed;
    if (typeof last.total_pages === "number") totalPages = last.total_pages;
    if (typeof last.percent === "number") percent = last.percent;
    if (typeof last.current_stage === "string") stage = last.current_stage;
    if (totalPages > 0 && (!percent || percent === 0)) {
      percent = (pagesCompleted / totalPages) * 100;
    }
  }
  return { pagesCompleted, totalPages, percent, stage };
}

function statusLabel(status: WSStatus): string {
  switch (status) {
    case "open":
      return "Live";
    case "connecting":
    case "authenticating":
      return "Connecting…";
    case "reconnecting":
      return "Reconnecting…";
    case "closed":
      return "Disconnected";
    case "error":
      return "Connection error";
    default:
      return "Idle";
  }
}

export function JobProgress({ job, wsStatus, lastMessage, onReconnect }: JobProgressProps) {
  const derived = useMemo(() => deriveProgress(job, lastMessage), [job, lastMessage]);

  // Throughput: pages-per-minute computed from incremental updates.
  const samplesRef = useRef<Array<{ t: number; pages: number }>>([]);
  const [ppm, setPpm] = useState<number | null>(null);
  const [eta, setEta] = useState<string | null>(null);

  useEffect(() => {
    samplesRef.current.push({ t: Date.now(), pages: derived.pagesCompleted });
    // Keep only the last 60 seconds of samples.
    const cutoff = Date.now() - 60_000;
    samplesRef.current = samplesRef.current.filter((s) => s.t >= cutoff);
    if (samplesRef.current.length >= 2) {
      const first = samplesRef.current[0];
      const last = samplesRef.current[samplesRef.current.length - 1];
      const dt = (last.t - first.t) / 1000;
      const dp = last.pages - first.pages;
      if (dt > 0 && dp > 0) {
        const pagesPerMinute = (dp / dt) * 60;
        setPpm(pagesPerMinute);
        const remaining = Math.max(0, derived.totalPages - derived.pagesCompleted);
        if (pagesPerMinute > 0 && remaining > 0) {
          const minutes = remaining / pagesPerMinute;
          if (minutes < 1) setEta(`${(minutes * 60).toFixed(0)}s`);
          else if (minutes < 60) setEta(`${minutes.toFixed(1)}m`);
          else setEta(`${(minutes / 60).toFixed(1)}h`);
        } else {
          setEta(null);
        }
      }
    }
  }, [derived.pagesCompleted, derived.totalPages]);

  const isTerminal =
    job.status === "completed" || job.status === "failed" || job.status === "cancelled";

  return (
    <div className="rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Live progress
        </h2>
        <div className="flex items-center gap-2 text-xs">
          <span
            className={cn(
              "inline-flex items-center gap-1 rounded-full px-2 py-0.5 ring-1 ring-inset",
              wsStatus === "open"
                ? "bg-green-100 text-green-800 ring-green-300"
                : wsStatus === "error"
                  ? "bg-red-100 text-red-800 ring-red-300"
                  : "bg-amber-100 text-amber-900 ring-amber-300"
            )}
            data-testid="ws-status-indicator"
          >
            <span
              className={cn(
                "h-1.5 w-1.5 rounded-full",
                wsStatus === "open" ? "bg-green-600" : "bg-amber-600"
              )}
            />
            {statusLabel(wsStatus)}
          </span>
          {wsStatus !== "open" && !isTerminal ? (
            <Button
              variant="outline"
              size="sm"
              onClick={onReconnect}
              data-testid="ws-reconnect"
            >
              Reconnect
            </Button>
          ) : null}
        </div>
      </div>

      {wsStatus === "closed" || wsStatus === "error" ? (
        <p
          data-testid="ws-disconnect-banner"
          className="mt-2 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-900 ring-1 ring-inset ring-amber-300"
        >
          Live updates are paused. Showing the last known state.
        </p>
      ) : null}

      <dl className="mt-3 grid gap-3 sm:grid-cols-4">
        <div>
          <dt className="text-xs text-muted-foreground">Pages</dt>
          <dd className="font-medium" data-testid="progress-pages">
            {derived.pagesCompleted}
            {derived.totalPages > 0 ? ` / ${derived.totalPages}` : ""}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-muted-foreground">Percent</dt>
          <dd className="font-medium" data-testid="progress-percent">
            {derived.percent ? `${derived.percent.toFixed(1)}%` : "—"}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-muted-foreground">Throughput</dt>
          <dd className="font-medium" data-testid="progress-ppm">
            {ppm !== null ? `${ppm.toFixed(1)} ppm` : "—"}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-muted-foreground">ETA</dt>
          <dd className="font-medium" data-testid="progress-eta">
            {eta ?? "—"}
          </dd>
        </div>
      </dl>

      <div
        className="mt-3 h-2 w-full overflow-hidden rounded-full bg-muted"
        role="progressbar"
        aria-valuenow={Math.round(derived.percent)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className="h-full bg-primary transition-all"
          style={{ width: `${Math.min(100, Math.max(0, derived.percent))}%` }}
          data-testid="progress-bar-fill"
        />
      </div>
      <p className="mt-2 text-xs text-muted-foreground">Stage: {derived.stage}</p>
    </div>
  );
}
