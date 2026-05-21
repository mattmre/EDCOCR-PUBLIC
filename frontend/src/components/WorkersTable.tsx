"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { CapabilityBadge } from "@/components/CapabilityBadge";
import { RelativeTime } from "@/components/RelativeTime";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/cn";
import type { Worker, WorkerStatus } from "@/lib/types";

const STATUS_PALETTE: Record<WorkerStatus | "stale", string> = {
  online: "bg-emerald-100 text-emerald-800 border-emerald-300",
  busy: "bg-emerald-100 text-emerald-800 border-emerald-300",
  idle: "bg-sky-100 text-sky-800 border-sky-300",
  draining: "bg-amber-100 text-amber-800 border-amber-300",
  offline: "bg-gray-200 text-gray-700 border-gray-300",
  error: "bg-red-100 text-red-800 border-red-300",
  stale: "bg-amber-100 text-amber-800 border-amber-300",
};

export interface WorkersTableProps {
  workers: Worker[];
  staleThresholdSeconds: number;
  /** When set, the table filters to only stale workers regardless of chips. */
  staleOnly?: boolean;
  /** Optional initial selections, used in tests for deterministic state. */
  initialCapabilityFilter?: string[];
  initialStatusFilter?: Array<WorkerStatus | "stale">;
  pageSize?: number;
}

type SortKey = "heartbeat" | "hostname" | "uptime";

function isStale(worker: Worker, thresholdSec: number, nowSec: number): boolean {
  if (worker.state === "offline") return false;
  if (worker.last_heartbeat <= 0) return true;
  return nowSec - worker.last_heartbeat > thresholdSec;
}

function uniqueCapabilities(workers: Worker[]): string[] {
  const set = new Set<string>();
  for (const w of workers) for (const c of w.capabilities) set.add(c);
  return Array.from(set).sort();
}

const ALL_STATUSES: Array<WorkerStatus | "stale"> = [
  "online",
  "busy",
  "idle",
  "stale",
  "offline",
  "error",
];

export function WorkersTable({
  workers,
  staleThresholdSeconds,
  staleOnly = false,
  initialCapabilityFilter,
  initialStatusFilter,
  pageSize = 25,
}: WorkersTableProps) {
  const allCapabilities = useMemo(() => uniqueCapabilities(workers), [workers]);
  const [capFilter, setCapFilter] = useState<string[]>(initialCapabilityFilter ?? []);
  const [statusFilter, setStatusFilter] = useState<Array<WorkerStatus | "stale">>(
    initialStatusFilter ?? []
  );
  const [sortKey, setSortKey] = useState<SortKey>("heartbeat");
  const [sortDesc, setSortDesc] = useState(true);
  const [page, setPage] = useState(0);

  const nowSec = Date.now() / 1000;

  const filtered = useMemo(() => {
    return workers.filter((worker) => {
      if (staleOnly && !isStale(worker, staleThresholdSeconds, nowSec)) return false;
      if (capFilter.length > 0) {
        const hit = worker.capabilities.some((c) => capFilter.includes(c));
        if (!hit) return false;
      }
      if (statusFilter.length > 0) {
        const stale = isStale(worker, staleThresholdSeconds, nowSec);
        const matches = statusFilter.some((s) => {
          if (s === "stale") return stale;
          return s === worker.state;
        });
        if (!matches) return false;
      }
      return true;
    });
  }, [workers, staleOnly, staleThresholdSeconds, nowSec, capFilter, statusFilter]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      let va: string | number;
      let vb: string | number;
      switch (sortKey) {
        case "hostname":
          va = a.hostname;
          vb = b.hostname;
          break;
        case "uptime":
          va = a.uptime_seconds;
          vb = b.uptime_seconds;
          break;
        case "heartbeat":
        default:
          va = a.last_heartbeat;
          vb = b.last_heartbeat;
      }
      if (va < vb) return sortDesc ? 1 : -1;
      if (va > vb) return sortDesc ? -1 : 1;
      return 0;
    });
    return arr;
  }, [filtered, sortKey, sortDesc]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const safePage = Math.min(page, totalPages - 1);
  const pageRows = sorted.slice(safePage * pageSize, safePage * pageSize + pageSize);

  const toggleCap = (cap: string) => {
    setPage(0);
    setCapFilter((cur) =>
      cur.includes(cap) ? cur.filter((c) => c !== cap) : [...cur, cap]
    );
  };
  const toggleStatus = (status: WorkerStatus | "stale") => {
    setPage(0);
    setStatusFilter((cur) =>
      cur.includes(status) ? cur.filter((s) => s !== status) : [...cur, status]
    );
  };
  const setSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDesc((d) => !d);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Workers</CardTitle>
        <div className="flex flex-wrap gap-2 pt-2">
          {allCapabilities.length > 0 && (
            <div
              className="flex flex-wrap items-center gap-1"
              role="group"
              aria-label="Filter by capability"
            >
              <span className="text-xs text-muted-foreground">capability:</span>
              {allCapabilities.map((cap) => {
                const active = capFilter.includes(cap);
                return (
                  <button
                    key={cap}
                    type="button"
                    onClick={() => toggleCap(cap)}
                    aria-pressed={active}
                    data-testid={`cap-chip-${cap}`}
                    className={cn(
                      "rounded-full border px-2 py-0.5 text-xs",
                      active
                        ? "border-primary bg-primary text-primary-foreground"
                        : "border-border bg-background hover:bg-accent"
                    )}
                  >
                    {cap}
                  </button>
                );
              })}
            </div>
          )}
          <div
            className="flex flex-wrap items-center gap-1"
            role="group"
            aria-label="Filter by status"
          >
            <span className="text-xs text-muted-foreground">status:</span>
            {ALL_STATUSES.map((status) => {
              const active = statusFilter.includes(status);
              return (
                <button
                  key={status}
                  type="button"
                  onClick={() => toggleStatus(status)}
                  aria-pressed={active}
                  data-testid={`status-chip-${status}`}
                  className={cn(
                    "rounded-full border px-2 py-0.5 text-xs capitalize",
                    active
                      ? "border-primary bg-primary text-primary-foreground"
                      : "border-border bg-background hover:bg-accent"
                  )}
                >
                  {status}
                </button>
              );
            })}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {sorted.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">No workers match the current filters.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm" data-testid="workers-table">
              <thead>
                <tr className="border-b border-border text-left text-xs text-muted-foreground">
                  <Th onClick={() => setSort("hostname")} active={sortKey === "hostname"} desc={sortDesc}>
                    Host
                  </Th>
                  <Th>Capabilities</Th>
                  <Th>Status</Th>
                  <Th onClick={() => setSort("heartbeat")} active={sortKey === "heartbeat"} desc={sortDesc}>
                    Heartbeat
                  </Th>
                  <Th>Current job</Th>
                  <Th>GPU</Th>
                  <Th onClick={() => setSort("uptime")} active={sortKey === "uptime"} desc={sortDesc}>
                    Uptime
                  </Th>
                </tr>
              </thead>
              <tbody>
                {pageRows.map((worker) => {
                  const stale = isStale(worker, staleThresholdSeconds, nowSec);
                  const displayState: WorkerStatus | "stale" = stale ? "stale" : worker.state;
                  const palette = STATUS_PALETTE[displayState];
                  const gpuLabel = worker.gpus.length > 0
                    ? worker.gpus.map((g) => `#${g.gpu_id}`).join(", ")
                    : "—";
                  return (
                    <tr
                      key={worker.worker_id}
                      data-testid={`worker-row-${worker.worker_id}`}
                      className="border-b border-border/60 last:border-0"
                    >
                      <td className="py-2 font-medium">{worker.hostname || worker.worker_id}</td>
                      <td className="py-2">
                        <div className="flex flex-wrap gap-1">
                          {worker.capabilities.length === 0 ? (
                            <span className="text-xs text-muted-foreground">—</span>
                          ) : (
                            worker.capabilities.map((c) => (
                              <CapabilityBadge key={c} capability={c} />
                            ))
                          )}
                        </div>
                      </td>
                      <td className="py-2">
                        <span
                          data-testid={`worker-status-${worker.worker_id}`}
                          className={cn(
                            "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium capitalize",
                            palette
                          )}
                        >
                          {displayState}
                        </span>
                      </td>
                      <td className="py-2 text-muted-foreground">
                        <RelativeTime epochSeconds={worker.last_heartbeat} />
                      </td>
                      <td className="py-2">
                        {worker.current_job_id ? (
                          <Link
                            href={`/jobs/${worker.current_job_id}`}
                            className="text-primary underline-offset-2 hover:underline"
                            data-testid={`worker-job-link-${worker.worker_id}`}
                          >
                            {worker.current_job_id}
                          </Link>
                        ) : (
                          <span className="text-muted-foreground">—</span>
                        )}
                      </td>
                      <td className="py-2 text-muted-foreground">{gpuLabel}</td>
                      <td className="py-2 text-muted-foreground">
                        {worker.uptime_seconds > 0
                          ? `${Math.floor(worker.uptime_seconds / 60)}m`
                          : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        {totalPages > 1 && (
          <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
            <span>
              Page {safePage + 1} of {totalPages} · {sorted.length} workers
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                disabled={safePage === 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                className="rounded-md border border-border px-2 py-1 disabled:opacity-50"
              >
                Prev
              </button>
              <button
                type="button"
                disabled={safePage >= totalPages - 1}
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                className="rounded-md border border-border px-2 py-1 disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

interface ThProps {
  children: React.ReactNode;
  onClick?: => void;
  active?: boolean;
  desc?: boolean;
}

function Th({ children, onClick, active, desc }: ThProps) {
  if (!onClick) {
    return <th className="py-2 pr-3 font-medium">{children}</th>;
  }
  return (
    <th className="py-2 pr-3 font-medium">
      <button
        type="button"
        onClick={onClick}
        className={cn(
          "inline-flex items-center gap-1 hover:text-foreground",
          active && "text-foreground"
        )}
      >
        {children}
        {active ? <span aria-hidden>{desc ? "↓" : "↑"}</span> : null}
      </button>
    </th>
  );
}
