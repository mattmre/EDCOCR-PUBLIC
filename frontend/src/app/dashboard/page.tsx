"use client";

import { useCallback } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { MetricCard } from "@/components/metric-card";
import { RecentJobsTable } from "@/components/recent-jobs-table";
import { StatusBadge } from "@/components/status-badge";
import { ApiError, get } from "@/lib/api-client";
import { useRequireAuth } from "@/lib/auth";
import { useAutoRefresh } from "@/lib/hooks";
import type {
  DashboardSnapshot,
  DetailedHealthResponse,
  FleetSnapshot,
  JobListResponse,
  SubsystemCheck,
} from "@/lib/types";

const REFRESH_INTERVAL_MS = 5_000;
const RECENT_JOBS_LIMIT = 10;

/**
 * Treat 404 responses as "endpoint not enabled" rather than an error
 * so the dashboard still renders when ENABLE_DASHBOARD is unset on the
 * backend. Other ApiErrors propagate.
 */
async function getOrNull<T>(path: string, signal: AbortSignal): Promise<T | null> {
  try {
    return await get<T>(path, { signal });
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      return null;
    }
    throw err;
  }
}

function formatNumber(n: number, fractionDigits = 0): string {
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, {
    maximumFractionDigits: fractionDigits,
    minimumFractionDigits: 0,
  });
}

function pipelineHealthTone(status: string | undefined): {
  tone: "success" | "warning" | "danger";
  label: string;
} {
  if (status === "healthy") return { tone: "success", label: "Healthy" };
  if (status === "degraded") return { tone: "warning", label: "Degraded" };
  if (status === "unhealthy") return { tone: "danger", label: "Down" };
  return { tone: "warning", label: status ?? "Unknown" };
}

function subsystemTone(check: SubsystemCheck | undefined): {
  tone: "success" | "warning" | "danger";
  label: string;
} {
  if (!check) return { tone: "warning", label: "Unknown" };
  if (check.status === "healthy") return { tone: "success", label: "Ready" };
  if (check.status === "degraded") return { tone: "warning", label: "Check" };
  if (check.status === "unhealthy") return { tone: "danger", label: "Down" };
  return { tone: "warning", label: check.status || "Unknown" };
}

function describeError(err: Error): string {
  if (err instanceof ApiError) {
    return `${err.status} ${err.message}`;
  }
  return err.message || "unknown error";
}

export default function DashboardPage() {
  useRequireAuth();

  const fetchHealth = useCallback(
    (signal: AbortSignal) =>
      get<DetailedHealthResponse>("/api/v1/health/detailed", { signal }),
    []
  );
  const fetchSnapshot = useCallback(
    (signal: AbortSignal) =>
      getOrNull<DashboardSnapshot>("/api/v1/dashboard", signal),
    []
  );
  const fetchFleet = useCallback(
    (signal: AbortSignal) => getOrNull<FleetSnapshot>("/api/v1/fleet", signal),
    []
  );
  const fetchJobs = useCallback(
    (signal: AbortSignal) =>
      get<JobListResponse>(`/api/v1/jobs?limit=${RECENT_JOBS_LIMIT}&offset=0`, {
        signal,
      }),
    []
  );

  const health = useAutoRefresh(fetchHealth, REFRESH_INTERVAL_MS);
  const snapshot = useAutoRefresh(fetchSnapshot, REFRESH_INTERVAL_MS);
  const fleet = useAutoRefresh(fetchFleet, REFRESH_INTERVAL_MS);
  const jobs = useAutoRefresh(fetchJobs, REFRESH_INTERVAL_MS);

  const refreshAll = => {
    health.refresh();
    snapshot.refresh();
    fleet.refresh();
    jobs.refresh();
  };

  // Pipeline status card values.
  const healthInfo = pipelineHealthTone(health.data?.status);
  const externalTranslation = health.data?.checks?.external_translation;
  const externalTranslationInfo = subsystemTone(externalTranslation);

  // Queue counts: prefer dashboard snapshot when present, fall back to
  // counts from /health/detailed.
  const queued =
    snapshot.data?.jobs?.queued ?? health.data?.jobs?.["submitted"] ?? 0;
  const processing =
    snapshot.data?.jobs?.active ??
    health.data?.jobs?.["processing"] ??
    0;
  const completed =
    snapshot.data?.jobs?.completed ??
    health.data?.jobs?.["completed"] ??
    0;

  // Throughput.
  const ppm = snapshot.data?.throughput?.pages_per_minute;
  const dph = snapshot.data?.throughput?.docs_per_hour;

  // Fleet.
  const totalWorkers = fleet.data?.summary?.total_workers ?? 0;
  const onlineWorkers = fleet.data?.summary?.online ?? 0;
  const totalGpus = fleet.data?.gpu?.total_gpus ?? 0;

  // Surface the most user-actionable error (health > jobs > snapshot/fleet).
  const fatalError = health.error ?? jobs.error;

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Dashboard</h1>
          <p className="text-sm text-muted-foreground">
            Live pipeline summary. Refreshes every {REFRESH_INTERVAL_MS / 1000}s.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={refreshAll}>
          Refresh
        </Button>
      </div>

      {fatalError ? (
        <Card data-testid="dashboard-error" role="alert">
          <CardHeader>
            <CardTitle className="text-destructive">
              Failed to load dashboard
            </CardTitle>
            <CardDescription>{describeError(fatalError)}</CardDescription>
          </CardHeader>
          <CardContent>
            <Button variant="outline" size="sm" onClick={refreshAll}>
              Retry
            </Button>
          </CardContent>
        </Card>
      ) : null}

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Pipeline status"
          value={healthInfo.label}
          tone={healthInfo.tone}
          loading={health.loading && !health.data}
          description={
            health.data
              ? `version ${health.data.version} · uptime ${formatNumber(
                  health.data.uptime_seconds
                )}s`
              : "Checking /health/detailed"
          }
          trailing={
            health.data ? <StatusBadge status={health.data.status} withRing={false} /> : null
          }
        />

        <MetricCard
          label="Jobs in queue"
          value={`${formatNumber(queued)} queued`}
          loading={health.loading && !health.data && snapshot.loading && !snapshot.data}
          description={`${formatNumber(processing)} processing · ${formatNumber(
            completed
          )} completed`}
        />

        <MetricCard
          label="Processing rate"
          value={
            ppm === undefined
              ? "—"
              : `${formatNumber(ppm, 1)} pages/min`
          }
          loading={snapshot.loading && !snapshot.data}
          description={
            dph === undefined
              ? "Dashboard endpoint disabled"
              : `${formatNumber(dph, 1)} docs/hr`
          }
        />

        <MetricCard
          label="Fleet health"
          value={
            fleet.data
              ? `${formatNumber(onlineWorkers)} / ${formatNumber(
                  totalWorkers
                )} online`
              : "—"
          }
          loading={fleet.loading && !fleet.data}
          description={
            fleet.data
              ? `${formatNumber(totalGpus)} GPU${totalGpus === 1 ? "" : "s"} tracked`
              : "Fleet endpoint disabled"
          }
        />

        <MetricCard
          label="EXTERNAL_TRANSLATION"
          value={externalTranslationInfo.label}
          tone={externalTranslationInfo.tone}
          loading={health.loading && !health.data}
          description={externalTranslation?.message ?? "Readiness unavailable"}
          trailing={
            externalTranslation ? (
              <StatusBadge status={externalTranslation.status} withRing={false} />
            ) : null
          }
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent jobs</CardTitle>
          <CardDescription>
            Last {RECENT_JOBS_LIMIT} submissions. Click a row to open job detail.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <RecentJobsTable
            jobs={jobs.data?.jobs ?? []}
            loading={jobs.loading && !jobs.data}
          />
        </CardContent>
      </Card>
    </div>
  );
}
