"use client";

import { useMemo } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/cn";
import type { FleetSnapshot, QueueSnapshot, Worker } from "@/lib/types";

export const STALE_DEFAULT_SECONDS = 60;

export interface FleetSummaryProps {
  fleet: FleetSnapshot | null;
  queues: QueueSnapshot | null;
  staleThresholdSeconds?: number;
  /** Pages-per-minute throughput for backlog ETA. Pass null when unknown. */
  throughputPerMinute?: number | null;
  /** When non-null, the stale card is rendered as a toggle button. */
  staleFilterActive?: boolean;
  onToggleStaleFilter?: => void;
}

interface CapabilityCount {
  capability: string;
  count: number;
}

function countCapabilities(workers: Worker[]): CapabilityCount[] {
  const counts = new Map<string, number>();
  for (const worker of workers) {
    if (worker.state === "offline") continue;
    for (const cap of worker.capabilities) {
      counts.set(cap, (counts.get(cap) ?? 0) + 1);
    }
  }
  return Array.from(counts.entries())
    .map(([capability, count]) => ({ capability, count }))
    .sort((a, b) => b.count - a.count);
}

function countStale(workers: Worker[], thresholdSec: number, nowSec: number): number {
  let n = 0;
  for (const worker of workers) {
    if (worker.last_heartbeat <= 0) {
      // Unset heartbeat counts as stale only when state is not offline (offline
      // is its own card; stale highlights live workers that stopped reporting).
      if (worker.state !== "offline") n += 1;
      continue;
    }
    if (nowSec - worker.last_heartbeat > thresholdSec && worker.state !== "offline") {
      n += 1;
    }
  }
  return n;
}

function formatEta(depth: number, throughputPerMinute: number | null | undefined): string {
  if (!throughputPerMinute || throughputPerMinute <= 0 || depth <= 0) return "—";
  const minutes = depth / throughputPerMinute;
  if (minutes < 1) return "<1m";
  if (minutes < 60) return `${Math.round(minutes)}m`;
  const hours = minutes / 60;
  if (hours < 24) return `${hours.toFixed(1)}h`;
  return `${(hours / 24).toFixed(1)}d`;
}

export function FleetSummary({
  fleet,
  queues,
  staleThresholdSeconds = STALE_DEFAULT_SECONDS,
  throughputPerMinute = null,
  staleFilterActive = false,
  onToggleStaleFilter,
}: FleetSummaryProps) {
  const capabilityCounts = useMemo( => countCapabilities(fleet?.workers ?? []),
    [fleet]
  );
  const staleCount = useMemo(() => {
    if (!fleet) return 0;
    return countStale(fleet.workers ?? [], staleThresholdSeconds, Date.now() / 1000);
  }, [fleet, staleThresholdSeconds]);

  const onlineTotal = fleet?.summary
    ? fleet.summary.online + fleet.summary.busy + fleet.summary.idle
    : 0;
  const totalDepth = queues?.total_depth ?? 0;

  const StaleCard = (
    <Card
      data-testid="card-stale"
      className={cn(
        "transition-colors",
        staleFilterActive && "ring-2 ring-amber-500",
        onToggleStaleFilter && "cursor-pointer hover:border-amber-400"
      )}
    >
      <CardHeader>
        <CardTitle>Stale heartbeat</CardTitle>
        <CardDescription>{`>${staleThresholdSeconds}s since last beat`}</CardDescription>
      </CardHeader>
      <CardContent>
        <p
          data-testid="stale-count"
          className={cn(
            "text-3xl font-semibold",
            staleCount > 0 ? "text-amber-700" : "text-foreground"
          )}
        >
          {staleCount}
        </p>
        {onToggleStaleFilter && (
          <p className="mt-1 text-xs text-muted-foreground">
            {staleFilterActive ? "Showing only stale workers" : "Click to filter table"}
          </p>
        )}
      </CardContent>
    </Card>
  );

  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      <Card data-testid="card-online">
        <CardHeader>
          <CardTitle>Online workers</CardTitle>
          <CardDescription>Healthy + busy + idle</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-3xl font-semibold">{onlineTotal}</p>
          <p className="mt-2 text-xs text-muted-foreground">
            {capabilityCounts.length === 0
              ? "No capabilities reported"
              : capabilityCounts
                  .map((c) => `${c.count} ${c.capability}`)
                  .join(" + ")}
          </p>
        </CardContent>
      </Card>

      {onToggleStaleFilter ? (
        <button
          type="button"
          onClick={onToggleStaleFilter}
          aria-pressed={staleFilterActive}
          className="text-left"
        >
          {StaleCard}
        </button>
      ) : (
        StaleCard
      )}

      <Card data-testid="card-queue-depth">
        <CardHeader>
          <CardTitle>Queue depth</CardTitle>
          <CardDescription>Sum across all queues</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-3xl font-semibold">{totalDepth}</p>
          <p className="mt-2 text-xs text-muted-foreground">
            {queues?.queues.length ?? 0} queues monitored
          </p>
        </CardContent>
      </Card>

      <Card data-testid="card-eta">
        <CardHeader>
          <CardTitle>Backlog ETA</CardTitle>
          <CardDescription>depth ÷ throughput</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-3xl font-semibold">{formatEta(totalDepth, throughputPerMinute)}</p>
          <p className="mt-2 text-xs text-muted-foreground">
            {throughputPerMinute && throughputPerMinute > 0
              ? `${throughputPerMinute.toFixed(1)} ppm`
              : "throughput unavailable"}
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
