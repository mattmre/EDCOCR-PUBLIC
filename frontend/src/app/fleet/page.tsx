"use client";

import { useCallback, useState } from "react";
import { FleetSummary, STALE_DEFAULT_SECONDS } from "@/components/FleetSummary";
import { QueuesPanel } from "@/components/QueuesPanel";
import { WorkersTable } from "@/components/WorkersTable";
import { ApiError, get, put } from "@/lib/api-client";
import { useRequireAuth } from "@/lib/auth";
import { useAutoRefresh } from "@/lib/hooks";
import type { FleetSnapshot, QueueSnapshot, QueueThreshold } from "@/lib/types";

const FLEET_INTERVAL_MS = 10_000;
const QUEUES_INTERVAL_MS = 5_000;

export default function FleetPage() {
  useRequireAuth();
  const [staleOnly, setStaleOnly] = useState(false);

  const fleetFetcher = useCallback(
    (signal: AbortSignal) => get<FleetSnapshot>("/api/v1/fleet", { signal }),
    []
  );
  const queuesFetcher = useCallback(
    (signal: AbortSignal) => get<QueueSnapshot>("/api/v1/alerts", { signal }),
    []
  );

  const fleet = useAutoRefresh<FleetSnapshot>(fleetFetcher, FLEET_INTERVAL_MS);
  const queues = useAutoRefresh<QueueSnapshot>(queuesFetcher, QUEUES_INTERVAL_MS);

  const fleetError = formatError(fleet.error);
  const queueError = formatError(queues.error);
  const updateQueueThreshold = useCallback(
    async (queueName: string, threshold: Omit<QueueThreshold, "queue_name">) => {
      await put<QueueThreshold>(
        `/api/v1/queues/${encodeURIComponent(queueName)}/threshold`,
        threshold
      );
      queues.refresh();
    },
    [queues]
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Fleet</h1>
        <p className="text-sm text-muted-foreground">
          Worker fleet health and queue depth. Workers refresh every {FLEET_INTERVAL_MS / 1000}s,
          queues every {QUEUES_INTERVAL_MS / 1000}s. Polling pauses when the tab is hidden.
        </p>
      </div>

      {(fleetError || queueError) && (
        <div
          role="alert"
          className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive"
        >
          {fleetError && <p>Fleet snapshot error: {fleetError}</p>}
          {queueError && <p>Queue snapshot error: {queueError}</p>}
        </div>
      )}

      <FleetSummary
        fleet={fleet.data}
        queues={queues.data}
        staleThresholdSeconds={STALE_DEFAULT_SECONDS}
        staleFilterActive={staleOnly}
        onToggleStaleFilter={() => setStaleOnly((v) => !v)}
      />

      <div className="grid gap-6 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <WorkersTable
            workers={fleet.data?.workers ?? []}
            staleThresholdSeconds={STALE_DEFAULT_SECONDS}
            staleOnly={staleOnly}
          />
        </div>
        <QueuesPanel
          queues={queues.data?.queues ?? []}
          onUpdateThreshold={updateQueueThreshold}
        />
      </div>
    </div>
  );
}

function formatError(err: Error | null): string | null {
  if (!err) return null;
  if (err instanceof ApiError) {
    return `${err.status} ${err.message}`;
  }
  return err.message;
}
