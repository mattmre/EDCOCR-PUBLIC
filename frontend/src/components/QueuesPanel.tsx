"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/cn";
import type { Queue, QueueThreshold } from "@/lib/types";

export const DEFAULT_WARN_DEPTH = 500;
export const DEFAULT_ALERT_DEPTH = 1000;

export type QueueSeverity = "green" | "amber" | "red";

export function classifyQueue(
  queue: Queue,
  warnDefault = DEFAULT_WARN_DEPTH,
  alertDefault = DEFAULT_ALERT_DEPTH
): QueueSeverity {
  const critical = queue.critical_threshold ?? alertDefault;
  const warning = queue.warning_threshold ?? warnDefault;
  if (queue.depth >= critical) return "red";
  if (queue.depth >= warning) return "amber";
  return "green";
}

const BAR_PALETTE: Record<QueueSeverity, string> = {
  red: "bg-red-500",
  amber: "bg-amber-500",
  green: "bg-emerald-500",
};

const TEXT_PALETTE: Record<QueueSeverity, string> = {
  red: "text-red-700",
  amber: "text-amber-700",
  green: "text-emerald-700",
};

export interface QueuesPanelProps {
  queues: Queue[];
  warnDefault?: number;
  alertDefault?: number;
  onUpdateThreshold?: (
    queueName: string,
    threshold: Omit<QueueThreshold, "queue_name">
  ) => Promise<void>;
}

export function QueuesPanel({
  queues,
  warnDefault = DEFAULT_WARN_DEPTH,
  alertDefault = DEFAULT_ALERT_DEPTH,
  onUpdateThreshold,
}: QueuesPanelProps) {
  const maxDepth = Math.max(
    1,
    ...queues.map((q) => Math.max(q.depth, q.critical_threshold ?? alertDefault))
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle>Queues</CardTitle>
      </CardHeader>
      <CardContent>
        {queues.length === 0 ? (
          <p className="py-6 text-center text-sm text-muted-foreground">
            No queues reporting depth right now.
          </p>
        ) : (
          <ul className="space-y-3" data-testid="queues-list">
            {queues.map((queue) => {
              const severity = classifyQueue(queue, warnDefault, alertDefault);
              const widthPct = Math.min(100, Math.round((queue.depth / maxDepth) * 100));
              return (
                <li
                  key={queue.queue_name}
                  data-testid={`queue-row-${queue.queue_name}`}
                  data-severity={severity}
                  className="space-y-1"
                >
                  <div className="flex items-baseline justify-between text-sm">
                    <span className="font-medium">{queue.queue_name}</span>
                    <span
                      data-testid={`queue-depth-${queue.queue_name}`}
                      className={cn("font-mono text-xs", TEXT_PALETTE[severity])}
                    >
                      {queue.depth.toLocaleString()}
                    </span>
                  </div>
                  <div
                    className="relative h-2 w-full overflow-hidden rounded-full bg-muted"
                    role="progressbar"
                    aria-label={`${queue.queue_name} depth ${queue.depth}`}
                    aria-valuenow={queue.depth}
                    aria-valuemin={0}
                    aria-valuemax={maxDepth}
                  >
                    <div
                      data-testid={`queue-bar-${queue.queue_name}`}
                      className={cn("h-full transition-all", BAR_PALETTE[severity])}
                      style={{ width: `${widthPct}%` }}
                    />
                  </div>
                  <div className="flex flex-wrap gap-x-3 text-[11px] text-muted-foreground">
                    {typeof queue.consumers === "number" && (
                      <span>consumers: {queue.consumers}</span>
                    )}
                    {typeof queue.in_flight === "number" && (
                      <span>in-flight: {queue.in_flight}</span>
                    )}
                    {typeof queue.oldest_item_age_seconds === "number" &&
                      queue.oldest_item_age_seconds > 0 && (
                        <span>
                          oldest: {Math.round(queue.oldest_item_age_seconds)}s
                        </span>
                      )}
                    {queue.warning_threshold !== null &&
                      queue.warning_threshold !== undefined && (
                        <span>
                          warn: {queue.warning_threshold} / crit:{" "}
                          {queue.critical_threshold ?? alertDefault}
                        </span>
                      )}
                  </div>
                  {onUpdateThreshold && (
                    <QueueThresholdForm
                      queue={queue}
                      warnDefault={warnDefault}
                      alertDefault={alertDefault}
                      onUpdateThreshold={onUpdateThreshold}
                    />
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

interface QueueThresholdFormProps {
  queue: Queue;
  warnDefault: number;
  alertDefault: number;
  onUpdateThreshold: QueuesPanelProps["onUpdateThreshold"];
}

function QueueThresholdForm({
  queue,
  warnDefault,
  alertDefault,
  onUpdateThreshold,
}: QueueThresholdFormProps) {
  const [warningDepth, setWarningDepth] = useState(
    String(queue.warning_threshold ?? warnDefault)
  );
  const [criticalDepth, setCriticalDepth] = useState(
    String(queue.critical_threshold ?? alertDefault)
  );
  const [warningWaitSeconds, setWarningWaitSeconds] = useState(
    String(queue.warning_wait_seconds ?? 300)
  );
  const [criticalWaitSeconds, setCriticalWaitSeconds] = useState(
    String(queue.critical_wait_seconds ?? 600)
  );
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    setWarningDepth(String(queue.warning_threshold ?? warnDefault));
    setCriticalDepth(String(queue.critical_threshold ?? alertDefault));
    setWarningWaitSeconds(String(queue.warning_wait_seconds ?? 300));
    setCriticalWaitSeconds(String(queue.critical_wait_seconds ?? 600));
  }, [
    queue.warning_threshold,
    queue.critical_threshold,
    queue.warning_wait_seconds,
    queue.critical_wait_seconds,
    warnDefault,
    alertDefault,
  ]);

  const disabled = !onUpdateThreshold;

  return (
    <form
      className="mt-2 grid gap-2 rounded-md border border-border/70 p-2 sm:grid-cols-[1fr_1fr_auto]"
      aria-label={`${queue.queue_name} threshold settings`}
      onSubmit={async (event) => {
        event.preventDefault();
        if (!onUpdateThreshold) return;
        setStatus("Saving");
        try {
          await onUpdateThreshold(queue.queue_name, {
            warning_depth: Number(warningDepth),
            critical_depth: Number(criticalDepth),
            warning_wait_seconds: Number(warningWaitSeconds),
            critical_wait_seconds: Number(criticalWaitSeconds),
          });
          setStatus("Saved");
        } catch (err: unknown) {
          setStatus(err instanceof Error ? err.message : "Save failed");
        }
      }}
    >
      <label className="grid gap-1 text-[11px] text-muted-foreground">
        Warn depth
        <input
          className="h-8 rounded-md border border-input bg-background px-2 font-mono text-xs text-foreground"
          type="number"
          min={0}
          value={warningDepth}
          disabled={disabled}
          onChange={(event) => setWarningDepth(event.target.value)}
        />
      </label>
      <label className="grid gap-1 text-[11px] text-muted-foreground">
        Critical depth
        <input
          className="h-8 rounded-md border border-input bg-background px-2 font-mono text-xs text-foreground"
          type="number"
          min={1}
          value={criticalDepth}
          disabled={disabled}
          onChange={(event) => setCriticalDepth(event.target.value)}
        />
      </label>
      <button
        className="h-8 self-end rounded-md bg-primary px-3 text-xs font-medium text-primary-foreground disabled:opacity-60"
        type="submit"
        disabled={disabled}
      >
        Save
      </button>
      <label className="grid gap-1 text-[11px] text-muted-foreground">
        Warn wait
        <input
          className="h-8 rounded-md border border-input bg-background px-2 font-mono text-xs text-foreground"
          type="number"
          min={0}
          value={warningWaitSeconds}
          disabled={disabled}
          onChange={(event) => setWarningWaitSeconds(event.target.value)}
        />
      </label>
      <label className="grid gap-1 text-[11px] text-muted-foreground">
        Critical wait
        <input
          className="h-8 rounded-md border border-input bg-background px-2 font-mono text-xs text-foreground"
          type="number"
          min={0}
          value={criticalWaitSeconds}
          disabled={disabled}
          onChange={(event) => setCriticalWaitSeconds(event.target.value)}
        />
      </label>
      {status && (
        <p className="self-end text-[11px] text-muted-foreground" role="status">
          {status}
        </p>
      )}
    </form>
  );
}
