"use client";

import { useMemo, useState } from "react";
import { cn } from "@/lib/cn";
import { Button } from "@/components/ui/button";
import { Pagination } from "@/components/Pagination";
import type { Alert, AdminAlertState, AlertSeverity } from "@/lib/types";

const SEVERITY_TONE: Record<AlertSeverity, string> = {
  critical: "bg-red-100 text-red-800 ring-red-300",
  warning: "bg-amber-100 text-amber-900 ring-amber-300",
  info: "bg-blue-100 text-blue-800 ring-blue-300",
};

const STATE_TONE: Record<AdminAlertState, string> = {
  firing: "bg-red-100 text-red-800",
  pending: "bg-amber-100 text-amber-900",
  inactive: "bg-slate-100 text-slate-700",
  muted: "bg-zinc-200 text-zinc-700",
};

function relativeAge(iso: string, nowMs: number): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const sec = Math.max(0, Math.floor((nowMs - t) / 1000));
  if (sec < 5) return "just now";
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
  return `${Math.floor(sec / 86400)}d ago`;
}

export interface SeverityBadgeProps {
  severity: AlertSeverity;
  className?: string;
}

export function SeverityBadge({ severity, className }: SeverityBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset",
        SEVERITY_TONE[severity],
        className
      )}
      data-testid={`severity-badge-${severity}`}
      data-severity={severity}
    >
      {severity}
    </span>
  );
}

export interface MutePayload {
  reason: string;
}

export interface AlertsListProps {
  alerts: Alert[];
  loading?: boolean;
  /** ISO 8601 reference clock so relative ages re-render predictably in tests. */
  nowMs?: number;
  pageSize?: number;
  /** Async; component awaits, then optimistically updates row state. */
  onMute?: (alertId: string, payload: MutePayload) => Promise<void> | void;
  onUnmute?: (alertId: string) => Promise<void> | void;
}

const DEFAULT_PAGE_SIZE = 25;

export function AlertsList({
  alerts,
  loading,
  nowMs,
  pageSize = DEFAULT_PAGE_SIZE,
  onMute,
  onUnmute,
}: AlertsListProps) {
  const [page, setPage] = useState<number>(1);
  const [muteTarget, setMuteTarget] = useState<Alert | null>(null);
  const [muteReason, setMuteReason] = useState<string>("");
  const [busyId, setBusyId] = useState<string | null>(null);
  const [errorId, setErrorId] = useState<string | null>(null);

  const total = alerts.length;
  const start = (page - 1) * pageSize;
  const visible = useMemo( => alerts.slice(start, start + pageSize),
    [alerts, start, pageSize]
  );
  const clock = nowMs ?? Date.now();

  async function commitMute() {
    if (!muteTarget || !onMute) return;
    const id = muteTarget.id;
    const reason = muteReason.trim();
    if (!reason) {
      setErrorId(id);
      return;
    }
    setBusyId(id);
    setErrorId(null);
    try {
      await onMute(id, { reason });
      setMuteTarget(null);
      setMuteReason("");
    } catch {
      setErrorId(id);
    } finally {
      setBusyId(null);
    }
  }

  async function handleUnmute(alert: Alert) {
    if (!onUnmute) return;
    setBusyId(alert.id);
    setErrorId(null);
    try {
      await onUnmute(alert.id);
    } catch {
      setErrorId(alert.id);
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="space-y-3" data-testid="alerts-list">
      <div className="overflow-x-auto rounded-md border border-border">
        <table className="w-full text-sm" aria-label="Active alerts">
          <thead className="bg-muted/40 text-xs uppercase text-muted-foreground">
            <tr>
              <th className="px-3 py-2 text-left">Severity</th>
              <th className="px-3 py-2 text-left">Rule</th>
              <th className="px-3 py-2 text-left">Tenant</th>
              <th className="px-3 py-2 text-left">State</th>
              <th className="px-3 py-2 text-left">Started</th>
              <th className="px-3 py-2 text-left">Message</th>
              <th className="px-3 py-2 text-right">Action</th>
            </tr>
          </thead>
          <tbody>
            {loading && visible.length === 0 ? (
              <tr>
                <td
                  className="px-3 py-3 text-muted-foreground"
                  colSpan={7}
                  data-testid="alerts-loading"
                >
                  Loading alerts…
                </td>
              </tr>
            ) : visible.length === 0 ? (
              <tr>
                <td
                  className="px-3 py-3 text-muted-foreground"
                  colSpan={7}
                  data-testid="alerts-empty"
                >
                  No active alerts.
                </td>
              </tr>
            ) : (
              visible.map((alert) => {
                const isMuted = alert.state === "muted";
                return (
                  <tr
                    key={alert.id}
                    className="border-t border-border hover:bg-muted/20"
                    data-testid={`alert-row-${alert.id}`}
                  >
                    <td className="px-3 py-2">
                      <SeverityBadge severity={alert.severity} />
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">{alert.rule_id}</td>
                    <td className="px-3 py-2 text-xs">
                      {alert.tenant_id ?? <span className="text-muted-foreground">—</span>}
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={cn(
                          "inline-flex items-center rounded-full px-2 py-0.5 text-xs",
                          STATE_TONE[alert.state]
                        )}
                        data-testid={`alert-state-${alert.id}`}
                      >
                        {alert.state}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-xs text-muted-foreground">
                      {relativeAge(alert.started_at, clock)}
                    </td>
                    <td className="px-3 py-2 text-xs">{alert.message}</td>
                    <td className="px-3 py-2 text-right">
                      {isMuted ? (
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          disabled={busyId === alert.id}
                          onClick={() => handleUnmute(alert)}
                          data-testid={`alert-unmute-${alert.id}`}
                        >
                          Unmute
                        </Button>
                      ) : (
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          disabled={busyId === alert.id}
                          onClick={() => {
                            setMuteTarget(alert);
                            setMuteReason("");
                          }}
                          data-testid={`alert-mute-${alert.id}`}
                        >
                          Mute
                        </Button>
                      )}
                      {errorId === alert.id ? (
                        <p
                          className="mt-1 text-xs text-destructive"
                          role="alert"
                          data-testid={`alert-action-error-${alert.id}`}
                        >
                          Action failed
                        </p>
                      ) : null}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      <Pagination
        page={page}
        pageSize={pageSize}
        total={total}
        onPageChange={setPage}
      />

      {muteTarget ? (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Confirm mute"
          data-testid="mute-dialog"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
        >
          <div className="w-full max-w-md space-y-4 rounded-md border border-border bg-background p-6 shadow-lg">
            <div>
              <h2 className="text-lg font-semibold">Mute alert</h2>
              <p className="text-sm text-muted-foreground">
                Suppress notifications for{" "}
                <span className="font-mono">{muteTarget.rule_id}</span>. The
                alert keeps firing in the backend; only operator
                notifications are silenced.
              </p>
            </div>
            <label className="block text-sm">
              Reason
              <textarea
                className="mt-1 block w-full rounded-md border border-input bg-background p-2 text-sm"
                rows={3}
                value={muteReason}
                onChange={(e) => setMuteReason(e.target.value)}
                data-testid="mute-reason"
                autoFocus
              />
            </label>
            <div className="flex justify-end gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  setMuteTarget(null);
                  setMuteReason("");
                }}
                data-testid="mute-cancel"
              >
                Cancel
              </Button>
              <Button
                type="button"
                disabled={busyId === muteTarget.id || muteReason.trim().length === 0}
                onClick={commitMute}
                data-testid="mute-confirm"
              >
                {busyId === muteTarget.id ? "Muting…" : "Mute"}
              </Button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
