"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { FlagChangeRequestDialog } from "@/components/FlagChangeRequestDialog";
import { cn } from "@/lib/cn";
import type {
  FeatureFlag,
  FeatureFlagHistoryEntry,
  FeatureFlagRequestStatus,
} from "@/lib/types";

const STATUS_BADGE_CLASS: Record<FeatureFlagRequestStatus, string> = {
  pending: "bg-amber-100 text-amber-800 border-amber-200",
  approved: "bg-blue-100 text-blue-800 border-blue-200",
  rejected: "bg-rose-100 text-rose-800 border-rose-200",
  applied: "bg-emerald-100 text-emerald-800 border-emerald-200",
  rolled_back: "bg-gray-100 text-gray-700 border-gray-200",
};

interface BakeBannerProps {
  bakeHours: number;
  lastChangedAt: string;
}

/**
 * Returns the bake-window status when the flag last changed inside the
 * documented soak window. The detail page surfaces this banner so operators
 * know not to re-flip the flag mid-bake.
 */
function bakeStatus(
  bakeHours: number,
  lastChangedAt: string
): { active: boolean; expiresAt: Date | null } {
  if (!bakeHours || bakeHours <= 0) return { active: false, expiresAt: null };
  const ts = Date.parse(lastChangedAt);
  if (!Number.isFinite(ts)) return { active: false, expiresAt: null };
  const expires = new Date(ts + bakeHours * 60 * 60 * 1000);
  const active = expires.getTime() > Date.now();
  return { active, expiresAt: expires };
}

function BakeBanner({ bakeHours, lastChangedAt }: BakeBannerProps) {
  const status = bakeStatus(bakeHours, lastChangedAt);
  if (!status.active || !status.expiresAt) return null;
  return (
    <div
      data-testid="flag-detail-bake-banner"
      className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900"
    >
      <strong>Bake window in progress.</strong>{" "}
      This flag is in its mandated {bakeHours}-hour bake window. Do not request
      another change until {status.expiresAt.toISOString()}.
    </div>
  );
}

export interface FeatureFlagDetailProps {
  flag: FeatureFlag;
  history: FeatureFlagHistoryEntry[] | null;
  historyLoading?: boolean;
  historyError?: Error | null;
  onChangeRequested?: => void;
}

export function FeatureFlagDetail({
  flag,
  history,
  historyLoading,
  historyError,
  onChangeRequested,
}: FeatureFlagDetailProps) {
  const [dialogOpen, setDialogOpen] = useState<boolean>(false);

  function handleSubmitted(_: FeatureFlagHistoryEntry) {
    setDialogOpen(false);
    if (onChangeRequested) onChangeRequested();
  }

  return (
    <div className="space-y-4" data-testid={`flag-detail-${flag.key}`}>
      {flag.requires_bake_hours && flag.last_changed_at ? (
        <BakeBanner
          bakeHours={flag.requires_bake_hours}
          lastChangedAt={flag.last_changed_at}
        />
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle className="font-mono text-sm">{flag.key}</CardTitle>
          <CardDescription>{flag.description || "No description provided."}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <dl className="grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
            <div>
              <dt className="text-xs uppercase text-muted-foreground">Current value</dt>
              <dd className="mt-1 font-mono" data-testid="flag-detail-current">
                {String(flag.current_value)}
              </dd>
            </div>
            <div>
              <dt className="text-xs uppercase text-muted-foreground">Default</dt>
              <dd className="mt-1 font-mono" data-testid="flag-detail-default">
                {String(flag.default_value)}
              </dd>
            </div>
            <div>
              <dt className="text-xs uppercase text-muted-foreground">Source</dt>
              <dd className="mt-1" data-testid="flag-detail-source">
                {flag.source}
              </dd>
            </div>
            <div>
              <dt className="text-xs uppercase text-muted-foreground">Category</dt>
              <dd className="mt-1">{String(flag.category)}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase text-muted-foreground">Type</dt>
              <dd className="mt-1">{flag.value_type}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase text-muted-foreground">Strong-auth</dt>
              <dd className="mt-1">
                {flag.requires_strong_auth ? (
                  <span className="font-medium text-amber-700">Required</span>
                ) : (
                  <span className="text-muted-foreground">Not required</span>
                )}
              </dd>
            </div>
            {flag.requires_bake_hours ? (
              <div>
                <dt className="text-xs uppercase text-muted-foreground">Bake window</dt>
                <dd className="mt-1">{flag.requires_bake_hours} hours</dd>
              </div>
            ) : null}
            {flag.last_changed_at ? (
              <div>
                <dt className="text-xs uppercase text-muted-foreground">
                  Last changed
                </dt>
                <dd className="mt-1 font-mono text-xs">
                  {flag.last_changed_at}
                </dd>
              </div>
            ) : null}
          </dl>

          <div>
            <Button
              type="button"
              onClick={() => setDialogOpen(true)}
              data-testid="flag-detail-request-change"
            >
              Request change
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Recent change history</CardTitle>
          <CardDescription>
            Up to 20 most recent change requests. Each row is a custody event
            in the audit chain.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {historyLoading && !history ? (
            <p className="text-sm text-muted-foreground">Loading history…</p>
          ) : historyError ? (
            <p
              className="text-sm text-destructive"
              role="alert"
              data-testid="flag-history-error"
            >
              {historyError.message}
            </p>
          ) : !history || history.length === 0 ? (
            <p
              className="text-sm text-muted-foreground"
              data-testid="flag-history-empty"
            >
              No history yet.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table
                className="w-full text-sm"
                data-testid="flag-history-table"
              >
                <thead className="bg-muted/30 text-xs uppercase text-muted-foreground">
                  <tr>
                    <th className="px-3 py-2 text-left">Status</th>
                    <th className="px-3 py-2 text-left">From → To</th>
                    <th className="px-3 py-2 text-left">Reason</th>
                    <th className="px-3 py-2 text-left">Requester</th>
                    <th className="px-3 py-2 text-left">Requested</th>
                  </tr>
                </thead>
                <tbody>
                  {history.slice(0, 20).map((entry) => (
                    <tr
                      key={entry.request_id}
                      className="border-t border-border"
                      data-testid={`flag-history-row-${entry.request_id}`}
                    >
                      <td className="px-3 py-2">
                        <span
                          className={cn(
                            "inline-flex rounded-full border px-2 py-0.5 text-[11px] font-medium",
                            STATUS_BADGE_CLASS[entry.status as FeatureFlagRequestStatus] ??
                              "bg-gray-100 text-gray-700 border-gray-200"
                          )}
                        >
                          {entry.status}
                        </span>
                      </td>
                      <td className="px-3 py-2 font-mono text-xs">
                        {String(entry.previous_value)} → {String(entry.new_value)}
                      </td>
                      <td className="px-3 py-2 text-xs text-muted-foreground">
                        {entry.reason}
                      </td>
                      <td className="px-3 py-2 text-xs">{entry.requested_by}</td>
                      <td className="px-3 py-2 font-mono text-xs">
                        {entry.requested_at}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      <FlagChangeRequestDialog
        flag={flag}
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        onSubmitted={handleSubmitted}
      />
    </div>
  );
}
