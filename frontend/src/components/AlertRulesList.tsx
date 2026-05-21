"use client";

import Link from "next/link";
import { cn } from "@/lib/cn";
import { SeverityBadge } from "@/components/AlertsList";
import type { AlertRule, AdminAlertState } from "@/lib/types";

const STATE_TONE: Record<AdminAlertState, string> = {
  firing: "text-red-700 bg-red-100",
  pending: "text-amber-800 bg-amber-100",
  inactive: "text-slate-700 bg-slate-100",
  muted: "text-zinc-700 bg-zinc-200",
};

function formatThreshold(rule: AlertRule): string {
  switch (rule.threshold_unit) {
    case "bytes":
      return formatBytes(rule.threshold_value);
    case "seconds":
      return `${rule.threshold_value}s`;
    case "percent":
      return `${rule.threshold_value}%`;
    case "count":
    default:
      return String(rule.threshold_value);
  }
}

function formatBytes(n: number): string {
  if (n === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = Math.abs(n);
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatLastTriggered(iso: string | null | undefined): string {
  if (!iso) return "Never";
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

export interface AlertRulesListProps {
  rules: AlertRule[];
  loading?: boolean;
}

export function AlertRulesList({ rules, loading }: AlertRulesListProps) {
  return (
    <div className="overflow-x-auto rounded-md border border-border" data-testid="alert-rules-list">
      <table className="w-full text-sm" aria-label="Alert rules">
        <thead className="bg-muted/40 text-xs uppercase text-muted-foreground">
          <tr>
            <th className="px-3 py-2 text-left">Name</th>
            <th className="px-3 py-2 text-left">Severity</th>
            <th className="px-3 py-2 text-left">State</th>
            <th className="px-3 py-2 text-left">Threshold</th>
            <th className="px-3 py-2 text-left">Last triggered</th>
            <th className="px-3 py-2 text-left">Enabled</th>
          </tr>
        </thead>
        <tbody>
          {loading && rules.length === 0 ? (
            <tr>
              <td
                colSpan={6}
                className="px-3 py-3 text-muted-foreground"
                data-testid="alert-rules-loading"
              >
                Loading rules…
              </td>
            </tr>
          ) : rules.length === 0 ? (
            <tr>
              <td
                colSpan={6}
                className="px-3 py-3 text-muted-foreground"
                data-testid="alert-rules-empty"
              >
                No alert rules defined.
              </td>
            </tr>
          ) : (
            rules.map((rule) => (
              <tr
                key={rule.id}
                className="border-t border-border hover:bg-muted/20"
                data-testid={`alert-rule-row-${rule.id}`}
              >
                <td className="px-3 py-2">
                  <Link
                    href={`/admin/alerts/${encodeURIComponent(rule.id)}`}
                    className="text-primary hover:underline"
                    data-testid={`alert-rule-link-${rule.id}`}
                  >
                    {rule.name}
                  </Link>
                </td>
                <td className="px-3 py-2">
                  <SeverityBadge severity={rule.severity} />
                </td>
                <td className="px-3 py-2">
                  <span
                    className={cn(
                      "inline-flex items-center rounded-full px-2 py-0.5 text-xs",
                      STATE_TONE[rule.current_state]
                    )}
                  >
                    {rule.current_state}
                  </span>
                </td>
                <td className="px-3 py-2 font-mono text-xs">{formatThreshold(rule)}</td>
                <td className="px-3 py-2 text-xs text-muted-foreground">
                  {formatLastTriggered(rule.last_triggered_at)}
                </td>
                <td className="px-3 py-2 text-xs">
                  {rule.enabled ? (
                    <span className="text-green-700">enabled</span>
                  ) : (
                    <span className="text-muted-foreground">disabled</span>
                  )}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

export { formatThreshold as formatRuleThreshold };
