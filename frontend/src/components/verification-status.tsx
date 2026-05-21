"use client";

import { cn } from "@/lib/cn";
import type { VerificationResult, VerificationStatus } from "@/lib/audit-verify";

interface VerificationStatusProps {
  result: VerificationResult | null;
  isVerifying: boolean;
  className?: string;
}

interface BadgeStyle {
  label: string;
  className: string;
  symbol: string;
}

const STYLES: Record<VerificationStatus | "idle", BadgeStyle> = {
  idle: {
    label: "Not yet verified",
    className: "bg-muted text-muted-foreground border-border",
    symbol: "·",
  },
  verifying: {
    label: "Verifying",
    className: "bg-amber-50 text-amber-900 border-amber-200",
    symbol: "...",
  },
  intact: {
    label: "Chain intact",
    className: "bg-emerald-50 text-emerald-900 border-emerald-200",
    symbol: "OK",
  },
  broken: {
    label: "Chain broken",
    className: "bg-red-50 text-red-900 border-red-200",
    symbol: "X",
  },
  empty: {
    label: "No events",
    className: "bg-muted text-muted-foreground border-border",
    symbol: "—",
  },
};

export function VerificationStatusBadge({
  result,
  isVerifying,
  className,
}: VerificationStatusProps) {
  let key: VerificationStatus | "idle" = "idle";
  if (isVerifying) {
    key = "verifying";
  } else if (result) {
    key = result.status;
  }
  const style = STYLES[key];

  const tooltip =
    result && result.status === "broken"
      ? result.reason ?? "A custody event failed verification."
      : result && result.status === "intact"
      ? `Verified ${result.verifiedEvents} of ${result.totalEvents} event(s).`
      : result && result.status === "empty"
      ? "Custody log contains no events."
      : isVerifying
      ? "Recomputing SHA-256 chain in the browser."
      : "Click 'Verify hash chain' to check integrity client-side.";

  return (
    <span
      role="status"
      aria-live="polite"
      title={tooltip}
      className={cn(
        "inline-flex items-center gap-2 rounded-md border px-3 py-1 text-xs font-semibold",
        style.className,
        className
      )}
    >
      <span aria-hidden className="font-mono">
        {style.symbol}
      </span>
      <span>{style.label}</span>
      {result && result.status === "broken" && typeof result.brokenAtIndex === "number" ? (
        <span className="font-normal text-red-800">@ event #{result.brokenAtIndex}</span>
      ) : null}
    </span>
  );
}
