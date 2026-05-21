"use client";

import { useEffect, useState } from "react";

/**
 * Format the gap between `epochSeconds` and now as a short human string.
 * Returns "—" when the timestamp is unset or in the future.
 *
 * Exported separately so tests can pin the formatter without rendering.
 */
export function formatRelative(epochSeconds: number, nowMs: number): string {
  if (!epochSeconds || epochSeconds <= 0) return "—";
  const deltaSec = Math.max(0, Math.floor(nowMs / 1000 - epochSeconds));
  if (deltaSec < 5) return "just now";
  if (deltaSec < 60) return `${deltaSec}s ago`;
  if (deltaSec < 3600) return `${Math.floor(deltaSec / 60)}m ago`;
  if (deltaSec < 86400) return `${Math.floor(deltaSec / 3600)}h ago`;
  return `${Math.floor(deltaSec / 86400)}d ago`;
}

interface RelativeTimeProps {
  epochSeconds: number;
  /** Re-render cadence; tests pass a smaller value. */
  intervalMs?: number;
  className?: string;
}

export function RelativeTime({ epochSeconds, intervalMs = 1000, className }: RelativeTimeProps) {
  const [now, setNow] = useState<number>(() => Date.now());

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), intervalMs);
    return => window.clearInterval(id);
  }, [intervalMs]);

  return (
    <span className={className} title={epochSeconds > 0 ? new Date(epochSeconds * 1000).toISOString() : undefined}>
      {formatRelative(epochSeconds, now)}
    </span>
  );
}
