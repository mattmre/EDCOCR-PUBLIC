import { cn } from "@/lib/cn";

type Tone = "neutral" | "success" | "warning" | "danger" | "info";

/**
 * Tone-bucketed colour classes. Each tone supports both a solid (no-ring)
 * and a ring-bordered presentation; callers pick the variant via the
 * `withRing` prop. Default is `withRing` true to match the dominant
 * usage across the operator console.
 */
const TONE_CLASSES: Record<Tone, { solid: string; ring: string }> = {
  neutral: {
    solid: "bg-muted text-muted-foreground",
    ring: "bg-slate-100 text-slate-700 ring-slate-300",
  },
  success: {
    solid: "bg-green-100 text-green-800",
    ring: "bg-green-100 text-green-800 ring-green-300",
  },
  warning: {
    solid: "bg-yellow-100 text-yellow-800",
    ring: "bg-amber-100 text-amber-900 ring-amber-300",
  },
  danger: {
    solid: "bg-red-100 text-red-800",
    ring: "bg-red-100 text-red-800 ring-red-300",
  },
  info: {
    solid: "bg-blue-100 text-blue-800",
    ring: "bg-blue-100 text-blue-800 ring-blue-300",
  },
};

/**
 * Map a job/health status string to a tone bucket. Unknown statuses fall
 * through to `neutral` so we never throw on new backend states.
 */
export function statusTone(status: string): Tone {
  const s = status.toLowerCase();
  if (s === "completed" || s === "healthy" || s === "online" || s === "ok") {
    return "success";
  }
  if (s === "processing" || s === "busy" || s === "running") {
    return "info";
  }
  if (
    s === "submitted" ||
    s === "queued" ||
    s === "idle" ||
    s === "draining" ||
    s === "degraded"
  ) {
    return "warning";
  }
  if (
    s === "failed" ||
    s === "error" ||
    s === "unhealthy" ||
    s === "cancelled" ||
    s === "down"
  ) {
    return "danger";
  }
  return "neutral";
}

export interface StatusBadgeProps {
  status: string | null | undefined;
  className?: string;
  /**
   * When `true` (default), renders with a ring border and lowercase text --
   * this matches the visual used by jobs/review tables.
   *
   * When `false`, renders the legacy "solid pill" treatment with capitalized
   * text -- this matches the dashboard summary cards.
   */
  withRing?: boolean;
}

export function StatusBadge({
  status,
  className,
  withRing = true,
}: StatusBadgeProps) {
  const normalized = (status ?? "unknown").toString();
  const display = withRing ? normalized.toLowerCase() : normalized || "unknown";
  const tone = statusTone(normalized);
  const variant = withRing ? TONE_CLASSES[tone].ring : TONE_CLASSES[tone].solid;
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full text-xs font-medium",
        withRing
          ? "px-2.5 py-0.5 ring-1 ring-inset"
          : "px-2 py-0.5 capitalize",
        variant,
        className
      )}
      data-status={normalized}
      data-tone={tone}
      data-testid="status-badge"
    >
      {display}
    </span>
  );
}
