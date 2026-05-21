/**
 * Backward-compat re-export. The canonical `StatusBadge` lives at
 * `@/components/status-badge` and supports both ring-bordered and solid
 * presentations via the `withRing` prop (default `true`, matching the
 * visual this module shipped with).
 *
 * New call sites should import directly from `@/components/status-badge`.
 */
export {
  StatusBadge,
  statusTone,
  type StatusBadgeProps,
} from "@/components/status-badge";
