import type { ReactNode } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/skeleton";
import { cn } from "@/lib/cn";

export interface MetricCardProps {
  label: string;
  value: ReactNode;
  description?: ReactNode;
  /** When true, render a skeleton in place of the value. */
  loading?: boolean;
  /** Optional tone tint applied to the value text. */
  tone?: "default" | "success" | "warning" | "danger";
  /** Optional trailing slot (e.g. a status badge, sparkline, sub-metric). */
  trailing?: ReactNode;
}

const TONE_CLASSES: Record<NonNullable<MetricCardProps["tone"]>, string> = {
  default: "text-foreground",
  success: "text-green-700",
  warning: "text-yellow-700",
  danger: "text-red-700",
};

/**
 * Reusable metric card for the dashboard top row. Shows a label, a large
 * value, optional supporting text, and an optional trailing slot for a
 * status badge or trend indicator.
 */
export function MetricCard({
  label,
  value,
  description,
  loading = false,
  tone = "default",
  trailing,
}: MetricCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {label}
        </CardTitle>
        {description ? (
          <CardDescription>{description}</CardDescription>
        ) : null}
      </CardHeader>
      <CardContent>
        <div className="flex items-end justify-between gap-3">
          {loading ? (
            <Skeleton className="h-8 w-24" />
          ) : (
            <span className={cn("text-2xl font-semibold", TONE_CLASSES[tone])}>
              {value}
            </span>
          )}
          {trailing ? <div className="shrink-0">{trailing}</div> : null}
        </div>
      </CardContent>
    </Card>
  );
}
