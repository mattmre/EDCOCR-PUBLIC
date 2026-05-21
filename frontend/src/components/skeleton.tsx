import { type HTMLAttributes } from "react";
import { cn } from "@/lib/cn";

/**
 * Tailwind-only loading shimmer. Uses `animate-pulse` (built into Tailwind)
 * so we avoid pulling in a motion library for .
 */
export function Skeleton({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("animate-pulse rounded-md bg-muted", className)}
      aria-hidden="true"
      {...props}
    />
  );
}
