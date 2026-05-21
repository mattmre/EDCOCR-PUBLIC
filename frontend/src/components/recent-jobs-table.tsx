"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import type { KeyboardEvent } from "react";
import { StatusBadge } from "@/components/status-badge";
import { Skeleton } from "@/components/skeleton";
import type { Job } from "@/lib/types";

export interface RecentJobsTableProps {
  jobs: Job[];
  loading?: boolean;
  emptyMessage?: string;
}

function formatTimestamp(value?: string | null): string {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) {
    return value;
  }
  return d.toLocaleString();
}

function formatDuration(job: Job): string {
  if (!job.started_at) return "—";
  const start = new Date(job.started_at).getTime();
  const end = job.completed_at
    ? new Date(job.completed_at).getTime()
    : Date.now();
  if (Number.isNaN(start) || Number.isNaN(end) || end < start) {
    return "—";
  }
  const seconds = Math.round((end - start) / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remSec = seconds % 60;
  if (minutes < 60) return `${minutes}m ${remSec}s`;
  const hours = Math.floor(minutes / 60);
  const remMin = minutes % 60;
  return `${hours}h ${remMin}m`;
}

/**
 * Compact table of recent jobs. Rows are keyboard-navigable and route to
 * the job detail page when activated. The detail route lands in PR D3;
 * for now we link to `/jobs?selected={id}` so the link is forward-
 * compatible.
 */
export function RecentJobsTable({
  jobs,
  loading = false,
  emptyMessage = "No jobs yet.",
}: RecentJobsTableProps) {
  const router = useRouter();

  const navigateTo = (jobId: string) => {
    router.push(`/jobs?selected=${encodeURIComponent(jobId)}`);
  };

  const onRowKeyDown =
    (jobId: string) => (event: KeyboardEvent<HTMLTableRowElement>) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        navigateTo(jobId);
      }
    };

  if (loading) {
    return (
      <div className="space-y-2" data-testid="recent-jobs-loading">
        <Skeleton className="h-9 w-full" />
        <Skeleton className="h-9 w-full" />
        <Skeleton className="h-9 w-full" />
      </div>
    );
  }

  if (jobs.length === 0) {
    return (
      <p className="text-sm text-muted-foreground" data-testid="recent-jobs-empty">
        {emptyMessage}
      </p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="border-b border-border text-xs uppercase text-muted-foreground">
          <tr>
            <th className="px-3 py-2 font-medium">Job</th>
            <th className="px-3 py-2 font-medium">Status</th>
            <th className="px-3 py-2 font-medium">Created</th>
            <th className="px-3 py-2 font-medium">Duration</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <tr
              key={job.job_id}
              data-testid={`recent-jobs-row-${job.job_id}`}
              className="cursor-pointer border-b border-border/60 hover:bg-accent/40 focus:bg-accent/40 focus:outline-none"
              tabIndex={0}
              role="link"
              aria-label={`Open job ${job.job_id}`}
              onClick={() => navigateTo(job.job_id)}
              onKeyDown={onRowKeyDown(job.job_id)}
            >
              <td className="px-3 py-2 font-mono text-xs">
                <Link
                  href={`/jobs?selected=${encodeURIComponent(job.job_id)}`}
                  className="text-primary hover:underline"
                  // Stop bubble so the row click handler doesn't fire twice.
                  onClick={(e) => e.stopPropagation()}
                >
                  {job.job_id}
                </Link>
                <div className="truncate text-[11px] text-muted-foreground">
                  {job.source_file}
                </div>
              </td>
              <td className="px-3 py-2">
                <StatusBadge status={job.status} withRing={false} />
              </td>
              <td className="px-3 py-2 text-xs text-muted-foreground">
                {formatTimestamp(job.created_at)}
              </td>
              <td className="px-3 py-2 text-xs text-muted-foreground">
                {formatDuration(job)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
