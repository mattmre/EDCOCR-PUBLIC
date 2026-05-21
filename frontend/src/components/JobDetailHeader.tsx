"use client";

import { StatusBadge } from "@/components/ui/StatusBadge";
import type { Job } from "@/lib/types";

export interface JobDetailHeaderProps {
  job: Job;
}

function fmt(value: string | null | undefined): string {
  if (!value) return "—";
  try {
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return value;
    return d.toLocaleString();
  } catch {
    return value;
  }
}

export function JobDetailHeader({ job }: JobDetailHeaderProps) {
  const sourceSize = (job.settings?.["source_size_bytes"] as number | undefined) ?? null;
  const sourceSizeMb =
    sourceSize !== null && Number.isFinite(sourceSize)
      ? `${(sourceSize / (1024 * 1024)).toFixed(1)} MB`
      : "—";

  return (
    <div className="rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold" data-testid="job-detail-id">
          {job.job_id}
        </h1>
        <StatusBadge status={job.status} />
        <span className="text-xs text-muted-foreground">priority: {job.priority}</span>
      </div>
      <p className="mt-1 truncate text-sm text-muted-foreground" title={job.source_file}>
        {job.source_file}
      </p>
      <dl className="mt-3 grid gap-3 text-xs sm:grid-cols-3">
        <div>
          <dt className="text-muted-foreground">Created</dt>
          <dd className="font-medium">{fmt(job.created_at)}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Started</dt>
          <dd className="font-medium">{fmt(job.started_at)}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Completed</dt>
          <dd className="font-medium">{fmt(job.completed_at)}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Source size</dt>
          <dd className="font-medium">{sourceSizeMb}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Webhook</dt>
          <dd className="font-medium">{job.webhook_status ?? "—"}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Stage</dt>
          <dd className="font-medium">{job.progress?.current_stage ?? "—"}</dd>
        </div>
      </dl>
    </div>
  );
}
