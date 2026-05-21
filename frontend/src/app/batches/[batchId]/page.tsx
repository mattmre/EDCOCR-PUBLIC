"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { fetchBatch } from "@/lib/batches-api";
import { useRequireAuth } from "@/lib/auth";
import type { BatchStatusResponse } from "@/lib/types";

export default function BatchDetailPage() {
  useRequireAuth();
  const params = useParams<{ batchId: string }>();
  const batchId = (params?.batchId ?? "") as string;
  const [batch, setBatch] = useState<BatchStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const response = await fetchBatch(batchId, controller.signal);
        setBatch(response);
        setError(null);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "Unable to load batch");
        }
      }
    }
    if (batchId) void load();
    return => controller.abort();
  }, [batchId]);

  if (error) {
    return <div className="rounded-md border border-destructive/50 p-4 text-sm text-destructive">{error}</div>;
  }

  if (!batch) {
    return <p className="text-sm text-muted-foreground" data-testid="batch-detail-loading">Loading batch...</p>;
  }

  return (
    <div className="space-y-6">
      <Link className="text-sm text-primary hover:underline" href="/batches">
        Back to batches
      </Link>
      <div>
        <h1 className="font-mono text-2xl font-semibold" data-testid="batch-detail-id">
          {batch.batch_id}
        </h1>
        <div className="mt-2">
          <StatusBadge status={batch.status} />
        </div>
      </div>
      <div className="grid gap-3 sm:grid-cols-4">
        <Metric label="Total" value={batch.total_jobs} />
        <Metric label="Completed" value={batch.progress.completed} />
        <Metric label="Failed" value={batch.progress.failed} />
        <Metric label="Progress" value={`${batch.progress.percent_complete.toFixed(0)}%`} />
      </div>
      <div className="overflow-x-auto rounded-md border border-border">
        <table className="min-w-full divide-y divide-border text-sm" data-testid="batch-jobs-table">
          <thead className="bg-muted/40 text-xs uppercase text-muted-foreground">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Job ID</th>
              <th className="px-3 py-2 text-left font-medium">Status</th>
              <th className="px-3 py-2 text-left font-medium">Source</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {batch.jobs.map((job) => (
              <tr key={job.job_id}>
                <td className="px-3 py-2 font-mono text-xs">
                  <Link className="text-primary hover:underline" href={`/jobs/${job.job_id}`}>
                    {job.job_id}
                  </Link>
                </td>
                <td className="px-3 py-2">
                  <StatusBadge status={job.status} />
                </td>
                <td className="px-3 py-2 text-muted-foreground">{job.source_file}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-md border border-border bg-background p-4">
      <p className="text-xs uppercase text-muted-foreground">{label}</p>
      <p className="mt-1 text-xl font-semibold">{value}</p>
    </div>
  );
}
