"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { JobDetailHeader } from "@/components/JobDetailHeader";
import { JobLogs } from "@/components/JobLogs";
import { JobOutputs } from "@/components/JobOutputs";
import { JobProgress } from "@/components/JobProgress";
import { Tabs, type TabDefinition } from "@/components/Tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ApiError, get } from "@/lib/api-client";
import { useRequireAuth } from "@/lib/auth";
import type { Job } from "@/lib/types";
import { useJobWebSocket } from "@/lib/ws-client";

const POLL_INTERVAL_MS = 15_000;

export default function JobDetailPage() {
  useRequireAuth();
  const params = useParams<{ id: string }>();
  const jobId = (params?.id ?? "") as string;

  const [job, setJob] = useState<Job | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const { status: wsStatus, lastMessage, reconnect } = useJobWebSocket(jobId);

  // Initial fetch + low-frequency poll as a fallback when the WS is down.
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    async function load() {
      try {
        const response = await get<Job>(`/api/v1/jobs/${jobId}`);
        if (!cancelled) {
          setJob(response);
          setLoadError(null);
          setLoading(false);
        }
      } catch (err) {
        if (cancelled) return;
        const message =
          err instanceof ApiError
            ? `${err.status} ${err.message}`
            : err instanceof Error
              ? err.message
              : "unknown error";
        setLoadError(message);
        setLoading(false);
      }
    }
    void load();
    const id = setInterval(() => void load(), POLL_INTERVAL_MS);
    return => {
      cancelled = true;
      clearInterval(id);
    };
  }, [jobId]);

  // Apply WS messages to the local job state so the header status updates
  // even before the next REST poll lands.
  useEffect(() => {
    if (!lastMessage || !job) return;
    if (lastMessage.type === "completed" && job.status !== "completed") {
      setJob({ ...job, status: "completed" });
    } else if (lastMessage.type === "failed" && job.status !== "failed") {
      setJob({ ...job, status: "failed" });
    } else if (lastMessage.type === "cancelled" && job.status !== "cancelled") {
      setJob({ ...job, status: "cancelled" });
    } else if (lastMessage.type === "progress" && lastMessage.status && job.status !== lastMessage.status) {
      setJob({ ...job, status: lastMessage.status });
    }
  }, [lastMessage, job]);

  if (loading && !job) {
    return (
      <div className="space-y-6">
        <p className="text-sm text-muted-foreground" data-testid="detail-loading">
          Loading job…
        </p>
      </div>
    );
  }

  if (loadError && !job) {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Failed to load job</CardTitle>
            <CardDescription>{loadError}</CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/jobs" className="text-sm text-primary hover:underline">
              ← Back to jobs
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!job) return null;

  const tabs: TabDefinition[] = [
    {
      id: "overview",
      label: "Overview",
      content: <OverviewTab job={job} />,
    },
    {
      id: "pages",
      label: "Pages",
      content: <PagesTab job={job} />,
    },
    {
      id: "outputs",
      label: "Outputs",
      content: <JobOutputs jobId={job.job_id} />,
    },
    {
      id: "logs",
      label: "Logs",
      content: <JobLogs jobId={job.job_id} />,
    },
    {
      id: "audit",
      label: "Audit",
      content: <AuditTab jobId={job.job_id} />,
    },
  ];

  return (
    <div className="space-y-6">
      <Link href="/jobs" className="text-sm text-primary hover:underline">
        ← Back to jobs
      </Link>
      <JobDetailHeader job={job} />
      <JobProgress
        job={job}
        wsStatus={wsStatus}
        lastMessage={lastMessage}
        onReconnect={reconnect}
      />
      <Tabs tabs={tabs} defaultTab="overview" />
    </div>
  );
}

function OverviewTab({ job }: { job: Job }) {
  const settings = job.settings ?? {};
  const settingEntries = Object.entries(settings);
  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4 text-sm">
      <h3 className="text-sm font-semibold">Settings & metadata</h3>
      {settingEntries.length === 0 ? (
        <p className="text-muted-foreground">No additional metadata.</p>
      ) : (
        <dl className="grid gap-2 sm:grid-cols-2">
          {settingEntries.map(([key, value]) => (
            <div key={key} className="flex flex-col">
              <dt className="text-xs uppercase tracking-wide text-muted-foreground">{key}</dt>
              <dd className="break-words font-mono text-xs">{formatSettingValue(value)}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

function formatSettingValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function PagesTab({ job }: { job: Job }) {
  const total = job.progress?.total_pages ?? 0;
  const completed = job.progress?.pages_completed ?? 0;
  return (
    <div className="rounded-md border border-border bg-background p-4 text-sm">
      <h3 className="text-sm font-semibold">Pages</h3>
      <p className="mt-2 text-muted-foreground">
        {total > 0
          ? `${completed} of ${total} pages processed.`
          : "Page-level details will appear once the document has been extracted."}
      </p>
      <p className="mt-2 text-xs text-muted-foreground">
        Per-page confidence and language data is exposed by the result endpoint after
        completion. A full inline page list arrives in a later wave.
      </p>
    </div>
  );
}

function AuditTab({ jobId }: { jobId: string }) {
  return (
    <div className="rounded-md border border-border bg-background p-4 text-sm">
      <h3 className="text-sm font-semibold">Audit timeline</h3>
      <p className="mt-2 text-muted-foreground">
        The custody event timeline lives on the dedicated audit page (D4).
      </p>
      <Link
        href={`/audit/${jobId}`}
        className="mt-3 inline-block text-sm text-primary hover:underline"
        data-testid="audit-link"
      >
        Open audit timeline for {jobId}
      </Link>
    </div>
  );
}
