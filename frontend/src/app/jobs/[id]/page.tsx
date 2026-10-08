"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
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
const TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);

export default function JobDetailPage() {
  useRequireAuth();
  const params = useParams<{ id: string }>();
  const jobId = (params?.id ?? "") as string;

  const [job, setJob] = useState<Job | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const terminalEvent = useRef<{ jobId: string; status: string } | null>(null);

  const { status: wsStatus, lastMessage, reconnect } = useJobWebSocket(jobId);

  // Initial fetch + low-frequency poll as a fallback when the WS is down.
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    let inFlight = false;
    terminalEvent.current = null;
    // Drop the previous route's record before the next fetch. A 404 for the
    // new id must not keep rendering the job the operator just left.
    setJob(null);
    setLoadError(null);
    setLoading(true);
    async function load() {
      // Serialize fallback polls: a slow request must not overwrite a newer
      // response. The API client supplies the request deadline.
      if (inFlight) return;
      inFlight = true;
      try {
        const response = await get<Job>(`/api/v1/jobs/${jobId}`);
        if (cancelled) return;
        if (response.job_id !== jobId) {
          throw new Error("Job response does not match the requested job");
        }
        setJob((current) => {
          // A poll begun before the terminal socket event may arrive later.
          // API retries create a new job ID, so this does not hide retries.
          if (current?.job_id === jobId && TERMINAL_STATUSES.has(current.status)
              && !TERMINAL_STATUSES.has(response.status)) return current;
          // The socket can complete before the initial REST request returns.
          const terminal = terminalEvent.current;
          if (terminal?.jobId === jobId && !TERMINAL_STATUSES.has(response.status)) {
            return { ...response, status: terminal.status };
          }
          return response;
        });
        setLoadError(null);
        setLoading(false);
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
        setJob((current) => (current && current.job_id === jobId ? current : null));
      } finally {
        inFlight = false;
      }
    }
    void load();
    const id = setInterval(() => void load(), POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [jobId]);

  // Consume each socket message when it arrives, not again when a REST poll
  // changes job. Merge progress into the same state so an old socket frame
  // cannot keep overriding the newer REST progress in the progress panel.
  useEffect(() => {
    if (!lastMessage || !("job_id" in lastMessage) || lastMessage.job_id !== jobId) return;
    if (TERMINAL_STATUSES.has(lastMessage.status)) {
      terminalEvent.current = { jobId, status: lastMessage.status };
    }
    setJob((current) => {
      if (!current || current.job_id !== jobId || TERMINAL_STATUSES.has(current.status)) return current;
      if (lastMessage.type === "completed" || lastMessage.type === "failed" || lastMessage.type === "cancelled") {
        return { ...current, status: lastMessage.type };
      }
      if (lastMessage.type !== "progress") return current;
      return {
        ...current,
        status: lastMessage.status,
        progress: {
          total_pages: lastMessage.total_pages ?? current.progress?.total_pages ?? 0,
          pages_completed: lastMessage.pages_completed ?? current.progress?.pages_completed ?? 0,
          percent_complete: lastMessage.percent ?? current.progress?.percent_complete ?? 0,
          current_stage: lastMessage.current_stage ?? current.progress?.current_stage ?? lastMessage.status,
        },
      };
    });
  }, [lastMessage, jobId]);

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
      {loadError ? <p role="alert" className="text-sm text-destructive">Failed to refresh job: {loadError}</p> : null}
      <Link href="/jobs" className="text-sm text-primary hover:underline">
        ← Back to jobs
      </Link>
      <JobDetailHeader job={job} />
      <JobProgress
        job={job}
        wsStatus={wsStatus}
        lastMessage={null}
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
