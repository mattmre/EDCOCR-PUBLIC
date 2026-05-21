"use client";

import { type FormEvent, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useRequireAuth } from "@/lib/auth";
import { listRecentJobs, type JobSummary } from "@/lib/audit-api";
import { ApiError } from "@/lib/api-client";

export default function AuditPage() {
  useRequireAuth();
  const router = useRouter();
  const [search, setSearch] = useState("");
  const [recentJobs, setRecentJobs] = useState<JobSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    listRecentJobs(20)
      .then((res) => {
        if (cancelled) return;
        setRecentJobs(res.jobs);
        setLoadError(null);
      })
      .catch((err) => {
        if (cancelled) return;
        setRecentJobs([]);
        setLoadError(
          err instanceof ApiError
            ? err.message
            : err instanceof Error
            ? err.message
            : "Could not load recent jobs"
        );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return => {
      cancelled = true;
    };
  }, []);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = search.trim();
    if (trimmed.length === 0) return;
    router.push(`/audit/${encodeURIComponent(trimmed)}`);
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Audit</h1>
        <p className="text-sm text-muted-foreground">
          Forensic chain-of-custody timeline with client-side hash-chain verification.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Open a job&apos;s audit log</CardTitle>
          <CardDescription>
            Enter a job id to view its custody timeline, or pick one of the recent jobs below.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="flex gap-2">
            <Input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="job-abc123…"
              aria-label="Job id"
              className="font-mono"
            />
            <Button type="submit">Open</Button>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Recent jobs</CardTitle>
          <CardDescription>
            Last 20 jobs known to the API. Custody logs are produced when the pipeline runs with
            chain-of-custody enabled.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {loading ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : loadError ? (
            <p className="text-sm text-destructive" role="alert">
              {loadError}
            </p>
          ) : recentJobs.length === 0 ? (
            <p className="text-sm text-muted-foreground">No jobs yet.</p>
          ) : (
            <ul className="divide-y divide-border" data-testid="recent-jobs-list">
              {recentJobs.map((job) => (
                <li key={job.job_id} className="flex items-center justify-between py-2 text-sm">
                  <div>
                    <Link
                      href={`/audit/${encodeURIComponent(job.job_id)}`}
                      className="font-mono font-medium text-primary hover:underline"
                    >
                      {job.job_id}
                    </Link>
                    {job.filename ? (
                      <span className="ml-2 text-muted-foreground">{job.filename}</span>
                    ) : null}
                  </div>
                  <span className="text-xs text-muted-foreground">{job.status}</span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>What hash-chain verification proves</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm leading-relaxed text-muted-foreground">
          <p>
            Every step of the OCR pipeline appends an event to a per-document custody log. Each
            event records the document id, event type, ISO-8601 UTC timestamp, payload, and a
            SHA-256 hash that covers all of those fields plus the previous event&apos;s hash. That
            forward-link is what makes the log a chain: changing any earlier event changes its
            hash, which breaks the link of every event that came after it.
          </p>
          <p>
            When you open a job below and click <em>Verify hash chain</em>, your browser refetches
            the JSONL log, recomputes each SHA-256 from scratch using the Web Crypto API, and
            compares it against the recorded hash. A green <em>Chain intact</em> badge means the
            log we just read matches the producer&apos;s hashes byte-for-byte. A red{" "}
            <em>Chain broken</em> badge identifies the first event that fails to verify so you know
            exactly where to investigate.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
