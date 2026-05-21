"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useRequireAuth } from "@/lib/auth";
import { fetchCustodyLog } from "@/lib/audit-api";
import {
  type CustodyEvent,
  type VerificationResult,
  verifyChain,
} from "@/lib/audit-verify";
import { ApiError } from "@/lib/api-client";
import { AuditTimeline } from "@/components/audit-timeline";
import { VerificationStatusBadge } from "@/components/verification-status";

function downloadBlob(filename: string, mime: string, content: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export default function AuditDetailPage() {
  useRequireAuth();
  const params = useParams<{ id: string }>();
  const jobId = typeof params?.id === "string" ? params.id : "";

  const [events, setEvents] = useState<CustodyEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [verification, setVerification] = useState<VerificationResult | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    setLoading(true);
    setLoadError(null);
    setEvents([]);
    setVerification(null);
    fetchCustodyLog(jobId)
      .then((parsed) => {
        if (cancelled) return;
        setEvents(parsed);
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) {
          setLoadError(
            "No custody log found for this job. Either the job id is wrong or the pipeline did not run with chain-of-custody enabled."
          );
        } else {
          setLoadError(
            err instanceof ApiError
              ? err.message
              : err instanceof Error
              ? err.message
              : "Could not load custody log"
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return => {
      cancelled = true;
    };
  }, [jobId]);

  const filename = useMemo(() => {
    for (const event of events) {
      if (event.event_type === "file_ingested" && event.data && typeof event.data === "object") {
        const sourcePath = (event.data as Record<string, unknown>).source_path;
        if (typeof sourcePath === "string") {
          const parts = sourcePath.replace(/\\/g, "/").split("/");
          return parts[parts.length - 1];
        }
      }
    }
    return null;
  }, [events]);

  const verify = useCallback(async => {
    setIsVerifying(true);
    try {
      const result = await verifyChain(events);
      setVerification(result);
    } catch (err) {
      setVerification({
        status: "broken",
        totalEvents: events.length,
        verifiedEvents: 0,
        reason:
          err instanceof Error ? err.message : "Verification failed with an unknown error.",
      });
    } finally {
      setIsVerifying(false);
    }
  }, [events]);

  const handleExportJson = useCallback(() => {
    downloadBlob(`${jobId}.custody.json`, "application/json", JSON.stringify(events, null, 2));
  }, [events, jobId]);

  const handleExportNdjson = useCallback(() => {
    const ndjson = events.map((e) => JSON.stringify(e)).join("\n");
    downloadBlob(`${jobId}.custody.jsonl`, "application/jsonl", ndjson);
  }, [events, jobId]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <Link
            href="/audit"
            className="text-xs text-muted-foreground hover:text-primary"
          >
            ← Back to audit
          </Link>
          <h1 className="text-2xl font-semibold">
            Audit: <span className="font-mono">{jobId || "(unknown)"}</span>
          </h1>
          {filename ? (
            <p className="text-sm text-muted-foreground">{filename}</p>
          ) : null}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <VerificationStatusBadge result={verification} isVerifying={isVerifying} />
          <Button
            type="button"
            onClick={verify}
            disabled={isVerifying || loading || events.length === 0}
            data-testid="verify-button"
          >
            {isVerifying ? "Verifying…" : "Verify hash chain"}
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={handleExportJson}
            disabled={events.length === 0}
          >
            Export JSON
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={handleExportNdjson}
            disabled={events.length === 0}
          >
            Export NDJSON
          </Button>
        </div>
      </div>

      {loading ? (
        <Card>
          <CardContent className="p-6 text-sm text-muted-foreground">Loading custody log…</CardContent>
        </Card>
      ) : loadError ? (
        <Card>
          <CardHeader>
            <CardTitle>Could not load custody log</CardTitle>
            <CardDescription>The audit endpoint returned an error.</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-destructive" role="alert">
              {loadError}
            </p>
          </CardContent>
        </Card>
      ) : events.length === 0 ? (
        <Card>
          <CardContent className="p-6 text-sm text-muted-foreground">
            The custody log is empty.
          </CardContent>
        </Card>
      ) : (
        <>
          {verification?.status === "broken" ? (
            <div
              role="alert"
              className="rounded border border-red-200 bg-red-50 p-3 text-sm text-red-900"
            >
              <p className="font-semibold">Chain broken at event #{verification.brokenAtIndex}</p>
              <p className="mt-1">{verification.reason}</p>
            </div>
          ) : null}
          <AuditTimeline events={events} verification={verification} />
        </>
      )}
    </div>
  );
}
