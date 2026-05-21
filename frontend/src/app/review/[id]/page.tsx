"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { CertifyDialog } from "@/components/CertifyDialog";
import { ReviewDecisionPanel } from "@/components/ReviewDecisionPanel";
import { useRequireAuth } from "@/lib/auth";
import { useReviewItem } from "@/lib/hooks";
import type { ReviewItem } from "@/lib/types";

function formatTimestamp(value: string): string {
  if (!value) return "—";
  try {
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return value;
    return d.toLocaleString();
  } catch {
    return value;
  }
}

function MetadataRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 border-b border-border/50 py-1 text-sm last:border-b-0">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-medium">{value || "—"}</dd>
    </div>
  );
}

export default function ReviewDetailPage() {
  useRequireAuth();
  const params = useParams<{ id: string }>();
  const reviewId = typeof params?.id === "string" ? params.id : "";

  const { data, error, loading, refresh } = useReviewItem(reviewId);
  const [certifyOpen, setCertifyOpen] = useState<boolean>(false);
  const [optimisticItem, setOptimisticItem] = useState<ReviewItem | null>(null);

  const item = optimisticItem ?? data;

  function handleDecision(updated: ReviewItem) {
    setOptimisticItem(updated);
    // Refresh from server so we drop the optimistic copy once authoritative
    // data arrives.
    setTimeout(() => {
      setOptimisticItem(null);
      refresh();
    }, 0);
  }

  function handleCertified(updated: ReviewItem) {
    setOptimisticItem(updated);
    setTimeout(() => {
      setOptimisticItem(null);
      refresh();
    }, 0);
  }

  if (loading && !item) {
    return (
      <div className="space-y-2" data-testid="review-detail-loading">
        <h1 className="text-2xl font-semibold">Review item</h1>
        <p className="text-sm text-muted-foreground">Loading…</p>
      </div>
    );
  }

  if (error && !item) {
    return (
      <div className="space-y-3">
        <h1 className="text-2xl font-semibold">Review item</h1>
        <p
          role="alert"
          className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
          data-testid="review-detail-error"
        >
          Failed to load review item: {error.message}
        </p>
        <Button variant="outline" size="sm" onClick={() => refresh()}>
          Retry
        </Button>
      </div>
    );
  }

  if (!item) {
    return (
      <div className="space-y-2" data-testid="review-detail-missing">
        <h1 className="text-2xl font-semibold">Review item</h1>
        <p className="text-sm text-muted-foreground">Item not found.</p>
        <Link className="text-sm text-primary hover:underline" href="/review">
          ← Back to review queue
        </Link>
      </div>
    );
  }

  const certified = Boolean(item.metadata?.certified);
  const ocrText =
    typeof item.metadata?.ocr_text === "string"
      ? (item.metadata.ocr_text as string)
      : "";
  const translation =
    typeof item.metadata?.translation === "string"
      ? (item.metadata.translation as string)
      : "";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Link
          className="text-sm text-primary hover:underline"
          href="/review"
          data-testid="review-detail-back"
        >
          ← Back
        </Link>
        <h1 className="text-2xl font-semibold" data-testid="review-detail-id">
          {item.review_id}
        </h1>
        <StatusBadge status={item.status} />
        {certified ? (
          <span
            className="inline-flex items-center rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-medium text-emerald-800 ring-1 ring-inset ring-emerald-300"
            data-testid="review-certified-badge"
          >
            certified
          </span>
        ) : null}
      </div>

      <section
        className="rounded-md border border-border bg-background p-4"
        aria-labelledby="review-summary-heading"
      >
        <h2 id="review-summary-heading" className="text-sm font-semibold">
          Summary
        </h2>
        <dl className="mt-3 grid gap-1 md:grid-cols-2">
          <MetadataRow
            label="Job"
            value={item.job_id}
          />
          <MetadataRow label="Reason" value={item.reason} />
          <MetadataRow
            label="Confidence"
            value={`${(item.confidence * 100).toFixed(1)}%`}
          />
          <MetadataRow
            label="Quality"
            value={item.quality_classification}
          />
          <MetadataRow
            label="Submitted"
            value={formatTimestamp(item.created_at)}
          />
          <MetadataRow
            label="Reviewed"
            value={formatTimestamp(item.reviewed_at)}
          />
          <MetadataRow label="Reviewer" value={item.reviewer} />
        </dl>
        <p className="mt-3 text-sm">
          <Link
            href={`/jobs/${encodeURIComponent(item.job_id)}`}
            className="text-primary hover:underline"
            data-testid="review-detail-job-link"
          >
            Open original document →
          </Link>
        </p>
      </section>

      {ocrText ? (
        <details
          className="rounded-md border border-border bg-background p-4"
          data-testid="review-ocr-section"
        >
          <summary className="cursor-pointer text-sm font-semibold">
            OCR text
          </summary>
          <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap rounded bg-muted/40 p-3 text-xs">
            {ocrText}
          </pre>
        </details>
      ) : null}

      {translation ? (
        <details
          className="rounded-md border border-border bg-background p-4"
          data-testid="review-translation-section"
        >
          <summary className="cursor-pointer text-sm font-semibold">
            Translation
          </summary>
          <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap rounded bg-muted/40 p-3 text-xs">
            {translation}
          </pre>
        </details>
      ) : null}

      <ReviewDecisionPanel
        reviewId={reviewId}
        item={item}
        onDecision={handleDecision}
      />

      <section
        className="rounded-md border border-border bg-background p-4"
        aria-labelledby="review-certify-heading"
      >
        <h2 id="review-certify-heading" className="text-sm font-semibold">
          Translation certification
        </h2>
        <p className="mt-2 text-xs text-muted-foreground">
          Certification flips this item to <code>certified=true</code> and emits an
          immutable strong-auth custody event. Required for translations entering
          legal/forensic record.
        </p>
        <div className="mt-3">
          <Button
            variant="outline"
            disabled={certified}
            onClick={() => setCertifyOpen(true)}
            data-testid="review-certify-open"
          >
            {certified ? "Already certified" : "Certify Translation"}
          </Button>
        </div>
      </section>

      <CertifyDialog
        reviewId={reviewId}
        open={certifyOpen}
        onClose={() => setCertifyOpen(false)}
        onCertified={handleCertified}
      />

      <section
        className="rounded-md border border-border bg-background p-4"
        aria-labelledby="review-audit-heading"
        data-testid="review-audit-timeline"
      >
        <h2 id="review-audit-heading" className="text-sm font-semibold">
          Custody timeline
        </h2>
        <p className="mt-2 text-xs text-muted-foreground">
          Full chain-of-custody for this job is available on the{" "}
          <Link
            className="text-primary hover:underline"
            href={`/audit/${encodeURIComponent(item.job_id)}`}
          >
            audit page
          </Link>
          .
        </p>
        {Array.isArray(item.metadata?.events) && (item.metadata.events as unknown[]).length > 0 ? (
          <ol className="mt-3 space-y-2">
            {(item.metadata.events as Array<Record<string, unknown>>).map((evt, idx) => (
              <li
                key={idx}
                className="rounded border border-border/60 p-2 text-xs"
                data-testid={`review-event-${idx}`}
              >
                <div className="flex justify-between gap-2">
                  <span className="font-mono">
                    {String(evt.event_type ?? evt.type ?? "event")}
                  </span>
                  <span className="text-muted-foreground">
                    {formatTimestamp(String(evt.timestamp ?? ""))}
                  </span>
                </div>
                {evt.actor ? (
                  <div className="mt-1 text-muted-foreground">
                    actor: {String(evt.actor)}
                  </div>
                ) : null}
              </li>
            ))}
          </ol>
        ) : (
          <p className="mt-2 text-xs italic text-muted-foreground">
            No inline events for this review item.
          </p>
        )}
      </section>
    </div>
  );
}
