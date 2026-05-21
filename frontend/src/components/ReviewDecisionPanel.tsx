"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { ApiError, post } from "@/lib/api-client";
import type {
  ReviewDecision,
  ReviewDecisionRequest,
  ReviewItem,
} from "@/lib/types";

const DECISION_BUTTONS: Array<{
  decision: ReviewDecision;
  label: string;
  variant: "default" | "destructive" | "outline";
  description: string;
}> = [
  {
    decision: "approved",
    label: "Approve",
    variant: "default",
    description: "Accept the OCR/translation output as-is.",
  },
  {
    decision: "rejected",
    label: "Reject",
    variant: "destructive",
    description: "Discard the output. The job is marked failed for downstream consumers.",
  },
  {
    decision: "reprocess",
    label: "Escalate / Reprocess",
    variant: "outline",
    description: "Send back through the pipeline (e.g. higher DPI, alternate engine).",
  },
];

export interface ReviewDecisionPanelProps {
  reviewId: string;
  /** Current item state -- used for optimistic UI and to disable when already decided. */
  item: ReviewItem;
  /** Called with the API-returned item after a successful decision. */
  onDecision: (updated: ReviewItem) => void;
  /** Optional reviewer label (defaults to empty string -> backend records as anonymous). */
  reviewer?: string;
}

/**
 * Decision panel: notes textarea + Approve/Reject/Escalate buttons.
 *
 * - Disabled while submitting and after a non-pending decision lands.
 * - Optimistic update fires before the network round-trip; on error we
 *   surface the message and the parent can refresh the item.
 */
export function ReviewDecisionPanel({
  reviewId,
  item,
  onDecision,
  reviewer = "",
}: ReviewDecisionPanelProps) {
  const [notes, setNotes] = useState<string>("");
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [pendingDecision, setPendingDecision] = useState<ReviewDecision | null>(null);
  const [error, setError] = useState<string | null>(null);

  const alreadyDecided = item.status !== "pending";

  async function submit(decision: ReviewDecision) {
    if (alreadyDecided || submitting) return;
    setSubmitting(true);
    setPendingDecision(decision);
    setError(null);
    const body: ReviewDecisionRequest = {
      status: decision,
      reviewer,
      notes,
    };
    try {
      const updated = await post<ReviewItem>(
        `/api/v1/review/${encodeURIComponent(reviewId)}/decision`,
        body
      );
      onDecision(updated);
    } catch (err: unknown) {
      const message =
        err instanceof ApiError
          ? `${err.status} ${err.message}`
          : err instanceof Error
            ? err.message
            : "Decision failed";
      setError(message);
    } finally {
      setSubmitting(false);
      setPendingDecision(null);
    }
  }

  return (
    <section
      className="space-y-3 rounded-md border border-border bg-background p-4"
      aria-labelledby="review-decision-heading"
      data-testid="review-decision-panel"
    >
      <h2 id="review-decision-heading" className="text-sm font-semibold">
        Decision
      </h2>

      {alreadyDecided ? (
        <p
          className="text-xs text-muted-foreground"
          data-testid="review-decision-locked"
        >
          This item was already decided as <strong>{item.status}</strong>
          {item.reviewer ? ` by ${item.reviewer}` : ""}
          {item.reviewed_at ? ` at ${item.reviewed_at}` : ""}.
          Decisions are immutable once recorded.
        </p>
      ) : null}

      <label className="block text-xs font-medium text-muted-foreground">
        Reviewer notes
        <textarea
          data-testid="review-decision-notes"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          disabled={alreadyDecided || submitting}
          placeholder="Why are you making this call? (optional, recorded in custody log)"
          className="mt-1 block w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          rows={3}
          maxLength={2000}
          aria-label="Reviewer notes"
        />
      </label>

      <div className="flex flex-wrap gap-2" role="group" aria-label="Decision actions">
        {DECISION_BUTTONS.map((btn) => (
          <Button
            key={btn.decision}
            variant={btn.variant}
            disabled={alreadyDecided || submitting}
            onClick={() => void submit(btn.decision)}
            data-testid={`review-decision-${btn.decision}`}
            aria-label={`${btn.label}: ${btn.description}`}
          >
            {submitting && pendingDecision === btn.decision ? "Submitting..." : btn.label}
          </Button>
        ))}
      </div>

      {error ? (
        <p
          role="alert"
          className="text-xs text-destructive"
          data-testid="review-decision-error"
        >
          {error}
        </p>
      ) : null}
    </section>
  );
}
