import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ReviewDecisionPanel } from "@/components/ReviewDecisionPanel";
import { clearApiKey, setApiKey } from "@/lib/auth";
import type { ReviewItem } from "@/lib/types";

function itemFixture(partial: Partial<ReviewItem> = {}): ReviewItem {
  return {
    review_id: "rev_bbbbbbbbbbbb",
    job_id: "job_abc",
    reason: "low_confidence",
    confidence: 0.5,
    quality_classification: "review_required",
    status: partial.status ?? "pending",
    reviewer: partial.reviewer ?? "",
    decision_notes: partial.decision_notes ?? "",
    created_at: "2026-04-25T10:00:00Z",
    reviewed_at: partial.reviewed_at ?? "",
    metadata: partial.metadata ?? {},
    ...partial,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("ReviewDecisionPanel reviewer identity", () => {
  beforeEach(() => {
    clearApiKey();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    clearApiKey();
  });

  it("posts a non-empty derived reviewer when prop is empty", async () => {
    setApiKey("sk-live-abcdefghijklmnop");
    const onDecision = vi.fn();
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(
        itemFixture({
          status: "approved",
          reviewer: "sk-l…mnop",
          reviewed_at: "2026-04-25T11:00:00Z",
        })
      )
    );

    render(
      <ReviewDecisionPanel
        reviewId="rev_bbbbbbbbbbbb"
        item={itemFixture()}
        onDecision={onDecision}
      />
    );

    expect(screen.getByTestId("review-decision-recording-as")).toHaveTextContent(
      "sk-l…mnop"
    );

    fireEvent.click(screen.getByTestId("review-decision-approved"));

    await waitFor(() => {
      expect(onDecision).toHaveBeenCalled();
    });

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    const body = JSON.parse(String(init.body)) as { reviewer: string; status: string };
    expect(body.status).toBe("approved");
    expect(body.reviewer).toBe("sk-l…mnop");
    expect(body.reviewer.trim().length).toBeGreaterThan(0);
  });

  it("falls back to operator when no API key is cached", async () => {
    const onDecision = vi.fn();
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(
        itemFixture({
          status: "rejected",
          reviewer: "operator",
          reviewed_at: "2026-04-25T11:00:00Z",
        })
      )
    );

    render(
      <ReviewDecisionPanel
        reviewId="rev_bbbbbbbbbbbb"
        item={itemFixture()}
        onDecision={onDecision}
        reviewer=""
      />
    );

    expect(screen.getByTestId("review-decision-recording-as")).toHaveTextContent(
      /Recording as:\s*operator/
    );

    fireEvent.click(screen.getByTestId("review-decision-rejected"));

    await waitFor(() => {
      expect(onDecision).toHaveBeenCalled();
    });

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    const body = JSON.parse(String(init.body)) as { reviewer: string };
    expect(body.reviewer).toBe("operator");
  });

  it("honors an explicit reviewer prop over the derived label", async () => {
    setApiKey("sk-live-abcdefghijklmnop");
    const onDecision = vi.fn();
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(
        itemFixture({
          status: "reprocess",
          reviewer: "alice@example.com",
          reviewed_at: "2026-04-25T11:00:00Z",
        })
      )
    );

    render(
      <ReviewDecisionPanel
        reviewId="rev_bbbbbbbbbbbb"
        item={itemFixture()}
        onDecision={onDecision}
        reviewer="alice@example.com"
      />
    );

    expect(screen.getByTestId("review-decision-recording-as")).toHaveTextContent(
      "alice@example.com"
    );

    fireEvent.click(screen.getByTestId("review-decision-reprocess"));

    await waitFor(() => {
      expect(onDecision).toHaveBeenCalled();
    });

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    const body = JSON.parse(String(init.body)) as { reviewer: string };
    expect(body.reviewer).toBe("alice@example.com");
  });
});
