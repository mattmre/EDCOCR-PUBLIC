import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ReviewDetailPage from "@/app/review/[id]/page";
import { setApiKey } from "@/lib/auth";
import type { ReviewItem } from "@/lib/types";

vi.mock("next/link", => ({
  __esModule: true,
  default: ({
    href,
    children,
    onClick,
    ...rest
  }: {
    href: string;
    children: React.ReactNode;
    onClick?: (e: React.MouseEvent) => void;
  }) => (
    // eslint-disable-next-line jsx-a11y/anchor-has-content
    <a href={href} onClick={onClick} {...rest}>
      {children}
    </a>
  ),
}));

vi.mock("next/navigation", => ({
  __esModule: true,
  useParams: => ({ id: "rev_aaaaaaaaaaaa" }),
}));

function itemFixture(partial: Partial<ReviewItem> = {}): ReviewItem {
  return {
    review_id: "rev_aaaaaaaaaaaa",
    job_id: "job_xyz",
    reason: "low_confidence",
    confidence: 0.42,
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

describe("ReviewDetailPage", => {
  beforeEach(() => {
    setApiKey("test-key");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders summary fields for a pending item", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(itemFixture()));

    render(<ReviewDetailPage />);

    await waitFor(() => {
      expect(screen.getByTestId("review-detail-id")).toHaveTextContent(
        "rev_aaaaaaaaaaaa"
      );
    });
    expect(screen.getByTestId("review-detail-job-link")).toHaveAttribute(
      "href",
      "/jobs/job_xyz"
    );
    expect(screen.getByTestId("review-decision-panel")).toBeInTheDocument();
    expect(screen.getByTestId("review-decision-approved")).not.toBeDisabled();
  });

  it("submits an approve decision and updates the panel optimistically", async => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    fetchMock.mockResolvedValueOnce(jsonResponse(itemFixture()));
    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        itemFixture({
          status: "approved",
          reviewer: "operator@example.com",
          reviewed_at: "2026-04-25T10:05:00Z",
        })
      )
    );
    // Subsequent refresh fetch.
    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        itemFixture({
          status: "approved",
          reviewer: "operator@example.com",
          reviewed_at: "2026-04-25T10:05:00Z",
        })
      )
    );

    render(<ReviewDetailPage />);

    await waitFor(() =>
      expect(screen.getByTestId("review-decision-approved")).toBeInTheDocument()
    );

    fireEvent.change(screen.getByTestId("review-decision-notes"), {
      target: { value: "Looks good." },
    });
    fireEvent.click(screen.getByTestId("review-decision-approved"));

    await waitFor(() => {
      expect(screen.getByTestId("review-decision-locked")).toBeInTheDocument();
    });

    // The decision endpoint was called with the proper body.
    const decisionCall = fetchMock.mock.calls.find(([url]) =>
      String(url).includes("/decision")
    );
    expect(decisionCall).toBeTruthy();
    const init = decisionCall?.[1] as RequestInit;
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toMatchObject({
      status: "approved",
      notes: "Looks good.",
    });
  });

  it("surfaces a decision error and keeps the panel actionable", async => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    fetchMock.mockResolvedValueOnce(jsonResponse(itemFixture()));
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ detail: "queue is busy" }, 409)
    );

    render(<ReviewDetailPage />);

    await waitFor(() =>
      expect(screen.getByTestId("review-decision-rejected")).toBeInTheDocument()
    );

    fireEvent.click(screen.getByTestId("review-decision-rejected"));

    await waitFor(() => {
      expect(screen.getByTestId("review-decision-error")).toHaveTextContent(/409/);
    });
    expect(screen.getByTestId("review-decision-rejected")).not.toBeDisabled();
  });

  it("locks the decision panel when the item was already decided", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(
        itemFixture({
          status: "approved",
          reviewer: "operator@example.com",
          reviewed_at: "2026-04-25T10:05:00Z",
        })
      )
    );

    render(<ReviewDetailPage />);

    await waitFor(() => {
      expect(screen.getByTestId("review-decision-locked")).toBeInTheDocument();
    });
    expect(screen.getByTestId("review-decision-approved")).toBeDisabled();
  });
});
