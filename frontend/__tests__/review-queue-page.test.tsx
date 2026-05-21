import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ReviewQueuePage from "@/app/review/page";
import { setApiKey } from "@/lib/auth";
import type { ReviewItem, ReviewQueueResponse } from "@/lib/types";

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

const replaceMock = vi.fn();
const pushMock = vi.fn();
const searchParamsHolder = { current: new URLSearchParams() };

vi.mock("next/navigation", => ({
  __esModule: true,
  useRouter: => ({ replace: replaceMock, push: pushMock }),
  useSearchParams: => searchParamsHolder.current,
}));

function reviewFixture(partial: Partial<ReviewItem> = {}): ReviewItem {
  return {
    review_id: partial.review_id ?? "rev_aaaaaaaaaaaa",
    job_id: partial.job_id ?? "job_111",
    reason: partial.reason ?? "low_confidence",
    confidence: partial.confidence ?? 0.55,
    quality_classification: partial.quality_classification ?? "review_required",
    status: partial.status ?? "pending",
    reviewer: partial.reviewer ?? "",
    decision_notes: partial.decision_notes ?? "",
    created_at: partial.created_at ?? "2026-04-25T10:00:00Z",
    reviewed_at: partial.reviewed_at ?? "",
    metadata: partial.metadata ?? {},
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("ReviewQueuePage", => {
  beforeEach(() => {
    setApiKey("test-key");
    replaceMock.mockClear();
    pushMock.mockClear();
    searchParamsHolder.current = new URLSearchParams();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the rows returned from the API", async => {
    const payload: ReviewQueueResponse = {
      items: [
        reviewFixture(),
        reviewFixture({ review_id: "rev_bbbbbbbbbbbb", job_id: "job_222" }),
      ],
      total: 2,
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(payload));

    render(<ReviewQueuePage />);

    await waitFor(() => {
      expect(screen.queryByTestId("review-loading")).toBeNull();
    });

    expect(screen.getByTestId("review-link-rev_aaaaaaaaaaaa")).toHaveAttribute(
      "href",
      "/review/rev_aaaaaaaaaaaa"
    );
    expect(screen.getByTestId("review-link-rev_bbbbbbbbbbbb")).toBeInTheDocument();
    expect(screen.getByTestId("pagination-summary")).toHaveTextContent(
      "Showing 1-2 of 2"
    );
  });

  it("shows the empty state when API returns no items", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ items: [], total: 0 } satisfies ReviewQueueResponse)
    );
    render(<ReviewQueuePage />);
    await waitFor(() => {
      expect(screen.getByTestId("review-empty")).toBeInTheDocument();
    });
  });

  it("forwards a single status filter as a server query param", async => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(
        jsonResponse({ items: [], total: 0 } satisfies ReviewQueueResponse)
      );

    render(<ReviewQueuePage />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    // Toggle off pending and toggle on approved -> single status filter "approved".
    fireEvent.click(screen.getByTestId("review-chip-pending"));
    fireEvent.click(screen.getByTestId("review-chip-approved"));

    await waitFor(() => {
      const lastCallUrl = fetchMock.mock.calls.at(-1)?.[0] as string;
      expect(lastCallUrl).toMatch(/status=approved/);
    });
  });

  it("syncs filter state to the URL via router.replace", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ items: [], total: 0 } satisfies ReviewQueueResponse)
    );
    render(<ReviewQueuePage />);
    await waitFor(() => expect(replaceMock).toHaveBeenCalled());
    // Initial sync writes the default pending filter.
    const firstCall = replaceMock.mock.calls[0]?.[0] as string;
    expect(firstCall).toMatch(/^\/review/);
    expect(firstCall).toMatch(/status=pending/);
  });

  it("paginates by sending offset on next-page click", async => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        items: Array.from({ length: 25 }, (_, i) =>
          reviewFixture({
            review_id: `rev_${i.toString(16).padStart(12, "0")}`,
          })
        ),
        total: 60,
      } satisfies ReviewQueueResponse)
    );

    render(<ReviewQueuePage />);
    await waitFor(() => expect(screen.queryByTestId("review-loading")).toBeNull());

    fireEvent.click(screen.getByTestId("pagination-next"));

    await waitFor(() => {
      const lastCallUrl = fetchMock.mock.calls.at(-1)?.[0] as string;
      expect(lastCallUrl).toMatch(/offset=25/);
    });
  });
});
