import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import AuditDetailPage from "@/app/audit/[id]/page";
import { setApiKey } from "@/lib/auth";

const useParamsMock = vi.fn();

vi.mock("next/navigation", => ({
  useParams: => useParamsMock(),
  useRouter: => ({ push: vi.fn() }),
  usePathname: => "/audit/job-1",
}));

// The detail page calls fetchCustodyLog which uses fetch directly. Mock fetch
// to return a known-good 3-event chain so the verifier has something real to
// compute against.
const event1 = {
  document_id: "doc-1",
  event_type: "file_ingested",
  timestamp: "2026-04-26T10:00:00.000+00:00",
  data: { source_path: "/in/foo.pdf", size: 1024 },
  prev_hash: null,
  hash: "b7a5c5b6ad0c3fe333c619f4de1d3609bcad7d06a960dc59ed36e40c39ad10b0",
};
const event2 = {
  document_id: "doc-1",
  event_type: "page_extracted",
  timestamp: "2026-04-26T10:00:01.000+00:00",
  data: { page: 1 },
  prev_hash: event1.hash,
  hash: "e16331b7a8e40241b18aae80be56cf6cb8fe970b4e5b5e4beae4d425ba435667",
};
const event3 = {
  document_id: "doc-1",
  event_type: "ocr_primary",
  timestamp: "2026-04-26T10:00:02.000+00:00",
  data: { page: 1, engine: "paddleocr", confidence: 0.97 },
  prev_hash: event2.hash,
  hash: "11b5d32beec9958459ca9e784c7f2e5fb1acd57e2b17d1781106144874a75207",
};

function ndjson(events: object[]): string {
  return events.map((e) => JSON.stringify(e)).join("\n");
}

function ndjsonResponse(events: object[]): Response {
  return new Response(ndjson(events), {
    status: 200,
    headers: { "content-type": "application/jsonl" },
  });
}

describe("AuditDetailPage", => {
  beforeEach(() => {
    setApiKey("test-key");
    useParamsMock.mockReturnValue({ id: "job-1" });
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      ndjsonResponse([event1, event2, event3])
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the timeline once the custody log loads", async => {
    render(<AuditDetailPage />);

    await waitFor(() =>
      expect(screen.getByTestId("audit-timeline")).toBeInTheDocument()
    );

    const cards = screen.getAllByTestId("audit-event-card");
    expect(cards).toHaveLength(3);
    expect(cards[0]).toHaveAttribute("data-event-type", "file_ingested");
    expect(cards[1]).toHaveAttribute("data-event-type", "page_extracted");
    expect(cards[2]).toHaveAttribute("data-event-type", "ocr_primary");
  });

  it("expands an event payload when 'Show payload' is clicked", async => {
    const user = userEvent.setup();
    render(<AuditDetailPage />);

    await waitFor(() =>
      expect(screen.getByTestId("audit-timeline")).toBeInTheDocument()
    );

    const cards = screen.getAllByTestId("audit-event-card");
    const showButton = within(cards[0]).getByRole("button", { name: /show payload/i });
    await user.click(showButton);

    // After expansion, the payload string from `data` should appear.
    expect(within(cards[0]).getByText(/source_path/)).toBeInTheDocument();
  });

  it("filters timeline entries by event type", async => {
    const user = userEvent.setup();
    render(<AuditDetailPage />);

    await waitFor(() =>
      expect(screen.getByTestId("audit-timeline")).toBeInTheDocument()
    );

    expect(screen.getAllByTestId("audit-event-card")).toHaveLength(3);

    const filterButton = screen.getByRole("button", { name: "page_extracted" });
    await user.click(filterButton);

    const filtered = screen.getAllByTestId("audit-event-card");
    expect(filtered).toHaveLength(1);
    expect(filtered[0]).toHaveAttribute("data-event-type", "page_extracted");
  });

  it("flips the verification badge to 'Chain intact' after Verify", async => {
    const user = userEvent.setup();
    render(<AuditDetailPage />);

    await waitFor(() =>
      expect(screen.getByTestId("audit-timeline")).toBeInTheDocument()
    );

    expect(screen.getByText(/not yet verified/i)).toBeInTheDocument();

    await user.click(screen.getByTestId("verify-button"));

    await waitFor(() =>
      expect(screen.getByText(/chain intact/i)).toBeInTheDocument()
    );
  });

  it("shows a clear error when the audit endpoint returns 404", async => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Response(JSON.stringify({ detail: "no_outputs" }), {
        status: 404,
        headers: { "content-type": "application/json" },
      })
    );

    render(<AuditDetailPage />);

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(/No custody log found/i)
    );
  });
});
