import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BatchesTable } from "@/components/BatchesTable";
import { setApiKey } from "@/lib/auth";

vi.mock("next/link", => ({
  __esModule: true,
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const batch = {
  batch_id: "batch_abc123def456",
  status: "processing",
  created_at: "2026-05-14T01:00:00Z",
  completed_at: null,
  processing_time: null,
  total_jobs: 2,
  progress: {
    submitted: 0,
    processing: 1,
    completed: 1,
    failed: 0,
    cancelled: 0,
    percent_complete: 50,
  },
  jobs: [
    { job_id: "job_abc123def456", source_file: "one.pdf", status: "completed" },
    { job_id: "job_def456abc123", source_file: "two.pdf", status: "processing" },
  ],
  settings: {},
  webhook_status: null,
};

describe("BatchesTable", => {
  beforeEach(() => {
    setApiKey("test-key");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders batches returned by the API", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ batches: [batch], total: 1, limit: 25, offset: 0 })
    );

    render(<BatchesTable />);

    await waitFor(() => expect(screen.queryByTestId("batches-loading")).toBeNull());
    expect(screen.getByText("batch_abc123def456")).toHaveAttribute(
      "href",
      "/batches/batch_abc123def456"
    );
    expect(screen.getByText("50%")).toBeInTheDocument();
  });

  it("shows an empty state", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ batches: [], total: 0, limit: 25, offset: 0 })
    );

    render(<BatchesTable />);

    await waitFor(() => expect(screen.getByTestId("batches-empty")).toBeInTheDocument());
  });

  it("submits selected files and prepends the returned batch", async => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ batches: [], total: 0, limit: 25, offset: 0 }))
      .mockResolvedValueOnce(
        jsonResponse({
          batch_id: "batch_new123456789",
          status: "submitted",
          created_at: "2026-05-14T01:01:00Z",
          total_jobs: 1,
          priority: "urgent",
          jobs: [{ job_id: "job_new123456789", source_file: "new.pdf", status: "submitted" }],
        }, 201)
      );

    render(<BatchesTable />);
    await waitFor(() => expect(screen.getByTestId("batches-empty")).toBeInTheDocument());

    const file = new File(["pdf"], "new.pdf", { type: "application/pdf" });
    fireEvent.change(screen.getByTestId("batch-file-input"), {
      target: { files: [file] },
    });
    fireEvent.change(screen.getByTestId("batch-priority"), {
      target: { value: "urgent" },
    });
    fireEvent.click(screen.getByTestId("batch-submit"));

    await waitFor(() => expect(screen.getByText("batch_new123456789")).toBeInTheDocument());
    const submitCall = fetchMock.mock.calls[1];
    expect(String(submitCall[0])).toContain("/api/v1/jobs/batch");
    expect((submitCall[1]?.body as FormData).get("priority")).toBe("urgent");
  });
});
