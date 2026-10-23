import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BatchesTable } from "@/components/BatchesTable";
import { getApiKey, setApiKey } from "@/lib/auth";

vi.mock("next/link", () => ({
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

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
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

describe("BatchesTable", () => {
  beforeEach(() => {
    setApiKey("test-key");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders batches returned by the API", async () => {
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


  it("shows an empty state", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ batches: [], total: 0, limit: 25, offset: 0 })
    );

    render(<BatchesTable />);

    await waitFor(() => expect(screen.getByTestId("batches-empty")).toBeInTheDocument());
  });

  it("submits selected files and prepends the returned batch", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ batches: [], total: 0, limit: 25, offset: 0 }))
      .mockResolvedValueOnce(
        jsonResponse(
          {
            batch_id: "batch_new123456789",
            status: "submitted",
            created_at: "2026-05-14T01:01:00Z",
            total_jobs: 1,
            priority: "urgent",
            jobs: [{ job_id: "job_new123456789", source_file: "new.pdf", status: "submitted" }],
          },
          201
        )
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

  it("clears a rejected key on 401 but keeps a valid key on 403", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ batches: [], total: 0, limit: 25, offset: 0 }))
      .mockResolvedValueOnce(jsonResponse({ detail: "bad key" }, 401));

    const { unmount } = render(<BatchesTable />);
    const file = new File(["pdf"], "unauthorized.pdf", { type: "application/pdf" });
    fireEvent.change(screen.getByTestId("batch-file-input"), {
      target: { files: [file] },
    });
    fireEvent.click(screen.getByTestId("batch-submit"));
    await waitFor(() => expect(getApiKey()).toBeNull());
    unmount();

    setApiKey("test-key");
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ batches: [], total: 0, limit: 25, offset: 0 }))
      .mockResolvedValueOnce(jsonResponse({ detail: "submit capability required" }, 403));
    render(<BatchesTable />);
    await waitFor(() => expect(screen.queryByTestId("batches-loading")).toBeNull());
    const forbiddenFile = new File(["pdf"], "forbidden.pdf", { type: "application/pdf" });
    fireEvent.change(screen.getByTestId("batch-file-input"), {
      target: { files: [forbiddenFile] },
    });
    fireEvent.click(screen.getByTestId("batch-submit"));
    await waitFor(() => expect(screen.getByTestId("batch-submit-error")).toBeInTheDocument());
    expect(getApiKey()).toBe("test-key");
  });

  it("returns to page one on submit and ignores an old page response", async () => {
    const oldPage = deferred<Response>();
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ batches: [batch], total: 30, limit: 25, offset: 0 }))
      .mockImplementationOnce(() => oldPage.promise)
      .mockResolvedValueOnce(jsonResponse({
        batch_id: "batch_new123456789", status: "submitted", created_at: "2026-05-14T01:01:00Z",
        total_jobs: 1, jobs: [{ job_id: "job_new123456789", source_file: "new.pdf", status: "submitted" }],
      }, 201))
      .mockResolvedValue(jsonResponse({ batches: [batch], total: 30, limit: 25, offset: 0 }));
    render(<BatchesTable />);
    await waitFor(() => expect(screen.getByText("batch_abc123def456")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("pagination-next"));
    await waitFor(() => expect(String(fetchMock.mock.calls[1][0])).toContain("offset=25"));
    fireEvent.change(screen.getByTestId("batch-file-input"), {
      target: { files: [new File(["pdf"], "new.pdf", { type: "application/pdf" })] },
    });
    fireEvent.click(screen.getByTestId("batch-submit"));
    await waitFor(() => expect(screen.getByText("batch_new123456789")).toBeInTheDocument());
    await waitFor(() => expect(screen.getByTestId("pagination-current")).toHaveTextContent("Page 1"));
    oldPage.resolve(jsonResponse({ batches: [{ ...batch, batch_id: "batch_oldpage123" }], total: 30 }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(screen.queryByText("batch_oldpage123")).toBeNull();
  });
});
