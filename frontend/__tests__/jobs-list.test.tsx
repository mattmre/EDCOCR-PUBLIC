import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { JobsTable } from "@/components/JobsTable";
import { setApiKey } from "@/lib/auth";
import type { Job, JobListResponse } from "@/lib/types";

vi.mock("next/link", => ({
  __esModule: true,
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    // eslint-disable-next-line jsx-a11y/anchor-has-content
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

function jobFixture(partial: Partial<Job> = {}): Job {
  return {
    job_id: partial.job_id ?? "job_abcdef012345",
    status: partial.status ?? "processing",
    created_at: partial.created_at ?? "2026-04-25T10:00:00Z",
    started_at: partial.started_at ?? "2026-04-25T10:00:01Z",
    completed_at: partial.completed_at ?? null,
    priority: partial.priority ?? "normal",
    source_file: partial.source_file ?? "report.pdf",
    progress: partial.progress ?? {
      total_pages: 10,
      pages_completed: 3,
      percent_complete: 30,
      current_stage: "ocr",
    },
    settings: partial.settings ?? {},
    webhook_status: partial.webhook_status ?? null,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("<JobsTable />", => {
  beforeEach(() => {
    setApiKey("test-key");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders jobs returned from the API", async => {
    const payload: JobListResponse = {
      jobs: [jobFixture(), jobFixture({ job_id: "job_111122223333", source_file: "two.pdf" })],
      total: 2,
      limit: 25,
      offset: 0,
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(payload));

    render(<JobsTable refresh={false} />);

    await waitFor(() => {
      expect(screen.queryByTestId("jobs-loading")).toBeNull();
    });

    expect(screen.getByTestId("job-link-job_abcdef012345")).toHaveAttribute(
      "href",
      "/jobs/job_abcdef012345"
    );
    expect(screen.getByTestId("job-link-job_111122223333")).toBeInTheDocument();
    expect(screen.getByTestId("pagination-summary")).toHaveTextContent("Showing 1-2 of 2");
  });

  it("shows the empty state when the API returns no jobs", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ jobs: [], total: 0, limit: 25, offset: 0 } satisfies JobListResponse)
    );
    render(<JobsTable refresh={false} />);
    await waitFor(() => {
      expect(screen.getByTestId("jobs-empty")).toBeInTheDocument();
    });
  });

  it("forwards the status filter as a query parameter", async => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(
        jsonResponse({ jobs: [], total: 0, limit: 25, offset: 0 } satisfies JobListResponse)
      );
    render(<JobsTable refresh={false} />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    fireEvent.click(screen.getByTestId("status-chip-failed"));

    await waitFor(() => {
      const lastCallUrl = fetchMock.mock.calls.at(-1)?.[0] as string;
      expect(lastCallUrl).toMatch(/status=failed/);
    });
  });

  it("forwards the free-text search as a server-side q param", async => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(
        jsonResponse({ jobs: [], total: 0, limit: 25, offset: 0 } satisfies JobListResponse)
      );
    render(<JobsTable refresh={false} />);
    await waitFor(() => expect(screen.queryByTestId("jobs-loading")).toBeNull());

    fireEvent.change(screen.getByTestId("jobs-filter-search"), {
      target: { value: "beta" },
    });

    // The bar debounces 300 ms by default; allow waitFor to retry.
    await waitFor( => {
        const lastCallUrl = fetchMock.mock.calls.at(-1)?.[0] as string;
        expect(lastCallUrl).toMatch(/q=beta/);
      },
      { timeout: 2000 }
    );
  });

  it("paginates by sending offset on next-page click", async => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        jobs: Array.from({ length: 25 }, (_, i) =>
          jobFixture({ job_id: `job_${i.toString(16).padStart(12, "0")}` })
        ),
        total: 60,
        limit: 25,
        offset: 0,
      } satisfies JobListResponse)
    );
    render(<JobsTable refresh={false} />);
    await waitFor(() => expect(screen.queryByTestId("jobs-loading")).toBeNull());

    fireEvent.click(screen.getByTestId("pagination-next"));

    await waitFor(() => {
      const lastCallUrl = fetchMock.mock.calls.at(-1)?.[0] as string;
      expect(lastCallUrl).toMatch(/offset=25/);
    });
  });
});
