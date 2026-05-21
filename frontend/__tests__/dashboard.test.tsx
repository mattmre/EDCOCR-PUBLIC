import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DashboardPage from "@/app/dashboard/page";
import { setApiKey } from "@/lib/auth";

const pushMock = vi.fn();

vi.mock("next/navigation", => ({
  useRouter: => ({ push: pushMock }),
  usePathname: => "/dashboard",
}));

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const HEALTH_OK = {
  status: "healthy",
  version: "4.1.0",
  uptime_seconds: 12345,
  jobs: { submitted: 2, processing: 1, completed: 9, failed: 0 },
  checks: {
    external_translation: {
      status: "healthy",
      message: "external translation preference is disabled",
    },
  },
};

const SNAPSHOT_OK = {
  timestamp: 1700000000,
  throughput: { pages_per_minute: 24.5, docs_per_hour: 6, bytes_per_second: 0 },
  latency: { avg_ms: 0, p50_ms: 0, p95_ms: 0, p99_ms: 0 },
  jobs: { total: 12, active: 1, completed: 9, failed: 0, queued: 2 },
};

const FLEET_OK = {
  timestamp: 1700000000,
  summary: {
    total_workers: 4,
    online: 3,
    busy: 1,
    idle: 2,
    offline: 1,
    error: 0,
    draining: 0,
  },
  gpu: {
    total_gpus: 4,
    avg_utilization_pct: 45,
    avg_memory_pct: 60,
    total_memory_mb: 0,
    used_memory_mb: 0,
  },
  workers: [],
};

const JOBS_OK = {
  jobs: [
    {
      job_id: "job_aaaaaaaaaaaa",
      status: "completed",
      created_at: "2026-04-26T10:00:00Z",
      started_at: "2026-04-26T10:00:01Z",
      completed_at: "2026-04-26T10:00:31Z",
      priority: "normal",
      source_file: "/in/sample.pdf",
      progress: { total_pages: 5, pages_completed: 5, percent_complete: 100, current_stage: "done" },
      settings: {},
      webhook_status: null,
    },
    {
      job_id: "job_bbbbbbbbbbbb",
      status: "processing",
      created_at: "2026-04-26T10:01:00Z",
      started_at: "2026-04-26T10:01:01Z",
      completed_at: null,
      priority: "normal",
      source_file: "/in/another.pdf",
      progress: null,
      settings: {},
      webhook_status: null,
    },
  ],
  total: 2,
  limit: 10,
  offset: 0,
};

function routeFetch(url: string): Response {
  if (url.includes("/api/v1/health/detailed")) return jsonResponse(200, HEALTH_OK);
  if (url.includes("/api/v1/dashboard")) return jsonResponse(200, SNAPSHOT_OK);
  if (url.includes("/api/v1/fleet")) return jsonResponse(200, FLEET_OK);
  if (url.includes("/api/v1/jobs")) return jsonResponse(200, JOBS_OK);
  return jsonResponse(404, { detail: "not found" });
}

describe("DashboardPage", => {
  beforeEach(() => {
    setApiKey("test-key");
    pushMock.mockClear();
    vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders metrics from the API", async => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockImplementation(
      async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        return routeFetch(url);
      }
    );

    render(<DashboardPage />);

    // Pipeline status card description appears once the health probe lands.
    await waitFor(() => {
      expect(screen.getByText(/version 4.1.0/i)).toBeInTheDocument();
    });

    // Throughput cell.
    expect(screen.getByText(/24\.5 pages\/min/i)).toBeInTheDocument();

    // Fleet "3 / 4 online".
    expect(screen.getByText(/3 \/ 4 online/i)).toBeInTheDocument();

    expect(screen.getByText("EXTERNAL_TRANSLATION")).toBeInTheDocument();
    expect(screen.getByText(/external translation preference is disabled/i)).toBeInTheDocument();

    // Recent jobs table renders rows.
    expect(
      screen.getByTestId("recent-jobs-row-job_aaaaaaaaaaaa")
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("recent-jobs-row-job_bbbbbbbbbbbb")
    ).toBeInTheDocument();
  });

  it("falls back when /dashboard and /fleet return 404", async => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockImplementation(
      async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/v1/dashboard")) return jsonResponse(404, { detail: "disabled" });
        if (url.includes("/api/v1/fleet")) return jsonResponse(404, { detail: "disabled" });
        return routeFetch(url);
      }
    );

    render(<DashboardPage />);

    // When dashboard is disabled, processing rate falls back to em-dash and
    // description mentions the disabled endpoint.
    await waitFor(() => {
      expect(screen.getByText(/Dashboard endpoint disabled/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/Fleet endpoint disabled/i)).toBeInTheDocument();

    // Queue counts fall back to /health/detailed jobs map (submitted=2).
    expect(screen.getByText(/2 queued/i)).toBeInTheDocument();
  });

  it("shows an error card when /health/detailed fails", async => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockImplementation(
      async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/v1/health/detailed")) {
          return jsonResponse(500, { detail: "boom" });
        }
        if (url.includes("/api/v1/jobs")) return jsonResponse(200, JOBS_OK);
        return jsonResponse(404, { detail: "not found" });
      }
    );

    render(<DashboardPage />);

    const errorCard = await screen.findByTestId("dashboard-error");
    expect(within(errorCard).getByText(/Failed to load dashboard/i)).toBeInTheDocument();
    expect(within(errorCard).getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("navigates to /jobs?selected=... on row click", async => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockImplementation(
      async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        return routeFetch(url);
      }
    );

    const user = userEvent.setup();
    render(<DashboardPage />);

    const row = await screen.findByTestId("recent-jobs-row-job_aaaaaaaaaaaa");
    await user.click(row);

    expect(pushMock).toHaveBeenCalledWith(
      "/jobs?selected=job_aaaaaaaaaaaa"
    );
  });
});
