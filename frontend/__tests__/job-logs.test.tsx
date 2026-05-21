import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { JobLogs } from "@/components/JobLogs";
import { setApiKey } from "@/lib/auth";

function ndjsonResponse(records: unknown[], status = 200): Response {
  const body = records.map((r) => JSON.stringify(r)).join("\n");
  return new Response(body, {
    status,
    headers: { "content-type": "application/x-ndjson" },
  });
}

function notFoundResponse(): Response {
  return new Response(
    JSON.stringify({ detail: { detail: "no job logs available", code: "NO_PER_JOB_LOGS" } }),
    { status: 404, headers: { "content-type": "application/json" } }
  );
}

describe("<JobLogs />", => {
  beforeEach(() => {
    setApiKey("test-key");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders log records returned from the NDJSON endpoint", async => {
    const records = [
      {
        ts: "2026-04-27T12:00:00+00:00",
        level: "INFO",
        code: "JOB_STARTED",
        job_id: "job_aaaaaaaa1111",
        message: "Pipeline started",
      },
      {
        ts: "2026-04-27T12:00:01+00:00",
        level: "ERROR",
        code: "JOB_FAILED",
        job_id: "job_aaaaaaaa1111",
        message: "Boom",
      },
    ];
    vi.spyOn(globalThis, "fetch").mockResolvedValue(ndjsonResponse(records));

    render(<JobLogs jobId="job_aaaaaaaa1111" pollingEnabled={false} />);

    await waitFor(() => {
      expect(screen.getByTestId("job-logs-pre")).toBeInTheDocument();
    });
    const pre = screen.getByTestId("job-logs-pre");
    expect(pre.textContent).toContain("JOB_STARTED");
    expect(pre.textContent).toContain("JOB_FAILED");
  });

  it("shows the missing state on 404", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(notFoundResponse());

    render(<JobLogs jobId="job_aaaaaaaa2222" pollingEnabled={false} />);

    await waitFor(() => {
      expect(screen.getByTestId("job-logs-missing")).toBeInTheDocument();
    });
  });

  it("auto-tails by passing the last-seen ts as since=", async => {
    const t0 = "2026-04-27T12:00:00+00:00";
    const t1 = "2026-04-27T12:00:05+00:00";
    const fetchMock = vi.spyOn(globalThis, "fetch");
    fetchMock.mockResolvedValueOnce(
      ndjsonResponse([
        { ts: t0, level: "INFO", code: "X", job_id: "job_t", message: "first" },
      ])
    );
    fetchMock.mockResolvedValueOnce(
      ndjsonResponse([
        { ts: t1, level: "INFO", code: "X", job_id: "job_t", message: "second" },
      ])
    );

    render(
      <JobLogs jobId="job_t" pollingEnabled={true} pollIntervalMs={50} />
    );

    await waitFor(() => {
      expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(2);
    });

    const secondCallUrl = fetchMock.mock.calls[1][0] as string;
    expect(secondCallUrl).toContain(`since=${encodeURIComponent(t0)}`);
  });

  it("pauses polling when pause is clicked", async => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(
        ndjsonResponse([
          {
            ts: "2026-04-27T12:00:00+00:00",
            level: "INFO",
            code: "X",
            job_id: "job_p",
            message: "ok",
          },
        ])
      );

    render(<JobLogs jobId="job_p" pollingEnabled={true} pollIntervalMs={20} />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    fireEvent.click(screen.getByTestId("job-logs-pause"));
    const callsAfterPause = fetchMock.mock.calls.length;

    await new Promise((resolve) => setTimeout(resolve, 80));
    // Polling should be stopped, so fetch count should not climb.
    expect(fetchMock.mock.calls.length).toBe(callsAfterPause);
  });

  it("clears retained log records on Clear", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      ndjsonResponse([
        {
          ts: "2026-04-27T12:00:00+00:00",
          level: "INFO",
          code: "X",
          job_id: "job_c",
          message: "to be cleared",
        },
      ])
    );

    render(<JobLogs jobId="job_c" pollingEnabled={false} />);
    await waitFor(() => {
      expect(screen.getByTestId("job-logs-pre").textContent).toContain(
        "to be cleared"
      );
    });

    fireEvent.click(screen.getByTestId("job-logs-clear"));
    expect(screen.queryByTestId("job-logs-pre")).toBeNull();
    expect(screen.getByTestId("job-logs-empty")).toBeInTheDocument();
  });
});
