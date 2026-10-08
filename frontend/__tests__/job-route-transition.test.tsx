import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import JobDetailPage from "@/app/jobs/[id]/page";
import { ApiError } from "@/lib/api-client";
import type { Job, JobWSMessage } from "@/lib/types";

const route = vi.hoisted(() => ({ id: "job_aaaaaaaaaaaa" }));
const getMock = vi.hoisted(() => vi.fn());
const socket = vi.hoisted(() => ({ message: null as JobWSMessage | null }));

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: route.id }),
  useRouter: () => ({ push: vi.fn() }),
}));

vi.mock("@/lib/auth", () => ({
  useRequireAuth: () => undefined,
}));

vi.mock("@/lib/ws-client", () => ({
  useJobWebSocket: () => ({
    status: "closed",
    lastMessage: socket.message,
    reconnect: vi.fn(),
  }),
}));

vi.mock("@/lib/api-client", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-client")>("@/lib/api-client");
  return { ...actual, get: getMock };
});

function job(id: string): Job {
  return {
    job_id: id,
    status: "completed",
    priority: "normal",
    source_file: `${id}.pdf`,
    created_at: "2026-09-26T00:00:00Z",
    started_at: null,
    completed_at: "2026-09-26T00:01:00Z",
    settings: {},
    webhook_status: null,
    progress: {
      total_pages: 1,
      pages_completed: 1,
      percent_complete: 100,
      current_stage: "done",
    },
  };
}

describe("job detail route identity", () => {
  beforeEach(() => {
    route.id = "job_aaaaaaaaaaaa";
    getMock.mockReset();
    socket.message = null;
  });

  afterEach(() => vi.useRealTimers());

  it("shows the job returned for the requested id", async () => {
    getMock.mockResolvedValue(job("job_aaaaaaaaaaaa"));
    render(<JobDetailPage />);
    expect(await screen.findByTestId("job-detail-id")).toHaveTextContent("job_aaaaaaaaaaaa");
    expect(getMock).toHaveBeenCalledWith("/api/v1/jobs/job_aaaaaaaaaaaa");
  });

  it("drops the previous job while the next id is still loading", async () => {
    let release: (value: Job) => void = () => undefined;
    getMock.mockImplementation((path: string) => {
      if (path.endsWith("job_aaaaaaaaaaaa")) return Promise.resolve(job("job_aaaaaaaaaaaa"));
      return new Promise<Job>((resolve) => {
        release = resolve;
      });
    });
    const view = render(<JobDetailPage />);
    expect(await screen.findByTestId("job-detail-id")).toHaveTextContent("job_aaaaaaaaaaaa");

    route.id = "job_bbbbbbbbbbbb";
    view.rerender(<JobDetailPage />);
    expect(await screen.findByTestId("detail-loading")).toBeInTheDocument();
    expect(screen.queryByText("job_aaaaaaaaaaaa")).not.toBeInTheDocument();

    release(job("job_bbbbbbbbbbbb"));
    await waitFor(() => {
      expect(screen.getByTestId("job-detail-id")).toHaveTextContent("job_bbbbbbbbbbbb");
    });
  });

  it("shows the new id's error instead of the previous job", async () => {
    getMock.mockImplementation((path: string) => {
      if (path.endsWith("job_aaaaaaaaaaaa")) return Promise.resolve(job("job_aaaaaaaaaaaa"));
      return Promise.reject(new ApiError(404, "not found"));
    });
    const view = render(<JobDetailPage />);
    expect(await screen.findByTestId("job-detail-id")).toHaveTextContent("job_aaaaaaaaaaaa");

    route.id = "job_bbbbbbbbbbbb";
    view.rerender(<JobDetailPage />);
    expect(await screen.findByText("Failed to load job")).toBeInTheDocument();
    expect(screen.queryByText("job_aaaaaaaaaaaa")).not.toBeInTheDocument();
    expect(screen.getByText(/404 not found/)).toBeInTheDocument();
  });

  it("reports a mismatched response instead of loading forever", async () => {
    getMock.mockResolvedValue(job("job_bbbbbbbbbbbb"));
    render(<JobDetailPage />);
    expect(await screen.findByText("Failed to load job")).toBeInTheDocument();
    expect(screen.getByText(/does not match the requested job/)).toBeInTheDocument();
    expect(screen.queryByTestId("detail-loading")).not.toBeInTheDocument();
    expect(screen.queryByTestId("job-detail-id")).not.toBeInTheDocument();
  });

  it("keeps a mismatched poll visible as an error without showing the foreign job", async () => {
    vi.useFakeTimers();
    getMock.mockResolvedValueOnce(job(route.id)).mockResolvedValue(job("job_bbbbbbbbbbbb"));
    render(<JobDetailPage />);
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await vi.advanceTimersByTimeAsync(15_000); });
    expect(screen.getByRole("alert")).toHaveTextContent(/does not match/);
    expect(screen.getByTestId("job-detail-id")).toHaveTextContent(route.id);
    expect(screen.queryByText("job_bbbbbbbbbbbb")).not.toBeInTheDocument();
  });

  it("lets REST complete a job when the last socket progress is stale", async () => {
    vi.useFakeTimers();
    getMock.mockResolvedValueOnce({ ...job(route.id), status: "processing" })
      .mockResolvedValue(job(route.id));
    const view = render(<JobDetailPage />);
    await act(async () => { await Promise.resolve(); });
    socket.message = { type: "progress", job_id: route.id, status: "processing", percent: 10 };
    view.rerender(<JobDetailPage />);
    expect(screen.getByTestId("status-badge")).toHaveAttribute("data-status", "processing");
    await act(async () => { await vi.advanceTimersByTimeAsync(15_000); });
    expect(screen.getByTestId("status-badge")).toHaveAttribute("data-status", "completed");
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100");
  });

  it("rejects foreign-job socket messages", async () => {
    getMock.mockResolvedValue({ ...job(route.id), status: "processing" });
    const view = render(<JobDetailPage />);
    await screen.findByTestId("job-detail-id");
    socket.message = { type: "failed", job_id: "job_bbbbbbbbbbbb", status: "failed", error: "foreign" };
    view.rerender(<JobDetailPage />);
    expect(screen.getByTestId("status-badge")).toHaveAttribute("data-status", "processing");
  });

  it("does not roll back a terminal socket event when an older poll resolves", async () => {
    vi.useFakeTimers();
    let release: (value: Job) => void = () => undefined;
    getMock.mockResolvedValueOnce({ ...job(route.id), status: "processing" })
      .mockImplementation(() => new Promise<Job>((resolve) => { release = resolve; }));
    const view = render(<JobDetailPage />);
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await vi.advanceTimersByTimeAsync(15_000); });
    socket.message = { type: "completed", job_id: route.id, status: "completed", output_path: "output" };
    view.rerender(<JobDetailPage />);
    await act(async () => { release({ ...job(route.id), status: "processing" }); });
    expect(screen.getByTestId("status-badge")).toHaveAttribute("data-status", "completed");
  });

  it("retains a terminal event received before the initial REST response", async () => {
    let release: (value: Job) => void = () => undefined;
    getMock.mockImplementation(() => new Promise<Job>((resolve) => { release = resolve; }));
    const view = render(<JobDetailPage />);
    socket.message = { type: "completed", job_id: route.id, status: "completed", output_path: "output" };
    view.rerender(<JobDetailPage />);
    await act(async () => { release({ ...job(route.id), status: "processing" }); });
    expect(screen.getByTestId("status-badge")).toHaveAttribute("data-status", "completed");
  });

  it("serializes slow polls and resumes after the request settles", async () => {
    vi.useFakeTimers();
    let release: (value: Job) => void = () => undefined;
    getMock.mockImplementation(() => new Promise<Job>((resolve) => { release = resolve; }));
    render(<JobDetailPage />);
    await act(async () => { await vi.advanceTimersByTimeAsync(45_000); });
    expect(getMock).toHaveBeenCalledTimes(1);
    await act(async () => { release(job(route.id)); });
    await act(async () => { await vi.advanceTimersByTimeAsync(15_000); });
    expect(getMock).toHaveBeenCalledTimes(2);
  });

  it("shows a retry's new job ID even after a terminal job", async () => {
    getMock.mockResolvedValueOnce(job(route.id))
      .mockResolvedValue({ ...job("job_bbbbbbbbbbbb"), status: "queued" });
    const view = render(<JobDetailPage />);
    await screen.findByTestId("job-detail-id");
    route.id = "job_bbbbbbbbbbbb";
    view.rerender(<JobDetailPage />);
    await waitFor(() => expect(screen.getByTestId("status-badge")).toHaveAttribute("data-status", "queued"));
    expect(screen.getByTestId("job-detail-id")).toHaveTextContent(route.id);
  });
});
