import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { WorkersTable } from "@/components/WorkersTable";
import type { Worker } from "@/lib/types";

vi.mock("next/link", => ({
  default: ({
    href,
    children,
    ...rest
  }: { href: string; children: React.ReactNode } & Record<string, unknown>) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

function worker(overrides: Partial<Worker>): Worker {
  return {
    worker_id: "w-default",
    hostname: "host-default",
    state: "online",
    capabilities: ["ocr_gpu"],
    gpus: [],
    current_job_id: "",
    jobs_completed: 0,
    jobs_failed: 0,
    uptime_seconds: 0,
    last_heartbeat: 0,
    is_healthy: true,
    queue_name: "ocr_gpu",
    ...overrides,
  };
}

describe("<WorkersTable />", => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-26T12:00:00Z"));
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("filters by capability chip", async => {
    const nowSec = Math.floor(Date.now() / 1000);
    const workers = [
      worker({ worker_id: "g1", hostname: "gpu-1", capabilities: ["ocr_gpu"], last_heartbeat: nowSec }),
      worker({ worker_id: "c1", hostname: "cpu-1", capabilities: ["ocr_cpu"], last_heartbeat: nowSec }),
      worker({ worker_id: "n1", hostname: "nlp-1", capabilities: ["nlp"], last_heartbeat: nowSec }),
    ];
    vi.useRealTimers();
    const user = userEvent.setup();
    render(<WorkersTable workers={workers} staleThresholdSeconds={60} />);

    expect(screen.getByTestId("worker-row-g1")).toBeInTheDocument();
    expect(screen.getByTestId("worker-row-c1")).toBeInTheDocument();
    expect(screen.getByTestId("worker-row-n1")).toBeInTheDocument();

    await user.click(screen.getByTestId("cap-chip-ocr_cpu"));

    expect(screen.queryByTestId("worker-row-g1")).not.toBeInTheDocument();
    expect(screen.getByTestId("worker-row-c1")).toBeInTheDocument();
    expect(screen.queryByTestId("worker-row-n1")).not.toBeInTheDocument();
  });

  it("sorts by heartbeat descending by default", => {
    const nowSec = Math.floor(Date.now() / 1000);
    const workers = [
      worker({ worker_id: "old", hostname: "old-host", last_heartbeat: nowSec - 30 }),
      worker({ worker_id: "new", hostname: "new-host", last_heartbeat: nowSec - 1 }),
      worker({ worker_id: "mid", hostname: "mid-host", last_heartbeat: nowSec - 10 }),
    ];
    render(<WorkersTable workers={workers} staleThresholdSeconds={60} />);

    const table = screen.getByTestId("workers-table");
    const rows = within(table).getAllByRole("row").slice(1); // drop header
    expect(rows[0]).toHaveAttribute("data-testid", "worker-row-new");
    expect(rows[1]).toHaveAttribute("data-testid", "worker-row-mid");
    expect(rows[2]).toHaveAttribute("data-testid", "worker-row-old");
  });

  it("renders a job link to /jobs/{id} when worker has a current_job_id", => {
    const nowSec = Math.floor(Date.now() / 1000);
    render(
      <WorkersTable
        workers={[
          worker({
            worker_id: "busy",
            hostname: "busy-host",
            state: "busy",
            current_job_id: "job-42",
            last_heartbeat: nowSec,
          }),
        ]}
        staleThresholdSeconds={60}
      />
    );
    const link = screen.getByTestId("worker-job-link-busy");
    expect(link.tagName).toBe("A");
    expect(link).toHaveAttribute("href", "/jobs/job-42");
    expect(link).toHaveTextContent("job-42");
  });

  it("flags stale workers via the synthetic 'stale' status", => {
    const nowSec = Math.floor(Date.now() / 1000);
    render(
      <WorkersTable
        workers={[
          worker({ worker_id: "stale1", state: "online", last_heartbeat: nowSec - 9999 }),
          worker({ worker_id: "fresh", state: "online", last_heartbeat: nowSec }),
        ]}
        staleThresholdSeconds={60}
      />
    );
    expect(screen.getByTestId("worker-status-stale1")).toHaveTextContent(/stale/i);
    expect(screen.getByTestId("worker-status-fresh")).toHaveTextContent(/online/i);
  });

  it("respects staleOnly to hide non-stale workers", => {
    const nowSec = Math.floor(Date.now() / 1000);
    render(
      <WorkersTable
        workers={[
          worker({ worker_id: "stale1", state: "online", last_heartbeat: nowSec - 9999 }),
          worker({ worker_id: "fresh", state: "online", last_heartbeat: nowSec }),
        ]}
        staleThresholdSeconds={60}
        staleOnly
      />
    );
    expect(screen.getByTestId("worker-row-stale1")).toBeInTheDocument();
    expect(screen.queryByTestId("worker-row-fresh")).not.toBeInTheDocument();
  });
});
