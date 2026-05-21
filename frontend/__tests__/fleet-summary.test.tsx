import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FleetSummary } from "@/components/FleetSummary";
import { CapabilityBadge } from "@/components/CapabilityBadge";
import type { FleetSnapshot, QueueSnapshot, Worker } from "@/lib/types";

function worker(overrides: Partial<Worker>): Worker {
  return {
    worker_id: "w1",
    hostname: "host-1",
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

function makeFleet(workers: Worker[]): FleetSnapshot {
  let online = 0;
  let busy = 0;
  let idle = 0;
  let offline = 0;
  for (const w of workers) {
    if (w.state === "online") online += 1;
    else if (w.state === "busy") busy += 1;
    else if (w.state === "idle") idle += 1;
    else if (w.state === "offline") offline += 1;
  }
  return {
    timestamp: Date.now() / 1000,
    summary: { total_workers: workers.length, online, busy, idle, offline, error: 0, draining: 0 },
    gpu: {
      total_gpus: 0,
      avg_utilization_pct: 0,
      avg_memory_pct: 0,
      total_memory_mb: 0,
      used_memory_mb: 0,
    },
    workers,
  };
}

const emptyQueues: QueueSnapshot = {
  timestamp: 0,
  total_depth: 0,
  queues: [],
  active_alerts: [],
};

describe("<FleetSummary />", => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-26T12:00:00Z"));
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders online total + capability breakdown", => {
    const nowSec = Math.floor(Date.now() / 1000);
    const workers = [
      worker({ worker_id: "g1", state: "online", capabilities: ["ocr_gpu"], last_heartbeat: nowSec }),
      worker({ worker_id: "g2", state: "busy", capabilities: ["ocr_gpu"], last_heartbeat: nowSec }),
      worker({ worker_id: "c1", state: "idle", capabilities: ["ocr_cpu"], last_heartbeat: nowSec }),
      worker({ worker_id: "n1", state: "online", capabilities: ["nlp"], last_heartbeat: nowSec }),
      worker({ worker_id: "off", state: "offline", capabilities: ["ocr_gpu"], last_heartbeat: 0 }),
    ];
    render(<FleetSummary fleet={makeFleet(workers)} queues={emptyQueues} />);

    const onlineCard = screen.getByTestId("card-online");
    expect(onlineCard).toHaveTextContent("4");
    expect(onlineCard).toHaveTextContent(/2 ocr_gpu/);
    expect(onlineCard).toHaveTextContent(/1 ocr_cpu/);
    expect(onlineCard).toHaveTextContent(/1 nlp/);
  });

  it("counts stale workers (heartbeat older than threshold)", => {
    const nowSec = Math.floor(Date.now() / 1000);
    const workers = [
      worker({ worker_id: "fresh", state: "online", last_heartbeat: nowSec }),
      worker({ worker_id: "stale1", state: "online", last_heartbeat: nowSec - 300 }),
      worker({ worker_id: "stale2", state: "busy", last_heartbeat: nowSec - 120 }),
      worker({ worker_id: "off", state: "offline", last_heartbeat: nowSec - 9999 }),
    ];
    render(
      <FleetSummary
        fleet={makeFleet(workers)}
        queues={emptyQueues}
        staleThresholdSeconds={60}
      />
    );
    expect(screen.getByTestId("stale-count")).toHaveTextContent("2");
  });

  it("handles snapshots without a workers collection", => {
    const fleet = makeFleet([]);
    delete fleet.workers;

    render(<FleetSummary fleet={fleet} queues={emptyQueues} />);

    expect(screen.getByTestId("stale-count")).toHaveTextContent("0");
  });

  it("toggles stale filter via callback", async => {
    vi.useRealTimers();
    const user = userEvent.setup();
    const onToggle = vi.fn();
    render(
      <FleetSummary
        fleet={makeFleet([])}
        queues={emptyQueues}
        staleFilterActive={false}
        onToggleStaleFilter={onToggle}
      />
    );
    const button = screen.getByRole("button", { pressed: false });
    await user.click(button);
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  it("shows em-dash ETA when throughput is missing", => {
    render(
      <FleetSummary
        fleet={makeFleet([])}
        queues={{ ...emptyQueues, total_depth: 500 }}
        throughputPerMinute={null}
      />
    );
    expect(screen.getByTestId("card-eta")).toHaveTextContent("—");
  });

  it("computes ETA in minutes when throughput is provided", => {
    render(
      <FleetSummary
        fleet={makeFleet([])}
        queues={{ ...emptyQueues, total_depth: 600 }}
        throughputPerMinute={60}
      />
    );
    // 600 / 60 = 10 minutes
    expect(screen.getByTestId("card-eta")).toHaveTextContent("10m");
  });
});

describe("<CapabilityBadge />", => {
  it("uses a distinct palette per known capability", => {
    const { rerender } = render(<CapabilityBadge capability="ocr_gpu" />);
    const gpuPalette = screen.getByTestId("capability-badge-ocr_gpu").getAttribute("data-palette");

    rerender(<CapabilityBadge capability="ocr_cpu" />);
    const cpuPalette = screen.getByTestId("capability-badge-ocr_cpu").getAttribute("data-palette");

    rerender(<CapabilityBadge capability="nlp" />);
    const nlpPalette = screen.getByTestId("capability-badge-nlp").getAttribute("data-palette");

    expect(gpuPalette).not.toBe(cpuPalette);
    expect(cpuPalette).not.toBe(nlpPalette);
    expect(gpuPalette).not.toBe(nlpPalette);
  });

  it("falls back to a neutral palette for unknown capabilities", => {
    render(<CapabilityBadge capability="exotic_thing" />);
    const palette = screen
      .getByTestId("capability-badge-exotic_thing")
      .getAttribute("data-palette");
    expect(palette).toMatch(/gray/);
  });
});
