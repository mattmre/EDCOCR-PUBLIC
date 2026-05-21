import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, fireEvent, waitFor } from "@testing-library/react";
import { AlertsList, SeverityBadge } from "@/components/AlertsList";
import AlertsPage from "@/app/admin/alerts/page";
import { setApiKey } from "@/lib/auth";
import type { Alert } from "@/lib/types";

vi.mock("next/link", => ({
  __esModule: true,
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

vi.mock("next/navigation", async => ({
  useParams: => ({}),
  useRouter: => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: => new URLSearchParams(),
  usePathname: => "/admin/alerts",
}));

function mkAlert(overrides: Partial<Alert> = {}): Alert {
  return {
    id: "a1",
    rule_id: "queue_depth_critical",
    severity: "critical",
    state: "firing",
    tenant_id: null,
    started_at: "2026-04-27T00:00:00Z",
    last_seen: "2026-04-27T00:00:30Z",
    message: "ocr_gpu queue exceeds 100",
    labels: { queue: "ocr_gpu" },
    ...overrides,
  };
}

describe("<AlertsList />", => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders an empty-state when no alerts are present", => {
    render(<AlertsList alerts={[]} />);
    expect(screen.getByTestId("alerts-empty")).toBeInTheDocument();
  });

  it("renders rows for each alert with severity badges", => {
    const alerts: Alert[] = [
      mkAlert({ id: "a1", severity: "critical" }),
      mkAlert({ id: "a2", severity: "warning", rule_id: "sla_warning" }),
      mkAlert({ id: "a3", severity: "info", rule_id: "info_rule" }),
    ];
    render(<AlertsList alerts={alerts} />);
    expect(screen.getByTestId("alert-row-a1")).toBeInTheDocument();
    expect(screen.getByTestId("alert-row-a2")).toBeInTheDocument();
    expect(screen.getByTestId("alert-row-a3")).toBeInTheDocument();
  });

  it("uses red tone for critical, amber for warning, blue for info", => {
    render(
      <div>
        <SeverityBadge severity="critical" />
        <SeverityBadge severity="warning" />
        <SeverityBadge severity="info" />
      </div>
    );
    const critical = screen.getByTestId("severity-badge-critical");
    const warning = screen.getByTestId("severity-badge-warning");
    const info = screen.getByTestId("severity-badge-info");
    expect(critical.className).toMatch(/red/);
    expect(warning.className).toMatch(/amber/);
    expect(info.className).toMatch(/blue/);
  });

  it("shows a confirm dialog when the operator clicks Mute", => {
    render(<AlertsList alerts={[mkAlert()]} onMute={vi.fn()} />);
    fireEvent.click(screen.getByTestId("alert-mute-a1"));
    expect(screen.getByTestId("mute-dialog")).toBeInTheDocument();
    expect(screen.getByTestId("mute-reason")).toBeInTheDocument();
    // Confirm is disabled until a reason is supplied.
    expect(screen.getByTestId("mute-confirm")).toBeDisabled();
  });

  it("posts the reason and closes the dialog when mute succeeds", async => {
    const onMute = vi.fn().mockResolvedValue(undefined);
    render(<AlertsList alerts={[mkAlert()]} onMute={onMute} />);

    fireEvent.click(screen.getByTestId("alert-mute-a1"));
    fireEvent.change(screen.getByTestId("mute-reason"), {
      target: { value: "investigating with on-call" },
    });
    fireEvent.click(screen.getByTestId("mute-confirm"));

    await waitFor(() => {
      expect(onMute).toHaveBeenCalledWith("a1", {
        reason: "investigating with on-call",
      });
    });
    await waitFor(() => {
      expect(screen.queryByTestId("mute-dialog")).not.toBeInTheDocument();
    });
  });

  it("renders an unmute button for already-muted alerts", => {
    render(
      <AlertsList
        alerts={[mkAlert({ state: "muted" })]}
        onUnmute={vi.fn()}
      />
    );
    expect(screen.getByTestId("alert-unmute-a1")).toBeInTheDocument();
    expect(screen.queryByTestId("alert-mute-a1")).not.toBeInTheDocument();
  });

  it("calls onUnmute when the operator clicks Unmute", async => {
    const onUnmute = vi.fn().mockResolvedValue(undefined);
    render(
      <AlertsList alerts={[mkAlert({ state: "muted" })]} onUnmute={onUnmute} />
    );
    fireEvent.click(screen.getByTestId("alert-unmute-a1"));
    await waitFor(() => {
      expect(onUnmute).toHaveBeenCalledWith("a1");
    });
  });

  it("renders the alert state pill with the alert state", => {
    render(<AlertsList alerts={[mkAlert({ state: "pending" })]} />);
    expect(screen.getByTestId("alert-state-a1").textContent).toBe("pending");
  });

  it("computes a deterministic relative age when nowMs is supplied", => {
    const alerts = [
      mkAlert({ id: "fresh", started_at: "2026-04-27T00:04:30Z" }),
      mkAlert({ id: "old", started_at: "2026-04-26T20:00:00Z" }),
    ];
    const nowMs = new Date("2026-04-27T00:05:00Z").getTime();
    render(<AlertsList alerts={alerts} nowMs={nowMs} />);
    // 30 s ago -> "30s ago"
    expect(screen.getByTestId("alert-row-fresh").textContent).toMatch(/30s ago/);
    // ~5h ago
    expect(screen.getByTestId("alert-row-old").textContent).toMatch(/h ago/);
  });
});

describe("<AlertsPage />", => {
  beforeEach(() => {
    setApiKey("admin-key");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  function jsonResponse(status: number, body: unknown): Response {
    return new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json" },
    });
  }

  it("renders the access-denied empty-state on 403", async => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.includes("/api/v1/admin/alerts")) {
        return Promise.resolve(jsonResponse(403, { detail: "platform admin required" }));
      }
      if (url.includes("/api/v1/admin/alert-channels")) {
        return Promise.resolve(jsonResponse(403, {}));
      }
      return Promise.resolve(jsonResponse(404, {}));
    });

    render(<AlertsPage />);

    await waitFor(() => {
      expect(screen.getByTestId("alerts-forbidden")).toBeInTheDocument();
    });
  });

  it("renders the not-provisioned empty-state on 404", async => {
    vi.spyOn(globalThis, "fetch").mockImplementation(() =>
      Promise.resolve(jsonResponse(404, { detail: "not provisioned" }))
    );

    render(<AlertsPage />);

    await waitFor(() => {
      expect(screen.getByTestId("alerts-not-provisioned")).toBeInTheDocument();
    });
    expect(screen.getByTestId("alerts-grafana-link")).toHaveAttribute("href", "/grafana");
  });

  it("auto-refresh tick triggers a second listAlerts call", async => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation((input) => {
        const url = String(input);
        if (url.includes("/api/v1/admin/alerts/rules")) {
          return Promise.resolve(jsonResponse(200, []));
        }
        if (url.includes("/api/v1/admin/alert-channels")) {
          return Promise.resolve(jsonResponse(200, []));
        }
        if (url.includes("/api/v1/admin/alerts")) {
          return Promise.resolve(jsonResponse(200, []));
        }
        return Promise.resolve(jsonResponse(404, {}));
      });

    vi.useFakeTimers({ shouldAdvanceTime: true });

    render(<AlertsPage />);

    // Initial fetch.
    await waitFor(() => {
      const alertCalls = fetchMock.mock.calls.filter((c) =>
        String(c[0]).match(/\/api\/v1\/admin\/alerts(\?|$)/)
      );
      expect(alertCalls.length).toBeGreaterThanOrEqual(1);
    });

    const before = fetchMock.mock.calls.filter((c) =>
      String(c[0]).match(/\/api\/v1\/admin\/alerts(\?|$)/)
    ).length;

    // Advance past the 15 s polling interval.
    await act(async => {
      vi.advanceTimersByTime(16000);
      await Promise.resolve();
    });

    await waitFor(() => {
      const after = fetchMock.mock.calls.filter((c) =>
        String(c[0]).match(/\/api\/v1\/admin\/alerts(\?|$)/)
      ).length;
      expect(after).toBeGreaterThan(before);
    });

    vi.useRealTimers();
  });
});
