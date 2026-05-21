import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { JobDetailHeader } from "@/components/JobDetailHeader";
import { JobProgress } from "@/components/JobProgress";
import { Tabs } from "@/components/Tabs";
import type { Job, JobWSMessage } from "@/lib/types";

vi.mock("next/link", => ({
  __esModule: true,
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
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
    source_file: partial.source_file ?? "doc.pdf",
    progress: partial.progress ?? {
      total_pages: 20,
      pages_completed: 5,
      percent_complete: 25,
      current_stage: "ocr",
    },
    settings: partial.settings ?? { engine: "paddle" },
    webhook_status: partial.webhook_status ?? null,
  };
}

describe("JobDetailHeader", => {
  it("shows the job_id, status, and source filename", => {
    render(<JobDetailHeader job={jobFixture()} />);
    expect(screen.getByTestId("job-detail-id")).toHaveTextContent("job_abcdef012345");
    expect(screen.getByTestId("status-badge")).toHaveTextContent("processing");
    expect(screen.getByText("doc.pdf")).toBeInTheDocument();
  });
});

describe("JobProgress", => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders REST progress when no WS message has arrived yet", => {
    render(
      <JobProgress
        job={jobFixture()}
        wsStatus="connecting"
        lastMessage={null}
        onReconnect={() => {}}
      />
    );
    expect(screen.getByTestId("progress-pages")).toHaveTextContent("5 / 20");
    expect(screen.getByTestId("progress-percent")).toHaveTextContent("25.0%");
  });

  it("updates pages and percent when a WS progress frame is supplied", => {
    const msg: JobWSMessage = {
      type: "progress",
      job_id: "job_abcdef012345",
      status: "processing",
      pages_completed: 12,
      total_pages: 20,
      percent: 60,
    };
    render(
      <JobProgress
        job={jobFixture()}
        wsStatus="open"
        lastMessage={msg}
        onReconnect={() => {}}
      />
    );
    expect(screen.getByTestId("progress-pages")).toHaveTextContent("12 / 20");
    expect(screen.getByTestId("progress-percent")).toHaveTextContent("60.0%");
  });

  it("shows the disconnect banner and a reconnect button when the socket is closed", => {
    const onReconnect = vi.fn();
    render(
      <JobProgress
        job={jobFixture()}
        wsStatus="closed"
        lastMessage={null}
        onReconnect={onReconnect}
      />
    );
    expect(screen.getByTestId("ws-disconnect-banner")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("ws-reconnect"));
    expect(onReconnect).toHaveBeenCalledOnce();
  });

  it("hides the reconnect button on a terminal job state", => {
    render(
      <JobProgress
        job={jobFixture({ status: "completed" })}
        wsStatus="closed"
        lastMessage={null}
        onReconnect={() => {}}
      />
    );
    expect(screen.queryByTestId("ws-reconnect")).toBeNull();
  });
});

describe("Tabs", => {
  it("switches the visible panel when the tab is clicked", async => {
    render(
      <Tabs
        tabs={[
          { id: "a", label: "Alpha", content: <span data-testid="content-a">Alpha content</span> },
          { id: "b", label: "Beta", content: <span data-testid="content-b">Beta content</span> },
        ]}
        defaultTab="a"
      />
    );
    expect(screen.getByTestId("content-a")).toBeInTheDocument();
    expect(screen.queryByTestId("content-b")).toBeNull();
    fireEvent.click(screen.getByTestId("tab-b"));
    await waitFor(() => {
      expect(screen.queryByTestId("content-a")).toBeNull();
      expect(screen.getByTestId("content-b")).toBeInTheDocument();
    });
  });
});
