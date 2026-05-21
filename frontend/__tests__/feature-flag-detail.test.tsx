import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

import { FeatureFlagDetail } from "@/components/FeatureFlagDetail";
import { setApiKey } from "@/lib/auth";
import type { FeatureFlag, FeatureFlagHistoryEntry } from "@/lib/types";

vi.mock("next/link", => ({
  __esModule: true,
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

function flag(overrides: Partial<FeatureFlag>): FeatureFlag {
  return {
    key: "ENABLE_TRANSLATION",
    category: "translation",
    value_type: "boolean",
    current_value: false,
    default_value: false,
    source: "default",
    description: "Master translation toggle.",
    requires_strong_auth: true,
    requires_bake_hours: 48,
    ...overrides,
  };
}

function entry(overrides: Partial<FeatureFlagHistoryEntry>): FeatureFlagHistoryEntry {
  return {
    request_id: "req_1",
    flag_key: "ENABLE_TRANSLATION",
    previous_value: false,
    new_value: true,
    reason: "rollout to canary tenant",
    requested_by: "ops@example.com",
    requested_at: "2026-04-25T10:00:00Z",
    status: "pending",
    ...overrides,
  };
}

describe("<FeatureFlagDetail />", => {
  beforeEach(() => {
    setApiKey("test-key");
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders flag metadata", => {
    render(<FeatureFlagDetail flag={flag({})} history={[]} />);
    expect(screen.getByTestId("flag-detail-current")).toHaveTextContent("false");
    expect(screen.getByTestId("flag-detail-source")).toHaveTextContent("default");
  });

  it("shows the bake banner when last_changed_at is inside the bake window", => {
    const recent = new Date(Date.now() - 60 * 60 * 1000).toISOString();
    render(
      <FeatureFlagDetail
        flag={flag({ last_changed_at: recent })}
        history={[]}
      />
    );
    expect(screen.getByTestId("flag-detail-bake-banner")).toBeInTheDocument();
    expect(screen.getByText(/Bake window in progress/i)).toBeInTheDocument();
  });

  it("does NOT show the bake banner when the change is older than the bake window", => {
    const old = new Date(Date.now() - 100 * 60 * 60 * 1000).toISOString();
    render(
      <FeatureFlagDetail
        flag={flag({ last_changed_at: old })}
        history={[]}
      />
    );
    expect(screen.queryByTestId("flag-detail-bake-banner")).not.toBeInTheDocument();
  });

  it("renders an empty-history state", => {
    render(<FeatureFlagDetail flag={flag({})} history={[]} />);
    expect(screen.getByTestId("flag-history-empty")).toBeInTheDocument();
  });

  it("renders history rows when provided", => {
    const entries = [
      entry({ request_id: "req_a", status: "applied" }),
      entry({ request_id: "req_b", status: "rejected" }),
    ];
    render(<FeatureFlagDetail flag={flag({})} history={entries} />);
    expect(screen.getByTestId("flag-history-row-req_a")).toBeInTheDocument();
    expect(screen.getByTestId("flag-history-row-req_b")).toBeInTheDocument();
  });

  it("opens the change-request dialog when the action button is clicked", => {
    render(<FeatureFlagDetail flag={flag({})} history={[]} />);
    expect(screen.queryByTestId("flag-change-dialog")).not.toBeInTheDocument();
    fireEvent.click(screen.getByTestId("flag-detail-request-change"));
    expect(screen.getByTestId("flag-change-dialog")).toBeInTheDocument();
  });

  it("closes the dialog when cancel is clicked", => {
    render(<FeatureFlagDetail flag={flag({})} history={[]} />);
    fireEvent.click(screen.getByTestId("flag-detail-request-change"));
    expect(screen.getByTestId("flag-change-dialog")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("flag-change-cancel"));
    expect(screen.queryByTestId("flag-change-dialog")).not.toBeInTheDocument();
  });

  it("renders a history-error message when history fetch failed", => {
    render(
      <FeatureFlagDetail
        flag={flag({})}
        history={null}
        historyError={new Error("history boom")}
      />
    );
    expect(screen.getByTestId("flag-history-error")).toHaveTextContent("history boom");
  });

  it("shows 'Required' label for strong-auth flags and not otherwise", => {
    const { rerender } = render(
      <FeatureFlagDetail flag={flag({})} history={[]} />
    );
    expect(screen.getByText(/^Required$/)).toBeInTheDocument();
    rerender(
      <FeatureFlagDetail
        flag={flag({ requires_strong_auth: false })}
        history={[]}
      />
    );
    expect(screen.queryByText(/^Required$/)).not.toBeInTheDocument();
  });
});
