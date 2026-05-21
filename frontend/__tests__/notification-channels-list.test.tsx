import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import {
  NotificationChannelsList,
  redactChannelTarget,
} from "@/components/NotificationChannelsList";
import type { NotificationChannel } from "@/lib/types";

function mkChannel(overrides: Partial<NotificationChannel>): NotificationChannel {
  return {
    id: "ch1",
    type: "slack",
    target: "#ops-alerts",
    enabled: true,
    last_test_at: null,
    last_test_ok: null,
    ...overrides,
  };
}

describe("<NotificationChannelsList />", => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders an empty-state when no channels are configured", => {
    render(<NotificationChannelsList channels={[]} />);
    expect(screen.getByTestId("channels-empty")).toBeInTheDocument();
  });

  it("renders one row per channel with redacted target text", => {
    const channels = [
      mkChannel({ id: "ch_slack", type: "slack", target: "#ops-alerts" }),
      mkChannel({
        id: "ch_email",
        type: "email",
        target: "oncall@example.com",
      }),
      mkChannel({
        id: "ch_hook",
        type: "webhook",
        target: "https://hooks.internal/xxx/secret",
      }),
    ];
    render(<NotificationChannelsList channels={channels} />);

    const slackTarget = screen.getByTestId("channel-target-ch_slack").textContent || "";
    expect(slackTarget.startsWith("#")).toBe(true);
    expect(slackTarget).toContain("*");
    // Slack channel name is partially redacted but not entirely empty.
    expect(slackTarget).not.toBe("#ops-alerts");

    const emailTarget = screen.getByTestId("channel-target-ch_email").textContent || "";
    expect(emailTarget).toContain("@example.com");
    expect(emailTarget).not.toContain("oncall@");

    const hookTarget = screen.getByTestId("channel-target-ch_hook").textContent || "";
    expect(hookTarget).toContain("hooks.internal");
    expect(hookTarget).not.toContain("secret");
  });

  it("redactChannelTarget shape per channel type", => {
    expect(redactChannelTarget("slack", "#ops-alerts")).toMatch(/^#o\*+s$/);
    expect(redactChannelTarget("email", "alice@example.com")).toMatch(
      /\*+e@example\.com/
    );
    expect(redactChannelTarget("webhook", "https://example.com/hook/secret"))
      .toBe("https://example.com/***");
    // Unparseable webhook still gets meaningful redaction.
    expect(redactChannelTarget("webhook", "not-a-url")).toMatch(/\*\*\*/);
  });

  it("shows PASS with a fresh timestamp when test-channel succeeds", async => {
    const onTest = vi.fn().mockResolvedValue({ ok: true });
    render(
      <NotificationChannelsList
        channels={[mkChannel({ id: "ch_slack" })]}
        onTest={onTest}
      />
    );

    fireEvent.click(screen.getByTestId("channel-test-ch_slack"));

    await waitFor(() => {
      expect(onTest).toHaveBeenCalledWith("ch_slack");
    });
    await waitFor(() => {
      const result = screen.getByTestId("channel-test-result-ch_slack");
      expect(result.textContent).toContain("PASS");
    });
  });

  it("shows FAIL when the test-channel callback returns ok=false", async => {
    const onTest = vi
      .fn()
      .mockResolvedValue({ ok: false, message: "401 from slack" });
    render(
      <NotificationChannelsList
        channels={[mkChannel({ id: "ch_slack" })]}
        onTest={onTest}
      />
    );

    fireEvent.click(screen.getByTestId("channel-test-ch_slack"));

    await waitFor(() => {
      const result = screen.getByTestId("channel-test-result-ch_slack");
      expect(result.textContent).toContain("FAIL");
      expect(result.textContent).toContain("401 from slack");
    });
  });

  it("renders the test button as disabled when no onTest handler is supplied", => {
    render(
      <NotificationChannelsList channels={[mkChannel({ id: "ch_slack" })]} />
    );
    expect(screen.getByTestId("channel-test-ch_slack")).toBeDisabled();
  });

  it("renders the disabled state for channels with enabled=false", => {
    render(
      <NotificationChannelsList
        channels={[mkChannel({ id: "ch_slack", enabled: false })]}
      />
    );
    const row = screen.getByTestId("channel-row-ch_slack");
    expect(row.textContent).toContain("disabled");
  });

  it("falls back to last_test_at when no fresh result has been recorded", => {
    const channel = mkChannel({
      id: "ch_slack",
      last_test_at: "2026-04-26T12:00:00Z",
      last_test_ok: true,
    });
    render(<NotificationChannelsList channels={[channel]} />);
    const row = screen.getByTestId("channel-row-ch_slack");
    expect(row.textContent).toContain("PASS");
  });
});
