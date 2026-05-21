import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import {
  AlertRuleEditor,
  parseBytes,
} from "@/components/AlertRuleEditor";
import type { AlertRule, NotificationChannel } from "@/lib/types";

function mkRule(overrides: Partial<AlertRule> = {}): AlertRule {
  return {
    id: "queue_depth_critical",
    name: "OCR queue depth critical",
    severity: "critical",
    expression: "ocr_queue_depth{queue=\"ocr_gpu\"} > 100",
    threshold_value: 100,
    threshold_unit: "count",
    evaluation_window_seconds: 300,
    enabled: true,
    notification_channels: ["ch_ops"],
    last_triggered_at: null,
    current_state: "inactive",
    description: null,
    ...overrides,
  };
}

const channels: NotificationChannel[] = [
  { id: "ch_ops", type: "slack", target: "#ops-alerts", enabled: true },
  {
    id: "ch_oncall",
    type: "email",
    target: "oncall@example.com",
    enabled: true,
  },
];

describe("<AlertRuleEditor />", => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the PromQL expression read-only inside <pre>", => {
    const rule = mkRule();
    render(<AlertRuleEditor rule={rule} onSubmit={vi.fn()} />);
    const block = screen.getByTestId("rule-expression");
    expect(block.tagName.toLowerCase()).toBe("pre");
    expect(block.textContent).toContain("ocr_queue_depth");
  });

  it("Save button is disabled when the form is unchanged", => {
    render(<AlertRuleEditor rule={mkRule()} onSubmit={vi.fn()} />);
    expect(screen.getByTestId("rule-save-button")).toBeDisabled();
  });

  it("enables Save once a valid threshold change is entered", => {
    render(<AlertRuleEditor rule={mkRule()} onSubmit={vi.fn()} />);
    fireEvent.change(screen.getByTestId("threshold-input"), {
      target: { value: "200" },
    });
    expect(screen.getByTestId("rule-save-button")).not.toBeDisabled();
  });

  it("rejects negative count thresholds", => {
    render(<AlertRuleEditor rule={mkRule()} onSubmit={vi.fn()} />);
    fireEvent.change(screen.getByTestId("threshold-input"), {
      target: { value: "-5" },
    });
    expect(screen.getByTestId("threshold-error")).toBeInTheDocument();
    expect(screen.getByTestId("rule-save-button")).toBeDisabled();
  });

  it("renders a free-text input and parses 'MB' for bytes-typed thresholds", => {
    const rule = mkRule({ threshold_unit: "bytes", threshold_value: 1024 * 1024 });
    render(<AlertRuleEditor rule={rule} onSubmit={vi.fn()} />);
    const input = screen.getByTestId("threshold-input") as HTMLInputElement;
    expect(input.type).toBe("text");
    fireEvent.change(input, { target: { value: "16 MB" } });
    expect(screen.getByTestId("rule-save-button")).not.toBeDisabled();
  });

  it("parseBytes accepts B/KB/MB/GB/TB and rejects garbage", => {
    expect(parseBytes("0")).toBe(0);
    expect(parseBytes("1024")).toBe(1024);
    expect(parseBytes("1 KB")).toBe(1024);
    expect(parseBytes("2.5 MB")).toBe(Math.floor(2.5 * 1024 ** 2));
    expect(Number.isNaN(parseBytes("oops"))).toBe(true);
    expect(Number.isNaN(parseBytes("-1 KB"))).toBe(true);
  });

  it("renders a slider for evaluation window with valid bounds", => {
    render(<AlertRuleEditor rule={mkRule()} onSubmit={vi.fn()} />);
    const slider = screen.getByTestId("window-input") as HTMLInputElement;
    expect(slider.type).toBe("range");
    expect(slider.min).toBe("30");
    expect(slider.max).toBe(String(24 * 60 * 60));
  });

  it("offers all three severity options", => {
    render(<AlertRuleEditor rule={mkRule()} onSubmit={vi.fn()} />);
    const select = screen.getByTestId("severity-select") as HTMLSelectElement;
    const values = Array.from(select.options).map((o) => o.value);
    expect(values).toEqual(["critical", "warning", "info"]);
  });

  it("passes only the changed fields to onSubmit", async => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(
      <AlertRuleEditor rule={mkRule()} channels={channels} onSubmit={onSubmit} />
    );
    fireEvent.change(screen.getByTestId("threshold-input"), {
      target: { value: "250" },
    });
    fireEvent.click(screen.getByTestId("enabled-checkbox"));
    fireEvent.click(screen.getByTestId("rule-save-button"));

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledTimes(1);
    });
    const payload = onSubmit.mock.calls[0][0];
    expect(payload).toMatchObject({ threshold_value: 250, enabled: false });
    // No PromQL ever passed back.
    expect(payload).not.toHaveProperty("expression");
  });

  it("shows the saveError prop verbatim and surfaces a save spinner label", => {
    const { rerender } = render(
      <AlertRuleEditor rule={mkRule()} onSubmit={vi.fn()} saveError="boom" />
    );
    expect(screen.getByTestId("rule-save-error").textContent).toBe("boom");
    rerender(<AlertRuleEditor rule={mkRule()} onSubmit={vi.fn()} saving />);
    expect(screen.getByTestId("rule-save-button").textContent).toContain("Saving");
  });

  it("resets the draft when the upstream rule object changes", => {
    const onSubmit = vi.fn();
    const { rerender } = render(
      <AlertRuleEditor rule={mkRule({ threshold_value: 100 })} onSubmit={onSubmit} />
    );
    fireEvent.change(screen.getByTestId("threshold-input"), {
      target: { value: "999" },
    });
    expect(screen.getByTestId("rule-save-button")).not.toBeDisabled();
    // Simulate post-save refresh: parent passes an updated rule.
    rerender(
      <AlertRuleEditor rule={mkRule({ threshold_value: 999 })} onSubmit={onSubmit} />
    );
    // Button is disabled again -- draft now matches new upstream value.
    expect(screen.getByTestId("rule-save-button")).toBeDisabled();
  });
});
