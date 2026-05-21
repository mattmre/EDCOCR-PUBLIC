import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueuesPanel, classifyQueue } from "@/components/QueuesPanel";
import type { Queue } from "@/lib/types";

function queue(overrides: Partial<Queue>): Queue {
  return {
    queue_name: "ocr_gpu",
    depth: 0,
    warning_threshold: null,
    critical_threshold: null,
    ...overrides,
  };
}

describe("classifyQueue", => {
  it("classifies based on default thresholds when none configured", => {
    expect(classifyQueue(queue({ depth: 100 }))).toBe("green");
    expect(classifyQueue(queue({ depth: 700 }))).toBe("amber");
    expect(classifyQueue(queue({ depth: 1500 }))).toBe("red");
  });

  it("respects per-queue thresholds when present", => {
    const q = queue({ depth: 75, warning_threshold: 50, critical_threshold: 100 });
    expect(classifyQueue(q)).toBe("amber");
    const r = queue({ depth: 500, warning_threshold: 50, critical_threshold: 100 });
    expect(classifyQueue(r)).toBe("red");
  });

  it("treats depth equal to threshold as breached", => {
    expect(
      classifyQueue(queue({ depth: 100, warning_threshold: 50, critical_threshold: 100 }))
    ).toBe("red");
    expect(
      classifyQueue(queue({ depth: 50, warning_threshold: 50, critical_threshold: 100 }))
    ).toBe("amber");
  });
});

describe("<QueuesPanel />", => {
  it("renders one row per queue with severity-coded bars", => {
    const queues: Queue[] = [
      queue({ queue_name: "ocr_gpu", depth: 50 }),
      queue({ queue_name: "ocr_cpu", depth: 750 }),
      queue({ queue_name: "nlp", depth: 1200 }),
    ];
    render(<QueuesPanel queues={queues} />);

    const greenBar = screen.getByTestId("queue-bar-ocr_gpu");
    const amberBar = screen.getByTestId("queue-bar-ocr_cpu");
    const redBar = screen.getByTestId("queue-bar-nlp");

    expect(greenBar.className).toMatch(/emerald/);
    expect(amberBar.className).toMatch(/amber/);
    expect(redBar.className).toMatch(/red/);

    expect(screen.getByTestId("queue-row-ocr_gpu")).toHaveAttribute("data-severity", "green");
    expect(screen.getByTestId("queue-row-ocr_cpu")).toHaveAttribute("data-severity", "amber");
    expect(screen.getByTestId("queue-row-nlp")).toHaveAttribute("data-severity", "red");
  });

  it("formats depth with thousands separators", => {
    render(<QueuesPanel queues={[queue({ queue_name: "huge", depth: 12345 })]} />);
    expect(screen.getByTestId("queue-depth-huge")).toHaveTextContent("12,345");
  });

  it("shows an empty-state message when no queues are reported", => {
    render(<QueuesPanel queues={[]} />);
    expect(screen.getByText(/no queues reporting/i)).toBeInTheDocument();
  });

  it("submits queue threshold updates", async => {
    const user = userEvent.setup();
    const updates: Array<{
      queueName: string;
      warningDepth: number;
      criticalDepth: number;
      warningWait: number;
      criticalWait: number;
    }> = [];
    render(
      <QueuesPanel
        queues={[
          queue({
            queue_name: "ocr_gpu",
            depth: 5,
            warning_wait_seconds: 45,
            critical_wait_seconds: 90,
          }),
        ]}
        onUpdateThreshold={async (queueName, threshold) => {
          updates.push({
            queueName,
            warningDepth: threshold.warning_depth,
            criticalDepth: threshold.critical_depth,
            warningWait: threshold.warning_wait_seconds,
            criticalWait: threshold.critical_wait_seconds,
          });
        }}
      />
    );

    await user.clear(screen.getByLabelText("Warn depth"));
    await user.type(screen.getByLabelText("Warn depth"), "42");
    await user.clear(screen.getByLabelText("Critical depth"));
    await user.type(screen.getByLabelText("Critical depth"), "84");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(updates).toEqual([
      {
        queueName: "ocr_gpu",
        warningDepth: 42,
        criticalDepth: 84,
        warningWait: 45,
        criticalWait: 90,
      },
    ]);
    expect(await screen.findByText("Saved")).toBeInTheDocument();
  });
});
