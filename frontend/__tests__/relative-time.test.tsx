import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, act } from "@testing-library/react";
import { RelativeTime, formatRelative } from "@/components/RelativeTime";

describe("formatRelative", => {
  it("returns em-dash for unset timestamps", => {
    expect(formatRelative(0, Date.now())).toBe("—");
    expect(formatRelative(-1, Date.now())).toBe("—");
  });

  it("ladders through s/m/h/d", => {
    const now = 1_700_000_000_000; // ms
    expect(formatRelative(now / 1000 - 2, now)).toBe("just now");
    expect(formatRelative(now / 1000 - 30, now)).toBe("30s ago");
    expect(formatRelative(now / 1000 - 600, now)).toBe("10m ago");
    expect(formatRelative(now / 1000 - 3600 * 5, now)).toBe("5h ago");
    expect(formatRelative(now / 1000 - 86400 * 3, now)).toBe("3d ago");
  });
});

describe("<RelativeTime />", => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-26T12:00:00Z"));
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("updates after the configured interval", => {
    const epoch = Math.floor(new Date("2026-04-26T12:00:00Z").getTime() / 1000) - 30;
    render(<RelativeTime epochSeconds={epoch} intervalMs={1000} />);
    expect(screen.getByText("30s ago")).toBeInTheDocument();

    act(() => {
      vi.setSystemTime(new Date("2026-04-26T12:00:31Z"));
      vi.advanceTimersByTime(1000);
    });

    expect(screen.getByText("1m ago")).toBeInTheDocument();
  });
});
