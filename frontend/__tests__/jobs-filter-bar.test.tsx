import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  EMPTY_FILTER_STATE,
  JobsFilterBar,
  jobsFilterToQuery,
} from "@/components/JobsFilterBar";
import type { JobsFilterState } from "@/lib/types";

describe("<JobsFilterBar />", => {
  it("toggles status chips and emits onChange", => {
    const handleChange = vi.fn();
    render(<JobsFilterBar value={EMPTY_FILTER_STATE} onChange={handleChange} />);

    fireEvent.click(screen.getByTestId("status-chip-failed"));
    expect(handleChange).toHaveBeenLastCalledWith({
      ...EMPTY_FILTER_STATE,
      status: ["failed"],
    });
  });

  it("debounces the search input before emitting onChange", async => {
    const handleChange = vi.fn();
    render(
      <JobsFilterBar
        value={EMPTY_FILTER_STATE}
        onChange={handleChange}
        searchDebounceMs={20}
      />
    );

    const input = screen.getByTestId("jobs-filter-search");
    fireEvent.change(input, { target: { value: "alp" } });
    fireEvent.change(input, { target: { value: "alpha" } });

    // Should not fire immediately because of the debounce.
    expect(handleChange).not.toHaveBeenCalled();

    await waitFor( => {
        expect(handleChange).toHaveBeenCalledTimes(1);
      },
      { timeout: 1000 }
    );

    expect(handleChange).toHaveBeenLastCalledWith({
      ...EMPTY_FILTER_STATE,
      q: "alpha",
    });
  });

  it("clears all filters via the clear-filters button", => {
    const handleChange = vi.fn();
    const populated: JobsFilterState = {
      status: ["failed", "completed"],
      submitted_after: "2026-04-01T00:00",
      submitted_before: "2026-04-30T00:00",
      q: "alpha",
      sort: "duration_desc",
    };
    render(<JobsFilterBar value={populated} onChange={handleChange} />);

    fireEvent.click(screen.getByTestId("jobs-filter-clear"));
    expect(handleChange).toHaveBeenCalledWith(EMPTY_FILTER_STATE);
  });

  it("emits the sort selection on change", => {
    const handleChange = vi.fn();
    render(<JobsFilterBar value={EMPTY_FILTER_STATE} onChange={handleChange} />);
    fireEvent.change(screen.getByTestId("jobs-filter-sort"), {
      target: { value: "duration_desc" },
    });
    expect(handleChange).toHaveBeenLastCalledWith({
      ...EMPTY_FILTER_STATE,
      sort: "duration_desc",
    });
  });
});

describe("jobsFilterToQuery", => {
  it("omits empty/default fields", => {
    const params = jobsFilterToQuery(EMPTY_FILTER_STATE);
    expect(params.toString()).toBe("");
  });

  it("emits repeated status= entries for multi-select", => {
    const params = jobsFilterToQuery({
      ...EMPTY_FILTER_STATE,
      status: ["failed", "completed"],
    });
    const all = params.getAll("status");
    expect(all).toEqual(["failed", "completed"]);
  });

  it("includes q, dates, and non-default sort", => {
    const params = jobsFilterToQuery({
      status: [],
      submitted_after: "2026-04-01T00:00",
      submitted_before: "2026-04-30T00:00",
      q: "alpha",
      sort: "duration_desc",
    });
    expect(params.get("q")).toBe("alpha");
    expect(params.get("submitted_after")).toBe("2026-04-01T00:00");
    expect(params.get("submitted_before")).toBe("2026-04-30T00:00");
    expect(params.get("sort")).toBe("duration_desc");
  });
});
