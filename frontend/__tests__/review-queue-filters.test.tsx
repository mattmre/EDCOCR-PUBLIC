import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  EMPTY_REVIEW_FILTER_STATE,
  ReviewQueueFiltersBar,
} from "@/components/ReviewQueueFilters";
import type { ReviewQueueFilters } from "@/lib/types";

describe("<ReviewQueueFiltersBar />", => {
  it("renders the default pending chip as pressed", => {
    render(
      <ReviewQueueFiltersBar
        value={EMPTY_REVIEW_FILTER_STATE}
        onChange={() => {}}
      />
    );
    expect(screen.getByTestId("review-chip-pending")).toHaveAttribute(
      "aria-pressed",
      "true"
    );
    expect(screen.getByTestId("review-chip-approved")).toHaveAttribute(
      "aria-pressed",
      "false"
    );
  });

  it("toggles status chips on and off", => {
    const handleChange = vi.fn();
    render(
      <ReviewQueueFiltersBar
        value={EMPTY_REVIEW_FILTER_STATE}
        onChange={handleChange}
      />
    );

    fireEvent.click(screen.getByTestId("review-chip-rejected"));
    expect(handleChange).toHaveBeenLastCalledWith({
      ...EMPTY_REVIEW_FILTER_STATE,
      status: ["pending", "rejected"],
    });

    handleChange.mockClear();
    // Toggle off the pending chip from the default state.
    render(
      <ReviewQueueFiltersBar
        value={{ ...EMPTY_REVIEW_FILTER_STATE, status: ["pending", "rejected"] }}
        onChange={handleChange}
      />
    );
    fireEvent.click(screen.getAllByTestId("review-chip-pending")[1]);
    expect(handleChange).toHaveBeenLastCalledWith({
      status: ["rejected"],
      reason: "",
      q: "",
    });
  });

  it("emits the reason filter on dropdown change", => {
    const handleChange = vi.fn();
    render(
      <ReviewQueueFiltersBar
        value={EMPTY_REVIEW_FILTER_STATE}
        onChange={handleChange}
      />
    );
    fireEvent.change(screen.getByTestId("review-filter-reason"), {
      target: { value: "low_confidence" },
    });
    expect(handleChange).toHaveBeenLastCalledWith({
      ...EMPTY_REVIEW_FILTER_STATE,
      reason: "low_confidence",
    });
  });

  it("debounces the search input before emitting onChange", async => {
    const handleChange = vi.fn();
    render(
      <ReviewQueueFiltersBar
        value={EMPTY_REVIEW_FILTER_STATE}
        onChange={handleChange}
        searchDebounceMs={20}
      />
    );

    const input = screen.getByTestId("review-filter-search");
    fireEvent.change(input, { target: { value: "job_a" } });
    fireEvent.change(input, { target: { value: "job_abc" } });

    expect(handleChange).not.toHaveBeenCalled();
    await waitFor( => {
        expect(handleChange).toHaveBeenCalledTimes(1);
      },
      { timeout: 1000 }
    );
    expect(handleChange).toHaveBeenLastCalledWith({
      ...EMPTY_REVIEW_FILTER_STATE,
      q: "job_abc",
    });
  });

  it("clears all filters via the clear button", => {
    const handleChange = vi.fn();
    const populated: ReviewQueueFilters = {
      status: ["approved", "rejected"],
      reason: "manual_flag",
      q: "alpha",
    };
    render(
      <ReviewQueueFiltersBar value={populated} onChange={handleChange} />
    );
    fireEvent.click(screen.getByTestId("review-filter-clear"));
    expect(handleChange).toHaveBeenCalledWith(EMPTY_REVIEW_FILTER_STATE);
  });
});
