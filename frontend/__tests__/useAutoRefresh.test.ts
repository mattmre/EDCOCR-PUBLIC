import { afterEach, describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { useAutoRefresh } from "@/lib/hooks";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useAutoRefresh", => {
  it("calls the fetcher on mount and exposes data", async => {
    const fetcher = vi.fn(async => ({ value: 1 }));
    const { result } = renderHook(() => useAutoRefresh(fetcher, 60_000));

    await waitFor(() => {
      expect(result.current.data).toEqual({ value: 1 });
    });
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(result.current.error).toBeNull();
    expect(result.current.lastUpdated).not.toBeNull();
    expect(result.current.loading).toBe(false);
  });

  it("re-fetches on the interval", async => {
    let counter = 0;
    const fetcher = vi.fn(async => {
      counter += 1;
      return { value: counter };
    });

    // Use a tight interval (~30 ms) and real timers so testing-library's
    // waitFor (which itself relies on setTimeout) keeps working.
    const { result } = renderHook(() => useAutoRefresh(fetcher, 30));

    await waitFor( => {
        expect(fetcher.mock.calls.length).toBeGreaterThanOrEqual(3);
      },
      { timeout: 1000 }
    );
    expect(result.current.data?.value).toBeGreaterThanOrEqual(3);
  });

  it("captures fetcher errors without throwing", async => {
    const fetcher = vi.fn(async => {
      throw new Error("boom");
    });
    const { result } = renderHook(() => useAutoRefresh(fetcher, 60_000));

    await waitFor(() => {
      expect(result.current.error).toBeInstanceOf(Error);
    });
    expect(result.current.error?.message).toBe("boom");
    expect(result.current.data).toBeNull();
  });

  it("aborts in-flight requests on unmount", async => {
    const seenSignals: AbortSignal[] = [];
    const fetcher = vi.fn(async (signal: AbortSignal) => {
      seenSignals.push(signal);
      return await new Promise<{ value: number }>((_, reject) => {
        signal.addEventListener("abort", => {
          const err = new Error("aborted");
          (err as Error & { name: string }).name = "AbortError";
          reject(err);
        });
      });
    });

    const { unmount } = renderHook(() => useAutoRefresh(fetcher, 60_000));

    await waitFor(() => {
      expect(seenSignals.length).toBeGreaterThan(0);
    });
    expect(seenSignals[0].aborted).toBe(false);

    unmount();
    expect(seenSignals[0].aborted).toBe(true);
  });

  it("manual refresh triggers an immediate fetch", async => {
    const fetcher = vi.fn(async => ({ value: 42 }));
    const { result } = renderHook(() => useAutoRefresh(fetcher, 60_000));

    await waitFor(() => {
      expect(fetcher).toHaveBeenCalledTimes(1);
    });

    await act(async => {
      result.current.refresh();
    });
    await waitFor(() => {
      expect(fetcher).toHaveBeenCalledTimes(2);
    });
  });
});
