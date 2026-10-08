import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import BatchDetailPage from "@/app/batches/[batchId]/page";

const route = vi.hoisted(() => ({ batchId: "batch_aaaaaaaaaaaa" }));
const fetchBatch = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({ useParams: () => ({ batchId: route.batchId }) }));
vi.mock("@/lib/auth", () => ({ useRequireAuth: () => undefined }));
vi.mock("@/lib/batches-api", () => ({ fetchBatch }));

function batch(batchId: string) {
  return {
    batch_id: batchId, status: "completed", created_at: "2026-09-26T00:00:00Z",
    completed_at: null, processing_time: null, total_jobs: 0,
    progress: { submitted: 0, processing: 0, completed: 0, failed: 0, cancelled: 0, percent_complete: 100 },
    jobs: [], settings: {}, webhook_status: null,
  };
}

describe("batch detail route identity", () => {
  beforeEach(() => { route.batchId = "batch_aaaaaaaaaaaa"; fetchBatch.mockReset(); });

  it("ignores a completed response from the previous route", async () => {
    let releaseOld!: (value: ReturnType<typeof batch>) => void;
    fetchBatch.mockImplementation((id: string) => id === "batch_aaaaaaaaaaaa"
      ? new Promise((resolve) => { releaseOld = resolve; })
      : Promise.resolve(batch(id)));
    const view = render(<BatchDetailPage />);
    route.batchId = "batch_bbbbbbbbbbbb";
    view.rerender(<BatchDetailPage />);
    await waitFor(() => expect(screen.getByTestId("batch-detail-id")).toHaveTextContent("batch_bbbbbbbbbbbb"));
    releaseOld(batch("batch_aaaaaaaaaaaa"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(screen.getByTestId("batch-detail-id")).toHaveTextContent("batch_bbbbbbbbbbbb");
  });

  it("rejects a response with the wrong batch id", async () => {
    fetchBatch.mockResolvedValue(batch("batch_bbbbbbbbbbbb"));
    render(<BatchDetailPage />);
    await waitFor(() => expect(screen.getByText(/does not match the requested batch/)).toBeInTheDocument());
    expect(screen.queryByTestId("batch-detail-id")).toBeNull();
  });
});
