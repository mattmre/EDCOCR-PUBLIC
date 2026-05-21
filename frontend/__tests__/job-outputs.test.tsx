import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { JobOutputs } from "@/components/JobOutputs";

describe("JobOutputs", => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    process.env.NEXT_PUBLIC_API_BASE_URL = "http://api.test";
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("lists output artifacts and bundle links", async => {
    global.fetch = vi.fn(async =>
      new Response(
        JSON.stringify({
          job_id: "job_abc123def456",
          artifacts: [
            {
              output_type: "ocr_text",
              filename: "job_abc123def456.txt",
              relative_path: "job_abc123def456.txt",
              size_bytes: 1536,
              mime_type: "text/plain",
              schema_version: "ocr-text-v1",
            },
          ],
          schema_versions: { ocr_text: "ocr-text-v1" },
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      )
    ) as unknown as typeof fetch;

    render(<JobOutputs jobId="job_abc123def456" />);

    await waitFor(() => expect(screen.getByTestId("job-outputs")).toBeInTheDocument());
    expect(screen.getByText("ocr text")).toBeInTheDocument();
    expect(screen.getByText("job_abc123def456.txt")).toBeInTheDocument();
    expect(screen.getByText("1.5 KB")).toBeInTheDocument();
    expect(screen.getByTestId("output-link-ocr_text")).toHaveAttribute(
      "href",
      "http://localhost:8000/api/v1/jobs/job_abc123def456/outputs/ocr_text"
    );
    expect(screen.getByTestId("document-bundle-link")).toHaveAttribute(
      "href",
      "http://localhost:8000/api/v1/jobs/job_abc123def456/document-bundle"
    );
    expect(screen.getByTestId("evidence-bundle-link")).toHaveAttribute(
      "href",
      "http://localhost:8000/api/v1/jobs/job_abc123def456/evidence-bundle"
    );
  });

  it("shows empty output state", async => {
    global.fetch = vi.fn(async =>
      new Response(
        JSON.stringify({ job_id: "job_abc123def456", artifacts: [], schema_versions: {} }),
        { status: 200, headers: { "content-type": "application/json" } }
      )
    ) as unknown as typeof fetch;

    render(<JobOutputs jobId="job_abc123def456" />);

    await waitFor(() => expect(screen.getByTestId("job-outputs-empty")).toBeInTheDocument());
  });

  it("shows load errors", async => {
    global.fetch = vi.fn(async =>
      new Response(JSON.stringify({ detail: "no_outputs" }), {
        status: 404,
        headers: { "content-type": "application/json" },
      })
    ) as unknown as typeof fetch;

    render(<JobOutputs jobId="job_abc123def456" />);

    await waitFor(() => expect(screen.getByTestId("job-outputs-error")).toBeInTheDocument());
  });
});
