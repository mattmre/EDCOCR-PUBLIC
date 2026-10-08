import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { JobOutputs } from "@/components/JobOutputs";
import { setApiKey } from "@/lib/auth";

describe("JobOutputs", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    process.env.NEXT_PUBLIC_API_BASE_URL = "http://api.test";
    setApiKey("test-api-key");
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("lists output artifacts and authenticated download controls", async () => {
    global.fetch = vi.fn(async () =>
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

    // Download controls are buttons (not bare <a href>) so X-API-Key can be sent.
    const artifactBtn = screen.getByTestId("output-link-ocr_text");
    const documentBtn = screen.getByTestId("document-bundle-link");
    const evidenceBtn = screen.getByTestId("evidence-bundle-link");
    expect(artifactBtn.tagName).toBe("BUTTON");
    expect(documentBtn.tagName).toBe("BUTTON");
    expect(evidenceBtn.tagName).toBe("BUTTON");
    expect(artifactBtn).not.toHaveAttribute("href");
    expect(documentBtn).not.toHaveAttribute("href");
    expect(evidenceBtn).not.toHaveAttribute("href");
  });

  it("sends X-API-Key when downloading an artifact", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes("/outputs/ocr_text")) {
        return new Response("artifact-bytes", {
          status: 200,
          headers: {
            "content-type": "text/plain",
            "content-disposition": 'attachment; filename="job_abc123def456.txt"',
          },
        });
      }
      return new Response(
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
      );
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<JobOutputs jobId="job_abc123def456" />);
    await waitFor(() => expect(screen.getByTestId("job-outputs")).toBeInTheDocument());

    fireEvent.click(screen.getByTestId("output-link-ocr_text"));

    await waitFor(() => {
      const downloadCall = fetchMock.mock.calls.find(([input]) =>
        String(input).includes("/outputs/ocr_text")
      );
      expect(downloadCall).toBeDefined();
      const headers = new Headers(downloadCall?.[1]?.headers as HeadersInit);
      expect(headers.get("X-API-Key")).toBe("test-api-key");
    });
  });

  it("sends X-API-Key when downloading document and evidence bundles", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
      const url = String(input);
      if (url.includes("document-bundle") || url.includes("evidence-bundle")) {
        return new Response("zip-bytes", {
          status: 200,
          headers: {
            "content-type": "application/zip",
            "content-disposition": 'attachment; filename="bundle.zip"',
          },
        });
      }
      return new Response(
        JSON.stringify({
          job_id: "job_abc123def456",
          artifacts: [],
          schema_versions: {},
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<JobOutputs jobId="job_abc123def456" />);
    await waitFor(() => expect(screen.getByTestId("job-outputs")).toBeInTheDocument());

    fireEvent.click(screen.getByTestId("document-bundle-link"));
    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([input]) => String(input).includes("document-bundle"));
      expect(call).toBeDefined();
      expect(new Headers(call?.[1]?.headers as HeadersInit).get("X-API-Key")).toBe("test-api-key");
    });

    fireEvent.click(screen.getByTestId("evidence-bundle-link"));
    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([input]) => String(input).includes("evidence-bundle"));
      expect(call).toBeDefined();
      expect(new Headers(call?.[1]?.headers as HeadersInit).get("X-API-Key")).toBe("test-api-key");
    });
  });

  it("shows empty output state", async () => {
    global.fetch = vi.fn(async () =>
      new Response(
        JSON.stringify({ job_id: "job_abc123def456", artifacts: [], schema_versions: {} }),
        { status: 200, headers: { "content-type": "application/json" } }
      )
    ) as unknown as typeof fetch;

    render(<JobOutputs jobId="job_abc123def456" />);

    await waitFor(() => expect(screen.getByTestId("job-outputs-empty")).toBeInTheDocument());
  });

  it("shows load errors", async () => {
    global.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ detail: "no_outputs" }), {
        status: 404,
        headers: { "content-type": "application/json" },
      })
    ) as unknown as typeof fetch;

    render(<JobOutputs jobId="job_abc123def456" />);

    await waitFor(() => expect(screen.getByTestId("job-outputs-error")).toBeInTheDocument());
  });

  it("ignores an old job response after the route changes", async () => {
    let releaseOld!: (response: Response) => void;
    const oldRequest = new Promise<Response>((resolve) => { releaseOld = resolve; });
    global.fetch = vi.fn((input: RequestInfo | URL) =>
      String(input).includes("job_aaaaaaaaaaaa")
        ? oldRequest
        : Promise.resolve(new Response(JSON.stringify({
            job_id: "job_bbbbbbbbbbbb", artifacts: [], schema_versions: {},
          }), { status: 200, headers: { "content-type": "application/json" } }))
    ) as unknown as typeof fetch;
    const view = render(<JobOutputs jobId="job_aaaaaaaaaaaa" />);
    view.rerender(<JobOutputs jobId="job_bbbbbbbbbbbb" />);
    await waitFor(() => expect(screen.getByTestId("job-outputs-empty")).toBeInTheDocument());
    releaseOld(new Response(JSON.stringify({
      job_id: "job_aaaaaaaaaaaa", artifacts: [{
        output_type: "ocr_text", filename: "stale.txt", relative_path: "stale.txt",
        size_bytes: 1, mime_type: "text/plain", schema_version: "v1",
      }], schema_versions: {},
    }), { status: 200, headers: { "content-type": "application/json" } }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(screen.queryByText("stale.txt")).toBeNull();
  });
});
