import { handleUnauthorized } from "./api-client";
import { ApiError, UnauthorizedError, buildUrl, createTimeoutSignal, get } from "./api-client";
import { getApiKey, getAuthRequestCredentials } from "./auth";
import type { OutputManifest } from "./types";

export async function fetchJobOutputs(
  jobId: string,
  signal?: AbortSignal
): Promise<OutputManifest> {
  return get<OutputManifest>(`/api/v1/jobs/${encodeURIComponent(jobId)}/outputs`, { signal });
}

export function outputDownloadUrl(jobId: string, outputType: string): string {
  return buildUrl(
    `/api/v1/jobs/${encodeURIComponent(jobId)}/outputs/${encodeURIComponent(outputType)}`
  );
}

export function documentBundleUrl(jobId: string): string {
  return buildUrl(`/api/v1/jobs/${encodeURIComponent(jobId)}/document-bundle`);
}

export function evidenceBundleUrl(jobId: string): string {
  return buildUrl(`/api/v1/jobs/${encodeURIComponent(jobId)}/evidence-bundle`);
}

/**
 * Parse a Content-Disposition header value for a filename.
 * Supports `filename="..."` and RFC 5987 `filename*=UTF-8''...`.
 */
export function parseContentDispositionFilename(header: string | null): string | null {
  if (!header) return null;
  const starMatch = /filename\*\s*=\s*(?:UTF-8''|utf-8'')([^;]+)/i.exec(header);
  if (starMatch?.[1]) {
    try {
      return decodeURIComponent(starMatch[1].trim().replace(/^["']|["']$/g, ""));
    } catch {
      return starMatch[1].trim().replace(/^["']|["']$/g, "");
    }
  }
  const plainMatch = /filename\s*=\s*([^;]+)/i.exec(header);
  if (plainMatch?.[1]) {
    return plainMatch[1].trim().replace(/^["']|["']$/g, "");
  }
  return null;
}

/**
 * Authenticated binary download for job outputs / bundles.
 *
 * Bare `<a href>` cannot attach X-API-Key, so downloads must go through fetch
 * with the stored operator key (same pattern as audit-api custody fetch).
 * On success: blob → object URL → temporary anchor click → revoke.
 */
export async function downloadAuthenticated(
  url: string,
  fallbackFilename = "download"
): Promise<void> {
  const headers = new Headers();
  const apiKey = getApiKey();
  if (apiKey) {
    headers.set("X-API-Key", apiKey);
  }

  const { signal, cleanup } = createTimeoutSignal();
  try {
    const response = await fetch(url, {
      method: "GET",
      headers,
      credentials: getAuthRequestCredentials(),
      cache: "no-store",
      signal,
    });

    if (response.status === 401) {
      handleUnauthorized();
      throw new UnauthorizedError(response.status, null);
    }

    if (!response.ok) {
      let detail = `Request failed with status ${response.status}`;
      try {
        const body = await response.json();
        if (body && typeof body === "object" && "detail" in body) {
          detail = String((body as { detail: unknown }).detail);
        }
      } catch {
        // Non-JSON error body; keep generic message.
      }
      throw new ApiError(response.status, detail, null);
    }

    const blob = await response.blob();
    const filename =
      parseContentDispositionFilename(response.headers.get("content-disposition")) ||
      fallbackFilename;

    // No-op outside browser (unit tests without DOM download surface).
    if (typeof window === "undefined" || typeof document === "undefined") {
      return;
    }

    const objectUrl = URL.createObjectURL(blob);
    try {
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = filename;
      anchor.rel = "noopener";
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
    } finally {
      // Defer revocation so the browser can start the download.
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    }
  } finally {
    cleanup();
  }
}
