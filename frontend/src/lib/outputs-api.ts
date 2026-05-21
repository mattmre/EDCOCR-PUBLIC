import { apiClient, get } from "./api-client";
import type { OutputManifest } from "./types";

export async function fetchJobOutputs(jobId: string, signal?: AbortSignal): Promise<OutputManifest> {
  return get<OutputManifest>(`/api/v1/jobs/${encodeURIComponent(jobId)}/outputs`, { signal });
}

export function outputDownloadUrl(jobId: string, outputType: string): string {
  return `${apiClient.baseUrl.replace(/\/+$/, "")}/api/v1/jobs/${encodeURIComponent(jobId)}/outputs/${encodeURIComponent(outputType)}`;
}

export function documentBundleUrl(jobId: string): string {
  return `${apiClient.baseUrl.replace(/\/+$/, "")}/api/v1/jobs/${encodeURIComponent(jobId)}/document-bundle`;
}

export function evidenceBundleUrl(jobId: string): string {
  return `${apiClient.baseUrl.replace(/\/+$/, "")}/api/v1/jobs/${encodeURIComponent(jobId)}/evidence-bundle`;
}
