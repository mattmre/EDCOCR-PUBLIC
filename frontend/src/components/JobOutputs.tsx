"use client";

import { useEffect, useMemo, useState } from "react";
import {
  documentBundleUrl,
  evidenceBundleUrl,
  fetchJobOutputs,
  outputDownloadUrl,
} from "@/lib/outputs-api";
import type { OutputArtifact, OutputManifest } from "@/lib/types";

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value < 0) return "0 B";
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB"];
  let size = value / 1024;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(size >= 10 ? 0 : 1)} ${units[index]}`;
}

function artifactLabel(artifact: OutputArtifact): string {
  return artifact.output_type.replace(/_/g, " ");
}

export function JobOutputs({ jobId }: { jobId: string }) {
  const [manifest, setManifest] = useState<OutputManifest | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        setLoading(true);
        const data = await fetchJobOutputs(jobId, controller.signal);
        setManifest(data);
        setError(null);
      } catch (err) {
        if (controller.signal.aborted) return;
        setError(err instanceof Error ? err.message : "Unable to load outputs");
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void load();
    return => controller.abort();
  }, [jobId]);

  const artifacts = useMemo(() => manifest?.artifacts ?? [], [manifest]);

  if (loading) {
    return (
      <div className="rounded-md border border-border bg-background p-4 text-sm" data-testid="job-outputs-loading">
        Loading outputs...
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-md border border-destructive/40 bg-background p-4 text-sm" data-testid="job-outputs-error">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-4 rounded-md border border-border bg-background p-4 text-sm" data-testid="job-outputs">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold">Outputs</h3>
          <p className="mt-1 text-xs text-muted-foreground">
            {artifacts.length} artifact{artifacts.length === 1 ? "" : "s"} available
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <a
            className="rounded-md border border-border px-3 py-2 text-xs font-medium hover:bg-muted"
            data-testid="document-bundle-link"
            href={documentBundleUrl(jobId)}
          >
            DocumentBundle
          </a>
          <a
            className="rounded-md border border-border px-3 py-2 text-xs font-medium hover:bg-muted"
            data-testid="evidence-bundle-link"
            href={evidenceBundleUrl(jobId)}
          >
            Evidence
          </a>
        </div>
      </div>

      {artifacts.length === 0 ? (
        <p className="text-muted-foreground" data-testid="job-outputs-empty">
          No output artifacts are available yet.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-xs">
            <thead className="border-b border-border text-muted-foreground">
              <tr>
                <th className="py-2 pr-4 font-medium">Type</th>
                <th className="py-2 pr-4 font-medium">File</th>
                <th className="py-2 pr-4 font-medium">Size</th>
                <th className="py-2 pr-4 font-medium">Schema</th>
                <th className="py-2 font-medium">Action</th>
              </tr>
            </thead>
            <tbody>
              {artifacts.map((artifact) => (
                <tr key={`${artifact.output_type}:${artifact.relative_path}`} className="border-b border-border/60">
                  <td className="py-2 pr-4 capitalize">{artifactLabel(artifact)}</td>
                  <td className="py-2 pr-4 font-mono">{artifact.filename}</td>
                  <td className="py-2 pr-4">{formatBytes(artifact.size_bytes)}</td>
                  <td className="py-2 pr-4">{artifact.schema_version || "n/a"}</td>
                  <td className="py-2">
                    <a
                      className="text-primary hover:underline"
                      data-testid={`output-link-${artifact.output_type}`}
                      href={outputDownloadUrl(jobId, artifact.output_type)}
                    >
                      Download
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
