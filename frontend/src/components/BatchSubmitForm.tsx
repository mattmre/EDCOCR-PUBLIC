"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { submitBatch } from "@/lib/batches-api";
import type { BatchSubmitResponse } from "@/lib/types";

export function BatchSubmitForm({ onSubmitted }: { onSubmitted: (batch: BatchSubmitResponse) => void }) {
  const [files, setFiles] = useState<File[]>([]);
  const [priority, setPriority] = useState("normal");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (files.length === 0) {
      setError("Choose at least one document.");
      return;
    }
    try {
      setSubmitting(true);
      const batch = await submitBatch(files, priority);
      setError(null);
      setFiles([]);
      onSubmitted(batch);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Batch submit failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form
      className="space-y-3 rounded-md border border-border bg-background p-4"
      data-testid="batch-submit-form"
      onSubmit={handleSubmit}
    >
      <div className="grid gap-3 md:grid-cols-[1fr_160px_auto] md:items-end">
        <label className="block text-sm">
          <span className="mb-1 block font-medium">Documents</span>
          <input
            className="block w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            data-testid="batch-file-input"
            multiple
            type="file"
            onChange={(event) => setFiles(Array.from(event.currentTarget.files ?? []))}
          />
        </label>
        <label className="block text-sm">
          <span className="mb-1 block font-medium">Priority</span>
          <select
            className="block w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            data-testid="batch-priority"
            value={priority}
            onChange={(event) => setPriority(event.currentTarget.value)}
          >
            <option value="low">Low</option>
            <option value="normal">Normal</option>
            <option value="urgent">Urgent</option>
          </select>
        </label>
        <Button data-testid="batch-submit" disabled={submitting} type="submit">
          {submitting ? "Submitting" : "Submit"}
        </Button>
      </div>
      <p className="text-xs text-muted-foreground" data-testid="batch-file-count">
        {files.length} file{files.length === 1 ? "" : "s"} selected
      </p>
      {error ? (
        <p className="text-sm text-destructive" data-testid="batch-submit-error">
          {error}
        </p>
      ) : null}
    </form>
  );
}
