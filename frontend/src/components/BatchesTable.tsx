"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { BatchSubmitForm } from "@/components/BatchSubmitForm";
import { Pagination } from "@/components/Pagination";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { fetchBatches } from "@/lib/batches-api";
import type { BatchStatusResponse, BatchSubmitResponse } from "@/lib/types";

const PAGE_SIZE = 25;

function formatTimestamp(value: string | null): string {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export function BatchesTable() {
  const [batches, setBatches] = useState<BatchStatusResponse[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const offset = (page - 1) * PAGE_SIZE;

  async function load() {
    try {
      setLoading(true);
      const response = await fetchBatches(PAGE_SIZE, offset);
      setBatches(response.batches ?? []);
      setTotal(response.total ?? 0);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load batches");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page]);

  function handleSubmitted(batch: BatchSubmitResponse) {
    const progress = {
      submitted: batch.total_jobs,
      processing: 0,
      completed: 0,
      failed: 0,
      cancelled: 0,
      percent_complete: 0,
    };
    setBatches((current) => [
      {
        batch_id: batch.batch_id,
        status: batch.status,
        created_at: batch.created_at,
        completed_at: null,
        processing_time: null,
        total_jobs: batch.total_jobs,
        progress,
        jobs: batch.jobs,
        settings: {},
        webhook_status: null,
      },
      ...current,
    ]);
    setTotal((value) => value + 1);
  }

  return (
    <div className="space-y-4">
      <BatchSubmitForm onSubmitted={handleSubmitted} />
      {error ? (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive" data-testid="batches-error">
          {error}
        </div>
      ) : null}
      <div className="overflow-x-auto rounded-md border border-border">
        <table className="min-w-full divide-y divide-border" data-testid="batches-table">
          <thead className="bg-muted/40 text-xs uppercase text-muted-foreground">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Batch ID</th>
              <th className="px-3 py-2 text-left font-medium">Status</th>
              <th className="px-3 py-2 text-left font-medium">Jobs</th>
              <th className="px-3 py-2 text-left font-medium">Progress</th>
              <th className="px-3 py-2 text-left font-medium">Created</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border bg-background text-sm">
            {loading && batches.length === 0 ? (
              <tr>
                <td className="px-3 py-6 text-center text-muted-foreground" colSpan={5} data-testid="batches-loading">
                  Loading batches...
                </td>
              </tr>
            ) : batches.length === 0 ? (
              <tr>
                <td className="px-3 py-6 text-center text-muted-foreground" colSpan={5} data-testid="batches-empty">
                  No batches found
                </td>
              </tr>
            ) : (
              batches.map((batch) => (
                <tr className="hover:bg-muted/30" key={batch.batch_id}>
                  <td className="px-3 py-2 font-mono text-xs">
                    <Link className="text-primary hover:underline" href={`/batches/${batch.batch_id}`}>
                      {batch.batch_id}
                    </Link>
                  </td>
                  <td className="px-3 py-2">
                    <StatusBadge status={batch.status} />
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">{batch.total_jobs}</td>
                  <td className="px-3 py-2 text-muted-foreground">
                    {batch.progress.percent_complete.toFixed(0)}%
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">{formatTimestamp(batch.created_at)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      <Pagination
        disabled={loading}
        onPageChange={setPage}
        page={page}
        pageSize={PAGE_SIZE}
        total={total}
      />
    </div>
  );
}
