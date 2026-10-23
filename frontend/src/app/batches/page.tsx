"use client";

import { BatchesTable } from "@/components/BatchesTable";
import { useRequireAuth } from "@/lib/auth";

export default function BatchesPage() {
  useRequireAuth();
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Batches</h1>
        <p className="text-sm text-muted-foreground">
          Submit multiple documents and monitor grouped OCR progress. This list uses a fixed page size of 25 and loads when opened or when the page changes; it does not use saved Display page size or auto-refresh.
        </p>
      </div>
      <BatchesTable />
    </div>
  );
}
