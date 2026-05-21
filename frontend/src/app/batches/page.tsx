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
          Submit multiple documents and monitor grouped OCR progress.
        </p>
      </div>
      <BatchesTable />
    </div>
  );
}
