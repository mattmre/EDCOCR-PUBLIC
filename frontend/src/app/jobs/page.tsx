"use client";

import { JobsTable } from "@/components/JobsTable";
import { useRequireAuth } from "@/lib/auth";

export default function JobsPage() {
  useRequireAuth();
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Jobs</h1>
        <p className="text-sm text-muted-foreground">
          Submitted documents. Pagination and refresh timing use your saved Display preferences.
        </p>
      </div>
      <JobsTable />
    </div>
  );
}
