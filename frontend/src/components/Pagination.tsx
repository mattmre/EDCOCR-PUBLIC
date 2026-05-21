"use client";

import { Button } from "@/components/ui/button";

export interface PaginationProps {
  page: number;
  pageSize: number;
  total: number;
  onPageChange: (page: number) => void;
  disabled?: boolean;
}

export function Pagination({
  page,
  pageSize,
  total,
  onPageChange,
  disabled,
}: PaginationProps) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const safePage = Math.min(Math.max(page, 1), totalPages);
  const start = total === 0 ? 0 : (safePage - 1) * pageSize + 1;
  const end = Math.min(safePage * pageSize, total);
  const prevDisabled = disabled || safePage <= 1;
  const nextDisabled = disabled || safePage >= totalPages;

  return (
    <div className="flex items-center justify-between gap-4 px-1 py-2 text-sm">
      <p className="text-muted-foreground" data-testid="pagination-summary">
        {total === 0
          ? "No results"
          : `Showing ${start}-${end} of ${total}`}
      </p>
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={prevDisabled}
          onClick={() => onPageChange(safePage - 1)}
          aria-label="Previous page"
          data-testid="pagination-prev"
        >
          Previous
        </Button>
        <span className="text-xs text-muted-foreground" data-testid="pagination-current">
          Page {safePage} / {totalPages}
        </span>
        <Button
          variant="outline"
          size="sm"
          disabled={nextDisabled}
          onClick={() => onPageChange(safePage + 1)}
          aria-label="Next page"
          data-testid="pagination-next"
        >
          Next
        </Button>
      </div>
    </div>
  );
}
