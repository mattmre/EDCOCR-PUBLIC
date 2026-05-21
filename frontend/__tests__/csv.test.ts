import { describe, expect, it } from "vitest";
import { parseCsv, parseCsvRows, serializeCsv } from "@/lib/csv";

describe("parseCsv", => {
  it("parses a simple header + row", => {
    const out = parseCsv("a,b\n1,2\n");
    expect(out).toEqual([{ a: "1", b: "2" }]);
  });

  it("handles quoted commas", => {
    const csv = `name,city\n"Smith, John","New York"\n`;
    expect(parseCsv(csv)).toEqual([{ name: "Smith, John", city: "New York" }]);
  });

  it("handles embedded doubled quotes", => {
    const csv = `phrase\n"He said ""hi"""\n`;
    expect(parseCsv(csv)).toEqual([{ phrase: 'He said "hi"' }]);
  });

  it("handles CRLF line endings", => {
    const csv = "a,b\r\n1,2\r\n3,4\r\n";
    expect(parseCsv(csv)).toEqual([
      { a: "1", b: "2" },
      { a: "3", b: "4" },
    ]);
  });

  it("strips a leading BOM", => {
    const csv = "﻿a,b\n1,2\n";
    expect(parseCsv(csv)).toEqual([{ a: "1", b: "2" }]);
  });

  it("tolerates a missing trailing newline", => {
    expect(parseCsv("a,b\n1,2")).toEqual([{ a: "1", b: "2" }]);
  });

  it("yields empty string for missing trailing columns", => {
    expect(parseCsv("a,b,c\n1,2\n")).toEqual([{ a: "1", b: "2", c: "" }]);
  });

  it("skips fully blank lines", => {
    expect(parseCsv("a,b\n1,2\n\n3,4\n")).toEqual([
      { a: "1", b: "2" },
      { a: "3", b: "4" },
    ]);
  });

  it("throws on an unterminated quoted field", => {
    expect(() => parseCsv(`a,b\n"oops,no end\n`)).toThrow(/Unterminated/i);
  });

  it("preserves embedded newlines inside quoted fields", => {
    const csv = `a,b\n"line1\nline2",2\n`;
    expect(parseCsvRows(csv)).toEqual([
      ["a", "b"],
      ["line1\nline2", "2"],
    ]);
  });
});

describe("serializeCsv", => {
  it("emits a header row + records with CRLF terminators", => {
    const csv = serializeCsv(
      [{ a: "1", b: "2" }],
      ["a", "b"]
    );
    expect(csv).toBe("a,b\r\n1,2\r\n");
  });

  it("quotes fields with commas and doubles embedded quotes", => {
    const csv = serializeCsv(
      [{ name: "Smith, John", phrase: 'He said "hi"' }],
      ["name", "phrase"]
    );
    expect(csv).toBe(`name,phrase\r\n"Smith, John","He said ""hi"""\r\n`);
  });

  it("optionally prefixes with the UTF-8 BOM", => {
    const csv = serializeCsv([{ a: "x" }], ["a"], { bom: true });
    expect(csv.startsWith("﻿")).toBe(true);
  });

  it("round-trips with parseCsv", => {
    const original = [
      { source_term: "Alpha", target_term: "Alfa", notes: 'with "quotes"' },
      { source_term: "Bravo", target_term: "B,r,a,v,o", notes: "" },
    ];
    const csv = serializeCsv(original, ["source_term", "target_term", "notes"]);
    const parsed = parseCsv(csv);
    expect(parsed).toEqual(original);
  });
});
