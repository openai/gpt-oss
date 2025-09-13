import { parseArgs } from "node:util";
import { createWriteStream } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { runCase, RunCaseSummary } from "./runCase";
import { Listr, ListrTaskWrapper } from "listr2";
import { analyze, printAnalysis } from "./analysis";

function formatTimestamp(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  const yyyy = d.getFullYear();
  const mm = pad(d.getMonth() + 1);
  const dd = pad(d.getDate());
  const hh = pad(d.getHours());
  const mi = pad(d.getMinutes());
  const ss = pad(d.getSeconds());
  return `${yyyy}${mm}${dd}_${hh}${mi}${ss}`;
}

async function main() {
  const args = parseArgs({
    options: {
      cases: { type: "string", short: "c", default: "cases.jsonl" },
      provider: { type: "string", short: "p", default: "openai" },
      streaming: { type: "boolean", short: "s", default: false },
      maxTurns: { type: "string", short: "t", default: "10" },
      n: { type: "string", short: "n" },
      strict: { type: "boolean", short: "s", default: false },
      tries: { type: "string", short: "k", default: "1" },
    },
  });
  const casesPathArg = args.values.cases;
  const provider = args.values.provider as string;
  const streaming = Boolean(args.values.streaming);
  const maxTurns = Number(args.values.maxTurns ?? 10);
  const nRaw = args.values.n as string | undefined;
  const triesRaw = args.values.tries as string | undefined;
  const tries = triesRaw != null ? Number(triesRaw) : 1;
  const limit = nRaw != null ? Number(nRaw) : undefined;
  if (limit != null && (!Number.isFinite(limit) || limit <= 0)) {
    console.error("--n must be a positive integer");
    process.exitCode = 1;
    return;
  }

  if (!casesPathArg) {
    console.error("--cases is required (path to JSONL file)");
    process.exitCode = 1;
    return;
  }

  const casesPath = path.isAbsolute(casesPathArg)
    ? casesPathArg
    : path.join(process.cwd(), casesPathArg);

  const timestamp = formatTimestamp(new Date());
  const defaultFilename = `rollout_${provider}_${timestamp}.jsonl`;
  const outputFile = path.join(process.cwd(), defaultFilename);
  const analysisFile = path.join(
    process.cwd(),
    `analysis_${provider}_${timestamp}.json`
  );

  let fileContent: string;
  try {
    fileContent = await readFile(casesPath, "utf8");
  } catch (err: any) {
    console.error(
      `Failed to read cases file at ${casesPath}: ${err?.message ?? err}`
    );
    process.exitCode = 1;
    return;
  }

  const lines = fileContent
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  const selectedLines =
    typeof limit === "number" ? lines.slice(0, limit) : lines;

  const out = createWriteStream(outputFile, { flags: "w", encoding: "utf8" });

  const writeLine = (obj: any) =>
    new Promise<void>((resolve, reject) => {
      const str = JSON.stringify(obj) + "\n";
      out.write(str, (err) => (err ? reject(err) : resolve()));
    });

  // Accumulators for post-run analysis
  let skipped = 0; // invalid JSON lines
  const caseResults: Array<{
    run_id: string;
    success: boolean;
    provider: string;
    test_case: number;
    tool_name: string;
    input: string;
    result: RunCaseSummary;
  }> = [];

  async function processIndex(
    i: number,
    k: number,
    task: ListrTaskWrapper<any, any, any>
  ) {
    const line = selectedLines[i];
    let caseObj: any;
    try {
      caseObj = JSON.parse(line);
    } catch (err: any) {
      console.error(
        `Skipping invalid JSON on line ${i + 1}: ${err?.message ?? err}`
      );
      skipped++;
      return;
    }

    try {
      const summaries = await runCase(provider, caseObj, {
        maxTurns,
        streaming,
        strict: args.values.strict,
      });

      for (const summary of summaries) {
        const record = {
          run_id: `${i}_${k}`,
          success: summary.success,
          provider,
          test_case: i,
          tool_name: caseObj.tool_name,
          input: caseObj.input,
          result: summary,
        };
        task.output = `Case ${i} (attempt ${k + 1}): ${
          summary.success ? "Success" : "Failed"
        } ${summary.toolCallingDetails.warning || ""}`;
        caseResults.push(record);
        await writeLine(record);
      }
    } catch (err: any) {
      const record = {
        provider,
        test_case: i,
        tool_name: caseObj?.tool_name,
        input: caseObj?.input,
        expected_output: caseObj?.expected_output,
        instructions: caseObj?.instructions,
        error: String(err?.message ?? err),
      };
      await writeLine(record);
      task.output = `Case ${i} failed: ${err?.message ?? err}`;
    }
  }

  const listr = new Listr<{
    output: string;
  }>(
    selectedLines.flatMap((line, index) => {
      return Array.from({ length: tries }, (_, attempt) => ({
        title: `Processing case ${index} (attempt ${attempt + 1})`,
        task: async (_, task) => {
          await processIndex(index, attempt, task);
        },
        rendererOptions: { persistentOutput: true },
      }));
    }),
    {
      concurrent: 5,
    }
  );

  await listr.run();

  await new Promise((resolve) => out.end(resolve));
  console.log(`Results written to ${outputFile}`);
  const stats = analyze(caseResults, tries);
  await writeFile(analysisFile, JSON.stringify(stats, null, 2), "utf8");
  printAnalysis(
    stats,
    caseResults,
    provider,
    selectedLines,
    tries,
    skipped,
    analysisFile
  );
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
