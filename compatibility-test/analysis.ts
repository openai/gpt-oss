export function analyze(caseResults: any[], tries: number) {
  // Group results by unique task: test_case + apiType
  type TaskKey = string;
  const taskKeyFor = (r: any): TaskKey =>
    `${r.test_case}::${r.result?.apiType}`;

  const successesByTask: Map<TaskKey, Map<number, boolean>> = new Map();

  // Count wrong-input tool calls (schema correct but incorrect arguments)
  let wrongInputToolCalls = 0;

  // Count invalid response shapes per API type
  const totalByApiType: Record<string, number> = {};
  const invalidByApiType: Record<string, number> = {};

  for (const r of caseResults) {
    if (!r?.result || typeof r.result.apiType !== "string") continue;

    // Parse attempt index from run_id `${i}_${k}` safely
    let attemptIndex: number | undefined;
    if (typeof r.run_id === "string") {
      const parts = r.run_id.split("_");
      const k = Number(parts[1]);
      if (Number.isFinite(k)) attemptIndex = k;
    }

    const key = taskKeyFor(r);
    if (!successesByTask.has(key)) successesByTask.set(key, new Map());
    if (attemptIndex != null) {
      successesByTask.get(key)!.set(attemptIndex, Boolean(r.success));
    }

    const d = r.result.toolCallingDetails ?? {};
    const calledToolAtLeastOnce = Boolean(d.calledToolAtLeastOnce);
    const calledToolWithRightSchema = Boolean(d.calledToolWithRightSchema);
    const calledToolWithRightArguments = Boolean(
      d.calledToolWithRightArguments
    );
    if (
      calledToolAtLeastOnce &&
      calledToolWithRightSchema &&
      !calledToolWithRightArguments
    ) {
      wrongInputToolCalls++;
    }

    // Track invalid/total per apiType for response shape
    const apiType = r.result.apiType as string;
    totalByApiType[apiType] = (totalByApiType[apiType] ?? 0) + 1;
    const isValidResponse = r.result.validResponse === true;
    if (!isValidResponse) {
      invalidByApiType[apiType] = (invalidByApiType[apiType] ?? 0) + 1;
    }
  }

  const totalTasks = successesByTask.size;

  // Compute pass@k and pass^k for k = 1..tries
  const passAtKByK: number[] = [];
  const passHatKByK: number[] = [];

  for (let k = 1; k <= tries; k++) {
    let tasksSuccessfulK = 0; // any success in first k attempts
    let tasksAllSuccessfulK = 0; // all success in first k attempts

    for (const [, attemptsMap] of successesByTask) {
      let anySuccess = false;
      let allSuccess = true;
      for (let i = 0; i < k; i++) {
        const v = attemptsMap.get(i) === true;
        anySuccess = anySuccess || v;
        if (!v) allSuccess = false;
      }
      if (anySuccess) tasksSuccessfulK++;
      if (allSuccess) tasksAllSuccessfulK++;
    }

    const passAtK = totalTasks > 0 ? tasksSuccessfulK / totalTasks : 0;
    const passHatK = totalTasks > 0 ? tasksAllSuccessfulK / totalTasks : 0;
    passAtKByK.push(passAtK);
    passHatKByK.push(passHatK);
  }

  // Convenience: final k=tries values
  const passAtK = passAtKByK[tries - 1] ?? 0;
  const passHatK = passHatKByK[tries - 1] ?? 0;

  return {
    totalTasks,
    passAtKByK,
    passHatKByK,
    passAtK,
    passHatK,
    wrongInputToolCalls,
    // New stats for invalid response shapes per API
    invalidByApiType,
    totalByApiType,
  };
}

export function printAnalysis(
  stats: ReturnType<typeof analyze>,
  caseResults: any[],
  provider: string,
  selectedLines: string[],
  tries: number,
  skipped: number,
  analysisFile: string
) {
  const formatPerK = (arr: number[]) =>
    Array.from({ length: tries }, (_, i) => {
      const v = arr[i] ?? 0;
      return `${i + 1}=${v.toFixed(3)}`;
    }).join(", ");

  console.log("Summary:");
  console.log(`  Provider: ${provider}`);
  console.log(`  Total input cases: ${selectedLines.length}`);
  console.log(`  Tries: ${tries}`);
  console.log(`  Total tasks: ${stats.totalTasks}`);
  console.log(`  Total runs: ${caseResults.length}`);
  // Conditionally print invalid response shape stats per API type
  if ((stats.totalByApiType["responses"] ?? 0) > 0) {
    const bad = stats.invalidByApiType["responses"] ?? 0;
    const tot = stats.totalByApiType["responses"] ?? 0;
    console.log(`  Invalid Responses API responses: ${bad} (out of ${tot})`);
  }
  if ((stats.totalByApiType["chat"] ?? 0) > 0) {
    const bad = stats.invalidByApiType["chat"] ?? 0;
    const tot = stats.totalByApiType["chat"] ?? 0;
    console.log(
      `  Invalid Chat Completions API responses: ${bad} (out of ${tot})`
    );
  }
  console.log(`  pass@k (k=1..${tries}): ${formatPerK(stats.passAtKByK)}`);
  console.log(`  pass^k (k=1..${tries}): ${formatPerK(stats.passHatKByK)}`);
  console.log(`  pass@k (k=${tries}): ${stats.passAtK.toFixed(3)}`);
  console.log(`  pass^k (k=${tries}): ${stats.passHatK.toFixed(3)}`);
  console.log(`  Wrong-input tool calls: ${stats.wrongInputToolCalls}`);
  console.log(`  Invalid cases.jsonl lines: ${skipped}`);
  console.log(`  Analysis written to ${analysisFile}`);
}
