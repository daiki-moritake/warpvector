#!/usr/bin/env node

import * as fs from "fs";
import * as path from "path";
import { WarpPipeline } from "@warpvector/core";
import { QuantizationAdapter } from "@warpvector/extras";
import { evaluatePipeline, CorpusItem, EvalQuery } from "./evaluator";

// 評価に必要なアダプタを事前に登録
WarpPipeline.registerFinalStage("QuantizationAdapter", (state) =>
  QuantizationAdapter.importState(state as any),
);

interface ConfigFile {
  corpusPath: string;
  datasetPath: string;
  kList?: number[];
  pipelineStatePath?: string;
  intent?: string;
}

function printUsage() {
  console.log(`
Usage: npx warpvector-eval [options]

Options:
  --config <path>   Path to the configuration file (default: warpvector-eval.config.json)
  --help            Show help
`);
}

function formatPercent(value: number): string {
  if (value === 0) return "0.00%";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function formatDiff(value: number): string {
  if (value === 0) return "0.0000";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(4)}`;
}

function parseCSV(csvContent: string): any[] {
  const lines = csvContent
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  if (lines.length === 0) return [];
  const headers = lines[0].split(",").map((h) => h.trim());
  const result = [];
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i]
      .split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/)
      .map((v) => v.replace(/^"|"$/g, "").trim());
    const obj: any = {};
    headers.forEach((header, index) => {
      let val: any = values[index];
      if (val && val.startsWith("[") && val.endsWith("]")) {
        try {
          val = JSON.parse(val);
        } catch (e) {}
      } else if (header === "expectedIds" && typeof val === "string") {
        val = val
          .split(";")
          .map((s) => s.trim())
          .filter((s) => s.length > 0);
      }
      obj[header] = val;
    });
    result.push(obj);
  }
  return result;
}

function loadDataFile(filePath: string): any {
  const content = fs.readFileSync(filePath, "utf-8");
  const ext = path.extname(filePath).toLowerCase();

  if (ext === ".csv") {
    return parseCSV(content);
  } else if (ext === ".jsonl") {
    return content
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .map((line) => JSON.parse(line));
  } else {
    return JSON.parse(content);
  }
}

async function main() {
  const args = process.argv.slice(2);
  let configPath = "warpvector-eval.config.json";

  if (args.includes("--help") || args.includes("-h")) {
    printUsage();
    process.exit(0);
  }

  const configIndex = args.indexOf("--config");
  if (configIndex !== -1 && configIndex + 1 < args.length) {
    configPath = args[configIndex + 1];
  }

  const resolvedConfigPath = path.resolve(process.cwd(), configPath);

  if (!fs.existsSync(resolvedConfigPath)) {
    console.error(
      `Error: Configuration file not found at ${resolvedConfigPath}`,
    );
    process.exit(1);
  }

  console.log(`Loading configuration from: ${resolvedConfigPath}`);
  const configDir = path.dirname(resolvedConfigPath);

  let config: ConfigFile;
  try {
    const rawConfig = fs.readFileSync(resolvedConfigPath, "utf-8");
    config = JSON.parse(rawConfig);
  } catch (err) {
    console.error(
      `Error parsing configuration file: ${(err as Error).message}`,
    );
    process.exit(1);
  }

  // 必須ファイルの検証
  if (!config.corpusPath || !config.datasetPath) {
    console.error(
      "Error: 'corpusPath' and 'datasetPath' are required in the configuration file.",
    );
    process.exit(1);
  }

  const resolvedCorpusPath = path.resolve(configDir, config.corpusPath);
  const resolvedDatasetPath = path.resolve(configDir, config.datasetPath);

  if (!fs.existsSync(resolvedCorpusPath)) {
    console.error(`Error: Corpus file not found at ${resolvedCorpusPath}`);
    process.exit(1);
  }

  if (!fs.existsSync(resolvedDatasetPath)) {
    console.error(`Error: Dataset file not found at ${resolvedDatasetPath}`);
    process.exit(1);
  }

  let corpus: CorpusItem[];
  let dataset: EvalQuery[];

  try {
    corpus = loadDataFile(resolvedCorpusPath);
    dataset = loadDataFile(resolvedDatasetPath);
  } catch (err) {
    console.error(`Error reading data files: ${(err as Error).message}`);
    process.exit(1);
  }

  console.log(`Loaded Corpus : ${corpus.length} items`);
  console.log(`Loaded Dataset: ${dataset.length} queries`);

  let pipeline: WarpPipeline | undefined;
  if (config.pipelineStatePath) {
    const resolvedPipelineStatePath = path.resolve(
      configDir,
      config.pipelineStatePath,
    );
    if (!fs.existsSync(resolvedPipelineStatePath)) {
      console.error(
        `Error: Pipeline state file not found at ${resolvedPipelineStatePath}`,
      );
      process.exit(1);
    }
    try {
      const pipelineJson = JSON.parse(
        fs.readFileSync(resolvedPipelineStatePath, "utf-8"),
      );
      pipeline = WarpPipeline.importState(pipelineJson);
      await pipeline.init();
      console.log("Pipeline restored and initialized successfully.");
    } catch (err) {
      console.error(
        `Error restoring pipeline state: ${(err as Error).message}`,
      );
      process.exit(1);
    }
  }

  const kList = config.kList || [1, 3, 5, 10];
  const intentName = config.intent;

  console.log("Running evaluation...");
  const report = await evaluatePipeline({
    corpus,
    dataset,
    kList,
    pipeline,
    intentName,
  });

  // 結果の表示
  console.log("\n" + "=".repeat(70));
  console.log(" WarpVector RAG Evaluation Report");
  console.log("=".repeat(70));
  console.log(` Dataset Size : ${dataset.length} queries`);
  console.log(` Corpus Size  : ${corpus.length} items`);
  if (intentName) {
    console.log(` Intent       : ${intentName}`);
  }
  if (pipeline) {
    console.log(` Pipeline Steps:`);
    console.log(
      pipeline
        .inspect()
        .split("\n")
        .map((l) => "   " + l)
        .join("\n"),
    );
  } else {
    console.log(" Pipeline     : None (Vanilla search comparison only)");
  }
  console.log("=".repeat(70));
  console.log(
    " " +
      "Metric".padEnd(12) +
      " | " +
      "Vanilla".padEnd(10) +
      " | " +
      "Warped".padEnd(10) +
      " | " +
      "Improvement (Diff)",
  );
  console.log("-".repeat(70));

  const showMetricRow = (
    label: string,
    vanVal: number,
    warpVal: number,
    isLatency = false,
  ) => {
    const diff = warpVal - vanVal;
    const impPct = vanVal !== 0 ? (diff / vanVal) * 100 : 0;

    // 遅延の場合は低い方が良いので、符号や計算が逆になるが、単純な増減率として表示し
    // 括弧内でミリ秒を表示する
    let impStr = "";
    if (isLatency) {
      const pct = (diff / vanVal) * 100;
      const sign = diff > 0 ? "+" : "";
      impStr = `${sign}${pct.toFixed(2)}% (${sign}${diff.toFixed(2)} ms)`;
    } else {
      const pct = vanVal !== 0 ? (diff / vanVal) * 100 : 0;
      impStr = `${formatPercent(pct)} (${formatDiff(diff)})`;
    }

    const vanStr = isLatency ? `${vanVal.toFixed(2)} ms` : vanVal.toFixed(4);
    const warpStr = isLatency ? `${warpVal.toFixed(2)} ms` : warpVal.toFixed(4);

    console.log(
      " " +
        label.padEnd(12) +
        " | " +
        vanStr.padEnd(10) +
        " | " +
        warpStr.padEnd(10) +
        " | " +
        impStr,
    );
  };

  for (const k of kList) {
    showMetricRow(
      `Recall@${k}`,
      report.vanilla.recall[k],
      report.warped.recall[k],
    );
  }
  console.log("-".repeat(70));
  for (const k of kList) {
    showMetricRow(`NDCG@${k}`, report.vanilla.ndcg[k], report.warped.ndcg[k]);
  }
  console.log("-".repeat(70));
  showMetricRow("MRR", report.vanilla.mrr, report.warped.mrr);
  console.log("-".repeat(70));
  showMetricRow(
    "Avg Latency",
    report.vanilla.avgLatencyMs,
    report.warped.avgLatencyMs,
    true,
  );
  console.log("=".repeat(70) + "\n");
}

main().catch((err) => {
  console.error("Evaluation failed with error:", err);
  process.exit(1);
});
