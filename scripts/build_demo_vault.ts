/**
 * Provenance generator for the committed demo vault and evaluation assets.
 *
 * This script is committed for transparency, but its OUTPUT is also committed —
 * a clean clone never needs to run it and never touches the network. It converts
 * two public retrieval datasets, expected as local files under `scripts/data/`,
 * into the artifacts the plugin and eval consume:
 *
 *   - `demo-vault/`            one Markdown note per selected SciFact abstract
 *   - `eval/scifact_qrels.jsonl`  query → relevant demo-vault note paths
 *   - `eval/wikiqa_slice.jsonl`   question → answer sentence + candidate pool
 *
 * Source data (download once into scripts/data/, not committed):
 *   - SciFact (BeIR): https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
 *       unzipped to scripts/data/scifact/{corpus.jsonl,queries.jsonl,qrels/test.tsv}
 *       License: CC BY-NC 2.0 (see the BeIR/SciFact dataset card).
 *   - WikiQA (Microsoft): https://huggingface.co/datasets/microsoft/wiki_qa
 *       test split rows saved to scripts/data/wikiqa_test_raw.json
 *       License: Microsoft Research Data License (see the WikiQA dataset card).
 *
 * Selection is deterministic so the output is reproducible: every judged-relevant
 * SciFact document is included, plus the lowest-id distractors up to a fixed
 * corpus size, giving a realistic but tractable retrieval task.
 */

import { readFileSync, writeFileSync, mkdirSync, rmSync, existsSync } from "node:fs";
import { join } from "node:path";
import { readJsonl } from "../eval/io";

const DATA_DIR = join("scripts", "data");
const SCIFACT_DIR = join(DATA_DIR, "scifact");
const DEMO_VAULT_DIR = "demo-vault";
const EVAL_DIR = "eval";

/** Total number of notes in the demo vault (judged-relevant docs + distractors). */
const TARGET_NOTES = 1000;
/** Maximum number of WikiQA questions retained in the slice. */
const MAX_WIKIQA = 150;
/** Minimum candidate answers per retained WikiQA question. */
const MIN_WIKIQA_CANDIDATES = 4;

interface CorpusDoc {
  _id: string;
  title: string;
  text: string;
}

interface WikiQaRow {
  question_id: string;
  question: string;
  document_title: string;
  answer: string;
  label: number;
}

/** Slugify a title into a filesystem- and wikilink-friendly stem. */
function slugify(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 70)
    .replace(/-+$/g, "");
}

/** Escape a YAML scalar value by quoting and escaping embedded quotes. */
function yamlString(value: string): string {
  return `"${value.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
}

function buildDemoVault(): Map<string, string> {
  const corpus = readJsonl<CorpusDoc>(join(SCIFACT_DIR, "corpus.jsonl"));
  const queries = readJsonl<{ _id: string; text: string }>(join(SCIFACT_DIR, "queries.jsonl"));
  const queryText = new Map(queries.map((q) => [q._id, q.text]));

  const qrelsLines = readFileSync(join(SCIFACT_DIR, "qrels", "test.tsv"), "utf8")
    .split("\n")
    .slice(1)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  const relevantByQuery = new Map<string, string[]>();
  const relevantDocs = new Set<string>();
  for (const line of qrelsLines) {
    const [queryId, corpusId] = line.split("\t");
    relevantDocs.add(corpusId);
    const list = relevantByQuery.get(queryId) ?? [];
    list.push(corpusId);
    relevantByQuery.set(queryId, list);
  }

  // Deterministic selection: all relevant docs + lowest-id distractors.
  const corpusById = new Map(corpus.map((doc) => [doc._id, doc]));
  const distractors = corpus
    .filter((doc) => !relevantDocs.has(doc._id))
    .sort((a, b) => Number(a._id) - Number(b._id))
    .slice(0, Math.max(0, TARGET_NOTES - relevantDocs.size))
    .map((doc) => doc._id);
  const selected = new Set<string>([...relevantDocs, ...distractors]);

  // Reset and write the demo vault.
  rmSync(DEMO_VAULT_DIR, { recursive: true, force: true });
  mkdirSync(DEMO_VAULT_DIR, { recursive: true });

  const docIdToPath = new Map<string, string>();
  for (const id of selected) {
    const doc = corpusById.get(id);
    if (doc === undefined) {
      continue;
    }
    const fileName = `${slugify(doc.title) || "note"}-${id}.md`;
    docIdToPath.set(id, fileName);
    const frontmatter = [
      "---",
      `id: ${yamlString(id)}`,
      `title: ${yamlString(doc.title)}`,
      `source: ${yamlString("BeIR/SciFact")}`,
      "---",
      "",
    ].join("\n");
    const body = `# ${doc.title}\n\n${doc.text}\n`;
    writeFileSync(join(DEMO_VAULT_DIR, fileName), frontmatter + body, "utf8");
  }

  // Write qrels keyed by note path (only docs that made it into the vault).
  const qrelsOut: string[] = [];
  for (const [queryId, docIds] of relevantByQuery) {
    const relevant = docIds.map((id) => docIdToPath.get(id)).filter((p): p is string => Boolean(p));
    if (relevant.length === 0) {
      continue;
    }
    qrelsOut.push(
      JSON.stringify({ query_id: queryId, query: queryText.get(queryId) ?? "", relevant }),
    );
  }
  mkdirSync(EVAL_DIR, { recursive: true });
  writeFileSync(join(EVAL_DIR, "scifact_qrels.jsonl"), qrelsOut.join("\n") + "\n", "utf8");

  console.log(
    `demo-vault: ${selected.size} notes (${relevantDocs.size} relevant + ${distractors.length} distractors)`,
  );
  console.log(`eval/scifact_qrels.jsonl: ${qrelsOut.length} queries`);
  return docIdToPath;
}

function buildWikiQaSlice(): void {
  const rawPath = join(DATA_DIR, "wikiqa_test_raw.json");
  if (!existsSync(rawPath)) {
    console.warn(`Skipping WikiQA slice: ${rawPath} not found.`);
    return;
  }
  const rows = JSON.parse(readFileSync(rawPath, "utf8")) as WikiQaRow[];
  const byQuestion = new Map<string, WikiQaRow[]>();
  for (const row of rows) {
    const list = byQuestion.get(row.question_id) ?? [];
    list.push(row);
    byQuestion.set(row.question_id, list);
  }

  const out: string[] = [];
  for (const [questionId, group] of byQuestion) {
    if (group.length < MIN_WIKIQA_CANDIDATES) {
      continue;
    }
    const positives = group.filter((row) => row.label === 1).map((row) => row.answer);
    if (positives.length === 0) {
      continue;
    }
    out.push(
      JSON.stringify({
        id: questionId,
        question: group[0].question,
        document_title: group[0].document_title,
        answer: positives[0],
        answers: positives,
        candidates: group.map((row) => row.answer),
      }),
    );
    if (out.length >= MAX_WIKIQA) {
      break;
    }
  }
  mkdirSync(EVAL_DIR, { recursive: true });
  writeFileSync(join(EVAL_DIR, "wikiqa_slice.jsonl"), out.join("\n") + "\n", "utf8");
  console.log(`eval/wikiqa_slice.jsonl: ${out.length} questions`);
}

function main(): void {
  if (!existsSync(SCIFACT_DIR)) {
    throw new Error(
      `Missing ${SCIFACT_DIR}. Download the SciFact corpus (see the header comment) before running.`,
    );
  }
  buildDemoVault();
  buildWikiQaSlice();
}

main();
