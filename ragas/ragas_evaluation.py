"""
Unified RAGAS evaluation (replacing ragas_evaluation_liveqa.py AND ragas_evaluation_medredqa.py).

One process = ONE system + ONE Qdrant collection + ONE judge, run on every dataset in --datasets.

  # the pipeline under test (the collection is mandatory and is checked against Qdrant)
  python ragas/testing/ragas_evaluation.py --system rag --collection semantic_contextualised_v1 --datasets liveqa medredqa

  # RQ1 baseline: same generator model, same prompt style, no retrieval
  python ragas/testing/ragas_evaluation.py --system norag --datasets liveqa medredqa

  # supplementary closed-weights baseline (judge is still the same, independent model)
  python ragas/testing/ragas_evaluation.py --system norag --baseline-model openai/gpt-4o --datasets liveqa medredqa

  # compare runs on the questions ALL of them answered (paired, with bootstrap CIs)
  python evaluation_utils.py compare ragas/testing/runs/<rag_run> ragas/testing/runs/<norag_run>

Every run writes ragas/testing/runs/<run_name>/ :
  status.csv       one row per ATTEMPTED question: answered / abstained / error   (nothing is silently dropped)
  generations.json answers, contexts, raw model output (written after every question; --resume continues a run)
  scores.csv       RAGAS scores for the answered questions, keyed by qid
  summary.json     config, abstention/error rates with 95% Wilson CIs, per-metric means with n_scored
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import warnings
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# NOTE: append, never insert(0). The project has a folder called ./ragas which must not shadow the real `ragas` package.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from evaluation_utils import (  # noqa: E402
    assert_collection_exists, classify_answer, print_status_summary, strip_answer, summarize_status,
)

warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*(deprecated|ragas\.metrics\.collections).*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")

DATASETS = {
    "liveqa":   {"file": "liveqa_sampled_evaluated.csv",   "question": "message", "reference": "answer"},
    "medredqa": {"file": "medredqa_sampled_evaluated.csv", "question": "Body",    "reference": "Response"},
}

DEFAULT_JUDGE = "qwen/qwen-2.5-72b-instruct"   # must NOT be the same family as the model that wrote the answers
FAMILIES = ("llama", "qwen", "gpt", "gemini", "claude", "mistral", "deepseek", "gemma", "phi", "grok")


def model_family(name):
    n = name.lower()
    return next((f for f in FAMILIES if f in n), n.split("/")[0])


# --------------------------------------------------------------------------------------
# Systems under test
# --------------------------------------------------------------------------------------
def make_rag_runner(collection, top_k):
    from retriever.decomposer import decompose_query
    from retriever.retriever import MedicalRetriever
    from generator import generate_grounded_response

    engine = MedicalRetriever(collection_name=collection)
    assert_collection_exists(engine.q_client, collection)

    def run(question):
        atomic_queries = decompose_query(question)
        chunks, seen = [], set()
        for aq in atomic_queries:
            for chunk in engine.retrieve(user_query=aq, use_ontology=False, top_k=top_k):
                if chunk["text"] not in seen:
                    chunks.append(chunk)
                    seen.add(chunk["text"])
        n_retrieved = len(chunks)
        if not chunks:                      # same behaviour as before: the generator sees an empty-evidence marker and abstains
            chunks = [{"text": "Lack of medical data."}]
        out = generate_grounded_response(question, chunks)
        contexts = out["contexts"] if n_retrieved else []
        return {"answer": out["answer"], "contexts": contexts, "raw": out.get("raw_llm_output", ""), "n_retrieved": n_retrieved}

    return run, engine


def make_norag_runner(model):
    from generator import generate_ungrounded_response

    def run(question):
        out = generate_ungrounded_response(question, model=model)
        return {"answer": out["answer"], "contexts": [], "raw": out.get("raw_llm_output", ""), "n_retrieved": 0}

    return run, None


# --------------------------------------------------------------------------------------
# Judge + metrics (built and smoke-tested BEFORE the expensive generation phase)
# --------------------------------------------------------------------------------------
def build_judge(model, embed_model):
    from langchain_openai import ChatOpenAI
    from langchain_ollama import OllamaEmbeddings

    judge_llm = ChatOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model=model, temperature=0.0, max_retries=10, request_timeout=300,
    )
    judge_emb = OllamaEmbeddings(model=embed_model)
    return judge_llm, judge_emb


def build_metrics(system):
    # ragas 0.4.3: evaluate() only accepts the LEGACY metric objects from `ragas.metrics`
    # (served via a deprecated __getattr__). The classes in `ragas.metrics.collections` are a different API
    # (they need an `llm` argument and do not work inside evaluate()), so do NOT import from there.
    from ragas.metrics import answer_relevancy, context_precision, faithfulness, ContextRelevance

    if system == "rag":
        return [faithfulness, answer_relevancy, context_precision, ContextRelevance()]
    return [answer_relevancy]               # no retrieved context -> context metrics / faithfulness-vs-context are undefined


def score(items, metrics, judge_llm, judge_emb, workers, contexts_from="retrieved"):
    """items: list of dicts (question, answer, contexts, reference). Returns a DataFrame aligned with `items`."""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.run_config import RunConfig

    samples = []
    for it in items:
        ctx = it["contexts"] if contexts_from == "retrieved" else [it["reference"]]
        samples.append(SingleTurnSample(user_input=it["question"], response=it["answer"],
                                        retrieved_contexts=ctx, reference=it["reference"]))
    result = evaluate(
        dataset=EvaluationDataset(samples=samples), metrics=metrics, llm=judge_llm, embeddings=judge_emb,
        run_config=RunConfig(max_workers=workers, max_retries=10, timeout=300),
        raise_exceptions=False, show_progress=True,
    )
    df = result.to_pandas()
    if len(df) != len(items):
        raise RuntimeError(f"RAGAS returned {len(df)} rows for {len(items)} samples")
    return df.reset_index(drop=True)


def smoke_test(system, metrics, judge_llm, judge_emb, workers):
    """Runs the full scoring path on ONE dummy sample so import/key/embedding/column problems surface in seconds,
    not after 30 minutes of generation."""
    print("🧪 Smoke test of the scoring path (1 dummy sample)...")
    judge_llm.invoke("Reply with the single word: OK")
    judge_emb.embed_query("test")
    dummy = [{"question": "What is aspirin used for?",
              "answer": "Aspirin is used to relieve mild pain and reduce fever and inflammation.",
              "contexts": ["Aspirin is an analgesic, antipyretic and anti-inflammatory drug."] if system == "rag" else [],
              "reference": "Aspirin relieves pain, reduces fever and inflammation."}]
    df = score(dummy, metrics, judge_llm, judge_emb, workers)
    expected = {"rag": ["faithfulness", "answer_relevancy", "context_precision", "nv_context_relevance"],
                "norag": ["answer_relevancy"]}[system]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SystemExit(f"❌ RAGAS result has no column(s) {missing}; got {list(df.columns)}")
    print("✅ Scoring path OK:", {c: (None if pd.isna(df[c][0]) else round(float(df[c][0]), 3)) for c in expected})
    return expected


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def mean_table(scores, metric_cols):
    out = {}
    for dataset in sorted(scores["dataset"].unique()) + ["ALL"]:
        g = scores if dataset == "ALL" else scores[scores["dataset"] == dataset]
        out[dataset] = {m: {"mean": (None if g[m].dropna().empty else float(g[m].mean())),
                            "n_scored": int(g[m].notna().sum()), "n_nan": int(g[m].isna().sum())}
                        for m in metric_cols if m in g.columns}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=["rag", "norag"])
    ap.add_argument("--collection", help="Qdrant collection (REQUIRED for --system rag; no default on purpose)")
    ap.add_argument("--datasets", nargs="+", default=["liveqa", "medredqa"], choices=list(DATASETS))
    ap.add_argument("--n", type=int, default=50, help="first N questions of every dataset (0 = all)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--baseline-model", default=None, help="model for --system norag (default: the RAG generator model)")
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--allow-same-family-judge", action="store_true")
    ap.add_argument("--embeddings", default="nomic-embed-text")
    ap.add_argument("--workers", type=int, default=1, help="RAGAS max_workers (1 = as before; raise if the API allows)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", action="store_true", help="continue an existing run folder (skips already generated qids)")
    ap.add_argument("--skip-smoke-test", action="store_true")
    args = ap.parse_args()

    if args.system == "rag" and not args.collection:
        ap.error("--collection is required for --system rag (list yours with: curl localhost:6333/collections)")

    from generator import GENERATOR_MODEL
    answer_model = (args.baseline_model or GENERATOR_MODEL) if args.system == "norag" else GENERATOR_MODEL

    # Judge independence guard (self-preference bias)
    if model_family(args.judge) == model_family(answer_model) and not args.allow_same_family_judge:
        raise SystemExit(f"❌ Judge '{args.judge}' and answer model '{answer_model}' are from the same model family "
                         f"({model_family(answer_model)}). LLM judges favour their own family; pick another judge "
                         f"or pass --allow-same-family-judge (and state it as a limitation).")

    tag = args.collection if args.system == "rag" else f"{answer_model.split('/')[-1]}"
    run_name = args.run_name or f"{args.system}_{tag}_{args.judge.split('/')[-1]}_{datetime.now():%Y%m%d_%H%M}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    gen_path = os.path.join(run_dir, "generations.json")
    if os.path.exists(gen_path) and not args.resume:
        raise SystemExit(f"❌ {run_dir} already contains a run. Use --resume to continue it or another --run-name.")
    print(f"📁 Run folder: {run_dir}")

    # 1) judge + metrics first (fail fast)
    judge_llm, judge_emb = build_judge(args.judge, args.embeddings)
    metrics = build_metrics(args.system)
    if not args.skip_smoke_test:
        metric_cols = smoke_test(args.system, metrics, judge_llm, judge_emb, args.workers)
    else:
        metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "nv_context_relevance"] if args.system == "rag" else ["answer_relevancy"]

    # 2) system under test
    runner, engine = make_rag_runner(args.collection, args.top_k) if args.system == "rag" else make_norag_runner(answer_model)

    # 3) generation (every attempted question is recorded, whatever happens to it)
    records = {}
    if args.resume and os.path.exists(gen_path):
        records = {r["qid"]: r for r in json.load(open(gen_path, encoding="utf-8"))}
        print(f"↩️  Resuming: {len(records)} questions already generated")

    try:
        for ds_name in args.datasets:
            cfg = DATASETS[ds_name]
            df = pd.read_csv(os.path.join(SCRIPT_DIR, cfg["file"]))
            if args.n:
                df = df.head(args.n)
            for row_idx, row in df.iterrows():
                question = row.get(cfg["question"])
                if pd.isna(question) or not str(question).strip():
                    continue
                qid = f"{ds_name}:{row_idx}"
                if qid in records:
                    continue
                reference = "" if pd.isna(row.get(cfg["reference"])) else str(row[cfg["reference"]])
                print(f"\n[{ds_name} {row_idx + 1}/{len(df)}] {str(question)[:100]}...")
                t0 = time.time()
                rec = {"qid": qid, "dataset": ds_name, "question": str(question), "reference": reference,
                       "answer": "", "raw_answer": "", "contexts": [], "n_retrieved": 0, "error": "", "status": "error"}
                try:
                    out = runner(str(question))
                    rec.update(raw_answer=out["answer"], contexts=out["contexts"], n_retrieved=out["n_retrieved"],
                               status=classify_answer(out["answer"]), model_output=out["raw"])
                    rec["answer"] = strip_answer(out["answer"]) if rec["status"] == "answered" else ""
                    if rec["status"] == "answered" and not rec["answer"]:
                        rec["status"] = "error"
                        rec["error"] = "answer empty after stripping disclaimer"
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"
                    print(f"  ❌ {rec['error']}")
                rec["latency_s"] = round(time.time() - t0, 2)
                print(f"  → {rec['status']} ({rec['latency_s']} s)")
                records[qid] = rec
                json.dump(list(records.values()), open(gen_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    finally:
        if engine is not None and hasattr(engine, "q_client"):
            engine.q_client.close()

    recs = list(records.values())
    status_df = pd.DataFrame([{k: r.get(k) for k in ("qid", "dataset", "status", "latency_s", "n_retrieved", "error")}
                              for r in recs])
    status_df["n_contexts"] = [len(r["contexts"]) for r in recs]
    status_df.to_csv(os.path.join(run_dir, "status.csv"), index=False)
    status_summary = summarize_status(status_df)
    print_status_summary(status_summary)

    # 4) scoring: only 'answered' questions can be scored, but the denominators above keep the others visible
    answered = [r for r in recs if r["status"] == "answered"]
    scores = pd.DataFrame(columns=["qid", "dataset"])
    if answered:
        print(f"\n🚀 Scoring {len(answered)} answered questions with judge {args.judge}...")
        main_df = score(answered, metrics, judge_llm, judge_emb, args.workers)
        scores = pd.DataFrame({"qid": [r["qid"] for r in answered], "dataset": [r["dataset"] for r in answered]})
        for m in metric_cols:
            scores[m] = main_df[m].values

        # like-for-like faithfulness: the SAME judge scores every system's answer against the expert answer
        with_ref = [r for r in answered if r["reference"].strip()]
        scores["faithfulness_vs_reference"] = float("nan")
        if with_ref:
            from ragas.metrics import faithfulness
            print(f"\n🚀 Faithfulness vs the expert reference ({len(with_ref)} questions)...")
            ref_df = score(with_ref, [faithfulness], judge_llm, judge_emb, args.workers, contexts_from="reference")
            by_qid = dict(zip([r["qid"] for r in with_ref], ref_df["faithfulness"].values))
            scores["faithfulness_vs_reference"] = scores["qid"].map(by_qid)
    scores.to_csv(os.path.join(run_dir, "scores.csv"), index=False)

    all_metric_cols = metric_cols + (["faithfulness_vs_reference"] if answered else [])
    summary = {
        "config": {"run_name": run_name, "system": args.system, "collection": args.collection, "datasets": args.datasets,
                   "n_per_dataset": args.n, "top_k": args.top_k, "answer_model": answer_model, "judge": args.judge,
                   "embeddings": args.embeddings, "git_commit": git_commit(), "created": datetime.now().isoformat(timespec="seconds")},
        "status": status_summary,
        "scores_over_answered_only": mean_table(scores, all_metric_cols) if answered else {},
        "note": "Means cover ANSWERED questions only. Compare systems with `python evaluation_utils.py compare <run> <run> ...` "
                "(paired on the questions every run scored) and always report the abstention rate next to the scores.",
    }
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n=== MEAN SCORES (answered questions only) ===")
    for ds, table in summary["scores_over_answered_only"].items():
        print(f"  [{ds}] " + ", ".join(f"{m}={v['mean']:.3f} (n={v['n_scored']}, nan={v['n_nan']})" if v["mean"] is not None else f"{m}=n/a" for m, v in table.items()))
    print(f"\n💾 Saved to {run_dir}")


if __name__ == "__main__":
    main()
