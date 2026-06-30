import { describe, it, expect } from "bun:test";
import {
  FeedbackCollector,
  type SearchImpression,
} from "@warpvector/core";

function makeVec(seed: number, dim = 4): Float32Array {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = seed * 0.1 + i * 0.01;
  return v;
}

describe("FeedbackCollector", () => {
  it("recordImpression returns unique IDs", () => {
    const collector = new FeedbackCollector();
    const imp: SearchImpression = {
      queryVector: makeVec(1),
      resultVectors: [makeVec(2), makeVec(3)],
      timestamp: 1000,
    };
    const id1 = collector.recordImpression(imp);
    const id2 = collector.recordImpression({ ...imp, timestamp: 2000 });
    expect(id1).not.toBe(id2);
    expect(collector.pendingCount).toBe(2);
  });

  it("toTripletExamples converts click/skip to triplet pairs", () => {
    const collector = new FeedbackCollector();
    const query = makeVec(1);
    const doc0 = makeVec(2);
    const doc1 = makeVec(3);
    const doc2 = makeVec(4);

    const impId = collector.recordImpression({
      queryVector: query,
      resultVectors: [doc0, doc1, doc2],
      timestamp: Date.now(),
    });

    // doc0 をクリック、doc2 をスキップ
    collector.recordFeedback({
      impressionId: impId,
      resultIndex: 0,
      type: "click",
    });
    collector.recordFeedback({
      impressionId: impId,
      resultIndex: 2,
      type: "skip",
    });

    const examples = collector.toTripletExamples();

    // doc0 = positive, doc1(未操作) + doc2(skip) = negative → 2ペア
    expect(examples.length).toBe(2);
    expect(examples[0].anchor).toBe(query);
    expect(examples[0].positive).toBe(doc0);
  });

  it("toInfoNCEExamples groups negatives per positive", () => {
    const collector = new FeedbackCollector();
    const query = makeVec(1);
    const docs = [makeVec(2), makeVec(3), makeVec(4), makeVec(5)];

    const impId = collector.recordImpression({
      queryVector: query,
      resultVectors: docs,
      timestamp: Date.now(),
    });

    collector.recordFeedback({
      impressionId: impId,
      resultIndex: 0,
      type: "click",
    });

    const examples = collector.toInfoNCEExamples();
    expect(examples.length).toBe(1);
    expect(examples[0].positive).toBe(docs[0]);
    expect(examples[0].negatives.length).toBe(3); // docs[1], docs[2], docs[3]
  });

  it("dwell above threshold is positive, below is negative", () => {
    const collector = new FeedbackCollector({ dwellThresholdMs: 3000 });
    const query = makeVec(1);
    const docs = [makeVec(2), makeVec(3)];

    const impId = collector.recordImpression({
      queryVector: query,
      resultVectors: docs,
      timestamp: Date.now(),
    });

    // doc0: 5秒滞在 → positive
    collector.recordFeedback({
      impressionId: impId,
      resultIndex: 0,
      type: "dwell",
      value: 5000,
    });
    // doc1: 1秒滞在 → negative
    collector.recordFeedback({
      impressionId: impId,
      resultIndex: 1,
      type: "dwell",
      value: 1000,
    });

    const examples = collector.toTripletExamples();
    expect(examples.length).toBe(1);
    expect(examples[0].positive).toBe(docs[0]);
    expect(examples[0].negative).toBe(docs[1]);
  });

  it("returns empty when no positive exists", () => {
    const collector = new FeedbackCollector();
    const impId = collector.recordImpression({
      queryVector: makeVec(1),
      resultVectors: [makeVec(2), makeVec(3)],
      timestamp: Date.now(),
    });

    // skip のみ → positive がないので学習データなし
    collector.recordFeedback({
      impressionId: impId,
      resultIndex: 0,
      type: "skip",
    });
    collector.recordFeedback({
      impressionId: impId,
      resultIndex: 1,
      type: "skip",
    });

    expect(collector.toTripletExamples().length).toBe(0);
  });

  it("flush clears all impressions", () => {
    const collector = new FeedbackCollector();
    collector.recordImpression({
      queryVector: makeVec(1),
      resultVectors: [makeVec(2)],
      timestamp: Date.now(),
    });
    expect(collector.pendingCount).toBe(1);

    collector.flush();
    expect(collector.pendingCount).toBe(0);
  });

  it("evicts old impressions when maxImpressions exceeded", () => {
    const collector = new FeedbackCollector({ maxImpressions: 2 });

    collector.recordImpression({
      queryVector: makeVec(1),
      resultVectors: [makeVec(2)],
      timestamp: 1000,
    });
    collector.recordImpression({
      queryVector: makeVec(3),
      resultVectors: [makeVec(4)],
      timestamp: 2000,
    });
    const thirdId = collector.recordImpression({
      queryVector: makeVec(5),
      resultVectors: [makeVec(6)],
      timestamp: 3000,
    });

    expect(collector.pendingCount).toBe(2);
    // 最初のインプレッションは削除済み → 3番目は存在する
    expect(() => {
      collector.recordFeedback({
        impressionId: thirdId,
        resultIndex: 0,
        type: "click",
      });
    }).not.toThrow();
  });

  it("throws on invalid impressionId", () => {
    const collector = new FeedbackCollector();
    expect(() => {
      collector.recordFeedback({
        impressionId: "nonexistent",
        resultIndex: 0,
        type: "click",
      });
    }).toThrow("Impression not found");
  });

  it("throws on out-of-range resultIndex", () => {
    const collector = new FeedbackCollector();
    const impId = collector.recordImpression({
      queryVector: makeVec(1),
      resultVectors: [makeVec(2)],
      timestamp: Date.now(),
    });

    expect(() => {
      collector.recordFeedback({
        impressionId: impId,
        resultIndex: 5,
        type: "click",
      });
    }).toThrow("resultIndex 5 out of range");
  });
});
