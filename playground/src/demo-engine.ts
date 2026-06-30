/**
 * WarpVector Playground — Real Library Demo Engine
 *
 * This module uses the ACTUAL @warpvector/core library to perform
 * intent-based vector transformations.
 * Enhanced with Real LLM Embeddings, PCA Projection, Whitening, and Quantization.
 */
import { IntentAdapter, initWasm, cosineSimilarity, type IntentWeights } from '@warpvector/core';
import { WhiteningAdapter } from '@warpvector/ml';
import { QuantizationAdapter, type QuantizationType, TaskArithmetic } from '@warpvector/extras';
import { calculateNDCG, calculateRecall } from '@warpvector/eval';
import { FeedbackCollector, AdaptiveScheduler, FederatedAggregator } from '@warpvector/core';

// all-MiniLM-L6-v2 uses 384 dimensions
export const DIM = 384;

function getIdentityWeights(dim: number): IntentWeights {
  return {
    matrix: new Float32Array(dim * dim).map((_, i) => (i % (dim + 1) === 0 ? 1 : 0)),
    bias: new Float32Array(dim)
  };
}

// --- Federated Learning / Local Learning State ---
export const demoFeedbackCollector = new FeedbackCollector();
export const demoFederatedAggregator = new FederatedAggregator(getIdentityWeights(DIM), DIM);

export interface FedEdgeStatus {
  id: string;
  name: string;
  collectedFeedback: number;
  hasLocalUpdate: boolean;
}

export const fedEdges: FedEdgeStatus[] = [
  { id: 'local', name: 'You (Local)', collectedFeedback: 0, hasLocalUpdate: false },
  { id: 'edge-tokyo', name: 'Edge (Tokyo)', collectedFeedback: 12, hasLocalUpdate: true },
  { id: 'edge-ny', name: 'Edge (New York)', collectedFeedback: 25, hasLocalUpdate: true }
];

// Worker instance
let worker: Worker | null = null;
let messageIdCounter = 0;
type ProgressCallback = (status: string, data?: any) => void;

/** Initialize the Web Worker for Transformers.js */
function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./worker.ts', import.meta.url), {
      type: 'module'
    });
  }
  return worker;
}

/** Promisified message sending to Worker */
function runWorkerTask(type: string, payload: any, onProgress?: ProgressCallback): Promise<any> {
  return new Promise((resolve, reject) => {
    const id = messageIdCounter++;
    const w = getWorker();
    
    const listener = (event: MessageEvent) => {
      if (event.data.id === id) {
        if (event.data.status === 'progress' && onProgress) {
          onProgress(event.data.status, event.data.data);
        } else if (event.data.status === 'ready') {
          w.removeEventListener('message', listener);
          resolve(true);
        } else if (event.data.status === 'complete') {
          w.removeEventListener('message', listener);
          resolve(event.data.embeddings);
        } else if (event.data.status === 'error') {
          w.removeEventListener('message', listener);
          reject(new Error(event.data.error));
        }
      }
    };
    
    w.addEventListener('message', listener);
    w.postMessage({ id, type, ...payload });
  });
}

/** Preload the model */
export async function preloadModel(onProgress?: ProgressCallback) {
  await runWorkerTask('load', {}, onProgress);
}

/** Get embeddings for an array of texts */
export async function getEmbeddings(texts: string[]): Promise<Float32Array[]> {
  return await runWorkerTask('embed', { texts });
}

/** Generate a transformation matrix that pulls vectors in a category's direction */
export function generateIntentMatrix(
  dim: number,
  categoryDirection: Float32Array,
  strength: number,
): { matrix: Float32Array; bias: Float32Array } {
  const matrix = new Float32Array(dim * dim);
  const bias = new Float32Array(dim);

  for (let i = 0; i < dim; i++) {
    matrix[i * dim + i] = 1.0;
  }

  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      matrix[i * dim + j] += strength * categoryDirection[i] * categoryDirection[j];
    }
    bias[i] = strength * 0.1 * categoryDirection[i];
  }

  return { matrix, bias };
}

// Re-export cosineSimilarity for external consumers
export { cosineSimilarity };

export function projectTo2D(
  vec: Float32Array,
  basis1: Float32Array,
  basis2: Float32Array,
): { x: number; y: number } {
  let x = 0, y = 0;
  for (let i = 0; i < vec.length; i++) {
    x += vec[i] * basis1[i];
    y += vec[i] * basis2[i];
  }
  return { x, y };
}

export function computeDynamicBasis(vectors: Float32Array[], dim: number, iters: number = 10): { b1: Float32Array, b2: Float32Array } {
  if (vectors.length === 0) {
    const b1 = new Float32Array(dim); b1[0] = 1;
    const b2 = new Float32Array(dim); b2[1] = 1;
    return { b1, b2 };
  }

  const n = vectors.length;
  const mean = new Float32Array(dim);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < dim; d++) mean[d] += vectors[i][d];
  }
  for (let d = 0; d < dim; d++) mean[d] /= n;

  const centered: Float32Array[] = [];
  for (let i = 0; i < n; i++) {
    const c = new Float32Array(dim);
    for (let d = 0; d < dim; d++) c[d] = vectors[i][d] - mean[d];
    centered.push(c);
  }

  const powerIteration = (data: Float32Array[], excludeBasis?: Float32Array) => {
    const b = new Float32Array(dim);
    for (let d = 0; d < dim; d++) b[d] = Math.random() - 0.5;
    
    for (let iter = 0; iter < iters; iter++) {
      const nextB = new Float32Array(dim);
      for (let i = 0; i < n; i++) {
        let dot = 0;
        for (let d = 0; d < dim; d++) dot += data[i][d] * b[d];
        for (let d = 0; d < dim; d++) nextB[d] += data[i][d] * dot;
      }
      
      if (excludeBasis) {
        let proj = 0;
        for (let d = 0; d < dim; d++) proj += nextB[d] * excludeBasis[d];
        for (let d = 0; d < dim; d++) nextB[d] -= proj * excludeBasis[d];
      }

      let norm = 0;
      for (let d = 0; d < dim; d++) norm += nextB[d] * nextB[d];
      norm = Math.sqrt(norm);
      if (norm > 0) {
        for (let d = 0; d < dim; d++) b[d] = nextB[d] / norm;
      }
    }
    return b;
  };

  const b1 = powerIteration(centered);
  const b2 = powerIteration(centered, b1);

  return { b1, b2 };
}

// Optimization Helpers (Decode quantized vectors to float for 2D plotting & cosine sim)
function decodeInt8(arr: Int8Array, dim: number): Float32Array {
  const res = new Float32Array(dim);
  for (let i = 0; i < dim; i++) res[i] = arr[i] / 127.0;
  return res;
}
function decodeBinary(arr: Uint8Array, dim: number): Float32Array {
  const res = new Float32Array(dim);
  for (let i = 0, bi = 0; i < dim; i += 8, bi++) {
    const byte = arr[bi];
    res[i]   = (byte & 128) ? 1 : -1;
    res[i+1] = (byte & 64) ? 1 : -1;
    res[i+2] = (byte & 32) ? 1 : -1;
    res[i+3] = (byte & 16) ? 1 : -1;
    res[i+4] = (byte & 8) ? 1 : -1;
    res[i+5] = (byte & 4) ? 1 : -1;
    res[i+6] = (byte & 2) ? 1 : -1;
    res[i+7] = (byte & 1) ? 1 : -1;
  }
  return res;
}

export interface DocPoint {
  id: number;
  name: string;
  category: string;
  color: string;
  text: string;
  baseVector: Float32Array;
  currentVector: Float32Array;
  pos: { x: number; y: number };
}

export interface DemoState {
  docs: DocPoint[];
  query: { text: string; baseVector: Float32Array; currentVector: Float32Array; pos: { x: number; y: number } };
  adapter: IntentAdapter;
  whiteningAdapter: WhiteningAdapter;
  useWhitening: boolean;
  quantMode: 'none' | 'int8' | 'binary';
  basis1: Float32Array;
  basis2: Float32Array;
  wasmReady: boolean;
  intentsList: { key: string; name: string; desc: string; icon: string; color: string }[];
  lastMergedWeights?: { matrix: number[]; bias: number[] } | null;
}

export interface RankedDoc extends DocPoint {
  score: number;
}

export interface BenchmarkResult {
  batchSize: number;
  batchMs: number;
  individualMs: number;
  batchOpsPerSec: number;
  individualOpsPerSec: number;
  speedup: number;
}

// Greatly Expanded Sample Dataset
const INITIAL_DATA: Record<'en' | 'ja', SourceData> = {
  en: {
    tech: [
      { name: "TypeScript WASM Guide", text: "A comprehensive guide on running WebAssembly in TypeScript applications for high performance computing." },
      { name: "React Performance Tips", text: "Techniques for optimizing React rendering, including useMemo, useCallback, and virtual DOM strategies." },
      { name: "Edge Computing Patterns", text: "Architectural patterns for deploying serverless functions to the edge for low latency global applications." },
      { name: "Vector Database Internals", text: "Deep dive into HNSW indexes, cosine similarity, and distributed architectures of modern vector databases." },
      { name: "Rust Memory Safety", text: "Understanding Rust's ownership model and borrow checker to prevent memory leaks in systems programming." },
      { name: "GraphQL API Design", text: "Best practices for designing GraphQL schemas, handling resolvers, and managing query depth limits." },
      { name: "Kubernetes Orchestration", text: "Deploying microservices using Kubernetes pods, services, and ingress controllers in a multi-cloud environment." },
      { name: "WebGL Shaders", text: "Writing custom GLSL fragment and vertex shaders for hardware-accelerated 3D rendering in the browser." },
      { name: "Node.js Event Loop", text: "Deep architectural overview of the libuv event loop and asynchronous non-blocking I/O in Node.js." }
    ],
    business: [
      { name: "Q3 Revenue Analysis", text: "Financial breakdown of Q3 performance, highlighting key growth areas and operational cost reductions." },
      { name: "Market Entry Strategy", text: "Strategic framework for entering new emerging markets, including competitor analysis and pricing models." },
      { name: "Enterprise SaaS Pricing", text: "How to structure tiered pricing, usage-based billing, and enterprise negotiation tactics for SaaS products." },
      { name: "Startup Fundraising", text: "A guide to series A fundraising, creating pitch decks, and evaluating term sheets from venture capitalists." },
      { name: "B2B Lead Generation", text: "Inbound marketing strategies for capturing high-intent B2B leads through whitepapers and webinars." },
      { name: "Supply Chain Optimization", text: "Reducing logistics overhead by analyzing procurement data and optimizing warehouse distribution networks." },
      { name: "Employee Retention", text: "Human resources tactics for improving employee engagement and reducing churn in corporate environments." },
      { name: "Mergers and Acquisitions", text: "Evaluating M&A opportunities, due diligence processes, and post-merger cultural integration." },
      { name: "Product-Led Growth", text: "Using the product itself as the primary driver of customer acquisition, retention, and expansion." }
    ],
    medical: [
      { name: "Clinical Trial Design", text: "Methodology for designing phase III clinical trials, including patient cohort selection and statistical significance." },
      { name: "Drug Interaction DB", text: "A database outlining adverse effects and pharmacokinetic interactions between common prescription medications." },
      { name: "Patient Data Privacy", text: "Compliance guidelines for handling electronic health records (EHR) under HIPAA and GDPR regulations." },
      { name: "Genomics Pipeline", text: "Bioinformatics pipelines for analyzing next-generation sequencing data and identifying genetic variants." },
      { name: "Immunotherapy Advances", text: "Recent breakthroughs in CAR T-cell therapy and immune checkpoint inhibitors for cancer treatment." },
      { name: "Telemedicine Protocols", text: "Standardized protocols for conducting remote diagnostic consultations and digital prescription management." },
      { name: "Neurological Biomarkers", text: "Identifying early-stage biomarkers for Alzheimer's disease using cerebrospinal fluid analysis and MRI imaging." },
      { name: "Epidemiological Modeling", text: "Using SIR models to predict infectious disease spread and evaluate public health interventions." },
      { name: "Surgical Robotics", text: "Enhancing precision in minimally invasive surgeries through AI-assisted robotic manipulation systems." }
    ],
    general: [
      { name: "Machine Learning Basics", text: "Introduction to machine learning concepts: supervised learning, neural networks, and loss functions." },
      { name: "Data Privacy Compliance", text: "General overview of data protection laws and how companies must secure personally identifiable information." },
      { name: "Cloud Infrastructure", text: "Comparing AWS, Google Cloud, and Azure for hosting scalable web applications and databases." },
      { name: "Remote Work Productivity", text: "Tips and tools for maintaining high productivity and communication while working on distributed remote teams." },
      { name: "History of the Internet", text: "A brief history of the ARPANET, TCP/IP development, and the invention of the World Wide Web." },
      { name: "Sustainable Energy", text: "An overview of renewable energy sources including solar, wind, and the future of grid-scale battery storage." }
    ]
  },
  ja: {
    tech: [
      { name: "TypeScript WASM ガイド", text: "TypeScriptアプリケーションでハイパフォーマンスコンピューティングのためにWebAssemblyを実行する包括的ガイド。" },
      { name: "React パフォーマンス最適化", text: "useMemo, useCallback, 仮想DOM戦略を含むReactレンダリング最適化のテクニック。" },
      { name: "エッジコンピューティング設計", text: "低遅延のグローバルアプリケーションのためのエッジへのサーバーレス関数デプロイメントのアーキテクチャパターン。" },
      { name: "ベクトルDBの内部構造", text: "最新のベクトルデータベースのHNSWインデックス、コサイン類似度、分散アーキテクチャの徹底解説。" },
      { name: "Rustのメモリ安全性", text: "システムプログラミングにおけるメモリリークを防ぐためのRustの所有権モデルとボローチェッカーの理解。" },
      { name: "GraphQL API 設計", text: "GraphQLスキーマの設計、リゾルバの処理、クエリの深さ制限の管理に関するベストプラクティス。" },
      { name: "Kubernetes オーケストレーション", text: "マルチクラウド環境でのPod、Service、Ingress Controllerを使用したマイクロサービスのデプロイ。" },
      { name: "WebGL シェーダー入門", text: "ブラウザでのハードウェアアクセラレーションによる3DレンダリングのためのカスタムGLSLフラグメント/頂点シェーダーの記述。" },
      { name: "Node.js イベントループ", text: "Node.jsにおけるlibuvイベントループと非同期ノンブロッキングI/Oの深いアーキテクチャ概要。" }
    ],
    business: [
      { name: "Q3 収益分析レポート", text: "第3四半期の業績の財務的内訳、主要な成長分野と運営コスト削減のハイライト。" },
      { name: "市場参入戦略", text: "競合分析と価格モデルを含む、新興市場へ参入するための戦略的フレームワーク。" },
      { name: "エンタープライズ SaaS 価格設計", text: "SaaS製品における段階的価格設定、従量課金、エンタープライズ交渉戦術の構築方法。" },
      { name: "スタートアップ資金調達", text: "シリーズA資金調達、ピッチデック作成、VCからのタームシート評価のガイド。" },
      { name: "B2B リードジェネレーション", text: "ホワイトペーパーやウェビナーを通じて購買意欲の高いB2Bリードを獲得するためのインバウンドマーケティング戦略。" },
      { name: "サプライチェーン最適化", text: "調達データの分析と倉庫配送ネットワークの最適化による物流オーバーヘッドの削減。" },
      { name: "従業員リテンション", text: "企業環境における従業員のエンゲージメントを向上させ、離職率を低下させるための人事戦術。" },
      { name: "M&A（企業の合併・買収）", text: "M&Aの機会の評価、デューデリジェンスのプロセス、および合併後の文化的統合。" },
      { name: "プロダクト・レッド・グロース (PLG)", text: "顧客獲得、維持、拡大の主要な推進力としてプロダクトそのものを活用する戦略。" }
    ],
    medical: [
      { name: "臨床試験デザイン", text: "患者コホート選択と統計的有意性を含む、第III相臨床試験を設計するための方法論。" },
      { name: "薬物相互作用データベース", text: "一般的な処方薬間の副作用と薬物動態学的相互作用のデータベース。" },
      { name: "患者データプライバシー", text: "HIPAAおよびGDPR規制下での電子健康記録（EHR）取り扱いのためのコンプライアンスガイドライン。" },
      { name: "ゲノミクスパイプライン", text: "次世代シーケンシングデータを分析し、遺伝的変異を特定するためのバイオインフォマティクスパイプライン。" },
      { name: "免疫療法の進歩", text: "がん治療のためのCAR T細胞療法と免疫チェックポイント阻害剤における最近の画期的な進歩。" },
      { name: "遠隔医療プロトコル", text: "遠隔診断コンサルテーションとデジタル処方管理を実施するための標準化されたプロトコル。" },
      { name: "神経学的バイオマーカー", text: "脳脊髄液分析とMRI画像を用いたアルツハイマー病の早期バイオマーカーの特定。" },
      { name: "疫学モデリング", text: "感染症の拡大を予測し、公衆衛生の介入を評価するためのSIRモデルの使用。" },
      { name: "手術用ロボティクス", text: "AI支援のロボットマニピュレーションシステムによる低侵襲手術の精度向上。" }
    ],
    general: [
      { name: "機械学習入門", text: "教師あり学習、ニューラルネットワーク、損失関数などの機械学習の概念の入門。" },
      { name: "データプライバシー規制", text: "データ保護法の一般的な概要と、企業が個人を特定できる情報を保護する方法。" },
      { name: "クラウドインフラ構築", text: "スケーラブルなWebアプリケーションとデータベースをホストするためのAWS、Google Cloud、Azureの比較。" },
      { name: "リモートワーク生産性", text: "分散型リモートチームで働きながら高い生産性とコミュニケーションを維持するためのヒントとツール。" },
      { name: "インターネットの歴史", text: "ARPANET、TCP/IPの開発、およびWorld Wide Webの発明に関する短い歴史。" },
      { name: "持続可能なエネルギー", text: "太陽光、風力、およびグリッドスケールのバッテリーストレージの未来を含む再生可能エネルギー源の概要。" }
    ]
  }
};

export type DataCategory = 'tech' | 'business' | 'medical' | 'general';
export interface DataItem { name: string; text: string; }
export type SourceData = Record<DataCategory, DataItem[]>;

export interface CategoryMeta {
  icon: string;
  dotColor: string;
  bgColor: string;
  label: Record<'en' | 'ja', string>;
}

export const CATEGORY_META: Record<DataCategory, CategoryMeta> = {
  tech:     { icon: '💻', dotColor: '#3b82f6', bgColor: 'rgba(59,130,246,0.15)',  label: { en: 'Technology', ja: 'テクノロジー' } },
  business: { icon: '📊', dotColor: '#10b981', bgColor: 'rgba(16,185,129,0.15)',  label: { en: 'Business',   ja: 'ビジネス' } },
  medical:  { icon: '🏥', dotColor: '#f43f5e', bgColor: 'rgba(244,63,94,0.15)',   label: { en: 'Medical',    ja: '医療' } },
  general:  { icon: '📌', dotColor: '#94a3b8', bgColor: 'rgba(148,163,184,0.15)', label: { en: 'General',    ja: '一般' } },
};

const INTENT_STRENGTH = 1.2;

/** Maps intent keys to DataCategory for deriving icon/color from CATEGORY_META */
const INTENT_CATEGORY_MAP: Record<string, DataCategory> = {
  technology: 'tech',
  business: 'business',
  medical: 'medical',
};

interface IntentDefinition {
  key: string;
  name: string;
  desc: string;
  text: string;
}

const DEFAULT_INTENTS: Record<'en' | 'ja', IntentDefinition[]> = {
  en: [
    { key: 'technology', name: 'Technology', desc: 'Technical & programming focus', text: "I want highly technical documentation, software engineering concepts, coding techniques, and system architecture." },
    { key: 'business', name: 'Business', desc: 'Business & finance focus', text: "I am interested in business strategy, financial analysis, startup funding, and enterprise sales tactics." },
    { key: 'medical', name: 'Medical', desc: 'Healthcare & science focus', text: "Show me medical research, clinical trials, healthcare data compliance, and genomics information." }
  ],
  ja: [
    { key: 'technology', name: 'テクノロジー', desc: '技術とプログラミング重視', text: "高度な技術ドキュメント、ソフトウェアエンジニアリングの概念、コーディング手法、システムアーキテクチャが欲しいです。" },
    { key: 'business', name: 'ビジネス', desc: 'ビジネスと金融重視', text: "ビジネス戦略、財務分析、スタートアップ資金調達、エンタープライズ営業戦術に興味があります。" },
    { key: 'medical', name: '医療', desc: 'ヘルスケアと科学重視', text: "医学研究、臨床試験、ヘルスケアデータコンプライアンス、ゲノミクス情報を見せてください。" }
  ]
};

export async function createDemoState(
  lang: 'en' | 'ja', 
  onProgress?: ProgressCallback
): Promise<DemoState> {
  const wasmResult = await initWasm();
  const wasmReady = wasmResult !== null;

  if (onProgress) onProgress('init_model');
  await preloadModel(onProgress);

  const allDocs: { category: string, name: string, text: string, color: string }[] = [];
  const sourceData = INITIAL_DATA[lang];
  const dataCategories: DataCategory[] = ['tech', 'business', 'medical', 'general'];
  for (const cat of dataCategories) {
    if (!(cat in sourceData)) continue;
    for (const item of sourceData[cat]) {
      allDocs.push({ category: cat, name: item.name, text: item.text, color: CATEGORY_META[cat].dotColor });
    }
  }

  const queryText = lang === 'en' ? "Apple" : "機械学習とデータ分析"; 
  
  const defaultIntents = DEFAULT_INTENTS[lang];
  const intentTexts = defaultIntents.map(i => i.text);

  const allTextsToEmbed = [
    ...allDocs.map(d => d.text),
    queryText,
    ...intentTexts
  ];

  if (onProgress) onProgress('embedding');
  const embeddings = await getEmbeddings(allTextsToEmbed);

  const docs: DocPoint[] = [];
  let eIdx = 0;
  for (let i = 0; i < allDocs.length; i++) {
    const emb = embeddings[eIdx++];
    docs.push({
      id: i,
      name: allDocs[i].name,
      category: allDocs[i].category,
      color: allDocs[i].color,
      text: allDocs[i].text,
      baseVector: emb,
      currentVector: new Float32Array(emb),
      pos: { x: 0, y: 0 }
    });
  }

  const queryVec = embeddings[eIdx++];
  
  const adapter = new IntentAdapter(DIM);
  const intentsList: DemoState['intentsList'] = [];

  for (let i = 0; i < defaultIntents.length; i++) {
    const intentEmb = embeddings[eIdx++];
    const { matrix, bias } = generateIntentMatrix(DIM, intentEmb, INTENT_STRENGTH);
    adapter.addIntent(defaultIntents[i].key, { matrix, bias });
    const catKey = INTENT_CATEGORY_MAP[defaultIntents[i].key] ?? 'general';
    const meta = CATEGORY_META[catKey];
    intentsList.push({
      key: defaultIntents[i].key,
      name: defaultIntents[i].name,
      desc: defaultIntents[i].desc,
      icon: meta.icon,
      color: meta.bgColor
    });
  }

  // Initialize Whitening Adapter and learn from base vectors
  const whiteningAdapter = new WhiteningAdapter(DIM, { learningRate: 0.1, numComponents: 2 });
  // Oja's rule converges better with multiple passes over the dataset if the set is static
  for (let epoch = 0; epoch < 3; epoch++) {
    for (const doc of docs) {
      whiteningAdapter.update(doc.baseVector); // Online learning for anisotropy removal
    }
  }

  const allCurrentVecs = [...docs.map(d => d.currentVector), queryVec];
  const { b1, b2 } = computeDynamicBasis(allCurrentVecs, DIM);

  for (const doc of docs) {
    doc.pos = projectTo2D(doc.baseVector, b1, b2);
  }
  const queryPos = projectTo2D(queryVec, b1, b2);

  return {
    docs,
    query: { text: queryText, baseVector: queryVec, currentVector: new Float32Array(queryVec), pos: queryPos },
    adapter,
    whiteningAdapter,
    useWhitening: false,
    quantMode: 'none',
    basis1: b1,
    basis2: b2,
    wasmReady,
    intentsList
  };
}

export async function updateQuery(state: DemoState, text: string) {
  const [newVec] = await getEmbeddings([text]);
  state.query.text = text;
  state.query.baseVector = newVec;
  state.query.currentVector = new Float32Array(newVec);
}

export async function addCustomIntent(
  state: DemoState, 
  name: string, 
  text: string, 
  icon: string, 
  color: string
) {
  const [intentVec] = await getEmbeddings([text]);
  const key = 'custom_' + Date.now();
  const { matrix, bias } = generateIntentMatrix(DIM, intentVec, INTENT_STRENGTH);
  state.adapter.addIntent(key, { matrix, bias });
  
  state.intentsList.push({
    key,
    name,
    desc: 'Custom intent',
    icon,
    color
  });
  
  return key;
}

// Helper to apply optimizations pipeline
function applyPipeline(state: DemoState, baseVector: Float32Array, applyIntent: (v: Float32Array) => Float32Array, quantizer?: any): Float32Array {
  let v: Float32Array = new Float32Array(baseVector);
  
  // 1. Whitening (De-bias the space)
  if (state.useWhitening) {
    v = state.whiteningAdapter.tune(v);
  }

  // 2. Intent (Affine transform)
  v = applyIntent(v);

  // 3. Quantization (Compression)
  if (state.quantMode !== 'none' && quantizer) {
    const quantized = quantizer.encode(v);
    if (state.quantMode === 'int8' && quantized instanceof Int8Array) {
      v = decodeInt8(quantized, DIM);
    } else if (state.quantMode === 'binary' && quantized instanceof Uint8Array) {
      v = decodeBinary(quantized, DIM);
    }
  }

  return v;
}

function getExpectedCategory(intentKey: string): string | null {
  if (intentKey.includes('technology') || intentKey.includes('tech')) return 'tech';
  if (intentKey.includes('business')) return 'business';
  if (intentKey.includes('medical')) return 'medical';
  return null;
}

export interface EvalMetrics {
  ndcg3: number;
  recall3: number;
  expectedCategory: string | null;
}

function computeEvalMetrics(state: DemoState, rankings: RankedDoc[], activeIntentKey: string | null): EvalMetrics {
  if (!activeIntentKey || activeIntentKey === 'none') {
    return { ndcg3: 0, recall3: 0, expectedCategory: null };
  }
  const expectedCategory = getExpectedCategory(activeIntentKey);
  if (!expectedCategory) {
    return { ndcg3: 0, recall3: 0, expectedCategory: null };
  }

  const expectedIds = state.docs
    .filter(d => d.category === expectedCategory)
    .map(d => String(d.id));
  const retrievedIds = rankings.slice(0, 3).map(r => String(r.id));

  const ndcg3 = calculateNDCG(retrievedIds, expectedIds, 3);
  const recall3 = calculateRecall(retrievedIds, expectedIds, 3);

  return { ndcg3, recall3, expectedCategory };
}

/** Rank documents by cosine similarity to a query vector */
function rankByCosineSim(docs: DocPoint[], queryVec: Float32Array, useBase = false): RankedDoc[] {
  return docs.map((doc) => ({
    ...doc,
    score: cosineSimilarity(useBase ? doc.baseVector : doc.currentVector, queryVec),
  })).sort((a, b) => b.score - a.score);
}

/** Apply the optimization pipeline to all docs/query, update positions, and compute rankings */
function applyAndRank(
  state: DemoState,
  intentFunc: (v: Float32Array) => Float32Array,
): { latencyMs: number; rankings: RankedDoc[]; vanillaRankings: RankedDoc[] } {
  const t0 = performance.now();

  const quantizer = state.quantMode !== 'none' ? new QuantizationAdapter({ type: state.quantMode, dim: DIM }) : undefined;

  for (const doc of state.docs) {
    doc.currentVector = applyPipeline(state, doc.baseVector, intentFunc, quantizer);
  }
  state.query.currentVector = applyPipeline(state, state.query.baseVector, intentFunc, quantizer);

  for (const doc of state.docs) {
    doc.pos = projectTo2D(doc.currentVector, state.basis1, state.basis2);
  }
  state.query.pos = projectTo2D(state.query.currentVector, state.basis1, state.basis2);

  const latencyMs = performance.now() - t0;
  const rankings = rankByCosineSim(state.docs, state.query.currentVector);
  const vanillaRankings = rankByCosineSim(state.docs, state.query.baseVector, true);

  return { latencyMs, rankings, vanillaRankings };
}

export function transformWithIntent(
  state: DemoState,
  intent: string | null,
): { latencyMs: number; rankings: RankedDoc[]; vanillaRankings: RankedDoc[]; metrics: EvalMetrics } {
  const isActive = intent != null && intent !== 'none';
  const intentFunc = (v: Float32Array) => isActive ? state.adapter.tune(v, intent) : v;
  const { latencyMs, rankings, vanillaRankings } = applyAndRank(state, intentFunc);

  if (isActive) {
    try {
      const parsed = JSON.parse(state.adapter.exportState());
      state.lastMergedWeights = parsed.intents[intent] || null;
    } catch (e) {
      console.error(e);
      state.lastMergedWeights = null;
    }
  } else {
    state.lastMergedWeights = null;
  }

  const metrics = computeEvalMetrics(state, rankings, intent);
  return { latencyMs, rankings, vanillaRankings, metrics };
}

export function transformWithBlend(
  state: DemoState,
  weights: Record<string, number>,
): { latencyMs: number; rankings: RankedDoc[]; vanillaRankings: RankedDoc[]; codeSnippet: string; metrics: EvalMetrics } {
  const activeWeights: Record<string, number> = {};
  for (const [k, v] of Object.entries(weights)) {
    if (v > 0.01) activeWeights[k] = v;
  }
  const hasActive = Object.keys(activeWeights).length > 0;

  // Merge intent weights via TaskArithmetic
  if (hasActive) {
    try {
      const parsed = JSON.parse(state.adapter.exportState());
      const tasks = Object.entries(activeWeights).map(([key, scale]) => {
        const intentW = parsed.intents[key];
        if (!intentW) throw new Error(`Intent ${key} not found for merging`);
        return {
          weights: { matrix: new Float32Array(intentW.matrix), bias: new Float32Array(intentW.bias) },
          scale
        };
      });
      const merged = TaskArithmetic.merge(tasks);
      state.lastMergedWeights = {
        matrix: merged.matrix instanceof Float32Array ? Array.from(merged.matrix) : [],
        bias: merged.bias instanceof Float32Array ? Array.from(merged.bias) : []
      };
      state.adapter.addIntent("__merged__", merged);
    } catch (e) {
      console.error("TaskArithmetic merge failed:", e);
      state.lastMergedWeights = null;
    }
  } else {
    state.lastMergedWeights = null;
  }

  const intentFunc = (v: Float32Array) => hasActive ? state.adapter.tune(v, "__merged__") : v;
  const { latencyMs, rankings, vanillaRankings } = applyAndRank(state, intentFunc);

  // Clean up temporary merged intent
  if (hasActive) {
    try { state.adapter.removeIntent("__merged__"); } catch (e) { console.error(e); }
  }

  // Evaluate against the highest-weighted intent
  let topIntentKey: string | null = null;
  let maxWeight = 0;
  for (const [k, v] of Object.entries(activeWeights)) {
    if (v > maxWeight) { maxWeight = v; topIntentKey = k; }
  }
  const metrics = computeEvalMetrics(state, rankings, topIntentKey);

  // Generate code snippet
  const weightsStr = Object.entries(activeWeights)
    .map(([k, v]) => `  { weights: ${k}Weights, scale: ${v.toFixed(2)} }`)
    .join(',\n');
  const codeSnippet = hasActive
    ? `// Task Arithmetic (Model Merging)\nconst mergedWeights = TaskArithmetic.merge([\n${weightsStr}\n]);\n\nadapter.addIntent("merged", mergedWeights);\nconst warped = adapter.tune(baseVector, "merged");`
    : `// No intent applied\nconst result = baseVector;`;

  return { latencyMs, rankings, vanillaRankings, codeSnippet, metrics };
}

export function runBenchmark(
  state: DemoState,
  batchSize: number = 1000,
): BenchmarkResult {
  const vectors: Float32Array[] = [];
  for (let i = 0; i < batchSize; i++) {
    const v = new Float32Array(DIM);
    for (let j = 0; j < DIM; j++) v[j] = Math.random() * 2 - 1;
    vectors.push(v);
  }

  const benchIntent = state.intentsList[0].key;

  const t0 = performance.now();
  state.adapter.tuneBatch(vectors, benchIntent);
  const batchMs = performance.now() - t0;

  const t1 = performance.now();
  for (let i = 0; i < batchSize; i++) {
    state.adapter.tune(vectors[i], benchIntent);
  }
  const individualMs = performance.now() - t1;

  return {
    batchSize,
    batchMs,
    individualMs,
    batchOpsPerSec: Math.round(batchSize / (batchMs / 1000)),
    individualOpsPerSec: Math.round(batchSize / (individualMs / 1000)),
    speedup: individualMs / batchMs,
  };
}

/**
 * IntentMatrixFactory を使った自動Intent学習
 *
 * ドキュメントのカテゴリ別ベクトルから InfoNCE 対照学習で
 * 最適なIntent行列を自動生成し、既存のアダプタに適用します。
 */
export interface AutoLearnResult {
  categories: string[];
  trainingTimeMs: number;
  improved: boolean;
}

export async function autoLearnIntents(
  state: DemoState,
): Promise<AutoLearnResult> {
  // Dynamic import to keep the initial bundle lean
  const { IntentMatrixFactory } = await import('@warpvector/train');

  const t0 = performance.now();
  const factory = new IntentMatrixFactory(DIM);

  // Group document vectors by category
  const categories = new Set<string>();
  for (const doc of state.docs) {
    categories.add(doc.category);
  }

  for (const cat of categories) {
    const vecs = state.docs
      .filter(d => d.category === cat)
      .map(d => d.baseVector);
    if (vecs.length >= 2) {
      factory.addCategory(cat, vecs);
    }
  }

  // Train with InfoNCE
  const intents = await factory.build({
    training: { epochs: 80, learningRate: 0.01, patience: 8 },
  });

  const trainingTimeMs = performance.now() - t0;

  // Replace adapter intents with learned ones
  const learnedCategories: string[] = [];

  for (const cat of categories) {
    if (intents[cat]) {
      state.adapter.addIntent('auto_' + cat, intents[cat]);
      learnedCategories.push(cat);

      // Check if already in intentsList, replace if so
      const existingIdx = state.intentsList.findIndex(
        i => i.key === 'auto_' + cat,
      );
      const intentMeta = {
        key: 'auto_' + cat,
        name: `🤖 ${cat.charAt(0).toUpperCase() + cat.slice(1)}`,
        desc: 'Auto-learned (InfoNCE)',
        icon: (CATEGORY_META[cat as DataCategory] || CATEGORY_META.general).icon,
        color: (CATEGORY_META[cat as DataCategory] || CATEGORY_META.general).bgColor,
      };

      if (existingIdx >= 0) {
        state.intentsList[existingIdx] = intentMeta;
      } else {
        state.intentsList.push(intentMeta);
      }
    }
  }

  return {
    categories: learnedCategories,
    trainingTimeMs,
    improved: learnedCategories.length > 0,
  };
}

// --- Federated Learning / Local Learning Actions ---

export async function handleFeedbackAction(state: DemoState, docId: string, isGood: boolean) {
  const queryVec = state.query.baseVector;
  const targetDoc = state.docs.find(d => String(d.id) === docId);
  if (!targetDoc) return;
  
  // addInteraction(anchor, positive, negative)
  // 簡易デモのため、Goodならpositiveに、Badならnegativeにセットし、片方はゼロベクトルで埋める
  const zeroVec = new Float32Array(DIM);
  const impressionId = demoFeedbackCollector.recordImpression({
    queryVector: queryVec,
    resultVectors: isGood ? [targetDoc.baseVector, zeroVec] : [zeroVec, targetDoc.baseVector],
    timestamp: Date.now(),
  });
  demoFeedbackCollector.recordFeedback({
    impressionId,
    resultIndex: 0,
    type: "click",
  });
  
  const localEdge = fedEdges.find(e => e.id === 'local');
  if (localEdge) {
    localEdge.collectedFeedback++;
    if (localEdge.collectedFeedback >= 3) {
      localEdge.hasLocalUpdate = true;
    }
  }
}

export async function runFederatedAggregation(state: DemoState) {
  // モック: 各エッジの学習済み行列を統合
  for (const edge of fedEdges) {
    if (edge.hasLocalUpdate) {
      // 擬似的な学習済み行列(恒等行列ベースの微小変化)を追加
      const mat = new Float32Array(DIM * DIM);
      for(let i=0; i<DIM; i++) mat[i*DIM + i] = 1.0 + (Math.random() * 0.1 - 0.05);
      demoFederatedAggregator.submitUpdate({
        weights: { matrix: mat, bias: new Float32Array(DIM) },
        interactionCount: edge.collectedFeedback
      });
      
      // 送信後はリセット
      edge.hasLocalUpdate = false;
      edge.collectedFeedback = 0;
    }
  }
  
  const globalMatrix = demoFederatedAggregator.aggregate();
  
  // グローバル行列を `federated` インテントとして登録
  state.intentsList.push({
    key: 'federated',
    name: 'Federated Global',
    desc: 'Aggregated from all edges',
    icon: '🌐',
    color: '#0ea5e9'
  });
  state.adapter.addIntent('federated', globalMatrix);
  
  return globalMatrix;
}

