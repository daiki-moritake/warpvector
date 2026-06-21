/**
 * WarpVector Playground — Real Library Demo Engine
 *
 * This module uses the ACTUAL @warpvector/core library to perform
 * intent-based vector transformations.
 * Enhanced with Real LLM Embeddings via Transformers.js and Dynamic PCA Projection.
 */
import { IntentAdapter, initWasm } from '@warpvector/core';

// all-MiniLM-L6-v2 uses 384 dimensions
export const DIM = 384;

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

  // Identity matrix
  for (let i = 0; i < dim; i++) {
    matrix[i * dim + i] = 1.0;
  }

  // Add outer product: strength * direction ⊗ direction
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      matrix[i * dim + j] += strength * categoryDirection[i] * categoryDirection[j];
    }
    bias[i] = strength * 0.1 * categoryDirection[i]; // Small bias in the direction
  }

  return { matrix, bias };
}

/** Cosine similarity between two vectors */
export function cosineSim(a: Float32Array, b: Float32Array): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/** Project high-dimensional vector to 2D using two basis vectors */
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

/**
 * Compute the top 2 principal components (Basis vectors) of a set of vectors
 * using Power Iteration (NIPALS-like approach) for fast, lightweight PCA in JS.
 */
export function computeDynamicBasis(vectors: Float32Array[], dim: number, iters: number = 10): { b1: Float32Array, b2: Float32Array } {
  if (vectors.length === 0) {
    const b1 = new Float32Array(dim); b1[0] = 1;
    const b2 = new Float32Array(dim); b2[1] = 1;
    return { b1, b2 };
  }

  const n = vectors.length;
  // 1. Calculate mean vector
  const mean = new Float32Array(dim);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < dim; d++) mean[d] += vectors[i][d];
  }
  for (let d = 0; d < dim; d++) mean[d] /= n;

  // 2. Center the vectors
  const centered: Float32Array[] = [];
  for (let i = 0; i < n; i++) {
    const c = new Float32Array(dim);
    for (let d = 0; d < dim; d++) c[d] = vectors[i][d] - mean[d];
    centered.push(c);
  }

  // Helper for power iteration
  const powerIteration = (data: Float32Array[], excludeBasis?: Float32Array) => {
    let b = new Float32Array(dim);
    // Init with random
    for (let d = 0; d < dim; d++) b[d] = Math.random() - 0.5;
    
    for (let iter = 0; iter < iters; iter++) {
      const nextB = new Float32Array(dim);
      for (let i = 0; i < n; i++) {
        let dot = 0;
        for (let d = 0; d < dim; d++) dot += data[i][d] * b[d];
        for (let d = 0; d < dim; d++) nextB[d] += data[i][d] * dot;
      }
      
      // Orthogonalize against excludeBasis if provided (Gram-Schmidt)
      if (excludeBasis) {
        let proj = 0;
        for (let d = 0; d < dim; d++) proj += nextB[d] * excludeBasis[d];
        for (let d = 0; d < dim; d++) nextB[d] -= proj * excludeBasis[d];
      }

      // Normalize
      let norm = 0;
      for (let d = 0; d < dim; d++) norm += nextB[d] * nextB[d];
      norm = Math.sqrt(norm);
      if (norm > 0) {
        for (let d = 0; d < dim; d++) b[d] = nextB[d] / norm;
      }
    }
    return b;
  };

  // 1st Principal Component
  const b1 = powerIteration(centered);
  
  // 2nd Principal Component (Orthogonal to b1)
  const b2 = powerIteration(centered, b1);

  return { b1, b2 };
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
  query: { text: string; vector: Float32Array; pos: { x: number; y: number } };
  adapter: IntentAdapter;
  basis1: Float32Array;
  basis2: Float32Array;
  wasmReady: boolean;
  intentsList: { key: string; name: string; desc: string; icon: string; color: string }[];
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
const INITIAL_DATA = {
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

const CATEGORY_COLORS = {
  tech: '#3b82f6',
  business: '#10b981',
  medical: '#f43f5e',
  general: '#94a3b8'
};

const DEFAULT_INTENTS = {
  en: [
    { key: 'technology', name: 'Technology', desc: 'Technical & programming focus', icon: '💻', color: 'rgba(59,130,246,0.15)', text: "I want highly technical documentation, software engineering concepts, coding techniques, and system architecture." },
    { key: 'business', name: 'Business', desc: 'Business & finance focus', icon: '📊', color: 'rgba(16,185,129,0.15)', text: "I am interested in business strategy, financial analysis, startup funding, and enterprise sales tactics." },
    { key: 'medical', name: 'Medical', desc: 'Healthcare & science focus', icon: '🏥', color: 'rgba(244,63,94,0.15)', text: "Show me medical research, clinical trials, healthcare data compliance, and genomics information." }
  ],
  ja: [
    { key: 'technology', name: 'テクノロジー', desc: '技術とプログラミング重視', icon: '💻', color: 'rgba(59,130,246,0.15)', text: "高度な技術ドキュメント、ソフトウェアエンジニアリングの概念、コーディング手法、システムアーキテクチャが欲しいです。" },
    { key: 'business', name: 'ビジネス', desc: 'ビジネスと金融重視', icon: '📊', color: 'rgba(16,185,129,0.15)', text: "ビジネス戦略、財務分析、スタートアップ資金調達、エンタープライズ営業戦術に興味があります。" },
    { key: 'medical', name: '医療', desc: 'ヘルスケアと科学重視', icon: '🏥', color: 'rgba(244,63,94,0.15)', text: "医学研究、臨床試験、ヘルスケアデータコンプライアンス、ゲノミクス情報を見せてください。" }
  ]
};

/** Create the full demo state with real warpvector IntentAdapter and real embeddings */
export async function createDemoState(
  lang: 'en' | 'ja', 
  onProgress?: ProgressCallback
): Promise<DemoState> {
  // Initialize WASM
  const wasmResult = await initWasm();
  const wasmReady = wasmResult !== null;

  // Load Transformers.js model
  if (onProgress) onProgress('init_model');
  await preloadModel(onProgress);

  // Prepare all texts to embed
  const allDocs: { category: string, name: string, text: string, color: string }[] = [];
  const sourceData = INITIAL_DATA[lang];
  for (const cat of Object.keys(sourceData)) {
    for (const item of sourceData[cat as keyof typeof sourceData]) {
      allDocs.push({ category: cat, name: item.name, text: item.text, color: CATEGORY_COLORS[cat as keyof typeof CATEGORY_COLORS] });
    }
  }

  const queryText = lang === 'en' ? "Apple" : "Apple"; // Just a default initialization
  
  // Also embed the default intent directions
  const defaultIntents = DEFAULT_INTENTS[lang];
  const intentTexts = defaultIntents.map(i => i.text);

  const allTextsToEmbed = [
    ...allDocs.map(d => d.text),
    queryText,
    ...intentTexts
  ];

  if (onProgress) onProgress('embedding');
  const embeddings = await getEmbeddings(allTextsToEmbed);

  // Distribute embeddings back
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
  
  // Initialize IntentAdapter
  const adapter = new IntentAdapter(DIM);
  const intentsList: DemoState['intentsList'] = [];

  for (let i = 0; i < defaultIntents.length; i++) {
    const intentEmb = embeddings[eIdx++];
    const { matrix, bias } = generateIntentMatrix(DIM, intentEmb, 1.2);
    adapter.addIntent(defaultIntents[i].key, { matrix, bias });
    intentsList.push({
      key: defaultIntents[i].key,
      name: defaultIntents[i].name,
      desc: defaultIntents[i].desc,
      icon: defaultIntents[i].icon,
      color: defaultIntents[i].color
    });
  }

  // Define initial 2D projection basis dynamically using PCA on the current documents + query
  const allCurrentVecs = [...docs.map(d => d.currentVector), queryVec];
  const { b1, b2 } = computeDynamicBasis(allCurrentVecs, DIM);

  // Calculate initial positions
  for (const doc of docs) {
    doc.pos = projectTo2D(doc.baseVector, b1, b2);
  }
  const queryPos = projectTo2D(queryVec, b1, b2);

  return {
    docs,
    query: { text: queryText, vector: queryVec, pos: queryPos },
    adapter,
    basis1: b1,
    basis2: b2,
    wasmReady,
    intentsList
  };
}

/** Update the query dynamically */
export async function updateQuery(state: DemoState, text: string) {
  const [newVec] = await getEmbeddings([text]);
  state.query.text = text;
  state.query.vector = newVec;
  // Note: basis will be updated in transform functions
  state.query.pos = projectTo2D(newVec, state.basis1, state.basis2);
}

/** Add a new custom intent dynamically */
export async function addCustomIntent(
  state: DemoState, 
  name: string, 
  text: string, 
  icon: string, 
  color: string
) {
  const [intentVec] = await getEmbeddings([text]);
  const key = 'custom_' + Date.now();
  const { matrix, bias } = generateIntentMatrix(DIM, intentVec, 1.2);
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

/** Transform all document vectors and update dynamic basis */
export function transformWithIntent(
  state: DemoState,
  intent: string | null,
): { latencyMs: number; rankings: RankedDoc[] } {
  const t0 = performance.now();

  for (const doc of state.docs) {
    if (intent && intent !== 'none') {
      doc.currentVector = state.adapter.tune(doc.baseVector, intent);
    } else {
      doc.currentVector = new Float32Array(doc.baseVector);
    }
  }

  // Update Basis Dynamically via PCA
  const allCurrentVecs = [...state.docs.map(d => d.currentVector), state.query.vector];
  const { b1, b2 } = computeDynamicBasis(allCurrentVecs, DIM);
  state.basis1 = b1;
  state.basis2 = b2;

  // Project
  for (const doc of state.docs) {
    doc.pos = projectTo2D(doc.currentVector, state.basis1, state.basis2);
  }
  state.query.pos = projectTo2D(state.query.vector, state.basis1, state.basis2);

  const latencyMs = performance.now() - t0;

  const rankings = state.docs.map((doc) => ({
    ...doc,
    score: cosineSim(doc.currentVector, state.query.vector),
  })).sort((a, b) => b.score - a.score);

  return { latencyMs, rankings };
}

/** Transform using BLENDED intents and update dynamic basis */
export function transformWithBlend(
  state: DemoState,
  weights: Record<string, number>,
): { latencyMs: number; rankings: RankedDoc[]; codeSnippet: string } {
  const activeWeights: Record<string, number> = {};
  for (const [k, v] of Object.entries(weights)) {
    if (v > 0.01) activeWeights[k] = v;
  }

  const hasActive = Object.keys(activeWeights).length > 0;
  const t0 = performance.now();

  for (const doc of state.docs) {
    if (hasActive) {
      doc.currentVector = state.adapter.tuneBlended(doc.baseVector, activeWeights);
    } else {
      doc.currentVector = new Float32Array(doc.baseVector);
    }
  }

  // Update Basis Dynamically via PCA
  const allCurrentVecs = [...state.docs.map(d => d.currentVector), state.query.vector];
  const { b1, b2 } = computeDynamicBasis(allCurrentVecs, DIM);
  state.basis1 = b1;
  state.basis2 = b2;

  // Project
  for (const doc of state.docs) {
    doc.pos = projectTo2D(doc.currentVector, state.basis1, state.basis2);
  }
  state.query.pos = projectTo2D(state.query.vector, state.basis1, state.basis2);

  const latencyMs = performance.now() - t0;

  const rankings = state.docs.map((doc) => ({
    ...doc,
    score: cosineSim(doc.currentVector, state.query.vector),
  })).sort((a, b) => b.score - a.score);

  const weightsStr = Object.entries(activeWeights)
    .map(([k, v]) => `  "${k}": ${v.toFixed(2)}`)
    .join(',\n');
  const codeSnippet = hasActive
    ? `const warped = adapter.tuneBlended(\n  baseVector,\n  {\n${weightsStr}\n  }\n);`
    : `// No intent applied\nconst result = baseVector;`;

  return { latencyMs, rankings, codeSnippet };
}

/** Run batch benchmark */
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
