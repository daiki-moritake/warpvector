/**
 * テンプレート定義の共通インターフェース
 *
 * 新しいテンプレートを追加する場合は、この型に準拠した定義を
 * `src/templates/` 以下に作成し、barrel export (`index.ts`) に追加する。
 */
export interface TemplateDefinition {
  /** テンプレートの一意識別子 (例: 'minimal-intent')。prompts の select value としても使用される */
  id: string;

  /** プロンプトに表示するタイトル（カラー付き文字列） */
  title: string;

  /** プロンプトに表示する説明文 */
  description: string;

  /** プロジェクトファイルを生成する */
  generate: (dir: string, name: string, version: string) => void;

  /**
   * テンプレート固有の「次のステップ」コマンドを返す
   * @param pm - パッケージマネージャー名 (npm | pnpm | yarn | bun)
   */
  getNextSteps: (pm: string) => string[];
}
