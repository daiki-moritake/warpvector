import type { Choice } from 'prompts';

/**
 * テンプレート定義の共通インターフェース
 *
 * 新しいテンプレートを追加する場合は、この型に準拠した定義を
 * `src/templates/` 以下に作成し、`src/prompts.ts` のレジストリに登録する。
 */
export interface TemplateDefinition {
  /** テンプレートの一意識別子 (例: 'minimal-intent') */
  id: string;

  /** prompts ライブラリの Choice 形式で返す表示情報 */
  choice: Choice;

  /** プロジェクトファイルを生成する */
  generate: (dir: string, name: string) => void;

  /**
   * テンプレート固有の「次のステップ」コマンドを返す
   * @param pm - パッケージマネージャー名 (npm | pnpm | yarn | bun)
   * @param projectName - プロジェクト名
   */
  getNextSteps: (pm: string, projectName: string) => string[];
}
