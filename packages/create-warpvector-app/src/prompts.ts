import type { PromptObject } from "prompts";
import {
  type TemplateDefinition,
  minimalIntentTemplate,
  nextPrismaTemplate,
  cloudflareTemplate,
} from "./templates";

/** 登録済みテンプレートのレジストリ（追加時はここに追記するだけ） */
export const templates: TemplateDefinition[] = [
  minimalIntentTemplate,
  nextPrismaTemplate,
  cloudflareTemplate,
];

/** テンプレート ID からテンプレート定義を取得する */
export function findTemplate(id: string): TemplateDefinition | undefined {
  return templates.find((t) => t.id === id);
}

/** インタラクティブプロンプトの定義 */
export function createPromptQuestions(): PromptObject[] {
  return [
    {
      type: "text",
      name: "projectName",
      message: "What is your project named?",
      initial: "my-warpvector-app",
    },
    {
      type: "select",
      name: "template",
      message: "Which template would you like to use?",
      choices: templates.map((t) => ({
        title: t.title,
        value: t.id,
        description: t.description,
      })),
    },
    {
      type: "select",
      name: "packageManager",
      message: "Which package manager do you use?",
      choices: [
        { title: "bun", value: "bun" },
        { title: "npm", value: "npm" },
        { title: "pnpm", value: "pnpm" },
        { title: "yarn", value: "yarn" },
      ],
    },
  ];
}
