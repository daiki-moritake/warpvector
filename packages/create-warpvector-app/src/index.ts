import { Command } from "commander";
import prompts from "prompts";
import pc from "picocolors";
import fs from "fs";
import path from "path";
import { createPromptQuestions, findTemplate } from "./prompts";
import { createTsConfig } from "./scaffold";
import pkg from "../package.json";

const program = new Command();

program
  .name("create-warpvector-app")
  .description("Scaffold a new WarpVector application")
  .version(pkg.version);

program.action(async () => {
  console.log(pc.blue("🌌 Welcome to WarpVector!"));
  console.log("Let's create a new vector search application.\n");

  const response = await prompts(createPromptQuestions());

  if (!response.projectName || !response.template) {
    console.log(pc.red("Operation cancelled."));
    process.exit(1);
  }

  const projectDir = path.resolve(process.cwd(), response.projectName);

  if (fs.existsSync(projectDir)) {
    console.log(
      pc.red(`\nError: Directory ${response.projectName} already exists.`),
    );
    process.exit(1);
  }

  const template = findTemplate(response.template);
  if (!template) {
    console.log(pc.red(`\nError: Unknown template "${response.template}".`));
    process.exit(1);
  }

  // プロジェクト生成
  console.log(pc.cyan(`\nCreating project in ${projectDir}...`));
  fs.mkdirSync(projectDir, { recursive: true });
  template.generate(projectDir, response.projectName, pkg.version);
  createTsConfig(projectDir);

  // Next steps
  const pm = response.packageManager || "npm";
  const steps = template.getNextSteps(pm);

  console.log(pc.green("\n✔ Project created successfully!\n"));
  console.log("Next steps:");
  console.log(pc.cyan(`  cd ${response.projectName}`));
  console.log(pc.cyan(`  ${pm} install`));
  for (const step of steps) {
    console.log(pc.cyan(`  ${step}`));
  }

  console.log(
    pc.dim("\n📖 Docs: https://github.com/daiki-moritake/warpvector"),
  );
  console.log(
    pc.dim("🎮 Playground: https://daiki-moritake.github.io/warpvector/\n"),
  );
});

program.parse();
