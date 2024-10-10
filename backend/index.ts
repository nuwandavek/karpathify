import "dotenv/config";
import OpenAI from "openai";
import fs from "fs";
import path from "path";

const client = new OpenAI({
  apiKey: process.env["OPENAI_API_KEY"], // This is the default and can be omitted
});

const matchingRegex = ".*\\.py$"; // Only read python files
const excludedDirs = [".git", ".cache", "dist", ".vscode"]; // Directories to ignore
// const excludedFiles = [".gitignore", ".DS_Store", "package-lock.json"]; // Files to ignore

function readImportantFilesAsJson(dirPath) {
  const result = {};

  const items = fs.readdirSync(dirPath);

  items.forEach((item) => {
    const fullPath = path.join(dirPath, item);
    const stats = fs.statSync(fullPath);

    if (stats.isDirectory()) {
      if (!excludedDirs.includes(item)) {
        // Recursively read the directory only if it's not in the excluded list
        const newResult = readImportantFilesAsJson(fullPath);
        if (Object.keys(newResult).length > 0) {
          result[item] = newResult;
        }
      }
    } else if (stats.isFile()) {
      // if (!excludedFiles.includes(item)) {
      if (item.match(matchingRegex)) {
        // Add the file if it's not in the excluded list
        result[item] = fs.readFileSync(fullPath, "utf-8");
      }
    }
  });
  return result;
}

async function main() {
  const repoState = JSON.stringify(readImportantFilesAsJson("./repo"));
  const currentProficiency = "I understand calculus, python, pytorch";

  const prompt = `
  The user needs to understand a programming project.

Instructions:
- Go through the repo, and give me a 5 lesson plan to learn this. 
- The plan needs to be centered around 5 python programs
- Each lesson needs to introduce 1 or 2 a specific concepts in the file, and have code related to that. Also give a brief note about the concept. Tell me what else I can read up to understand it.
- Each lesson needs to build upon the previous lesson in complexity and length.

  <Repo>${repoState}</Repo>
${currentProficiency}`;
  const chatCompletion = await client.chat.completions.create({
    messages: [{ role: "user", content: prompt }],
    model: "o1-preview-2024-09-12",
  });
  console.log(chatCompletion.choices[0].message.content);
}

main()