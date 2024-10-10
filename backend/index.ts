import "dotenv/config";
import OpenAI from "openai";
import fs from "fs";
import path from "path";
import example8 from './examples/example8.json'

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

const fileInfo = {
  object: "file",
  id: "file-h83ykpO6P2akzhgD1GdbRTpe",
  purpose: "assistants",
  filename: "depth-pro.pdf",
  bytes: 27977407,
  created_at: 1728537170,
  status: "processed",
  status_details: null,
  _request_id: "req_2a3b263ab87e7144e88df1e0823002b2",
}

// async function convertFromPDF() {
//   // const filePath = './docs/depth-pro.pdf'
//   // const response = await client.files.create({
//   //   purpose: 'assistants',
//   //   file: fs.createReadStream(filePath)
//   // });
//   // console.log(response)
//   const prompt = `Convert this pdf (File ID: ${fileInfo.id}) to a text file`;
//   const chatCompletion = await client.chat.completions.create({
//     messages: [{ role: "user", content: prompt }],
//     model: "gpt-4o",
//   });
//   console.log(chatCompletion.choices[0].message.content);
// }

function jsonToMarkdown(jsonData) {
  let markdown = '';

  jsonData.lessons.forEach((lesson, index) => {
    // Add lesson title and description
    markdown += `## ${lesson.title}\n\n`;
    markdown += `${lesson.description}\n\n`;

    // Add content (code and notes)
    lesson.content.forEach((content, idx) => {
      // Add note (if present)
      if (content.notes) {
        markdown += `**Note:** ${content.notes}\n\n`;
      }
      if (content.code) {
        markdown += '```python\n'; // Assuming the code is in Python
        markdown += `${content.code}\n`;
        markdown += '```\n\n'; 
      }
    });
  });

  return markdown;
}

async function main() {
  const repoState = JSON.stringify(readImportantFilesAsJson("./repo"));
  const paper = fs.readFileSync("./docs/depth-pro.txt", "utf-8");
  const currentProficiency = "I understand calculus, python, pytorch";

  const prompt = `
  The user needs to understand a programming project and accompanying paper.

Instructions:
- Go through the repo, and give me a 5 lesson plan to learn this. The plan needs to be centered around 5 python programs.
- Each lesson needs to introduce 1 or 2 a specific concepts in the file, and have code related to that. Also give a brief note explaining the concept clearly.
- Each lesson needs to build upon the previous lesson in complexity and length.
- Output a json file containing an array of lessons. Each lesson should have a title, a brief description, and an array of objects containing 2 keys: "code" and "notes". The "code" key should contain the code block, and the "notes" key should contain any notes as Markdown.
- Ensure that the notes reference relevant sections of the paper so that the user understands the paper while going through the code. Use formulas from the paper in the notes.
- Use github flavored markdown for the notes since it supports mermaid diagrams, mathjax, and other features. (Use them!)
- Avoid large chunks of code or notes. Break a lesson into a large number of small steps.

  <Repo>${repoState}</Repo>
  <Paper>${paper}</Paper>
${currentProficiency}`;
  const chatCompletion = await client.chat.completions.create({
    messages: [{ role: "user", content: prompt }],
    model: "o1-preview-2024-09-12",
  });
  console.log(chatCompletion.choices[0].message.content);
}

// main()
console.log(jsonToMarkdown(example8))