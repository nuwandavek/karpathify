import "dotenv/config";
import OpenAI from "openai";
import fs from "fs";
import path from "path";

const client = new OpenAI({
  apiKey: process.env["OPENAI_API_KEY_O1"], // This is the default and can be omitted
});

const matchingRegex = ".*\\.py$"; // Only read python files
const excludedDirs = [".git", ".cache", "dist", ".vscode"]; // Directories to ignore
// const excludedFiles = [".gitignore", ".DS_Store", "package-lock.json"]; // Files to ignore

// Function to check if a file exists
function fileExists(filePath) {
  return fs.existsSync(filePath);
}

// Function to process files in a directory
function processJSONDirectory(directoryPath) {
  fs.readdir(directoryPath, (err, files) => {
    if (err) {
      console.error("Error reading directory:", err);
      return;
    }

    files.forEach((file) => {
      const jsonFilePath = path.join(directoryPath, file);

      if (path.extname(file) === ".json") {
        const mdFileName = path.basename(file, ".json") + ".md";
        const mdFilePath = path.join(directoryPath, mdFileName);

        // Check if the corresponding .md file exists
        if (!fileExists(mdFilePath)) {
          console.log(`Generating ${mdFileName} from ${file}`);
          const jsonContent = JSON.parse(
            fs.readFileSync(jsonFilePath, "utf-8")
          );
          const mdContent = jsonToMarkdown(jsonContent);
          fs.writeFileSync(mdFilePath, mdContent, "utf-8");
        }
      }
    });
  });
}

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
};

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

function markdownToIPythonNotebook(markdownFilePath, outputNotebookPath) {
  // Read the Markdown file content
  const markdownContent = fs.readFileSync(markdownFilePath, 'utf-8');

  // Split the content on code block indicators
  const segments = markdownContent.split(/```python([\s\S]*?)```/g);

  // Initialize an empty array for notebook cells
  let notebookCells = [];

  // Iterate through segments and categorize them as either markdown or code cells
  segments.forEach((segment, index) => {
      if (index % 2 === 0) {
          // Even index -> Markdown cell
          if (segment.trim()) {
              notebookCells.push({
                  cell_type: 'markdown',
                  metadata: {},
                  source: segment.split('\n'),
              });
          }
      } else {
          // Odd index -> Python code cell
          notebookCells.push({
              cell_type: 'code',
              execution_count: null,
              metadata: {},
              outputs: [],
              source: segment.split('\n'),
          });
      }
  });

  // Create the notebook structure
  const notebook = {
      cells: notebookCells,
      metadata: {
          language_info: {
              name: 'python',
          },
      },
      nbformat: 4,
      nbformat_minor: 5,
  };

  // Write the notebook to the output file
  fs.writeFileSync(outputNotebookPath, JSON.stringify(notebook, null, 2), 'utf-8');
}

function jsonToMarkdown(jsonData) {
  let markdown = "";

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
        markdown += "```python\n"; // Assuming the code is in Python
        markdown += `${content.code}\n`;
        markdown += "```\n\n";
      }
    });
  });

  return markdown;
}

// Previous Instrutions
// - Each lesson should have at least 5 code blocks and notes, with each code block not being more than 5-10 lines.

async function main() {
  const repoState = JSON.stringify(readImportantFilesAsJson("./repo"));
  const paper = fs.readFileSync("./docs/depth-pro_small.md", "utf-8");
  const currentProficiency =
    "I understand calculus, python, pytorch. I have some basic experience with neural nets.";

    const oai_auto_prompt1 = `You are an expert machine learning researcher and teacher like Andrej Karpathy. Create a 5 lesson plan as an iPython notebook to help the user understand the key insights and implementation details of a specific machine learning paper, based on their current proficiency, the paper itself, and the associated repository. Each lesson must be self contained. Expect the user to run each cell in the notebook as they go through the lessons.

    <MyCurrentProficiency>${currentProficiency}</MyCurrentProficiency>
    <Paper>${paper}</Paper>
    <Repo>${repoState}</Repo>
  
    # Instructions
    
    1. **Read and Understand the Paper**:
       - Extract key insights from the paper.
    
    2. **Review the Repository**:
       - Identify important code snippets relevant to the paper's insights.
       - Ignore irrelevant code.
    
    3. **Develop Lesson Plan**:
       - Divide the essential information from the paper and repository into 5 lessons.
       - Lesson 1 must start from the user's current proficiency level.
       - Each lesson introduces 1 or 2 specific concepts using the relevant code snippets. It builds on the previous lesson, increasing in complexity.
       - Each lesson must be self contained
       - Provide detailed notes explaining each concept through the code and explanation
       - Ensure that by lesson 5, all key insights from the paper and related code are covered
  
    # Notes
    - Use formulas from the paper where relevant for better understanding.
    - Notes should reference corresponding sections in the paper verbatim.
    - Keep lessons concise, logical, and progressive.
    - Ensure annotations are outside of the code to maintain clarity and focus on understanding through notes.
    - Use github flavored markdown for notes to support mermaid diagrams, mathjax, and other features.
    `;

  const oai_auto_prompt = `You are an expert machine learning researcher and teacher like Andrej Karpathy. Create a 5 lesson plan as an iPython notebook to help the user understand the key insights and implementation details of a specific machine learning paper, based on their current proficiency, the paper itself, and the associated repository. Each lesson must be self contained. Expect the user to run each cell in the notebook as they go through the lessons.

  <MyCurrentProficiency>${currentProficiency}</MyCurrentProficiency>
  <Paper>${paper}</Paper>
  <Repo>${repoState}</Repo>

  # Instructions
  
  1. **Read and Understand the Paper**:
     - Extract key insights from the paper.
  
  2. **Review the Repository**:
     - Identify important code snippets relevant to the paper's insights.
     - Ignore irrelevant code.
  
  3. **Develop Lesson Plan**:
     - Divide the essential information from the paper and repository into 5 lessons.
     - Put high level plan in <LessonPlan> tags. Tell me how each lesson takes the user from their current proficiency to understanding the key insights and implementation details of the paper. Tell how each lesson builds on the previous one.
     - Lesson 1 must start from the user's current proficiency level.
     - Each lesson introduces 1 or 2 specific concepts using the relevant code snippets. It builds on the previous lesson, increasing in complexity.
     - Provide detailed notes explaining each concept and the associated code thoroughly.
     - Ensure lesson 5 is almost all the code from <ImportantCode> tags and the key insights from the paper.
     - Each lesson should have at least 5 code blocks and notes, with each code block not being more than 5-10 lines.
     - Do not do ... in code. It should have the full code. This code will be executed by the user.
     - Do not hold back on writing code. The user should be able to run the code and understand the full lesson.
     - Do not put explanations in the code or code comments. The notes should explain the code.
     - Use github flavored markdown for the notes since it supports mermaid diagrams, mathjax, and other feature (Use them!)
  
  # Output Format
  - <PaperInsights> tags: Key insights from the paper.
  - <ImportantCode> tags: Important code snippets from the repository.
  - <LessonPlan> tags: High-level lesson
  - A JSON file containing an array of lessons.
  - Each lesson consists of:
    - A title,
    - A detailed description,
    - An array of objects each containing:
      - "code": the relevant code block.
      - "notes": explanatory notes in GitHub-Flavored Markdown.
  `
  
  const prompt = `You are an expert machine learning researcher and teacher. You have a paper and a repo. You should create a 5 lesson plan for me to go from where I am to understanding the key insights and implementation details of the paper.

  <MyCurrentProficiency>${currentProficiency}</MyCurrentProficiency>
  <Paper>${paper}</Paper>
  <Repo>${repoState}</Repo>

# Routine:
1. Read the paper and understand the key insights. Put them in <PaperInsights> tags.
2. Review the repo and understand the important code snippets, ignore the rest. Put them in <ImportantCode> tags.
3. Break this essential information into 5 lessons, each building on the previous one, building up in complexity. Each lesson needs to introduce 1 or 2 specific concepts using the code. Give a detailed note explaining the concept & code clearly. The first lesson needs to start from where I am currently. The last lesson should end with me understanding the key insights and implementation details of the paper.
4. Output a json file containing an array of lessons. Each lesson should have a title, a brief description, and an array of objects containing 2 keys: "code" and "notes". The "code" key should contain the code block, and the "notes" key should contain any notes as Markdown.

# General Guidelines:
- Ensure that the notes reference relevant sections of the paper so that the user understands the paper while going through the code. Use formulas from the paper in the notes.
- Use github flavored markdown for the notes since it supports mermaid diagrams, mathjax, and other features. (Use them!)
- Avoid large chunks of code or notes. No code block should be more than 5-10 lines long. Break them down into smaller blocks.
- Do not put explanations in the code or code comments. The notes should explain the code.
  `;

  const chatCompletion = await client.chat.completions.create({
    messages: [{ role: "user", content: oai_auto_prompt }],
    model: "o1-mini-2024-09-12",
  });
  console.log(JSON.stringify(chatCompletion.usage))
  console.log(chatCompletion.choices[0].message.content);
}

main()
// console.log(jsonToMarkdown(example))
// processJSONDirectory("./examples");
// processJSONDirectory("./examples");
// markdownToIPythonNotebook('./examples/example_12.md', './examples/example_12.ipynb');
