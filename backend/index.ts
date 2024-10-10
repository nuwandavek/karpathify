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

function markdownToCustomJson(markdown) {
  const lessons = [];
  const lines = markdown.split('\n');
  let currentLesson = null;
  let currentContent = null;

  lines.forEach(line => {
      // Check for new lesson title
      if (line.startsWith('# Lesson')) {
          if (currentLesson) {
              lessons.push(currentLesson); // Push the previous lesson
          }
          currentLesson = {
              title: line.replace('# ', '').trim(),
              description: '',
              content: []
          };
      }
      // Check for description
      else if (line.startsWith('*')) {
          if (currentLesson) {
              currentLesson.description = line.replace('*', '').trim();
          }
      }
      // Check for code block start
      else if (line.startsWith('```')) {
          if (currentContent && currentContent.code) {
              currentLesson.content.push(currentContent);
              currentContent = null;
          } else {
              currentContent = { code: "", notes: "" };
          }
      }
      // Handle code lines
      else if (currentContent && currentContent.code !== undefined) {
          currentContent.code += line + '\n';
      }
      // Handle markdown text
      else if (line.trim()) {
          if (!currentContent) {
              currentContent = { code: "", notes: "" };
          }
          currentContent.notes += line.trim() + ' ';
      }
  });

  // Push the last lesson and content
  if (currentContent) currentLesson.content.push(currentContent);
  if (currentLesson) lessons.push(currentLesson);

  return { lessons };
}

function splitMarkdownIntoLessons(markdown) {
  // Split the markdown content by the lesson headers
  const lessonRegex = /(^# Lesson \d+:.*$)/gm;
  const lessonMatches = [...markdown.matchAll(lessonRegex)];

  let parts = [];

  // Add introduction part (everything before the first lesson)
  if (lessonMatches.length > 0) {
      const introContent = markdown.slice(0, lessonMatches[0].index).trim();
      if (introContent) {
          parts.push({
              partTitle: "Introduction",
              content: introContent,
          });
      }

      // Process each lesson
      lessonMatches.forEach((match, index) => {
          const lessonTitle = match[0];
          const lessonStart = match.index;
          const lessonEnd = index < lessonMatches.length - 1
              ? lessonMatches[index + 1].index
              : markdown.length;

          const lessonContent = markdown.slice(lessonStart, lessonEnd).trim();

          parts.push({
              partTitle: lessonTitle.replace(/^# /, ''), // Remove the '# ' from the title
              content: lessonContent,
          });
      });
  }

  return parts;
}

function splitMarkdownFileIntoLessons(markdownFile, outputJSONFile) {
  const markdown = readMarkdownFile(markdownFile);
  const parts = splitMarkdownIntoLessons(markdown)
  fs.writeFileSync(outputJSONFile, JSON.stringify(parts, null, 2), "utf-8");
}

function readMarkdownFile(filePath) {
  return fs.readFileSync(filePath, "utf-8");
}

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
       - Divide the essential information from the paper and repository into N lessons. Should be at least 5 lessons.
       - Lesson 1 must start from the user's current proficiency level.
       - Each lesson introduces 1 or 2 specific concepts using the relevant code snippets. It builds on the previous lesson, increasing in complexity.
       - Each lesson must be self contained
       - Provide detailed notes explaining each concept through the code and explanation. Use excerpts and formulas from the paper where relevant.
       - Give very detailed explanations for each code block. Remember the users' proficiency level.
       - Ensure that by the last lesson, all key insights from the paper and related code are covered
  
    # Notes
    - Keep lessons concise, logical, and progressive.
    - Ensure annotations are outside of the code to maintain clarity and focus on understanding through notes.
    - Use github flavored markdown for notes to support mermaid diagrams, mathjax, and other features.
    `;
;

  const chatCompletion = await client.chat.completions.create({
    messages: [{ role: "user", content: oai_auto_prompt1 }],
    model: "o1-preview-2024-09-12",
  });
  // console.log(JSON.stringify(chatCompletion.usage))
  console.log(chatCompletion.choices[0].message.content);
}

// main()
// console.log(jsonToMarkdown(example))
// processJSONDirectory("./examples");
// processJSONDirectory("./examples");
// markdownToIPythonNotebook('./examples/example_12.md', './examples/example_12.ipynb');
// console.log(getJSONFromMDFile('./examples/example_12.md'))
splitMarkdownFileIntoLessons('./examples/example_12.md', './examples/example_12.json');