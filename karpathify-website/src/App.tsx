import React, { useState } from 'react';
import './App.css'; // Assuming you have a CSS file for styles
import Markdown from 'react-markdown'

const App: React.FC = () => {
  const [url, setUrl] = useState<string>('');
  const [prompt, setPrompt] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true); // Show spinner

    // Simulate a 5-second delay before download
    setTimeout(() => {
      // Fake content for the .ipynb file (can be modified to any content)
      const fileContent = {
        "cells": [
          {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
              `# Karpathified Notebook\n\n`,
              `## URL: ${url}\n`,
              `## Prompt: ${prompt}\n`
            ]
          }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
      };

      // Create a Blob from the file content
      const blob = new Blob([JSON.stringify(fileContent, null, 2)], { type: 'application/json' });

      // Create a link element for downloading
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'karpathified.ipynb'; // File name for download
      link.click(); // Programmatically click the link to trigger download

      // Reset loading state
      setIsLoading(false);
    }, 5000); // 5 seconds delay
  };

  return (
    <div className="App" style={{
      textAlign: "center",
      justifyContent: "center",
      display: "flex",
      flexDirection: "column",
      alignItems: "center"
  }}>
      <header className="App-header" style={{display: 'flex', alignItems: 'stretch', width: "800px"}}>
        {/* Logo and Title */}
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '20px', justifyContent: "center" }}>
          <img
            src="/karpathy.png" // Replace with your logo URL
            alt="Karpathify Logo"
            style={{ marginRight: '10px', width: '200px' }} // Adjust the width as needed
          />
          <h1>karpathify</h1>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="form-container">
          <div className="form-group" style={{justifyContent: "space-between"}}>
            <label htmlFor="url">paperswithcode URL:</label>
            <input
              type="url"
              id="url"
              name="url"
              placeholder="Enter a paperswithcode URL to the paper"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              required
              style={{ width: '80%' }} // Ensure it takes full width
            />
          </div>
          <div className="form-group"  style={{justifyContent: "space-between"}}>
            <label htmlFor="prompt">Current proficiency:</label>
            <textarea
              id="prompt"
              name="prompt"
              placeholder="Enter your current understanding of the topic"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              required
              rows={2} // Adjust the number of rows as needed
              style={{ width: '80%' }} // Ensure it takes full width
            />
          </div>
          <button type="submit" className="submit-btn" disabled={isLoading}>
            {isLoading ? (
              <div className="spinner"></div> // Display spinner when loading
            ) : (
              'Submit'
            )}
          </button>
        </form>
      </header>
      <div style={{ width: "60%", textAlign: "left",  background: "#333", padding: "10px", borderRadius: "5px"}}>
        <Markdown>{`You are an expert machine learning researcher and teacher like Andrej Karpathy. Create a 5 lesson plan as an iPython notebook to help the user understand the key insights and implementation details of a specific machine learning paper, based on their current proficiency, the paper itself, and the associated repository. Each lesson must be self contained. Expect the user to run each cell in the notebook as they go through the lessons.

<MyCurrentProficiency>currentProficiency</MyCurrentProficiency>
<Paper>paper</Paper>
<Repo>repoState</Repo>

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
        `}
        </Markdown>
        </div>
    </div>
  );
};

export default App;
