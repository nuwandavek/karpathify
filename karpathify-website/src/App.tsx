import React, { useState } from 'react';
import './App.css'; // Assuming you have a CSS file for styles

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
    <div className="App">
      <header className="App-header">
        {/* Logo and Title */}
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
          <img
            src="https://via.placeholder.com/50" // Replace with your logo URL
            alt="Karpathify Logo"
            style={{ marginRight: '10px' }}
          />
          <h1>Karpathify</h1>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="form-container">
          <div className="form-group">
            <label htmlFor="url">URL:</label>
            <input
              type="url"
              id="url"
              name="url"
              placeholder="Enter a URL"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="prompt">Prompt:</label>
            <textarea
              id="prompt"
              name="prompt"
              placeholder="Enter your prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              required
              rows={6} // Adjust the number of rows as needed
              style={{ width: '100%' }} // Ensure it takes full width
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
    </div>
  );
};

export default App;
