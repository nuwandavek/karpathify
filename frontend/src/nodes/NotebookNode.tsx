import { Handle, Position, type NodeProps } from '@xyflow/react';
import { type NotebookNode } from './types';
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python';
import vsd from 'react-syntax-highlighter/dist/esm/styles/prism/vsc-dark-plus';
import Markdown from 'react-markdown'
import { useState } from 'react';
import { Lesson, CodeBlock } from './types';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { dracula } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import 'katex/dist/katex.min.css';


SyntaxHighlighter.registerLanguage('python', python);

import dataFile from '../../../backend/examples/example_12.json'
const parts = dataFile;


const Block = ({ code, notes }: CodeBlock) => {
  const [isHovered, setIsHovered] = useState(false);
  return (
    <div style={{display: "flex", flexDirection: "row", justifyContent:"stretch",
      alignItems:"stretch", background: isHovered ? "#333" : "", padding: "10px"}}
      onMouseOver={()=>setIsHovered(true)} onMouseLeave={()=>setIsHovered(false)}>
      <div style={{width: "50%", overflow: "scroll", border: isHovered ? "1px solid #f1c40f" : "1px solid #eee", borderRadius: "5px",
      margin: "5px", background: "#1e1e1e"}}>
        <SyntaxHighlighter language={"python"} style={vsd}>
          {code}
        </SyntaxHighlighter>
      </div>
      <div style={{width: "50%", display: "flex", flexDirection: "column",
        justifyContent:"start", alignItems:"start", textAlign: "left", padding: "10px",
        border: isHovered ? "1px solid #f1c40f" : "1px solid #eee", borderRadius: "5px", margin: "5px",
        overflow:"scroll"}}>
        <Markdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex]}>
          {notes}
        </Markdown>
      </div>
    </div>
  )
}

export function NotebookNode({data}: NodeProps<NotebookNode>) {
  return (
    <div className="react-flow__node-default nowheel" style={{
      width: '100%',
      maxWidth: "1000px",
      display: 'flex',
      flexDirection: 'column',
      // alignItems: 'center',
    }}>
      <h1 style={{textAlign: "center"}}>{lessons[data.lessonIdx].title}</h1>
      <h3 style={{textAlign: "center"}}>{lessons[data.lessonIdx].description}</h3>
      {
        lessons[data.lessonIdx].content.map((block, idx) => (
          <Block key={idx} code={block.code} notes={block.notes} />
        ))
      }
      <Handle type="source" position={Position.Right} />
      <Handle type="target" position={Position.Left} />
    </div>
  );
}


export function NotebookMarkdownNode({data}: NodeProps<NotebookNode>) {
  return (
    <div className="react-flow__node-default nowheel" style={{
      width: '100%',
      maxWidth: "1000px",
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'stretch',
      textAlign: "left",
    }}>
      <Markdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          code({ node, inline, className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '');
  
            return !inline && match ? (
              <SyntaxHighlighter style={dracula} PreTag="div" language={match[1]} {...props}>
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            ) : (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
        }}>
          {parts[data.lessonIdx].content}
        </Markdown>
      <Handle type="source" position={Position.Right} />
      <Handle type="target" position={Position.Left} />
    </div>
  );
}
