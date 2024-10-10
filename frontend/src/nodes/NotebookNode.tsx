import { Handle, Position, type NodeProps } from '@xyflow/react';
import { type NotebookNode } from './types';
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python';
import vsd from 'react-syntax-highlighter/dist/esm/styles/prism/vsc-dark-plus';
import Markdown from 'react-markdown'
import { useState } from 'react';
import { Lesson, CodeBlock } from './types';
SyntaxHighlighter.registerLanguage('python', python);

import dataFile from '../../../backend/examples/example5.json'

const lessons: Lesson[] = dataFile.lessons

const Block = ({ code, notes }: CodeBlock) => {
  const [isHovered, setIsHovered] = useState(false);
  return (
    <div style={{display: "flex", flexDirection: "row", justifyContent:"stretch",
      alignItems:"stretch", background: isHovered ? "#333" : "", padding: "10px"}}
    onMouseOver={()=>setIsHovered(true)} onMouseLeave={()=>setIsHovered(false)}>
      <div style={{width: "50%", overflow: "scroll", border: "1px solid #eee", borderRadius: "5px",
    margin: "5px"}}>
        <SyntaxHighlighter language={"python"} style={vsd}>
          {code}
        </SyntaxHighlighter>
      </div>
      <div style={{width: "50%", display: "flex", flexDirection: "column",
        justifyContent:"start", alignItems:"start", textAlign: "left", padding: "10px",
        border: "1px solid #eee", borderRadius: "5px", margin: "5px"}}>
        <Markdown>{notes}</Markdown>
      </div>
    </div>
  )
}

export function NotebookNode({data}: NodeProps<NotebookNode>) {
  return (
    <div className="react-flow__node-default" style={{
      width: '100%',
      maxWidth: "1000px",
      display: 'flex',
      flexDirection: 'column',
      // alignItems: 'center',
    }}>
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
