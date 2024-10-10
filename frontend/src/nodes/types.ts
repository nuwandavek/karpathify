import type { Node, BuiltInNode } from '@xyflow/react';

export type NotebookNode = Node<{ lessonIdx: number }, 'notebook'>;
export type AppNode = BuiltInNode | NotebookNode;


export interface CodeBlock {
  code: string;
  notes: string;
}

export interface Lesson {
  title: string;
  description: string;
  content: CodeBlock[];
}
