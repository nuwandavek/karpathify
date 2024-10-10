import { NodeTypes, Position } from '@xyflow/react';

import { NotebookMarkdownNode } from './NotebookNode';
import { AppNode } from './types';

const numLessons = 5;

export const initialNodes: AppNode[] = [
  {
    id: 'start',
    type: 'input',
    position: { x: -300, y: 0 },
    data: { label: "welcome to karpathify!" },
    sourcePosition: Position.Right,
  }
]

for (let i = 0; i < numLessons; i++) {
  initialNodes.push({
    id: `lesson_${i}`,
    type: 'notebook',
    position: { x: 1100 * i, y: 0 },
    data: { lessonIdx: i },
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
  });
}

initialNodes.push({
  id: 'end',
  type: 'output',
  position: { x: 1100 * numLessons, y: 0 },
  data: { label: "fin!" },
  targetPosition: Position.Left,
});


export const nodeTypes = {
  'notebook': NotebookMarkdownNode,
  // Add any of your custom nodes here!
} satisfies NodeTypes;
