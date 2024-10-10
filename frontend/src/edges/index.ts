import type { Edge, EdgeTypes } from '@xyflow/react';

const numLessons = 5;

export const initialEdges: Edge[] = [
  {
    id: 'start->lesson_0',
    source: 'start',
    target: 'lesson_0',
    animated: true,
    label: 'Lesson 1',
    style: { stroke: 'white', strokeWidth: 2 }
  }
]

for (let i = 0; i < numLessons; i++) {
  initialEdges.push({
    id: `lesson_${i}->lesson_${i + 1}`,
    source: `lesson_${i}`,
    target: `lesson_${i + 1}`,
    animated: true,
    label: `Lesson ${i + 2}`,
    style: { stroke: 'white', strokeWidth: 2 }
  });
}
initialEdges.push({
  id: `lesson_${numLessons - 1}->end`,
  source: `lesson_${numLessons - 1}`,
  target: 'end',
  animated: true,
  label: 'End',
  style: { stroke: 'white', strokeWidth: 2 }
});

export const edgeTypes = {
  // Add your custom edge types here!
} satisfies EdgeTypes;
