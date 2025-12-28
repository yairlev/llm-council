import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import './Stage1.css';

export default function Stage1({ responses, loading = false }) {
  const [activeTab, setActiveTab] = useState(0);

  if (!responses || responses.length === 0) {
    return null;
  }

  // Ensure activeTab is valid when responses change
  const safeActiveTab = activeTab < responses.length ? activeTab : 0;

  return (
    <div className={`stage stage1 ${loading ? 'stage-loading-partial' : ''}`}>
      <h3 className="stage-title">
        Stage 1: Individual Responses
        {loading && <span className="loading-badge"> (collecting...)</span>}
      </h3>

      <div className="tabs">
        {responses.map((resp, index) => (
          <button
            key={resp.model}
            className={`tab ${safeActiveTab === index ? 'active' : ''}`}
            onClick={() => setActiveTab(index)}
          >
            {resp.model.split('/')[1] || resp.model}
          </button>
        ))}
      </div>

      <div className="tab-content">
        <div className="model-name">{responses[safeActiveTab].model}</div>
        <div className="response-text markdown-content" dir="auto">
          <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{responses[safeActiveTab].response}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
