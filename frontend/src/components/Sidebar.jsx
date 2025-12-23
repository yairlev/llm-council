import { useState } from 'react';
import './Sidebar.css';
import { api } from '../api';

export default function Sidebar({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
}) {
  const [exportMenuId, setExportMenuId] = useState(null);

  const handleExport = async (e, conversationId, format) => {
    e.stopPropagation();
    setExportMenuId(null);
    try {
      await api.exportConversation(conversationId, format);
    } catch (error) {
      console.error('Export failed:', error);
    }
  };

  const toggleExportMenu = (e, convId) => {
    e.stopPropagation();
    setExportMenuId(exportMenuId === convId ? null : convId);
  };

  return (
    <div className="sidebar" onClick={() => setExportMenuId(null)}>
      <div className="sidebar-header">
        <h1>LLM Council</h1>
        <button className="new-conversation-btn" onClick={onNewConversation}>
          + New Conversation
        </button>
      </div>

      <div className="conversation-list">
        {conversations.length === 0 ? (
          <div className="no-conversations">No conversations yet</div>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              className={`conversation-item ${
                conv.id === currentConversationId ? 'active' : ''
              }`}
              onClick={() => onSelectConversation(conv.id)}
            >
              <div className="conversation-content">
                <div className="conversation-title">
                  {conv.title || 'New Conversation'}
                </div>
                <div className="conversation-meta">
                  {conv.message_count} messages
                </div>
              </div>
              {conv.message_count > 0 && (
                <div className="conversation-actions">
                  <button
                    className="export-btn"
                    onClick={(e) => toggleExportMenu(e, conv.id)}
                    title="Export conversation"
                  >
                    ⬇
                  </button>
                  {exportMenuId === conv.id && (
                    <div className="export-menu">
                      <button onClick={(e) => handleExport(e, conv.id, 'markdown')}>
                        Export as Markdown
                      </button>
                      <button onClick={(e) => handleExport(e, conv.id, 'json')}>
                        Export as JSON
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
