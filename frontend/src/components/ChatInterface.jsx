import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import Stage1 from './Stage1';
import Stage2 from './Stage2';
import Stage3 from './Stage3';
import './ChatInterface.css';
import { api } from '../api';

export default function ChatInterface({
  conversation,
  conversationId,
  onSendMessage,
  isLoading,
}) {
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState([]);
  const [isDragActive, setIsDragActive] = useState(false);
  const [dropNotice, setDropNotice] = useState('');
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const dragDepthRef = useRef(0);
  const dropNoticeTimerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversation]);

  useEffect(() => {
    setAttachments([]);
    setInput('');
  }, [conversationId]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading || !conversationId) return;
    try {
      await onSendMessage(input, attachments);
      setInput('');
      setAttachments([]);
    } catch (error) {
      console.error('Send failed:', error);
    }
  };

  const handleKeyDown = (e) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey && conversationId) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFilesSelected = async (fileList) => {
    if (!conversationId) return;
    const files = Array.from(fileList);
    const added = [];
    for (const file of files) {
      try {
        const metadata = await api.uploadAttachment(conversationId, file);
        added.push(metadata);
      } catch (err) {
        console.error('Failed to upload attachment:', err);
      }
    }
    if (added.length) {
      setAttachments((prev) => [...prev, ...added]);
      showDropNotice(added);
    }
    return added;
  };

  const handleFileInputChange = async (e) => {
    if (e.target.files?.length) {
      await handleFilesSelected(e.target.files);
      e.target.value = '';
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    dragDepthRef.current = 0;
    setIsDragActive(false);
    if (isLoading || !conversationId) return;
    if (e.dataTransfer.files?.length) {
      await handleFilesSelected(e.dataTransfer.files);
    }
  };

  const handleDragOver = (e) => {
    if (e.dataTransfer?.types?.includes('Files')) {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    }
  };

  const handleDragEnter = (e) => {
    if (!e.dataTransfer?.types?.includes('Files')) return;
    e.preventDefault();
    dragDepthRef.current += 1;
    setIsDragActive(true);
  };

  const handleDragLeave = (e) => {
    if (!e.dataTransfer?.types?.includes('Files')) return;
    dragDepthRef.current = Math.max(dragDepthRef.current - 1, 0);
    if (dragDepthRef.current === 0) {
      setIsDragActive(false);
    }
  };

  const showDropNotice = (addedItems) => {
    if (dropNoticeTimerRef.current) {
      clearTimeout(dropNoticeTimerRef.current);
    }
    const names = addedItems.map((item) => item.filename || item.name || 'file');
    const label =
      names.length > 3
        ? `${names.slice(0, 3).join(', ')} +${names.length - 3} more`
        : names.join(', ');
    setDropNotice(`Added ${names.length} file${names.length > 1 ? 's' : ''}: ${label}`);
    dropNoticeTimerRef.current = setTimeout(() => setDropNotice(''), 2000);
  };

  const removeAttachment = async (attachment) => {
    if (!conversationId) return;
    try {
      await api.deleteAttachment(conversationId, attachment.id);
      setAttachments((prev) => prev.filter((item) => item.id !== attachment.id));
    } catch (error) {
      console.error('Failed to delete attachment:', error);
    }
  };

  const renderMessageAttachments = (items) => {
    if (!items || items.length === 0) return null;
    return (
      <div className="message-attachments">
        {items.map((att) => (
          <div key={att.id} className="attachment-item">
            <div className="attachment-title">
              <strong>{att.filename}</strong>
              <span>{att.mime_type}</span>
            </div>
            {att.id && (
              <div className="attachment-path">Attachment ID: {att.id}</div>
            )}
            {att.canonical_uri && (
              <div className="attachment-path">Artifact URI: {att.canonical_uri}</div>
            )}
            {att.text_excerpt && (
              <div className="attachment-excerpt">
                <em>{att.text_excerpt}</em>
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div
      className={`chat-interface ${isDragActive ? 'drag-active' : ''}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
    >
      {dropNotice && <div className="drop-toast">{dropNotice}</div>}
      {isDragActive && (
        <div className="drag-overlay">
          <div className="drag-overlay-inner">
            <div className="drag-overlay-icon">⇪</div>
            <div className="drag-overlay-text">Drop files to attach</div>
          </div>
        </div>
      )}
      <div className="messages-container">
        <div className="conversation-frame">
        {!conversation || conversation.messages.length === 0 ? (
          <div className="empty-state">
            <h2>Start a conversation</h2>
            <p>Ask a question to consult the LLM Council</p>
          </div>
        ) : (
          conversation.messages.map((msg, index) => {
            const isSingleMode = msg.metadata?.mode === 'single_agent';
            return (
              <div key={index} className="message-group">
                {msg.role === 'user' ? (
                  <div className="user-message">
                    <div className="message-label">You</div>
                    <div className="message-content">
                      <div className="markdown-content">
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                      {renderMessageAttachments(msg.attachments)}
                    </div>
                  </div>
                ) : (
                  <div className="assistant-message">
                    <div className="message-label">LLM Council</div>
                    {isSingleMode ? (
                      <SingleAgentResponse message={msg} />
                    ) : (
                      <>
                        {/* Stage 1 */}
                        {msg.loading?.stage1 && (
                          <div className="stage-loading">
                            <div className="spinner"></div>
                            <span>Running Stage 1: Collecting individual responses...</span>
                          </div>
                        )}
                        {msg.stage1 && <Stage1 responses={msg.stage1} />}

                        {/* Stage 2 */}
                        {msg.loading?.stage2 && (
                          <div className="stage-loading">
                            <div className="spinner"></div>
                            <span>Running Stage 2: Peer rankings...</span>
                          </div>
                        )}
                        {msg.stage2 && (
                          <Stage2
                            rankings={msg.stage2}
                            labelToModel={msg.metadata?.label_to_model}
                            aggregateRankings={msg.metadata?.aggregate_rankings}
                          />
                        )}

                        {/* Stage 3 */}
                        {msg.loading?.stage3 && (
                          <div className="stage-loading">
                            <div className="spinner"></div>
                            <span>Running Stage 3: Final synthesis...</span>
                          </div>
                        )}
                        {msg.stage3 && <Stage3 finalResponse={msg.stage3} />}
                      </>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}

        {isLoading && (
          <div className="loading-indicator">
            <div className="spinner"></div>
            <span>Consulting the council...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
        </div>
      </div>

      <form
        className="input-form"
        onSubmit={handleSubmit}
      >
        <div className="composer-area">
          <div className="attachment-dropzone">
            <p>Drop files here or</p>
            <button
              type="button"
              className="attachment-button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading || !conversationId}
            >
              Browse files
            </button>
            <input
              type="file"
              multiple
              ref={fileInputRef}
              onChange={handleFileInputChange}
              style={{ display: 'none' }}
            />
          </div>
          {attachments.length > 0 && (
            <div className="attachment-list">
              {attachments.map((att) => (
                <div key={att.id} className="attachment-chip">
                  <span>{att.filename}</span>
                  <button
                    type="button"
                    onClick={() => removeAttachment(att)}
                    aria-label={`Remove ${att.filename}`}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
          <textarea
            className="message-input"
            placeholder="Ask your question... (Shift+Enter for new line, Enter to send)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading || !conversationId}
            rows={3}
          />
        </div>
        <button
          type="submit"
          className="send-button"
          disabled={!input.trim() || isLoading || !conversationId}
        >
          Send
        </button>
      </form>
    </div>
  );
}

function SingleAgentResponse({ message }) {
  const agentName =
    message.metadata?.agent ||
    message.stage1?.[0]?.model ||
    message.stage3?.model ||
    'Agent';
  const body =
    message.stage3?.response ||
    message.stage3?.content ||
    message.stage1?.[0]?.response ||
    '';

  return (
    <div className="single-agent-response">
      <div className="single-agent-label">
        Single Agent Response ({agentName})
      </div>
      <div className="single-agent-content">
        <div className="markdown-content">
          <ReactMarkdown>{body}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
