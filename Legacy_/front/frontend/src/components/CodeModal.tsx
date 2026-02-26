/**
 * Code Modal — displays generated PyTorch code in a modal overlay.
 */
import { useState } from 'react';
import { useDesignerStore } from '../store/designerStore';
import { useToastStore } from '../store/toastStore';

export default function CodeModal() {
  const generatedCode = useDesignerStore((s) => s.generatedCode);
  const showCode = useDesignerStore((s) => s.showCode);
  const setShowCode = useDesignerStore((s) => s.setShowCode);
  const [copied, setCopied] = useState(false);

  if (!showCode || !generatedCode) return null;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(generatedCode);
      setCopied(true);
      useToastStore.getState().show('Code copied to clipboard', 'success');
      setTimeout(() => setCopied(false), 2000);
    } catch { /* ignore */ }
  };

  return (
    <div className="modal-overlay" onClick={() => setShowCode(false)}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>🐍 Generated PyTorch Code</h2>
          <div className="modal-actions">
            <button className="btn btn-secondary btn-sm" onClick={handleCopy}>
              {copied ? '✓ Copied' : '📋 Copy'}
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => setShowCode(false)}>✕</button>
          </div>
        </div>
        <pre className="code-block">
          <code>{generatedCode}</code>
        </pre>
      </div>
    </div>
  );
}
