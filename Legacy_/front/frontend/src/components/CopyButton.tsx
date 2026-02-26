/**
 * CopyButton — small inline button that copies text to clipboard with feedback.
 */
import { useState, useCallback } from 'react';
import { Copy, Check } from 'lucide-react';

interface Props {
  text: string;
  className?: string;
  label?: string;
}

export default function CopyButton({ text, className = '', label }: Props) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* fallback: ignore */
    }
  }, [text]);

  return (
    <button
      className={`inline-flex items-center justify-center p-1 rounded text-slate-500 hover:text-white hover:bg-slate-700 transition-colors cursor-pointer ${copied ? 'text-emerald-400' : ''} ${className}`}
      onClick={handleCopy}
      title={copied ? 'Copied!' : (label || 'Copy to clipboard')}
    >
      {copied ? <Check size={12} /> : <Copy size={12} />}
    </button>
  );
}
