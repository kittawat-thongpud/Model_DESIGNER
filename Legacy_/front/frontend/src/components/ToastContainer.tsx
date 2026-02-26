/**
 * ToastContainer — renders active toast notifications in bottom-right corner.
 */
import { useToastStore } from '../store/toastStore';
import { CheckCircle2, XCircle, Info } from 'lucide-react';

const TOAST_STYLES: Record<string, string> = {
  success: 'border-emerald-500/30 bg-emerald-500/10',
  error: 'border-red-500/30 bg-red-500/10',
  info: 'border-blue-500/30 bg-blue-500/10',
};

const TOAST_ICONS: Record<string, React.ReactNode> = {
  success: <CheckCircle2 size={16} className="text-emerald-400 shrink-0" />,
  error: <XCircle size={16} className="text-red-400 shrink-0" />,
  info: <Info size={16} className="text-blue-400 shrink-0" />,
};

export default function ToastContainer() {
  const toasts = useToastStore((s) => s.toasts);
  const dismiss = useToastStore((s) => s.dismiss);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-6 right-6 z-[9999] flex flex-col gap-2 pointer-events-auto">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`flex items-center gap-3 px-4 py-3 rounded-lg border shadow-xl backdrop-blur-sm cursor-pointer transition-all animate-slide-in ${TOAST_STYLES[t.type] || TOAST_STYLES.info}`}
          onClick={() => dismiss(t.id)}
        >
          {TOAST_ICONS[t.type] || TOAST_ICONS.info}
          <span className="text-sm text-white">{t.message}</span>
        </div>
      ))}
    </div>
  );
}
