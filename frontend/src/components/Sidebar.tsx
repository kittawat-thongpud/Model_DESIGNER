/**
 * Sidebar — persistent navigation with Tailwind CSS + Lucide icons.
 */
import type { PageName } from '../types';
import { useThemeStore } from '../store/themeStore';
import {
  LayoutDashboard, Network, Activity, Database,
  Sun, Moon, ArrowUp, PanelLeftClose, PanelLeftOpen, Weight,
  Workflow, Code2,
} from 'lucide-react';
import type { ReactNode } from 'react';

interface Props {
  currentPage: PageName;
  onNavigate: (page: PageName) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

const MENU_ITEMS: { page: PageName; icon: ReactNode; label: string; section: string }[] = [
  { page: 'dashboard', icon: <LayoutDashboard size={18} />, label: 'Dashboard', section: 'Menu' },
  { page: 'model-designer', icon: <Network size={18} />, label: 'Model Designer', section: 'Designers' },
  { page: 'module-designer', icon: <Code2 size={18} />, label: 'Module Designer', section: 'Designers' },
  { page: 'jobs', icon: <Activity size={18} />, label: 'Train Jobs', section: 'Operations' },
  { page: 'weights', icon: <Weight size={18} />, label: 'Weights', section: 'Operations' },
  { page: 'datasets', icon: <Database size={18} />, label: 'Datasets', section: 'Operations' },
];

export default function Sidebar({ currentPage, onNavigate, collapsed, onToggleCollapse }: Props) {
  const { theme, toggle: toggleTheme } = useThemeStore();

  const isActive = (page: PageName) =>
    currentPage === page
    || (page === 'jobs' && currentPage === 'job-detail')
    || (page === 'weights' && (currentPage === 'weight-detail' || currentPage === 'weight-editor'));

  const sections = MENU_ITEMS.reduce<Record<string, typeof MENU_ITEMS>>((acc, item) => {
    (acc[item.section] ??= []).push(item);
    return acc;
  }, {});

  return (
    <aside className={`${collapsed ? 'w-16' : 'w-56'} bg-slate-950 border-r border-slate-800 flex flex-col transition-all duration-200 shrink-0 overflow-hidden`}>
      {/* Brand */}
      <div className="h-14 flex items-center px-4 border-b border-slate-800 gap-3 shrink-0">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center shadow-lg shadow-orange-500/20 shrink-0">
          <Network size={16} className="text-white" />
        </div>
        {!collapsed && (
          <span className="text-white font-bold text-sm whitespace-nowrap">Model DESIGNER</span>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto py-3 px-2 space-y-4">
        {Object.entries(sections).map(([section, items]) => (
          <div key={section}>
            {!collapsed && (
              <p className="text-[10px] font-semibold text-slate-600 uppercase tracking-wider px-2 mb-1.5">{section}</p>
            )}
            <div className="space-y-0.5">
              {items.map((item) => (
                <button
                  key={item.page}
                  onClick={() => onNavigate(item.page)}
                  className={`flex items-center gap-3 w-full rounded-lg px-3 py-2 text-sm font-medium transition-colors cursor-pointer ${
                    isActive(item.page)
                      ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20'
                      : 'text-slate-400 hover:text-white hover:bg-slate-800/50 border border-transparent'
                  } ${collapsed ? 'justify-center px-0' : ''}`}
                  title={collapsed ? item.label : undefined}
                >
                  <span className="shrink-0">{item.icon}</span>
                  {!collapsed && <span className="truncate">{item.label}</span>}
                </button>
              ))}
            </div>
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="border-t border-slate-800 p-2 space-y-0.5 shrink-0">
        <button
          onClick={toggleTheme}
          className="flex items-center gap-3 w-full rounded-lg px-3 py-2 text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors cursor-pointer"
          title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
        >
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          {!collapsed && <span>{theme === 'dark' ? 'Light' : 'Dark'}</span>}
        </button>
        <button
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          className="flex items-center gap-3 w-full rounded-lg px-3 py-2 text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors cursor-pointer"
          title="Scroll to top"
        >
          <ArrowUp size={16} />
          {!collapsed && <span>Top</span>}
        </button>
        <button
          onClick={onToggleCollapse}
          className="flex items-center gap-3 w-full rounded-lg px-3 py-2 text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors cursor-pointer"
          title={collapsed ? 'Expand' : 'Collapse'}
        >
          {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  );
}
