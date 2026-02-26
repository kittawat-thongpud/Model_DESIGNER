/**
 * Sidebar — persistent navigation with Tailwind CSS + Lucide icons.
 * Matches the modern dark slate design.
 */
import type { PageName } from '../types';
import { useThemeStore } from '../store/themeStore';
import {
  LayoutDashboard,
  Network,
  Box,
  Activity,
  Settings,
  Database,
  Sun,
  Moon,
  ArrowUp,
  PanelLeftClose,
  PanelLeftOpen,
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
  { page: 'designer', icon: <Network size={18} />, label: 'Designer', section: 'Menu' },
  { page: 'models', icon: <Box size={18} />, label: 'Models', section: 'Menu' },
  { page: 'jobs', icon: <Activity size={18} />, label: 'Train Jobs', section: 'Operations' },
  { page: 'weights', icon: <Settings size={18} />, label: 'Weights', section: 'Operations' },
  { page: 'datasets', icon: <Database size={18} />, label: 'Datasets', section: 'Operations' },
];

export default function Sidebar({ currentPage, onNavigate, collapsed, onToggleCollapse }: Props) {
  const theme = useThemeStore((s) => s.theme);
  const toggleTheme = useThemeStore((s) => s.toggle);

  const scrollToTop = () => {
    document.getElementById('main-scroll')?.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const isActive = (page: PageName) =>
    currentPage === page || (page === 'jobs' && currentPage === 'job-detail');

  // Group items by section
  const sections = MENU_ITEMS.reduce<Record<string, typeof MENU_ITEMS>>((acc, item) => {
    (acc[item.section] ??= []).push(item);
    return acc;
  }, {});

  return (
    <aside className={`${collapsed ? 'w-16' : 'w-72'} bg-slate-950 border-r border-slate-800 flex flex-col transition-all duration-250 shrink-0 overflow-hidden`}>
      {/* Brand */}
      <div className="h-16 flex items-center px-4 border-b border-slate-800 gap-3 shrink-0">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center shadow-lg shadow-orange-500/20 shrink-0">
          <Network size={18} className="text-white" />
        </div>
        {!collapsed && (
          <span className="text-white font-bold text-base whitespace-nowrap">Model DESIGNER</span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-4">
        {Object.entries(sections).map(([section, items]) => (
          <div key={section}>
            {!collapsed && (
              <div className="px-4 mb-2 mt-4 first:mt-0 text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                {section}
              </div>
            )}
            {items.map((item) => (
              <button
                key={item.page}
                onClick={() => onNavigate(item.page)}
                title={collapsed ? item.label : undefined}
                className={`
                  flex items-center gap-3 w-full text-left text-sm transition-all duration-200 cursor-pointer
                  ${collapsed ? 'justify-center px-0 py-2.5 mx-auto my-0.5' : 'px-6 py-2.5 mx-2 my-0.5 rounded-lg'}
                  ${isActive(item.page)
                    ? 'bg-indigo-500/10 text-indigo-400 font-medium'
                    : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                  }
                `}
              >
                <span className="shrink-0">{item.icon}</span>
                {!collapsed && <span>{item.label}</span>}
              </button>
            ))}
          </div>
        ))}
      </nav>

      {/* Tool buttons */}
      <div className="border-t border-slate-800 p-2 flex flex-col gap-1 shrink-0">
        <button
          onClick={toggleTheme}
          title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
          className="flex items-center gap-3 px-3 py-2 rounded-lg text-slate-500 hover:bg-slate-800/50 hover:text-slate-300 transition-colors text-sm cursor-pointer"
        >
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          {!collapsed && <span>{theme === 'dark' ? 'Light' : 'Dark'}</span>}
        </button>
        <button
          onClick={scrollToTop}
          title="Scroll to top"
          className="flex items-center gap-3 px-3 py-2 rounded-lg text-slate-500 hover:bg-slate-800/50 hover:text-slate-300 transition-colors text-sm cursor-pointer"
        >
          <ArrowUp size={16} />
          {!collapsed && <span>Top</span>}
        </button>
        <button
          onClick={onToggleCollapse}
          title={collapsed ? 'Expand' : 'Collapse'}
          className="flex items-center gap-3 px-3 py-2 rounded-lg text-slate-500 hover:bg-slate-800/50 hover:text-slate-300 transition-colors text-sm cursor-pointer"
        >
          {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-slate-800 shrink-0">
        <div className={`flex items-center gap-3 ${collapsed ? 'justify-center' : 'px-2'}`}>
          <div className="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center border border-slate-700 shrink-0">
            <span className="text-xs font-medium text-white">US</span>
          </div>
          {!collapsed && (
            <div className="flex flex-col min-w-0">
              <span className="text-sm font-medium text-white truncate">Workspace</span>
              <span className="text-[10px] text-slate-500">v2.0</span>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
