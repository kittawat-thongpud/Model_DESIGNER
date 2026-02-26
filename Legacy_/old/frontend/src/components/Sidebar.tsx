/**
 * Sidebar — persistent navigation for the multi-page app.
 */
import type { PageName } from '../types';

interface Props {
  currentPage: PageName;
  onNavigate: (page: PageName) => void;
}

const NAV_ITEMS: { page: PageName; icon: string; label: string }[] = [
  { page: 'dashboard', icon: '📊', label: 'Dashboard' },
  { page: 'designer', icon: '🔥', label: 'Designer' },
  { page: 'models', icon: '🏗️', label: 'Models' },
  { page: 'jobs', icon: '🏋️', label: 'Train Jobs' },
  { page: 'weights', icon: '💾', label: 'Weights' },
  { page: 'datasets', icon: '📦', label: 'Datasets' },
];

export default function Sidebar({ currentPage, onNavigate }: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="sidebar-logo">🔥</span>
        <span className="sidebar-title">Model DESIGNER</span>
      </div>
      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.page}
            className={`sidebar-item ${
              currentPage === item.page || (item.page === 'jobs' && currentPage === 'job-detail') ? 'active' : ''
            }`}
            onClick={() => onNavigate(item.page)}
          >
            <span className="sidebar-icon">{item.icon}</span>
            <span className="sidebar-label">{item.label}</span>
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        <span className="sidebar-version">v2.0</span>
      </div>
    </aside>
  );
}
