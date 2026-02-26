/**
 * App.tsx — Root layout with sidebar navigation and page routing.
 */
import { useState, useEffect, useCallback } from 'react';
import { useNodeCatalogStore } from './store/nodeCatalogStore';
import Sidebar from './components/Sidebar';
import DashboardPage from './pages/DashboardPage';
import DesignerPage from './pages/DesignerPage';
import ModelsPage from './pages/ModelsPage';
import TrainJobsPage from './pages/TrainJobsPage';
import WeightsPage from './pages/WeightsPage';
import DatasetsPage from './pages/DatasetsPage';
import JobDetailPage from './pages/JobDetailPage';
import ToastContainer from './components/ToastContainer';
import type { PageName } from './types';

function getInitialCollapsed(): boolean {
  try { return localStorage.getItem('md-sidebar-collapsed') === 'true'; } catch { return false; }
}

export default function App() {
  const [currentPage, setCurrentPage] = useState<PageName>('dashboard');
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(getInitialCollapsed);

  useEffect(() => {
    // Load node catalog from backend on app init
    useNodeCatalogStore.getState().load();
  }, []);

  useEffect(() => {
    const handler = (e: MouseEvent) => e.preventDefault();
    document.addEventListener('contextmenu', handler);
    return () => document.removeEventListener('contextmenu', handler);
  }, []);

  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed((prev) => {
      const next = !prev;
      try { localStorage.setItem('md-sidebar-collapsed', String(next)); } catch { /* ignore */ }
      return next;
    });
  }, []);

  const navigateToJob = (jobId: string) => {
    setSelectedJobId(jobId);
    setCurrentPage('job-detail');
  };

  const handleBackFromDetail = () => {
    setCurrentPage('jobs');
    setSelectedJobId(null);
  };

  return (
    <div className="flex h-screen w-full bg-slate-950 text-slate-300 font-sans overflow-hidden">
      <Sidebar
        currentPage={currentPage}
        onNavigate={(p) => {
          setCurrentPage(p);
          if (p !== 'job-detail') setSelectedJobId(null);
        }}
        collapsed={sidebarCollapsed}
        onToggleCollapse={toggleSidebar}
      />
      <main id="main-scroll" className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden">
        {currentPage === 'dashboard' && <DashboardPage onNavigate={setCurrentPage} />}
        {currentPage === 'designer' && <DesignerPage />}
        {currentPage === 'models' && <ModelsPage />}
        {currentPage === 'jobs' && <TrainJobsPage onOpenJob={navigateToJob} />}
        {currentPage === 'job-detail' && selectedJobId && (
          <JobDetailPage jobId={selectedJobId} onBack={handleBackFromDetail} />
        )}
        {currentPage === 'weights' && <WeightsPage />}
        {currentPage === 'datasets' && <DatasetsPage />}
      </main>
      <ToastContainer />
    </div>
  );
}

