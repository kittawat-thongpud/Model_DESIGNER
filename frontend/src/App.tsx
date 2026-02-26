/**
 * App.tsx — Root layout with sidebar navigation and page routing.
 *
 * Two designers:
 *   - Model Designer: Ultralytics YAML architecture editor
 *   - Module Designer: Custom nn.Module block editor
 */
import { useState, useCallback, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import DashboardPage from './pages/DashboardPage';
import ModelDesignerPage from './pages/ModelDesignerPage';
import ModuleDesignerPage from './pages/ModuleDesignerPage';
import TrainJobsPage from './pages/TrainJobsPage';
import WeightsPage from './pages/WeightsPage';
import WeightDetailPage from './pages/WeightDetailPage';
import WeightEditorPage from './pages/WeightEditorPage';
import DatasetsPage from './pages/DatasetsPage';
import DatasetDetailPage from './pages/DatasetDetailPage';
import JobDetailPage from './pages/JobDetailPage';
import ToastContainer from './components/ToastContainer';
import type { PageName } from './types';

function getInitialCollapsed(): boolean {
  try { return localStorage.getItem('md-sidebar-collapsed') === 'true'; } catch { return false; }
}

export default function App() {
  const [currentPage, setCurrentPage] = useState<PageName>('dashboard');
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedWeightId, setSelectedWeightId] = useState<string | null>(null);
  const [selectedDatasetName, setSelectedDatasetName] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(getInitialCollapsed);

  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed((prev) => {
      const next = !prev;
      try { localStorage.setItem('md-sidebar-collapsed', String(next)); } catch { /* ignore */ }
      return next;
    });
  }, []);

  const navigateToJob = useCallback((jobId: string) => {
    setSelectedJobId(jobId);
    setCurrentPage('job-detail');
  }, []);

  useEffect(() => {
    const handler = (e: Event) => {
      const jobId = (e as CustomEvent<{ jobId: string }>).detail?.jobId;
      if (jobId) navigateToJob(jobId);
    };
    window.addEventListener('navigate-job', handler);
    return () => window.removeEventListener('navigate-job', handler);
  }, [navigateToJob]);

  const handleBackFromDetail = () => {
    setCurrentPage('jobs');
    setSelectedJobId(null);
  };

  const navigateToWeight = (weightId: string) => {
    setSelectedWeightId(weightId);
    setCurrentPage('weight-detail');
  };

  const handleBackFromWeight = () => {
    setCurrentPage('weights');
    setSelectedWeightId(null);
  };

  const navigateToDataset = (name: string) => {
    setSelectedDatasetName(name);
    setCurrentPage('dataset-detail');
  };

  const handleBackFromDataset = () => {
    setCurrentPage('datasets');
    setSelectedDatasetName(null);
  };

  return (
    <div className="flex h-screen w-full bg-slate-950 text-slate-300 font-sans overflow-hidden">
      <Sidebar
        currentPage={currentPage}
        onNavigate={(p) => {
          setCurrentPage(p);
          if (p !== 'job-detail') setSelectedJobId(null);
          if (p !== 'weight-detail' && p !== 'weight-editor') setSelectedWeightId(null);
          if (p !== 'dataset-detail') setSelectedDatasetName(null);
        }}
        collapsed={sidebarCollapsed}
        onToggleCollapse={toggleSidebar}
      />
      <main className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden">
        {currentPage === 'dashboard' && <DashboardPage onNavigate={setCurrentPage} />}
        {currentPage === 'model-designer' && <ModelDesignerPage onBack={() => setCurrentPage('dashboard')} />}
        {currentPage === 'module-designer' && <ModuleDesignerPage onBack={() => setCurrentPage('dashboard')} />}
        {currentPage === 'jobs' && <TrainJobsPage onOpenJob={navigateToJob} />}
        {currentPage === 'job-detail' && selectedJobId && (
          <JobDetailPage jobId={selectedJobId} onBack={handleBackFromDetail} />
        )}
        {currentPage === 'weights' && <WeightsPage onOpenWeight={navigateToWeight} />}
        {currentPage === 'weight-detail' && selectedWeightId && (
          <WeightDetailPage
            weightId={selectedWeightId}
            onBack={handleBackFromWeight}
            onOpenJob={navigateToJob}
            onOpenWeight={navigateToWeight}
            onEditWeight={(id: string) => { setSelectedWeightId(id); setCurrentPage('weight-editor'); }}
          />
        )}
        {currentPage === 'weight-editor' && selectedWeightId && (
          <WeightEditorPage
            weightId={selectedWeightId}
            onBack={() => { setCurrentPage('weight-detail'); }}
            onOpenWeight={navigateToWeight}
          />
        )}
        {currentPage === 'datasets' && <DatasetsPage onOpenDataset={navigateToDataset} />}
        {currentPage === 'dataset-detail' && selectedDatasetName && (
          <DatasetDetailPage datasetName={selectedDatasetName} onBack={handleBackFromDataset} />
        )}
      </main>
      <ToastContainer />
    </div>
  );
}
