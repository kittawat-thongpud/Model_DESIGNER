/**
 * App.tsx — Root layout with sidebar navigation and page routing.
 */
import { useState } from 'react';
import Sidebar from './components/Sidebar';
import DashboardPage from './pages/DashboardPage';
import DesignerPage from './pages/DesignerPage';
import ModelsPage from './pages/ModelsPage';
import TrainJobsPage from './pages/TrainJobsPage';
import WeightsPage from './pages/WeightsPage';
import DatasetsPage from './pages/DatasetsPage';
import JobDetailPage from './pages/JobDetailPage';
import type { PageName } from './types';

export default function App() {
  const [currentPage, setCurrentPage] = useState<PageName>('dashboard');
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  globalThis.document.addEventListener('contextmenu', (e) => e.preventDefault());

  const navigateToJob = (jobId: string) => {
    setSelectedJobId(jobId);
    setCurrentPage('job-detail');
  };

  const handleBackFromDetail = () => {
    setCurrentPage('jobs');
    setSelectedJobId(null);
  };

  return (
    <div className="app-shell">
      <Sidebar currentPage={currentPage} onNavigate={(p) => {
        setCurrentPage(p);
        if (p !== 'job-detail') setSelectedJobId(null);
      }} />
      <main className="main-content">
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
    </div>
  );
}

