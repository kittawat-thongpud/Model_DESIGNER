/**
 * Top Bar — model name/description, save/load/build/train controls.
 * Train button opens TrainConfigModal instead of inline popover.
 */
import { useState } from 'react';
import { useDesignerStore } from '../store/designerStore';
import TrainConfigModal from './TrainConfigModal';
import { ExportPackageModal } from './ExportPackageModal';

export default function TopBar() {
  const modelName = useDesignerStore((s) => s.modelName);
  const setModelName = useDesignerStore((s) => s.setModelName);
  const modelId = useDesignerStore((s) => s.modelId);
  const saveModel = useDesignerStore((s) => s.saveModel);
  const buildModel = useDesignerStore((s) => s.buildModel);
  const startTraining = useDesignerStore((s) => s.startTraining);
  const stopTraining = useDesignerStore((s) => s.stopTraining);
  const isTraining = useDesignerStore((s) => s.isTraining);
  const trainStatus = useDesignerStore((s) => s.trainStatus);
  const showLogs = useDesignerStore((s) => s.showLogs);
  const setShowLogs = useDesignerStore((s) => s.setShowLogs);

  const [showTrainModal, setShowTrainModal] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);

  const handleTrainClick = () => {
    if (isTraining) {
      stopTraining();
    } else {
      setShowTrainModal(true);
    }
  };

  const handleStartTraining = (config: Record<string, unknown>) => {
    startTraining(config);
    setShowTrainModal(false);
  };

  return (
    <div className="top-bar">
      <div className="top-bar-left">
        <span className="app-logo">🔥</span>
        <h1 className="app-title">Model DESIGNER</h1>
        <div className="model-name-field">
          <input
            type="text"
            className="model-name-input"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="Model name..."
          />
          {modelId && <span className="model-id-badge">{modelId}</span>}
        </div>
      </div>

      <div className="top-bar-right">
        {/* Training status indicator */}
        {trainStatus && isTraining && (
          <div className="train-status-badge">
            <span className="pulse-dot" />
            {trainStatus.train_accuracy > 0 && ` · ${trainStatus.train_accuracy}%`}
          </div>
        )}

        <button className="btn btn-secondary" onClick={() => setShowExportModal(true)}>📦 Export</button>
        <button className="btn btn-primary" onClick={saveModel}>💾 Save</button>
        <button className="btn btn-secondary" onClick={buildModel} disabled={!modelId}>⚙️ Build</button>
        <button className={`btn ${isTraining ? 'btn-danger' : 'btn-accent'}`} onClick={handleTrainClick} disabled={!modelId}>
          {isTraining ? '⏹ Stop' : '🚀 Train'}
        </button>
        <button className={`btn btn-ghost ${showLogs ? 'active' : ''}`} onClick={() => setShowLogs(!showLogs)}>
          📋 Logs
        </button>
      </div>

      {/* Train config modal */}
      <TrainConfigModal
        isOpen={showTrainModal}
        onClose={() => setShowTrainModal(false)}
        onStart={handleStartTraining}
      />
      
      <ExportPackageModal
        isOpen={showExportModal}
        onClose={() => setShowExportModal(false)}
      />
    </div>
  );
}
