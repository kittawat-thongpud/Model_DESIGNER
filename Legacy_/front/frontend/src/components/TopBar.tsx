/**
 * Top Bar — model name, save/build/train controls with Tailwind + Lucide.
 */
import { useState } from 'react';
import { useDesignerStore } from '../store/designerStore';
import TrainConfigModal from './TrainConfigModal';
import { ExportPackageModal } from './ExportPackageModal';
import CopyButton from './CopyButton';
import {
  ChevronRight,
  Download,
  Save,
  Cpu,
  Play,
  Square,
  ScrollText,
} from 'lucide-react';

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
    <header className="h-14 flex items-center justify-between px-5 border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm z-10 shrink-0">
      {/* Left: breadcrumb + model name */}
      <div className="flex items-center gap-2 text-sm min-w-0">
        <span className="text-slate-500">Projects</span>
        <ChevronRight size={14} className="text-slate-600 shrink-0" />
        <input
          type="text"
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          placeholder="Model name..."
          className="bg-transparent text-white font-medium text-sm border-none outline-none min-w-0 w-48 placeholder-slate-600"
        />
        {modelId && (
          <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-400 border border-slate-700">
            {modelId}
            <CopyButton text={modelId} label="Copy model ID" />
          </span>
        )}
        {isTraining && trainStatus && (
          <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20 ml-2">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
            Training{trainStatus.train_accuracy > 0 && ` · ${trainStatus.train_accuracy}%`}
          </span>
        )}
      </div>

      {/* Right: action buttons */}
      <div className="flex items-center gap-2 shrink-0">
        <button
          onClick={() => setShowExportModal(true)}
          title="Export model"
          className="px-3 py-1.5 text-sm font-medium text-slate-300 hover:text-white hover:bg-slate-800 rounded-md transition-colors flex items-center gap-2 cursor-pointer"
        >
          <Download size={15} /> Export
        </button>
        <button
          onClick={saveModel}
          title="Save model"
          className="px-3 py-1.5 text-sm font-medium text-slate-300 hover:text-white hover:bg-slate-800 rounded-md transition-colors flex items-center gap-2 cursor-pointer"
        >
          <Save size={15} /> Save
        </button>

        <div className="h-5 w-px bg-slate-800 mx-1" />

        <button
          onClick={buildModel}
          disabled={!modelId}
          title="Build model code"
          className="px-3.5 py-1.5 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-md shadow-lg shadow-indigo-600/20 transition-all flex items-center gap-2 cursor-pointer"
        >
          <Cpu size={15} /> Build
        </button>
        <button
          onClick={handleTrainClick}
          disabled={!modelId}
          title={isTraining ? 'Stop training' : 'Train model'}
          className={`px-3.5 py-1.5 text-sm font-medium text-white rounded-md shadow-lg transition-all flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer ${
            isTraining
              ? 'bg-red-600 hover:bg-red-500 shadow-red-600/20'
              : 'bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-400 hover:to-red-400 shadow-orange-500/20'
          }`}
        >
          {isTraining ? <><Square size={14} fill="currentColor" /> Stop</> : <><Play size={14} fill="currentColor" /> Train</>}
        </button>

        <div className="h-5 w-px bg-slate-800 mx-1" />

        <button
          onClick={() => setShowLogs(!showLogs)}
          title="Toggle logs"
          className={`p-1.5 rounded-md transition-colors cursor-pointer ${
            showLogs ? 'bg-slate-700 text-white' : 'text-slate-500 hover:text-white hover:bg-slate-800'
          }`}
        >
          <ScrollText size={16} />
        </button>
      </div>

      <TrainConfigModal
        isOpen={showTrainModal}
        onClose={() => setShowTrainModal(false)}
        onStart={handleStartTraining}
      />
      <ExportPackageModal
        isOpen={showExportModal}
        onClose={() => setShowExportModal(false)}
      />
    </header>
  );
}
