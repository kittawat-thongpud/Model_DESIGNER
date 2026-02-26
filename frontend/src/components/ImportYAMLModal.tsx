import { useState } from 'react';
import { X, Upload, FileText, Loader2, Check, AlertTriangle } from 'lucide-react';
import { api } from '../services/api';

interface Props {
  onClose: () => void;
  onImported: (modelId: string) => void;
}

export default function ImportYAMLModal({ onClose, onImported }: Props) {
  const [yamlContent, setYamlContent] = useState('');
  const [name, setName] = useState('Imported Model');
  const [task, setTask] = useState('detect');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  const handleImport = async () => {
    if (!yamlContent.trim()) {
      setError('Please paste YAML content');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess(false);

    try {
      const result = await api.importYAML(yamlContent, name, task);
      setSuccess(true);
      setTimeout(() => {
        onImported(result.model_id);
        onClose();
      }, 800);
    } catch (err: any) {
      setError(err.message || 'Import failed');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      setYamlContent(content);
      
      // Auto-set name from filename
      const filename = file.name.replace(/\.ya?ml$/i, '');
      setName(filename || 'Imported Model');
    };
    reader.readAsText(file);
  };

  const exampleYAML = `# Example: YOLOv8 Detection
nc: 80

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, C2f, [128, True]]

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 1], 1, Concat, [1]]
  - [[0, 1], 1, Detect, [nc]]`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-3xl mx-4 max-h-[90vh] overflow-hidden flex flex-col" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-500/20 rounded-lg">
              <Upload className="text-indigo-400" size={20} />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">Import YAML Model</h3>
              <p className="text-xs text-slate-400 mt-0.5">Convert Ultralytics YAML to graph format</p>
            </div>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors">
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {/* File Upload */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Upload YAML File</label>
            <div className="relative">
              <input
                type="file"
                accept=".yaml,.yml"
                onChange={handleFileUpload}
                className="hidden"
                id="yaml-file-input"
              />
              <label
                htmlFor="yaml-file-input"
                className="flex items-center justify-center gap-2 w-full px-4 py-3 bg-slate-800 hover:bg-slate-750 border border-slate-700 rounded-lg cursor-pointer transition-colors"
              >
                <FileText size={16} className="text-slate-400" />
                <span className="text-sm text-slate-300">Choose YAML file...</span>
              </label>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex-1 h-px bg-slate-800" />
            <span className="text-xs text-slate-500 uppercase tracking-wider">or paste content</span>
            <div className="flex-1 h-px bg-slate-800" />
          </div>

          {/* YAML Content */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">YAML Content</label>
            <textarea
              value={yamlContent}
              onChange={e => setYamlContent(e.target.value)}
              placeholder={exampleYAML}
              className="w-full h-64 px-3 py-2 bg-slate-950 border border-slate-700 rounded-lg text-slate-200 text-xs font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none"
            />
          </div>

          {/* Model Info */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Model Name</label>
              <input
                type="text"
                value={name}
                onChange={e => setName(e.target.value)}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Task</label>
              <select
                value={task}
                onChange={e => setTask(e.target.value)}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="detect">Detection</option>
                <option value="segment">Segmentation</option>
                <option value="classify">Classification</option>
                <option value="pose">Pose</option>
                <option value="obb">OBB</option>
              </select>
            </div>
          </div>

          {/* Info Box */}
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <div className="flex gap-3">
              <FileText className="text-blue-400 shrink-0 mt-0.5" size={16} />
              <div className="text-xs text-blue-200 space-y-1">
                <p className="font-medium">Supported formats:</p>
                <ul className="list-disc list-inside space-y-0.5 text-blue-300/80">
                  <li>Standard YAML with backbone/head sections</li>
                  <li>Sequential heads (e.g., Classify)</li>
                  <li>Multi-input layers (e.g., Concat, Detect)</li>
                  <li>All Ultralytics modules and PyTorch nn blocks</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 flex items-start gap-2">
              <AlertTriangle className="text-red-400 shrink-0 mt-0.5" size={16} />
              <p className="text-sm text-red-200">{error}</p>
            </div>
          )}

          {/* Success */}
          {success && (
            <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3 flex items-center gap-2">
              <Check className="text-green-400" size={16} />
              <p className="text-sm text-green-200">Import successful! Loading model...</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-6 border-t border-slate-800">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-slate-300 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleImport}
            disabled={loading || !yamlContent.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm font-medium rounded-lg transition-colors"
          >
            {loading ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Importing...
              </>
            ) : (
              <>
                <Upload size={16} />
                Import YAML
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
