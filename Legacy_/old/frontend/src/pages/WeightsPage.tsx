/**
 * Weights Page — lists all saved weights with parent model lineage.
 */
import { useEffect, useState } from 'react';
import { api } from '../services/api';
import type { WeightRecord } from '../types';

export default function WeightsPage() {
  const [weights, setWeights] = useState<WeightRecord[]>([]);
  const [loading, setLoading] = useState(true);

  const loadWeights = async () => {
    try {
      const data = await api.listWeights();
      setWeights(data);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  useEffect(() => {
    loadWeights();
  }, []);

  const handleDelete = async (weightId: string) => {
    try {
      await api.deleteWeight(weightId);
      setWeights((prev) => prev.filter((w) => w.weight_id !== weightId));
    } catch (e) { console.error(e); }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>💾 Weights</h1>
        <p className="page-subtitle">Trained model weights with parent model tracking</p>
      </div>

      {loading ? (
        <div className="page-loading">Loading weights...</div>
      ) : weights.length === 0 ? (
        <div className="empty-state-page">
          <span className="empty-icon">💾</span>
          <p>No trained weights yet</p>
          <p className="text-muted">Train a model to generate weights</p>
        </div>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Weight ID</th>
              <th>Parent Model</th>
              <th>Dataset</th>
              <th>Epochs</th>
              <th>Accuracy</th>
              <th>Loss</th>
              <th>Size</th>
              <th>Created</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {weights.map((w) => (
              <tr key={w.weight_id}>
                <td className="mono">{w.weight_id}</td>
                <td>
                  <div className="parent-info">
                    <span className="parent-name">{w.model_name}</span>
                    <span className="parent-id mono">{w.model_id}</span>
                  </div>
                </td>
                <td>{w.dataset}</td>
                <td>{w.epochs_trained}</td>
                <td className="mono">{w.final_accuracy != null ? `${w.final_accuracy}%` : '—'}</td>
                <td className="mono">{w.final_loss != null ? w.final_loss.toFixed(4) : '—'}</td>
                <td>{formatBytes(w.file_size_bytes)}</td>
                <td className="text-muted">{new Date(w.created_at).toLocaleString()}</td>
                <td>
                  <div className="action-btns">
                    {w.job_id && <span className="job-link mono" title={`From job ${w.job_id}`}>🔗 {w.job_id.slice(0,6)}</span>}
                    <button className="btn btn-sm btn-ghost" onClick={() => handleDelete(w.weight_id)}>🗑</button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
