/**
 * Datasets Page — displays available datasets with their metadata.
 */
import { useEffect, useState } from 'react';
import { api } from '../services/api';
import type { DatasetInfo } from '../types';

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await api.listDatasets();
        setDatasets(data);
      } catch (e) { console.error(e); }
      setLoading(false);
    }
    load();
  }, []);

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>📦 Datasets</h1>
        <p className="page-subtitle">Available datasets for training and validation</p>
      </div>

      {loading ? (
        <div className="page-loading">Loading datasets...</div>
      ) : datasets.length === 0 ? (
        <div className="empty-state-page">
          <span className="empty-icon">📦</span>
          <p>No datasets available</p>
        </div>
      ) : (
        <div className="dataset-grid">
          {datasets.map((ds) => (
            <div key={ds.name} className="dataset-card">
              <div className="dataset-card-header">
                <span className="dataset-icon">
                  {ds.name === 'mnist' ? '✍️' : '🖼️'}
                </span>
                <h3>{ds.display_name}</h3>
              </div>
              <div className="dataset-card-body">
                <div className="dataset-stat">
                  <span className="dataset-stat-label">Input Shape</span>
                  <span className="dataset-stat-value mono">{ds.input_shape.join(' × ')}</span>
                </div>
                <div className="dataset-stat">
                  <span className="dataset-stat-label">Classes</span>
                  <span className="dataset-stat-value">{ds.num_classes}</span>
                </div>
                <div className="dataset-stat">
                  <span className="dataset-stat-label">Train Size</span>
                  <span className="dataset-stat-value">{ds.train_size.toLocaleString()}</span>
                </div>
                <div className="dataset-stat">
                  <span className="dataset-stat-label">Test Size</span>
                  <span className="dataset-stat-value">{ds.test_size.toLocaleString()}</span>
                </div>
              </div>
              <div className="dataset-card-footer">
                <div className="class-tags">
                  {ds.classes.slice(0, 10).map((cls) => (
                    <span key={cls} className="class-tag">{cls}</span>
                  ))}
                  {ds.classes.length > 10 && <span className="class-tag more">+{ds.classes.length - 10}</span>}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
