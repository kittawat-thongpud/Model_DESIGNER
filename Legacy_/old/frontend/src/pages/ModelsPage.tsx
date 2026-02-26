/**
 * Models Page — shows built (compiled) models with code and layer details.
 */
import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import type { BuildSummary, BuildRecord } from '../types';

export default function ModelsPage() {
  const [builds, setBuilds] = useState<BuildSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedBuild, setSelectedBuild] = useState<BuildRecord | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const fetchBuilds = useCallback(async () => {
    setLoading(true);
    try {
      const list = await api.listBuilds();
      setBuilds(list);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchBuilds();
  }, [fetchBuilds]);

  const handleSelectBuild = async (buildId: string) => {
    if (selectedBuild?.build_id === buildId) {
      setSelectedBuild(null);
      return;
    }
    setDetailLoading(true);
    try {
      const detail = await api.getBuild(buildId);
      setSelectedBuild(detail);
    } catch {
      // ignore
    } finally {
      setDetailLoading(false);
    }
  };

  const handleDelete = async (buildId: string) => {
    try {
      await api.deleteBuild(buildId);
      setBuilds((prev) => prev.filter((b) => b.build_id !== buildId));
      if (selectedBuild?.build_id === buildId) setSelectedBuild(null);
    } catch {
      // ignore
    }
  };

  const formatDate = (d: string) => {
    try {
      return new Date(d).toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
      });
    } catch {
      return d;
    }
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h2 className="page-title">🏗️ Built Models</h2>
        <span className="page-subtitle">{builds.length} build{builds.length !== 1 ? 's' : ''}</span>
      </div>

      {loading ? (
        <div className="loading-text">Loading builds...</div>
      ) : builds.length === 0 ? (
        <div className="empty-state">
          <span className="empty-icon">🏗️</span>
          <h3>No builds yet</h3>
          <p>Go to Designer, create a model, and click Build to generate PyTorch code.</p>
        </div>
      ) : (
        <div className="builds-layout">
          {/* Builds list */}
          <div className="builds-list">
            {builds.map((b) => (
              <div
                key={b.build_id}
                className={`build-card ${selectedBuild?.build_id === b.build_id ? 'active' : ''}`}
                onClick={() => handleSelectBuild(b.build_id)}
              >
                <div className="build-card-header">
                  <span className="build-card-name">{b.model_name}</span>
                  <button
                    className="btn-icon danger"
                    onClick={(e) => { e.stopPropagation(); handleDelete(b.build_id); }}
                    title="Delete build"
                  >🗑️</button>
                </div>
                <div className="build-card-meta">
                  <span className="build-class-badge">{b.class_name}</span>
                  <span className="build-layer-count">{b.layer_count} layers</span>
                </div>
                <div className="build-card-layers">
                  {b.layer_types.map((lt, i) => (
                    <span key={i} className="layer-type-tag">{lt}</span>
                  ))}
                </div>
                <div className="build-card-date">{formatDate(b.created_at)}</div>
              </div>
            ))}
          </div>

          {/* Build detail panel */}
          {selectedBuild && (
            <div className="build-detail-panel">
              {detailLoading ? (
                <div className="loading-text">Loading...</div>
              ) : (
                <>
                  <div className="build-detail-header">
                    <h3>{selectedBuild.model_name}</h3>
                    <span className="build-class-badge large">{selectedBuild.class_name}</span>
                  </div>

                  <div className="build-detail-stats">
                    <div className="stat-mini">
                      <span className="stat-mini-label">Nodes</span>
                      <span className="stat-mini-value">{selectedBuild.node_count}</span>
                    </div>
                    <div className="stat-mini">
                      <span className="stat-mini-label">Edges</span>
                      <span className="stat-mini-value">{selectedBuild.edge_count}</span>
                    </div>
                    <div className="stat-mini">
                      <span className="stat-mini-label">Layers</span>
                      <span className="stat-mini-value">{selectedBuild.layers.length}</span>
                    </div>
                  </div>

                  <div className="build-detail-section">
                    <h4>📋 Layer Details</h4>
                    <div className="layer-detail-list">
                      {selectedBuild.layers.map((layer, i) => (
                        <div key={i} className="layer-detail-item">
                          <span className="layer-detail-type">{layer.layer_type}</span>
                          <span className="layer-detail-params">
                            {Object.entries(layer.params).map(([k, v]) =>
                              `${k}=${v}`
                            ).join(', ')}
                          </span>
                        </div>
                      ))}
                      {selectedBuild.layers.length === 0 && (
                        <div className="empty-layers">No layers (empty model)</div>
                      )}
                    </div>
                  </div>

                  <div className="build-detail-section">
                    <h4>🐍 Generated Code</h4>
                    <pre className="code-block"><code>{selectedBuild.code}</code></pre>
                  </div>

                  <div className="build-detail-footer">
                    <span className="build-detail-date">Built: {formatDate(selectedBuild.created_at)}</span>
                    <span className="build-detail-id">ID: {selectedBuild.build_id}</span>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
