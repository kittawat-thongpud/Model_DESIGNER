import { useEffect, useState, useCallback, useRef } from 'react';
import { api } from '../services/api';
import type { JobRecord, JobLogEntry, PerClassMetric } from '../types';
import LineChart from '../components/charts/LineChart';
import type { LineSeries } from '../components/charts/LineChart';
import HeatmapChart from '../components/charts/HeatmapChart';
import type { Colormap } from '../components/charts/HeatmapChart';
import BarChart from '../components/charts/BarChart';

type DetailTab = 'summary' | 'performance' | 'analysis' | 'logs' | 'weights';

interface Props {
  jobId: string;
  onBack: () => void;
}

export default function JobDetailPage({ jobId, onBack }: Props) {
  const [job, setJob] = useState<JobRecord | null>(null);
  const [jobLogs, setJobLogs] = useState<JobLogEntry[]>([]);
  const [logLimit, setLogLimit] = useState(20);
  const [logFilter, setLogFilter] = useState('');
  const [activeTab, setActiveTab] = useState<DetailTab>('summary');
  const [loading, setLoading] = useState(true);

  // Weight playback state
  const [snapshotLayers, setSnapshotLayers] = useState<string[]>([]);
  const [snapshotEpochs, setSnapshotEpochs] = useState<number[]>([]);
  const [selectedLayer, setSelectedLayer] = useState('');
  const [selectedEpoch, setSelectedEpoch] = useState(0);
  const [snapshotData, setSnapshotData] = useState<{
    values: number[][];
    min: number;
    max: number;
    mean: number;
    std: number;
    shape: number[];
    rows: number;
    cols: number;
  } | null>(null);
  const [weightColormap, setWeightColormap] = useState<Colormap>('viridis');
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(1);
  const playTimerRef = useRef<number | null>(null);
  const [snapshotStats, setSnapshotStats] = useState<Record<string, Record<string, unknown>[]>>({});

  // Confusion matrix colormap
  const [cmColormap, setCmColormap] = useState<Colormap>('viridis');

  const loadJobDetails = useCallback(async () => {
    try {
      const data = await api.listJobs();
      const found = data.find((j) => j.job_id === jobId);
      if (found) {
        setJob(found);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  const loadLogs = useCallback(async (limit: number) => {
    try {
      const logs = await api.getJobLogs(jobId, limit);
      setJobLogs(logs);
    } catch (e) {
      console.error(e);
    }
  }, [jobId]);

  useEffect(() => {
    loadJobDetails();
    loadLogs(20);
    const interval = setInterval(() => {
        loadJobDetails();
        // If on logs tab, maybe auto-refresh logs too?
        if (activeTab === 'logs') loadLogs(logLimit);
    }, 3000);
    return () => clearInterval(interval);
  }, [loadJobDetails, loadLogs, activeTab, logLimit]);

  const loadMoreLogs = async () => {
    const newLimit = logLimit + 20;
    setLogLimit(newLimit);
    loadLogs(newLimit);
  };

  const loadSnapshotData = useCallback(async (jobId: string, epoch: number, layer: string) => {
    if (!epoch || !layer) return;
    try {
      const data = await api.getSnapshotData(jobId, epoch, layer);
      setSnapshotData(data);
    } catch (e) {
      console.error('Failed to load snapshot:', e);
      setSnapshotData(null);
    }
  }, []);

  const [modelGraph, setModelGraph] = useState<any | null>(null);

  useEffect(() => {
    if (activeTab === 'weights' && jobId) {
      (async () => {
        try {
          // Fetch snapshots info
          const layers = await api.getSnapshotLayers(jobId);
          const epochs = await api.getSnapshotEpochs(jobId);
          setSnapshotLayers(layers);
          setSnapshotEpochs(epochs);
          if (layers.length > 0 && !selectedLayer) setSelectedLayer(layers[0]);
          if (epochs.length > 0 && !selectedEpoch) setSelectedEpoch(epochs[0]);

          const stats = await api.getSnapshotStats(jobId);
          setSnapshotStats(stats.timeline);

          // Fetch model metadata if not already loaded
          if (job?.model_id && !modelGraph) {
            const graph = await api.loadModel(job.model_id);
            setModelGraph(graph);
          }
        } catch (e) {
          console.error(e);
        }
      })();
    }
  }, [activeTab, jobId, job?.model_id, modelGraph, selectedLayer, selectedEpoch]);

  useEffect(() => {
    if (selectedLayer && selectedEpoch) {
      loadSnapshotData(jobId, selectedEpoch, selectedLayer);
    }
  }, [jobId, selectedLayer, selectedEpoch, loadSnapshotData]);

  useEffect(() => {
    if (isPlaying && snapshotEpochs.length > 0) {
      const delay = 1000 / playSpeed;
      playTimerRef.current = window.setInterval(() => {
        setSelectedEpoch((prev) => {
          const idx = snapshotEpochs.indexOf(prev);
          if (idx >= snapshotEpochs.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return snapshotEpochs[idx + 1];
        });
      }, delay);
    }
    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, [isPlaying, playSpeed, snapshotEpochs]);

  const statusColor: Record<string, string> = {
    pending: '#fb923c',
    running: '#4ade80',
    completed: '#6366f1',
    failed: '#f87171',
    stopped: '#9090b0',
  };

  if (loading && !job) {
    return <div className="page-loading">Loading job details...</div>;
  }

  if (!job) {
    return (
      <div className="page-container">
        <button className="btn btn-ghost" onClick={onBack}>← Back to Jobs</button>
        <div className="empty-state-page">
          <span className="empty-icon">⚠️</span>
          <h3>Job not found</h3>
          <p>The training job you requested does not exist or has been deleted.</p>
        </div>
      </div>
    );
  }

  // ── Chart builders ───────────────────────────────────────
  const buildLossChart = (): LineSeries[] => [
    { label: 'Train Loss', color: '#f87171', data: job.history.map((e) => ({ x: e.epoch, y: e.train_loss })) },
    { label: 'Val Loss', color: '#4ade80', data: job.history.filter((e) => e.val_loss != null).map((e) => ({ x: e.epoch, y: e.val_loss! })), dashed: true },
  ];

  const buildAccChart = (): LineSeries[] => [
    { label: 'Train Acc', color: '#60a5fa', data: job.history.map((e) => ({ x: e.epoch, y: e.train_accuracy })) },
    { label: 'Val Acc', color: '#c084fc', data: job.history.filter((e) => e.val_accuracy != null).map((e) => ({ x: e.epoch, y: e.val_accuracy! })), dashed: true },
  ];

  const buildPRChart = (): LineSeries[] => {
    const h = job.history.filter((e) => e.precision != null);
    return [
      { label: 'Precision', color: '#fb923c', data: h.map((e) => ({ x: e.epoch, y: e.precision! })) },
      { label: 'Recall', color: '#22d3ee', data: h.map((e) => ({ x: e.epoch, y: e.recall! })) },
      { label: 'F1', color: '#a78bfa', data: h.map((e) => ({ x: e.epoch, y: e.f1! })) },
    ];
  };

  const buildLRChart = (): LineSeries[] => {
    const h = job.history.filter((e) => e.lr != null);
    return [{ label: 'Learning Rate', color: '#fbbf24', data: h.map((e) => ({ x: e.epoch, y: e.lr })) }];
  };

  const filteredLogs = logFilter
    ? jobLogs.filter((l) => l.level.toLowerCase() === logFilter.toLowerCase())
    : jobLogs;

  const tabs: { key: DetailTab; label: string; icon: string }[] = [
    { key: 'summary', label: 'Summary', icon: '📋' },
    { key: 'performance', label: 'Performance', icon: '📊' },
    { key: 'analysis', label: 'Analysis', icon: '🔍' },
    { key: 'logs', label: 'Logs', icon: '📝' },
    { key: 'weights', label: 'Weights', icon: '🔬' },
  ];

  return (
    <div className="page-container job-detail-page">
      <div className="page-header sticky-header">
        <div className="header-left">
          <button className="btn btn-ghost" onClick={onBack}>← Back</button>
          <div className="header-title-group">
            <h2>{job.model_name}</h2>
            <span className="job-id-tag mono">{job.job_id}</span>
          </div>
        </div>
        <div className="header-status">
           <span className="status-badge" style={{ backgroundColor: `${statusColor[job.status]}20`, color: statusColor[job.status] }}>
             ● {job.status}
           </span>
        </div>
      </div>

      <div className="detail-layout">
        <div className="detail-tabs-full">
          {tabs.map((t) => (
            <button
              key={t.key}
              className={`tab-btn-lg ${activeTab === t.key ? 'active' : ''}`}
              onClick={() => setActiveTab(t.key)}
            >
              <span className="tab-icon">{t.icon}</span>
              <span className="tab-label">{t.label}</span>
            </button>
          ))}
        </div>

        <div className="detail-content-area">
          {activeTab === 'summary' && (
            <div className="tab-summary-full">
               <div className="metric-grid-full">
                  <div className="metric-item">
                    <label>Training Progress</label>
                    <div className="value-row">
                      <span className="value-huge">{job.epoch} / {job.total_epochs}</span>
                      <span className="unit">Epochs</span>
                    </div>
                    <div className="progress-bar-wide">
                      <div className="fill" style={{ width: `${(job.epoch / job.total_epochs) * 100}%` }} />
                    </div>
                  </div>
                  <div className="metric-item">
                    <label>Best Accuracy</label>
                    <div className="value-row">
                      <span className="value-huge">{job.best_val_accuracy ?? '—'}</span>
                      <span className="unit">%</span>
                    </div>
                  </div>
                  <div className="metric-item">
                    <label>Best Loss</label>
                    <div className="value-row">
                      <span className="value-huge mono">{job.best_val_loss?.toFixed(4) ?? '—'}</span>
                    </div>
                  </div>
               </div>

               <div className="summary-split">
                  <div className="config-section">
                    <h3>Configuration</h3>
                    <div className="config-list">
                      <div className="cfg-item"><span>Dataset</span><span>{job.config.dataset}</span></div>
                      <div className="cfg-item"><span>Batch Size</span><span>{job.config.batch_size}</span></div>
                      <div className="cfg-item"><span>Optimizer</span><span>{job.config.optimizer}</span></div>
                      <div className="cfg-item"><span>LR Initial</span><span className="mono">{job.config.lr0}</span></div>
                      <div className="cfg-item"><span>Device</span><span>{job.config.device}</span></div>
                      <div className="cfg-item"><span>AMP</span><span>{job.config.amp ? 'On' : 'Off'}</span></div>
                    </div>
                  </div>
                  <div className="chart-preview-section">
                    <div className="chart-box">
                      <label>Loss Curve</label>
                      <LineChart series={buildLossChart()} height={250} />
                    </div>
                    <div className="chart-box">
                      <label>Accuracy Curve</label>
                      <LineChart series={buildAccChart()} height={250} />
                    </div>
                  </div>
               </div>
            </div>
          )}

          {activeTab === 'performance' && (
            <div className="tab-performance-full">
               <div className="performance-grid-full">
                  <div className="chart-card-lg">
                    <h4>📉 Loss History</h4>
                    <LineChart series={buildLossChart()} xLabel="Epoch" yLabel="Loss" height={300} />
                  </div>
                  <div className="chart-card-lg">
                    <h4>📈 Accuracy History</h4>
                    <LineChart series={buildAccChart()} xLabel="Epoch" yLabel="%" height={300} />
                  </div>
                  <div className="chart-card-lg">
                    <h4>🎯 Precision / Recall</h4>
                    <LineChart series={buildPRChart()} xLabel="Epoch" yLabel="Score" height={300} />
                  </div>
                  <div className="chart-card-lg">
                    <h4>📐 Learning Rate</h4>
                    <LineChart series={buildLRChart()} xLabel="Epoch" yLabel="LR" height={300} showLegend={false} />
                  </div>
               </div>
            </div>
          )}

          {activeTab === 'analysis' && (
            <div className="tab-analysis-full">
               {job.status !== 'completed' && <div className="alert-info">Analysis will be available once training is completed.</div>}
               <div className="analysis-grid-full">
                  {job.confusion_matrix && (
                    <div className="analysis-card">
                      <h4>Confusion Matrix</h4>
                      <HeatmapChart
                        values={job.confusion_matrix}
                        rowLabels={job.class_names}
                        colLabels={job.class_names}
                        colormap={cmColormap}
                        onColormapChange={setCmColormap}
                        height={500}
                        showValues={job.class_names.length <= 15}
                      />
                    </div>
                  )}
                  {job.per_class_metrics && (
                    <div className="analysis-card">
                      <h4>Per-Class Metrics</h4>
                      <div className="per-class-viz">
                        <div className="viz-split">
                          <div className="viz-chart">
                             <BarChart 
                               data={job.per_class_metrics.map(m => ({ label: m.class, value: m.f1, color: '#a78bfa' }))}
                               yLabel="F1 Score"
                               height={250}
                             />
                          </div>
                          <div className="viz-table">
                            <table className="data-table">
                              <thead>
                                <tr>
                                  <th>Class</th>
                                  <th>P</th>
                                  <th>R</th>
                                  <th>F1</th>
                                  <th>Sup</th>
                                </tr>
                              </thead>
                              <tbody>
                                {job.per_class_metrics.map((m: PerClassMetric) => (
                                  <tr key={m.class}>
                                    <td>{m.class}</td>
                                    <td className="mono">{(m.precision*100).toFixed(0)}%</td>
                                    <td className="mono">{(m.recall*100).toFixed(0)}%</td>
                                    <td className="mono">{(m.f1*100).toFixed(0)}%</td>
                                    <td>{m.support}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
               </div>
            </div>
          )}

          {activeTab === 'logs' && (
            <div className="tab-logs-full">
              <div className="log-toolbar">
                <select value={logFilter} onChange={(e) => setLogFilter(e.target.value)}>
                  <option value="">All Levels</option>
                  <option value="INFO">INFO</option>
                  <option value="WARNING">WARNING</option>
                  <option value="ERROR">ERROR</option>
                </select>
                <span className="count">{filteredLogs.length} matching entries</span>
              </div>
              <div className="log-viewer">
                {filteredLogs.map((log, i) => (
                  <div key={i} className={`log-row level-${log.level.toLowerCase()}`}>
                    <span className="time">{new Date(log.timestamp).toLocaleTimeString()}</span>
                    <span className="level">{log.level}</span>
                    <span className="msg">{log.message}</span>
                  </div>
                ))}
                {filteredLogs.length === 0 && <div className="empty-logs">No log entries found.</div>}
                <button className="btn btn-secondary load-more" onClick={loadMoreLogs}>Load More Logs</button>
              </div>
            </div>
          )}

          {activeTab === 'weights' && (
             <div className="tab-weights-full">
                <div className="weight-sidebar">
                  <div className="layer-list-container">
                     <h4>Available Layers</h4>
                     <div className="layer-groups">
                       {Object.entries(
                         snapshotLayers.reduce((acc, layer) => {
                           const [nodeId, ...parts] = layer.split('.');
                           const type = parts.join('.');
                           if (!acc[nodeId]) acc[nodeId] = [];
                           acc[nodeId].push({ full: layer, type });
                           return acc;
                         }, {} as Record<string, { full: string, type: string }[]>)
                       ).sort((a, b) => {
                          // Sort by node number (node_1, node_2...)
                          const numA = parseInt(a[0].replace('node_', '')) || 0;
                          const numB = parseInt(b[0].replace('node_', '')) || 0;
                          return numA - numB;
                       }).map(([nodeId, items]) => {
                         // Find node info from graph
                         const nodeInfo = modelGraph?.nodes?.find((n: any) => n.id === nodeId);
                         const nodeLabel = nodeInfo?.data?.label;
                         const nodeType = nodeInfo?.data?.layerType;
                         
                         return (
                           <div key={nodeId} className="layer-group">
                             <div className="layer-group-header">
                               <span className="node-icon">📦</span>
                               <span className="node-name">
                                 {nodeType && <span className="node-type-tag">{nodeType}: </span>}
                                 {nodeLabel || nodeId}
                                 {nodeLabel && <span className="node-id-sub"> ({nodeId})</span>}
                               </span>
                             </div>
                             <div className="layer-group-items">
                               {items.map(item => (
                                 <button
                                   key={item.full}
                                   className={`layer-item-btn ${selectedLayer === item.full ? 'active' : ''}`}
                                   onClick={() => setSelectedLayer(item.full)}
                                 >
                                   <span className="layer-type">{item.type}</span>
                                 </button>
                               ))}
                             </div>
                           </div>
                         );
                       })}
                       {snapshotLayers.length === 0 && <div className="empty-msg">No layers found</div>}
                     </div>
                  </div>

                  <div className="control-group" style={{ marginTop: '1rem' }}>
                    <label>Epoch: {selectedEpoch}</label>
                    <input 
                       type="range" 
                       min={0} max={Math.max(0, snapshotEpochs.length-1)} 
                       value={snapshotEpochs.indexOf(selectedEpoch)} 
                       onChange={(e) => setSelectedEpoch(snapshotEpochs[parseInt(e.target.value)])}
                       disabled={snapshotEpochs.length === 0}
                    />
                  </div>
                  <div className="playback">
                    <button 
                      className={`btn ${isPlaying ? 'btn-danger' : 'btn-primary'}`} 
                      onClick={() => setIsPlaying(!isPlaying)}
                      disabled={snapshotEpochs.length === 0}
                    >
                      {isPlaying ? 'Pause' : 'Play'}
                    </button>
                    <select value={playSpeed} onChange={(e) => setPlaySpeed(parseFloat(e.target.value))}>
                       <option value={0.5}>0.5x</option>
                       <option value={1}>1.0x</option>
                       <option value={2}>2.0x</option>
                    </select>
                  </div>
                  {snapshotData && (
                    <div className="weight-mini-stats">
                       <div className="stat"><span>Mean</span><span>{snapshotData.mean.toFixed(6)}</span></div>
                       <div className="stat"><span>Std</span><span>{snapshotData.std.toFixed(6)}</span></div>
                       <div className="stat"><span>Min</span><span>{snapshotData.min.toFixed(6)}</span></div>
                       <div className="stat"><span>Max</span><span>{snapshotData.max.toFixed(6)}</span></div>
                    </div>
                  )}
                </div>
                <div className="weight-main">
                  {snapshotData ? (
                    <HeatmapChart
                      values={snapshotData.values}
                      colormap={weightColormap}
                      height={600}
                      onColormapChange={setWeightColormap}
                      title={`${selectedLayer} - Epoch ${selectedEpoch}`}
                    />
                  ) : <div className="empty-weights">Recording data not found.</div>}

                  {snapshotStats[selectedLayer] && (
                    <div className="weight-timeline">
                       <h4>Progression</h4>
                       <LineChart 
                          series={[
                            { label: 'Mean', color: '#60a5fa', data: snapshotStats[selectedLayer].map(s => ({ x: s.epoch as number, y: s.mean as number })) },
                            { label: 'Std', color: '#fb923c', data: snapshotStats[selectedLayer].map(s => ({ x: s.epoch as number, y: s.std as number })) }
                          ]}
                          height={200}
                       />
                    </div>
                  )}
                </div>
             </div>
          )}
        </div>
      </div>
    </div>
  );
}
