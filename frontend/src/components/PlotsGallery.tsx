import { useState, useEffect } from 'react';
import { Image, Download, X, ChevronLeft, ChevronRight, ImageOff } from 'lucide-react';
import { api } from '../services/api';

interface PlotsGalleryProps {
  jobId: string;
}

interface PlotInfo {
  name: string;
  path: string;
  size: number;
}

function PlotThumbnail({ src, alt, onClick }: { src: string; alt: string; onClick: () => void }) {
  const [status, setStatus] = useState<'loading' | 'loaded' | 'error'>('loading');
  return (
    <div
      className="aspect-video bg-slate-900 relative overflow-hidden cursor-pointer"
      onClick={onClick}
    >
      {/* Skeleton while loading */}
      {status === 'loading' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
          <div className="w-10 h-10 rounded-full border-2 border-slate-700 border-t-indigo-500 animate-spin" />
          <span className="text-[10px] text-slate-600 font-mono">loading…</span>
        </div>
      )}
      {/* Error state */}
      {status === 'error' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-600">
          <ImageOff size={24} />
          <span className="text-xs">Failed to load</span>
        </div>
      )}
      <img
        src={src}
        alt={alt}
        onLoad={() => setStatus('loaded')}
        onError={() => setStatus('error')}
        className={`w-full h-full object-contain transition-opacity duration-300 ${status === 'loaded' ? 'opacity-100' : 'opacity-0'}`}
      />
      {/* Hover overlay */}
      {status === 'loaded' && (
        <div className="absolute inset-0 bg-black/0 hover:bg-black/25 transition-all flex items-center justify-center">
          <Image className="w-7 h-7 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
        </div>
      )}
    </div>
  );
}

export default function PlotsGallery({ jobId }: PlotsGalleryProps) {
  const [plots, setPlots] = useState<PlotInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPlot, setSelectedPlot] = useState<PlotInfo | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [lightboxLoaded, setLightboxLoaded] = useState(false);

  useEffect(() => {
    loadPlots();
  }, [jobId]);

  const loadPlots = async () => {
    try {
      setLoading(true);
      const result = await api.getTrainPlots(jobId);
      setPlots(result.plots);
    } catch (error) {
      console.error('Failed to load plots:', error);
    } finally {
      setLoading(false);
    }
  };

  const getPlotUrl = (plotName: string) => {
    return api.getTrainPlotImage(jobId, plotName);
  };

  const openLightbox = (plot: PlotInfo, index: number) => {
    setSelectedPlot(plot);
    setSelectedIndex(index);
  };

  const closeLightbox = () => {
    setSelectedPlot(null);
  };

  const navigatePlot = (direction: 'prev' | 'next') => {
    if (!plots.length) return;
    let newIndex = selectedIndex;
    if (direction === 'prev') {
      newIndex = selectedIndex > 0 ? selectedIndex - 1 : plots.length - 1;
    } else {
      newIndex = selectedIndex < plots.length - 1 ? selectedIndex + 1 : 0;
    }
    setSelectedIndex(newIndex);
    setSelectedPlot(plots[newIndex]);
    setLightboxLoaded(false);
  };

  const openLightboxAt = (plot: PlotInfo, index: number) => {
    setSelectedPlot(plot);
    setSelectedIndex(index);
    setLightboxLoaded(false);
  };

  const downloadPlot = (plot: PlotInfo) => {
    const url = getPlotUrl(plot.name);
    const a = document.createElement('a');
    a.href = url;
    a.download = plot.name;
    a.click();
  };

  const getPlotTitle = (name: string) => {
    return name
      .replace('.png', '')
      .replace('.jpg', '')
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (l) => l.toUpperCase());
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (plots.length === 0) {
    return (
      <div className="text-center py-12 text-gray-400">
        <Image className="w-12 h-12 mx-auto mb-3 opacity-50" />
        <p>No plots available yet</p>
        <p className="text-sm mt-1">Plots will be generated during training</p>
      </div>
    );
  }

  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {plots.map((plot, index) => (
          <div
            key={plot.name}
            className="bg-slate-800 rounded-xl overflow-hidden hover:ring-2 hover:ring-indigo-500 transition-all cursor-pointer group border border-slate-700/50"
          >
            <PlotThumbnail
              src={getPlotUrl(plot.name)}
              alt={plot.name}
              onClick={() => openLightboxAt(plot, index)}
            />
            <div className="p-3">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium text-white truncate">
                  {getPlotTitle(plot.name)}
                </h4>
                <button
                  onClick={(e) => { e.stopPropagation(); downloadPlot(plot); }}
                  className="p-1 hover:bg-slate-700 rounded transition-colors"
                  title="Download"
                >
                  <Download className="w-4 h-4 text-slate-400" />
                </button>
              </div>
              <div className="text-xs text-slate-500 mt-1">
                {(plot.size / 1024).toFixed(1)} KB
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Lightbox */}
      {selectedPlot && (
        <div
          className="fixed inset-0 bg-black bg-opacity-90 z-50 flex items-center justify-center"
          onClick={closeLightbox}
        >
          <button
            onClick={closeLightbox}
            className="absolute top-4 right-4 p-2 bg-gray-800 hover:bg-gray-700 rounded-full transition-colors"
          >
            <X className="w-6 h-6 text-white" />
          </button>

          <button
            onClick={(e) => {
              e.stopPropagation();
              navigatePlot('prev');
            }}
            className="absolute left-4 p-2 bg-gray-800 hover:bg-gray-700 rounded-full transition-colors"
          >
            <ChevronLeft className="w-6 h-6 text-white" />
          </button>

          <button
            onClick={(e) => {
              e.stopPropagation();
              navigatePlot('next');
            }}
            className="absolute right-4 p-2 bg-gray-800 hover:bg-gray-700 rounded-full transition-colors"
          >
            <ChevronRight className="w-6 h-6 text-white" />
          </button>

          <div
            className="max-w-6xl max-h-[90vh] w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <div className="p-4 border-b border-gray-800 flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-white">
                    {getPlotTitle(selectedPlot.name)}
                  </h3>
                  <p className="text-sm text-gray-400">
                    {selectedIndex + 1} of {plots.length}
                  </p>
                </div>
                <button
                  onClick={() => downloadPlot(selectedPlot)}
                  className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded flex items-center gap-2 transition-colors"
                >
                  <Download className="w-4 h-4" />
                  Download
                </button>
              </div>
              <div className="p-4 bg-black relative min-h-[200px] flex items-center justify-center">
                {!lightboxLoaded && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black">
                    <div className="w-10 h-10 rounded-full border-2 border-slate-700 border-t-indigo-400 animate-spin" />
                    <span className="text-xs text-slate-500">Loading image…</span>
                  </div>
                )}
                <img
                  key={selectedPlot.name}
                  src={getPlotUrl(selectedPlot.name)}
                  alt={selectedPlot.name}
                  onLoad={() => setLightboxLoaded(true)}
                  className={`w-full h-auto max-h-[70vh] object-contain mx-auto transition-opacity duration-300 ${lightboxLoaded ? 'opacity-100' : 'opacity-0'}`}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
