import { useState, useEffect } from 'react';
import { Image, Download, X, ChevronLeft, ChevronRight } from 'lucide-react';
import { api } from '../services/api';

interface PlotsGalleryProps {
  jobId: string;
}

interface PlotInfo {
  name: string;
  path: string;
  size: number;
}

export default function PlotsGallery({ jobId }: PlotsGalleryProps) {
  const [plots, setPlots] = useState<PlotInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPlot, setSelectedPlot] = useState<PlotInfo | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);

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
            className="bg-gray-800 rounded-lg overflow-hidden hover:ring-2 hover:ring-blue-500 transition-all cursor-pointer group"
            onClick={() => openLightbox(plot, index)}
          >
            <div className="aspect-video bg-gray-900 relative">
              <img
                src={getPlotUrl(plot.name)}
                alt={plot.name}
                className="w-full h-full object-contain"
              />
              <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all flex items-center justify-center">
                <Image className="w-8 h-8 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            </div>
            <div className="p-3">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium text-white truncate">
                  {getPlotTitle(plot.name)}
                </h4>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    downloadPlot(plot);
                  }}
                  className="p-1 hover:bg-gray-700 rounded transition-colors"
                  title="Download"
                >
                  <Download className="w-4 h-4 text-gray-400" />
                </button>
              </div>
              <div className="text-xs text-gray-500 mt-1">
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
              <div className="p-4 bg-black">
                <img
                  src={getPlotUrl(selectedPlot.name)}
                  alt={selectedPlot.name}
                  className="w-full h-auto max-h-[70vh] object-contain mx-auto"
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
