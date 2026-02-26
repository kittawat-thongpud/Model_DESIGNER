import { useState, useEffect } from 'react';
import { Image as ImageIcon, Download, X, ChevronLeft, ChevronRight, Layers } from 'lucide-react';
import { api } from '../services/api';

interface ClassSamplesGalleryProps {
  jobId: string;
}

interface ClassSampleInfo {
  name: string;
  count: number;
  images: string[];
}

export default function ClassSamplesGallery({ jobId }: ClassSamplesGalleryProps) {
  const [classes, setClasses] = useState<ClassSampleInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedClass, setSelectedClass] = useState<string | null>(null);
  
  // Lightbox state
  const [lightboxImage, setLightboxImage] = useState<{className: string, filename: string} | null>(null);

  useEffect(() => {
    loadSamples();
  }, [jobId]);

  const loadSamples = async () => {
    try {
      setLoading(true);
      const result = await api.getClassSamples(jobId);
      setClasses(result.classes);
      if (result.classes.length > 0 && !selectedClass) {
        setSelectedClass(result.classes[0].name);
      }
    } catch (error) {
      console.error('Failed to load class samples:', error);
    } finally {
      setLoading(false);
    }
  };

  const getImageUrl = (className: string, filename: string) => {
    return api.getClassSampleImage(jobId, className, filename);
  };

  const currentClassInfo = classes.find(c => c.name === selectedClass);

  const openLightbox = (className: string, filename: string) => {
    setLightboxImage({ className, filename });
  };

  const closeLightbox = () => {
    setLightboxImage(null);
  };

  const navigateLightbox = (direction: 'prev' | 'next') => {
    if (!lightboxImage || !currentClassInfo) return;
    
    const currentIndex = currentClassInfo.images.indexOf(lightboxImage.filename);
    if (currentIndex === -1) return;
    
    let newIndex;
    if (direction === 'prev') {
      newIndex = currentIndex > 0 ? currentIndex - 1 : currentClassInfo.images.length - 1;
    } else {
      newIndex = currentIndex < currentClassInfo.images.length - 1 ? currentIndex + 1 : 0;
    }
    
    setLightboxImage({
      className: currentClassInfo.name,
      filename: currentClassInfo.images[newIndex]
    });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
      </div>
    );
  }

  if (classes.length === 0) {
    return null; // Don't show anything if no samples
  }

  return (
    <div className="space-y-4">
      {/* Class Selector Tabs */}
      <div className="flex flex-wrap gap-2 pb-2 border-b border-slate-800">
        {classes.map((cls) => (
          <button
            key={cls.name}
            onClick={() => setSelectedClass(cls.name)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center gap-2 ${
              selectedClass === cls.name
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/20'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
            }`}
          >
            {cls.name}
            <span className={`px-1.5 py-0.5 rounded-md text-[10px] ${
              selectedClass === cls.name ? 'bg-indigo-500/50' : 'bg-slate-700 text-slate-500'
            }`}>
              {cls.count}
            </span>
          </button>
        ))}
      </div>

      {/* Grid */}
      {currentClassInfo && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {currentClassInfo.images.map((img) => (
            <div
              key={img}
              className="group relative aspect-square bg-slate-900 rounded-xl overflow-hidden border border-slate-800 hover:border-indigo-500/50 transition-all cursor-pointer"
              onClick={() => openLightbox(currentClassInfo.name, img)}
            >
              <img
                src={getImageUrl(currentClassInfo.name, img)}
                alt={img}
                className="w-full h-full object-cover"
                loading="lazy"
              />
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-all flex items-center justify-center opacity-0 group-hover:opacity-100">
                <ImageIcon className="text-white w-6 h-6 drop-shadow-lg" />
              </div>
              <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent p-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <p className="text-[10px] text-slate-300 truncate font-mono">{img}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Lightbox */}
      {lightboxImage && (
        <div
          className="fixed inset-0 bg-black/95 z-[100] flex items-center justify-center backdrop-blur-sm"
          onClick={closeLightbox}
        >
          <button
            onClick={closeLightbox}
            className="absolute top-4 right-4 p-2 bg-slate-800/50 hover:bg-slate-700 text-slate-400 hover:text-white rounded-full transition-colors"
          >
            <X size={24} />
          </button>

          <button
            onClick={(e) => { e.stopPropagation(); navigateLightbox('prev'); }}
            className="absolute left-4 p-3 bg-slate-800/50 hover:bg-slate-700 text-slate-400 hover:text-white rounded-full transition-colors"
          >
            <ChevronLeft size={24} />
          </button>

          <button
            onClick={(e) => { e.stopPropagation(); navigateLightbox('next'); }}
            className="absolute right-4 p-3 bg-slate-800/50 hover:bg-slate-700 text-slate-400 hover:text-white rounded-full transition-colors"
          >
            <ChevronRight size={24} />
          </button>

          <div 
            className="max-w-7xl max-h-[90vh] w-full mx-4 flex flex-col items-center"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="relative rounded-lg overflow-hidden shadow-2xl border border-slate-800 bg-slate-900">
              <img
                src={getImageUrl(lightboxImage.className, lightboxImage.filename)}
                alt={lightboxImage.filename}
                className="max-h-[80vh] w-auto object-contain"
              />
              <div className="absolute top-0 inset-x-0 p-4 bg-gradient-to-b from-black/60 to-transparent flex justify-between items-start">
                <div>
                  <h3 className="text-white font-medium text-lg flex items-center gap-2">
                    <Layers size={18} className="text-indigo-400" />
                    {lightboxImage.className}
                  </h3>
                  <p className="text-slate-400 text-xs font-mono mt-1 opacity-80">{lightboxImage.filename}</p>
                </div>
                <a
                  href={getImageUrl(lightboxImage.className, lightboxImage.filename)}
                  download={lightboxImage.filename}
                  className="p-2 bg-white/10 hover:bg-white/20 text-white rounded-lg backdrop-blur-md transition-colors"
                  onClick={(e) => e.stopPropagation()}
                >
                  <Download size={18} />
                </a>
              </div>
            </div>
            
            {currentClassInfo && (
              <div className="mt-4 text-slate-500 text-sm font-medium bg-black/50 px-4 py-1.5 rounded-full border border-white/5 backdrop-blur-md">
                {currentClassInfo.images.indexOf(lightboxImage.filename) + 1} / {currentClassInfo.images.length}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
