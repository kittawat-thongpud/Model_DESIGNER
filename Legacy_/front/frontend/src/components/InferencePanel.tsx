import React, { useState, useRef, useEffect } from 'react';
import type { PredictResponse } from '../types';
import { Upload, Play, AlertCircle, CheckCircle, ImageIcon } from 'lucide-react';

interface Props {
  modelId: string;
  weightId?: string;
}

interface DetectionBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label_id: number;
  label_name?: string;
  score: number;
}

export default function InferencePanel({ modelId, weightId }: Props) {
  const [image, setImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result as string);
        setResult(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const runInference = async () => {
    if (!image) return;
    
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelId,
          weight_id: weightId,
          image_base64: image,
        }),
      });
      
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Prediction failed');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (result && result.task_type === 'detection' && result.boxes && imgRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const img = imgRef.current;
      
      if (ctx) {
        // Set canvas to image display size
        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        (result.boxes as DetectionBox[]).forEach(box => {
          // Box coords are normalized 0-1
          const x = box.x1 * canvas.width;
          const y = box.y1 * canvas.height;
          const w = (box.x2 - box.x1) * canvas.width;
          const h = (box.y2 - box.y1) * canvas.height;
          
          ctx.strokeStyle = '#ef4444';
          ctx.lineWidth = 2;
          ctx.strokeRect(x, y, w, h);
          
          // Label
          ctx.fillStyle = '#ef4444';
          const labelText = `${box.label_name || box.label_id} (${(box.score * 100).toFixed(1)}%)`;
          ctx.font = '12px Inter, sans-serif';
          const textWidth = ctx.measureText(labelText).width;
          ctx.fillRect(x, y - 18, textWidth + 10, 18);
          
          ctx.fillStyle = 'white';
          ctx.fillText(labelText, x + 5, y - 5);
        });
      }
    }
  }, [result]);

  return (
    <div className="flex flex-col h-full bg-slate-900/50">
      {/* Header */}
      <div className="p-4 border-b border-slate-800">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <ImageIcon size={16} className="text-indigo-400" />
          Model Inference
        </h3>
        <p className="text-xs text-slate-500 mt-1">Test your model with local images</p>
      </div>

      {/* Upload Area */}
      <div className="p-4">
        <label className="block cursor-pointer">
          <input type="file" accept="image/*" onChange={handleFileChange} hidden />
          {image ? (
            <div className="relative rounded-lg overflow-hidden border border-slate-700">
              <img 
                ref={imgRef}
                src={image} 
                alt="Preview" 
                className="w-full h-48 object-contain bg-slate-950" 
              />
              <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-32 rounded-lg border-2 border-dashed border-slate-700 hover:border-slate-600 hover:bg-slate-800/50 transition-colors">
              <Upload size={24} className="text-slate-500 mb-2" />
              <span className="text-sm text-slate-400">Click to upload image</span>
            </div>
          )}
        </label>
      </div>

      {/* Run Button */}
      <div className="px-4 pb-4">
        <button 
          className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
          onClick={runInference} 
          disabled={!image || isLoading}
        >
          <Play size={14} />
          {isLoading ? 'Running...' : 'Run Prediction'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="mx-4 mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2 text-sm text-red-400">
          <AlertCircle size={14} />
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="flex-1 overflow-y-auto px-4 pb-4">
          <div className="bg-slate-950 rounded-lg border border-slate-800 p-4 space-y-3">
            <div className="flex items-center gap-2 text-emerald-400">
              <CheckCircle size={16} />
              <span className="text-sm font-medium">Prediction Complete</span>
            </div>
            
            {result.task_type === 'classification' ? (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-500">Prediction:</span>
                  <span className="text-white font-medium">{result.class_name || result.class_id}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-500">Confidence:</span>
                  <span className="text-emerald-400 font-medium">{(result.confidence! * 100).toFixed(2)}%</span>
                </div>
              </div>
            ) : (
              <div className="flex justify-between text-sm">
                <span className="text-slate-500">Objects Detected:</span>
                <span className="text-white font-medium">{(result.boxes as DetectionBox[])?.length || 0}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
