import React, { useState, useRef, useEffect } from 'react';
import type { PredictResponse, DetectionBox } from '../types';

interface Props {
  modelId: string;
  weightId?: string;
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
      const response = await fetch('http://localhost:8000/api/predict', {
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
    } catch (err: any) {
      setError(err.message);
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
        
        result.boxes.forEach(box => {
          // Box coords are normalized 0-1
          const x = box.x1 * canvas.width;
          const y = box.y1 * canvas.height;
          const w = (box.x2 - box.x1) * canvas.width;
          const h = (box.y2 - box.y1) * canvas.height;
          
          ctx.strokeStyle = '#ff3860';
          ctx.lineWidth = 2;
          ctx.strokeRect(x, y, w, h);
          
          // Label
          ctx.fillStyle = '#ff3860';
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
    <div className="inference-panel">
      <div className="panel-header">
        <h3>Model Inference</h3>
        <p className="subtitle">Test your model with local images</p>
      </div>

      <div className="inference-upload">
        <label className="upload-box">
          <input type="file" accept="image/*" onChange={handleFileChange} hidden />
          {image ? (
            <div className="image-preview-container">
              <img 
                ref={imgRef}
                src={image} 
                alt="Preview" 
                className="image-preview" 
              />
              <canvas ref={canvasRef} className="detection-canvas" />
            </div>
          ) : (
            <div className="upload-placeholder">
              <span className="icon">🖼️</span>
              <span>Click to upload image</span>
            </div>
          )}
        </label>
      </div>

      <div className="actions">
        <button 
          className="btn btn-primary" 
          onClick={runInference} 
          disabled={!image || isLoading}
        >
          {isLoading ? 'Running...' : 'Run Prediction'}
        </button>
      </div>

      {error && <div className="error-msg">{error}</div>}

      {result && (
        <div className="inference-results">
          {result.task_type === 'classification' ? (
            <div className="classification-res">
              <div className="metric">
                <span className="label">Prediction:</span>
                <span className="value">{result.class_name || result.class_id}</span>
              </div>
              <div className="metric">
                <span className="label">Confidence:</span>
                <span className="value">{(result.confidence! * 100).toFixed(2)}%</span>
              </div>
            </div>
          ) : (
            <div className="detection-res">
              <span className="label">Objects Detected:</span>
              <span className="value">{result.boxes?.length || 0}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
