"use client";

import { useMemo, useRef, useState } from "react";
import { Activity, ArrowUpRight, CheckCircle2, ChevronRight, CloudUpload, Image as ImageIcon, Layers3, LoaderCircle, RotateCcw, ShieldCheck, Sparkles, Waves } from "lucide-react";

type Prediction = { class: string; confidence: number };
const classes = [
  { name: "Buildings", detail: "Urban structures and architecture", icon: "▥" },
  { name: "Forest", detail: "Woodland and dense vegetation", icon: "♧" },
  { name: "Glacier", detail: "Ice fields and frozen landscapes", icon: "❄" },
  { name: "Mountain", detail: "Mountain ranges and peaks", icon: "△" },
  { name: "Sea", detail: "Oceans, shores and open water", icon: "≈" },
  { name: "Street", detail: "Roads and city streets", icon: "▤" },
];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [results, setResults] = useState<Prediction[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;
  const primary = results?.[0];
  const ready = Boolean(file && apiUrl);
  const percent = (n: number) => (n * 100).toFixed(1) + "%";

  const setImage = (next: File | undefined) => {
    if (!next) return;
    if (!next.type.startsWith("image/")) { setError("Choose a valid image file."); return; }
    if (next.size > 10 * 1024 * 1024) { setError("Image must be 10 MB or smaller."); return; }
    setFile(next); setPreview(URL.createObjectURL(next)); setResults(null); setError(null);
  };

  const classify = async () => {
    if (!file) return;
    if (!apiUrl) { setError("The prediction service has not been connected yet. The website is ready, but the model API still needs its hosting URL."); return; }
    setLoading(true); setError(null); setResults(null);
    try {
      const body = new FormData(); body.append("file", file);
      const response = await fetch(apiUrl.replace(/\/$/, "") + "/predict", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Prediction failed. Please try another image.");
      if (!Array.isArray(payload.predictions) || payload.predictions.length === 0) throw new Error("The prediction service returned an unexpected response.");
      setResults(payload.predictions);
    } catch (e) { setError(e instanceof Error ? e.message : "Could not reach the prediction service."); }
    finally { setLoading(false); }
  };

  const reset = () => { setFile(null); setPreview(null); setResults(null); setError(null); if (inputRef.current) inputRef.current.value = ""; };
  const topLabel = useMemo(() => primary?.class.replaceAll("_", " ") ?? "", [primary]);

  return <main className="shell">
    <header className="topbar"><a className="brand" href="#"><span className="brand-mark"><Waves size={19}/></span><span>terrain<span className="brand-ai">.ai</span></span></a><nav><a href="#how-it-works">How it works</a><a href="#categories">Categories</a></nav><div className="top-status"><span className="status-dot"/> MODEL RESEARCH PROJECT</div></header>
    <section className="hero"><div className="hero-copy"><div className="eyebrow"><Sparkles size={14}/> COMPUTER VISION, MADE SIMPLE</div><h1>Give an image.<br/><span>Get its landscape.</span></h1><p className="hero-text">An image classification system trained to recognise six real-world scene categories, powered by a fine-tuned deep learning model.</p><div className="hero-metrics"><div><strong>91.70%</strong><span>Test accuracy</span></div><i/><div><strong>6</strong><span>Scene categories</span></div><i/><div><strong>150×150</strong><span>Input resolution</span></div></div></div><div className="hero-art" aria-hidden="true"><div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="art-sun"/><div className="mountain back"/><div className="mountain front"/><div className="art-glow"/></div></section>
    <section className="workspace"><div className="section-heading"><div><div className="eyebrow muted">LIVE CLASSIFICATION</div><h2>Try the classifier</h2><p>Upload a landscape or scene image to get started.</p></div><div className="model-pill"><span className="status-dot"/> MODEL READY <span className="pill-divider"/> INCEPTION FINE-TUNED</div></div>
      <div className="work-grid"><div className="upload-card"><div className="card-top"><div><span className="step">01</span><h3>Upload an image</h3></div><span className="file-hint">JPG, PNG, WEBP · MAX 10 MB</span></div>
        {!preview ? <button className={`dropzone ${dragging ? "dragging" : ""}`} onClick={() => inputRef.current?.click()} onDragOver={e => {e.preventDefault();setDragging(true);}} onDragLeave={() => setDragging(false)} onDrop={e => {e.preventDefault();setDragging(false);setImage(e.dataTransfer.files?.[0]);}}><span className="upload-icon"><CloudUpload size={25}/></span><strong>Drop your image here</strong><span>or <b>browse files</b> from your device</span><small>Your image is sent securely to the classifier for prediction.</small></button> : <div className="preview-wrap"><img src={preview} alt="Selected image preview"/><div className="preview-overlay"><span><ImageIcon size={15}/>{file?.name}</span><button onClick={reset} aria-label="Remove image"><RotateCcw size={16}/></button></div></div>}
        <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp,image/bmp" hidden onChange={e => setImage(e.target.files?.[0])}/>
        {error && <div className="error-box">{error}</div>}
        <button className="classify-btn" disabled={!file || loading} onClick={classify}>{loading ? <><LoaderCircle className="spin" size={17}/> Analysing image...</> : <>Classify image <ArrowUpRight size={17}/></>}</button>
        <div className="privacy-note"><ShieldCheck size={14}/> Images are processed for classification and are not stored by this interface.</div>
      </div>
      <div className="result-card"><div className="card-top"><div><span className="step">02</span><h3>Prediction result</h3></div><span className="result-state">{results ? "COMPLETE" : "AWAITING IMAGE"}</span></div>
        {!results ? <div className="empty-result"><div className="empty-icon"><Activity size={24}/></div><strong>Your result will appear here</strong><span>Upload an image and run the classifier to see its predicted category and confidence scores.</span><div className="empty-line"><i/><i/><i/></div></div> : <div className="results-content"><div className="prediction-label">MOST LIKELY CATEGORY</div><div className="prediction-main"><div><span className="prediction-icon">✦</span><strong>{topLabel}</strong></div><b>{percent(primary?.confidence ?? 0)}</b></div><div className="confidence-track"><i style={{width: percent(primary?.confidence ?? 0)}}/></div><div className="top-three-title">TOP PREDICTIONS <span>MODEL SCORES</span></div>{results.map((item, index) => <div className="score-row" key={item.class}><div className="score-name"><span>{String(index+1).padStart(2,"0")}</span>{item.class.replaceAll("_"," ")}</div><div className="score-bar"><i style={{width:percent(item.confidence)}}/></div><b>{percent(item.confidence)}</b></div>)}<button className="reset-btn" onClick={reset}><RotateCcw size={14}/> Classify another image</button></div>}
      </div></div>
    </section>
    <section className="categories" id="categories"><div className="section-heading compact"><div><div className="eyebrow muted">WHAT IT RECOGNISES</div><h2>Six scene categories</h2><p>The model predicts one of these classes for every uploaded image.</p></div><span className="category-count">01 — 06</span></div><div className="category-grid">{classes.map((item,index)=><div className="category-item" key={item.name}><span className="category-number">0{index+1}</span><span className="category-icon">{item.icon}</span><div><strong>{item.name}</strong><small>{item.detail}</small></div><ChevronRight size={16}/></div>)}</div></section>
    <section className="how" id="how-it-works"><div className="how-intro"><div className="eyebrow muted">UNDER THE HOOD</div><h2>From pixels<br/>to prediction.</h2><p>A fine-tuned Inception-based convolutional neural network extracts visual patterns and ranks the six scene categories.</p></div><div className="how-steps"><div><span>01</span><div><strong>Image input</strong><p>Your image is resized to 150 × 150 pixels and prepared for the model.</p></div></div><div><span>02</span><div><strong>Feature extraction</strong><p>The trained neural network analyses shapes, textures and visual features.</p></div></div><div><span>03</span><div><strong>Class prediction</strong><p>The model returns its three highest-scoring scene categories.</p></div></div></div></section>
    <footer><a className="brand" href="#"><span className="brand-mark"><Waves size={17}/></span><span>terrain<span className="brand-ai">.ai</span></span></a><span>Built as a computer vision research project.</span><span>MODEL TEST ACCURACY <b>91.70%</b></span></footer>
  </main>;
}
