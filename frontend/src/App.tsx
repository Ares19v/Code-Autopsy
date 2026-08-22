import { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Loader2, Play, CheckCircle2, AlertCircle, Copy, Check, Activity, ShieldAlert, Cpu } from 'lucide-react';

interface ReviewResponse {
  bug_identified: string;
  root_cause: string;
  fixed_code: string;
  confidence: number;
  language: string;
  latency_ms: number;
}

function App() {
  const [code, setCode] = useState('');
  const [language, setLanguage] = useState<'python' | 'javascript'>('python');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ReviewResponse | null>(null);
  const [copied, setCopied] = useState(false);

  const handleAnalyze = async () => {
    if (!code.trim()) {
      setError('Please enter some code to analyze.');
      return;
    }
    
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch('http://localhost:8000/review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, language }),
      });

      if (!res.ok) {
        throw new Error(`API Error: ${res.statusText}`);
      }

      const data: ReviewResponse = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to connect to the Code Autopsy API.');
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = () => {
    if (result?.fixed_code) {
      navigator.clipboard.writeText(result.fixed_code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="min-h-screen bg-brand-dark text-gray-200 font-sans relative overflow-hidden selection:bg-brand-blue/30">
      
      {/* Background ambient glow */}
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-brand-blue/20 blur-[120px] rounded-full pointer-events-none"></div>
      <div className="absolute bottom-[-20%] right-[-10%] w-[40%] h-[40%] bg-brand-teal/10 blur-[100px] rounded-full pointer-events-none"></div>
      
      <div className="max-w-7xl mx-auto p-6 md:p-12 relative z-10 space-y-10">
        
        {/* Header */}
        <header className="flex items-center justify-between pb-6 border-b border-brand-muted/30">
          <div className="flex items-center space-x-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-blue to-brand-teal p-[1px]">
              <div className="w-full h-full bg-brand-dark rounded-xl flex items-center justify-center">
                <Activity className="w-5 h-5 text-brand-teal" />
              </div>
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-white bg-clip-text text-transparent bg-gradient-to-r from-gray-100 to-gray-400">
              Code Autopsy
            </h1>
          </div>
          <div className="flex items-center space-x-3 bg-brand-muted/20 border border-brand-muted/30 backdrop-blur-sm px-4 py-2 rounded-full text-sm font-medium shadow-sm">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-teal opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-brand-teal"></span>
            </span>
            <span className="text-gray-300">Model Active</span>
          </div>
        </header>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Left Column: Input */}
          <div className="space-y-4">
            <div className="flex justify-between items-center px-1">
              <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-widest flex items-center space-x-2">
                <Cpu className="w-4 h-4 text-brand-blue" />
                <span>Source Analysis</span>
              </h2>
              <div className="relative">
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value as 'python' | 'javascript')}
                  className="appearance-none bg-brand-muted/20 border border-brand-muted/50 backdrop-blur-md text-sm rounded-lg px-4 py-2 pr-8 focus:outline-none focus:ring-2 focus:ring-brand-blue/50 text-gray-200 cursor-pointer hover:bg-brand-muted/40 transition-colors"
                >
                  <option value="python">Python 3</option>
                  <option value="javascript">JavaScript / ES6</option>
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400">
                  <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/></svg>
                </div>
              </div>
            </div>
            
            <div className="relative group rounded-2xl overflow-hidden border border-brand-muted/40 bg-brand-dark/60 backdrop-blur-md shadow-2xl focus-within:border-brand-blue/50 transition-all duration-300">
              <div className="absolute top-0 left-0 w-10 h-full bg-brand-muted/10 border-r border-brand-muted/20 pointer-events-none flex flex-col items-center py-4 space-y-1 text-xs text-gray-600 font-mono">
                {Array.from({ length: 20 }).map((_, i) => (
                  <span key={i}>{i + 1}</span>
                ))}
              </div>
              <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder={`// Paste your buggy ${language} code here...\n\ndef target_function():\n    pass`}
                className="w-full h-[550px] bg-transparent text-gray-200 font-mono text-[13px] sm:text-sm p-4 pl-14 resize-none focus:outline-none leading-relaxed"
                spellCheck="false"
              />
            </div>

            <button
              onClick={handleAnalyze}
              disabled={loading || !code.trim()}
              className="w-full relative group overflow-hidden rounded-xl p-[1px] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              <span className="absolute inset-0 bg-gradient-to-r from-brand-blue to-brand-teal opacity-70 group-hover:opacity-100 transition-opacity blur-sm"></span>
              <div className="relative flex items-center justify-center space-x-2 bg-gradient-to-r from-brand-blue to-brand-teal text-white py-4 px-6 rounded-xl font-bold tracking-wide shadow-lg group-hover:shadow-[0_0_25px_rgba(2,173,202,0.5)] transition-all">
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Processing Inference...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5 fill-current" />
                    <span>Execute Code Autopsy</span>
                  </>
                )}
              </div>
            </button>

            {error && (
              <div className="flex items-start space-x-3 text-red-300 bg-red-950/40 p-4 rounded-xl border border-red-500/30 backdrop-blur-sm animate-in fade-in slide-in-from-top-2">
                <AlertCircle className="w-5 h-5 shrink-0 mt-0.5 text-red-400" />
                <p className="text-sm">{error}</p>
              </div>
            )}
          </div>

          {/* Right Column: Output */}
          <div className="space-y-6 flex flex-col h-full">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-widest flex items-center space-x-2 px-1">
              <ShieldAlert className="w-4 h-4 text-brand-teal" />
              <span>Diagnostic Report</span>
            </h2>
            
            {!result && !loading && (
              <div className="flex-1 flex flex-col items-center justify-center text-brand-muted border-2 border-brand-muted/20 border-dashed rounded-2xl bg-brand-muted/5 backdrop-blur-sm p-8 text-center min-h-[550px]">
                <div className="w-20 h-20 mb-6 rounded-full bg-brand-muted/10 flex items-center justify-center">
                  <Activity className="w-10 h-10 opacity-50" />
                </div>
                <h3 className="text-xl font-medium text-gray-400 mb-2">Ready for Analysis</h3>
                <p className="text-gray-500 max-w-sm">Enter your code in the editor and run the autopsy to identify bugs, root causes, and get a fixed implementation.</p>
              </div>
            )}

            {loading && (
              <div className="flex-1 flex flex-col items-center justify-center text-brand-blue space-y-6 min-h-[550px] border border-brand-muted/20 rounded-2xl bg-brand-dark/40 backdrop-blur-md">
                <div className="relative">
                  <div className="absolute inset-0 border-4 border-brand-blue/30 rounded-full animate-ping"></div>
                  <Loader2 className="w-12 h-12 animate-spin text-brand-teal relative z-10" />
                </div>
                <p className="font-mono text-sm tracking-widest uppercase text-brand-blue/80 animate-pulse">Running Neural Diagnostics...</p>
              </div>
            )}

            {result && !loading && (
              <div className="space-y-5 animate-in fade-in slide-in-from-bottom-8 duration-700 pb-8">
                
                {/* Meta stats */}
                <div className="flex space-x-3 text-xs font-mono">
                  <div className="flex items-center space-x-1.5 bg-brand-muted/30 border border-brand-muted/40 px-3 py-1.5 rounded-lg text-brand-blue">
                    <span className="w-1.5 h-1.5 rounded-full bg-brand-blue"></span>
                    <span>Confidence: {(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex items-center space-x-1.5 bg-brand-muted/30 border border-brand-muted/40 px-3 py-1.5 rounded-lg text-brand-teal">
                    <span className="w-1.5 h-1.5 rounded-full bg-brand-teal"></span>
                    <span>Latency: {result.latency_ms}ms</span>
                  </div>
                </div>

                {/* Bug Identified */}
                <div className="relative overflow-hidden bg-brand-muted/10 border border-brand-muted/30 rounded-2xl p-6 backdrop-blur-sm group hover:border-brand-blue/30 transition-colors">
                  <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-rose-500 to-orange-500"></div>
                  <h3 className="flex items-center space-x-2 text-white font-semibold mb-3">
                    <AlertCircle className="w-5 h-5 text-rose-400" />
                    <span>Bug Identified</span>
                  </h3>
                  <p className="text-[15px] text-gray-300 leading-relaxed font-medium">{result.bug_identified}</p>
                </div>

                {/* Root Cause */}
                <div className="relative overflow-hidden bg-brand-muted/10 border border-brand-muted/30 rounded-2xl p-6 backdrop-blur-sm group hover:border-brand-teal/30 transition-colors">
                  <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-brand-blue to-brand-teal"></div>
                  <h3 className="flex items-center space-x-2 text-white font-semibold mb-3">
                    <span className="text-brand-blue text-lg">🔍</span>
                    <span>Root Cause</span>
                  </h3>
                  <p className="text-[15px] text-gray-300 leading-relaxed whitespace-pre-wrap">{result.root_cause}</p>
                </div>

                {/* Fixed Code */}
                <div className="bg-[#1e1e1e] border border-brand-muted/40 rounded-2xl overflow-hidden shadow-2xl">
                  <div className="flex items-center justify-between bg-brand-dark/90 px-5 py-3 border-b border-brand-muted/30">
                    <h3 className="flex items-center space-x-2 text-white font-semibold text-sm">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      <span>Optimized Code</span>
                    </h3>
                    <button 
                      onClick={copyToClipboard}
                      className="flex items-center space-x-1.5 text-xs font-medium text-gray-400 hover:text-white bg-brand-muted/20 hover:bg-brand-muted/40 px-3 py-1.5 rounded-md transition-all"
                      title="Copy to clipboard"
                    >
                      {copied ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                          <span className="text-emerald-400">Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5" />
                          <span>Copy</span>
                        </>
                      )}
                    </button>
                  </div>
                  <div className="text-[13px] sm:text-sm">
                    <SyntaxHighlighter
                      language={result.language}
                      style={vscDarkPlus}
                      showLineNumbers={true}
                      customStyle={{
                        margin: 0,
                        padding: '1.25rem',
                        background: 'transparent',
                      }}
                    >
                      {result.fixed_code}
                    </SyntaxHighlighter>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;
