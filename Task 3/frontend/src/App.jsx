import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import RatingPrediction from './pages/RatingPrediction';
import ModelComparison from './pages/ModelComparison';
import Analytics from './pages/Analytics';
import About from './pages/About';
import { fetchHealth, fetchAnalytics } from './services/api';
import { Film, RefreshCw, AlertTriangle } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [analyticsData, setAnalyticsData] = useState(null);
  const [isBackendHealthy, setIsBackendHealthy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const health = await fetchHealth();
      setIsBackendHealthy(health.status === 'healthy');

      const data = await fetchAnalytics();
      setAnalyticsData(data);
    } catch (err) {
      console.error('Failed to load backend data:', err);
      setIsBackendHealthy(false);
      setError('Could not connect to FastAPI backend server on port 8000. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans antialiased selection:bg-blue-600 selection:text-white">
      {/* Top Navigation */}
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        isBackendHealthy={isBackendHealthy} 
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-8">
        {loading && !analyticsData ? (
          <div className="min-h-[60vh] flex flex-col items-center justify-center space-y-4">
            <div className="w-12 h-12 rounded-xl bg-blue-50 border border-blue-200 flex items-center justify-center animate-pulse text-blue-600 shadow-sm">
              <Film className="w-6 h-6" />
            </div>
            <div className="text-center">
              <h3 className="text-base font-bold text-slate-800 tracking-tight">Loading Catalog & ML Pipeline...</h3>
              <p className="text-xs text-slate-500 mt-1">Retrieving empirical metrics and catalog distributions</p>
            </div>
          </div>
        ) : error && !analyticsData ? (
          <div className="min-h-[60vh] flex flex-col items-center justify-center text-center p-6">
            <div className="w-12 h-12 rounded-xl bg-red-50 border border-red-200 text-red-600 flex items-center justify-center mb-3">
              <AlertTriangle className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-slate-900 mb-1">Backend Service Offline</h3>
            <p className="text-xs text-slate-500 max-w-md mb-5">{error}</p>
            <button
              onClick={loadData}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold transition-all shadow-sm"
            >
              <RefreshCw className="w-3.5 h-3.5" /> Retry Connection
            </button>
          </div>
        ) : (
          <>
            {activeTab === 'dashboard' && (
              <Dashboard 
                analyticsData={analyticsData} 
                onNavigate={setActiveTab} 
              />
            )}
            {activeTab === 'predict' && <RatingPrediction />}
            {activeTab === 'models' && <ModelComparison analyticsData={analyticsData} />}
            {activeTab === 'analytics' && <Analytics analyticsData={analyticsData} />}
            {activeTab === 'about' && <About />}
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-6 text-center text-xs text-slate-500 mt-12">
        <div className="max-w-7xl mx-auto px-4 space-y-1.5">
            <span>Netflix Audience Rating Classification System</span>
          <p className="text-slate-500 text-[11px]">
            Engineered with Scikit-Learn (Random Forest, Gradient Boosting, ExtraTrees, Decision Tree, Logistic Regression), FastAPI, and React.js.
          </p>
          <p className="text-[11px] text-slate-400">
            Based on Netflix Dataset.csv (8,790 records, 14 multi-class categories).
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
