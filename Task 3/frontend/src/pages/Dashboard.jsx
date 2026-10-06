import React from 'react';
import { 
  Database, Tag, Award, Film, Tv, Sparkles, 
  ArrowRight, CheckCircle2, TrendingUp, Cpu
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  PieChart, Pie, Cell, Legend 
} from 'recharts';

const Dashboard = ({ analyticsData, onNavigate }) => {
  const dataset = analyticsData?.dataset || {};
  const models = analyticsData?.model_comparison || [];
  const bestModelName = analyticsData?.best_model_name || 'Random Forest (Tuned)';
  const bestModel = analyticsData?.best_model_metrics || {};

  const ratingDist = dataset?.rating_distribution || [];
  const typeDist = dataset?.type_distribution || [];

  return (
    <div className="space-y-8 animate-fade-in pb-12">
      {/* Hero Banner */}
      <div className="dashboard-card p-6 sm:p-8 bg-gradient-to-r from-blue-50/50 via-white to-slate-50">
        <div className="max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-100/70 border border-blue-200 text-blue-700 text-xs font-semibold uppercase tracking-wider">
            <Sparkles className="w-3.5 h-3.5" /> Netflix ML Classification Pipeline
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight leading-tight">
            Netflix Audience Rating <span className="text-blue-600">Classifier</span>
          </h1>
          <p className="text-slate-600 text-base leading-relaxed">
            Multi-class machine learning classification system engineered to analyze content attributes and predict parental audience rating categories across 8,790 official Netflix titles.
          </p>
          <div className="pt-2 flex flex-wrap gap-3">
            <button
              onClick={() => onNavigate('predict')}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-semibold text-sm transition-all duration-150 shadow-sm"
            >
              <Sparkles className="w-4 h-4" /> Try Rating Predictor
            </button>
            <button
              onClick={() => onNavigate('models')}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-white hover:bg-slate-50 text-slate-700 font-semibold text-sm transition-all duration-150 border border-slate-300 shadow-sm"
            >
              <Cpu className="w-4 h-4 text-slate-500" /> Compare Models
            </button>
          </div>
        </div>
      </div>

      {/* 4 Stat Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* Stat 1 */}
        <div className="dashboard-card dashboard-card-hover p-6">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-wider font-semibold text-slate-500">Catalog Volume</span>
            <div className="w-10 h-10 rounded-lg bg-blue-50 flex items-center justify-center text-blue-600">
              <Database className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-4">
            <div className="text-3xl font-bold text-slate-900">
              {dataset?.total_records?.toLocaleString() || '8,790'}
            </div>
            <p className="text-xs text-slate-500 mt-1 flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" /> 100% Complete Dataset (0 missing)
            </p>
          </div>
        </div>

        {/* Stat 2 */}
        <div className="dashboard-card dashboard-card-hover p-6">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-wider font-semibold text-slate-500">Target Categories</span>
            <div className="w-10 h-10 rounded-lg bg-amber-50 flex items-center justify-center text-amber-600">
              <Tag className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-4">
            <div className="text-3xl font-bold text-slate-900">14 Classes</div>
            <p className="text-xs text-slate-500 mt-1">TV-MA, TV-14, TV-PG, R, PG-13, TV-Y7...</p>
          </div>
        </div>

        {/* Stat 3 */}
        <div className="dashboard-card dashboard-card-hover p-6">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-wider font-semibold text-slate-500">Best Model Accuracy</span>
            <div className="w-10 h-10 rounded-lg bg-emerald-50 flex items-center justify-center text-emerald-600">
              <Award className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-4">
            <div className="text-3xl font-bold text-slate-900">
              {(bestModel.accuracy * 100).toFixed(2)}%
            </div>
            <p className="text-xs text-emerald-700 mt-1 flex items-center gap-1 font-medium">
              <TrendingUp className="w-3.5 h-3.5 text-emerald-600" /> F1 Weighted: {(bestModel.f1_weighted * 100).toFixed(2)}%
            </p>
          </div>
        </div>

        {/* Stat 4 */}
        <div className="dashboard-card dashboard-card-hover p-6">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-wider font-semibold text-slate-500">Content Balance</span>
            <div className="w-10 h-10 rounded-lg bg-indigo-50 flex items-center justify-center text-indigo-600">
              <Film className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-4">
            <div className="text-3xl font-bold text-slate-900">6,126 / 2,664</div>
            <p className="text-xs text-slate-500 mt-1">69.7% Movies vs 30.3% TV Shows</p>
          </div>
        </div>
      </div>

      {/* Charts Section: Rating Distribution & Content Type */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Rating Distribution Bar Chart */}
        <div className="lg:col-span-2 dashboard-card p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-base font-bold text-slate-900">Audience Rating Distribution</h2>
              <p className="text-xs text-slate-500">Official classification breakdown across 8,790 catalog titles</p>
            </div>
            <span className="text-xs px-2.5 py-1 rounded bg-slate-100 text-slate-600 border border-slate-200 font-mono">
              14 Categories
            </span>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ratingDist} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <XAxis dataKey="rating" stroke="#94a3b8" fontSize={11} angle={-30} textAnchor="end" />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                  itemStyle={{ color: '#0f172a' }}
                />
                <Bar dataKey="count" fill="#2563EB" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Content Type Donut */}
        <div className="dashboard-card p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h2 className="text-base font-bold text-slate-900">Catalog Composition</h2>
              <span className="text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">
                Format
              </span>
            </div>
            <p className="text-xs text-slate-500 mb-4">Proportion of feature films versus television series</p>
          </div>

          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={typeDist}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={85}
                  paddingAngle={4}
                  dataKey="value"
                >
                  <Cell fill="#2563EB" />
                  <Cell fill="#0EA5E9" />
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                />
                <Legend verticalAlign="bottom" height={36} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-2 gap-3 pt-3 border-t border-slate-100 text-center">
            <div className="p-2.5 rounded-lg bg-blue-50/60 border border-blue-100">
              <span className="text-xs text-slate-500 flex items-center justify-center gap-1 font-medium"><Film className="w-3.5 h-3.5 text-blue-600" /> Movies</span>
              <span className="text-sm font-bold text-slate-900">6,126 (69.7%)</span>
            </div>
            <div className="p-2.5 rounded-lg bg-sky-50/60 border border-sky-100">
              <span className="text-xs text-slate-500 flex items-center justify-center gap-1 font-medium"><Tv className="w-3.5 h-3.5 text-sky-600" /> TV Shows</span>
              <span className="text-sm font-bold text-slate-900">2,664 (30.3%)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Model Leaderboard Snapshot */}
      <div className="dashboard-card p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
              <Award className="w-5 h-5 text-amber-500" /> Machine Learning Model Evaluation Leaderboard
            </h2>
            <p className="text-xs sm:text-sm text-slate-500 mt-0.5">
              Empirical metrics evaluated on 20% held-out test split (1,758 samples) with 5-fold cross-validation.
            </p>
          </div>
          <button
            onClick={() => onNavigate('models')}
            className="inline-flex items-center gap-1.5 text-sm font-semibold text-blue-600 hover:text-blue-700 transition-colors"
          >
            Full Comparison <ArrowRight className="w-4 h-4" />
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-xs font-semibold uppercase text-slate-500 bg-slate-50">
                <th className="py-3 px-4">Rank & Model</th>
                <th className="py-3 px-4">Accuracy</th>
                <th className="py-3 px-4">Weighted F1</th>
                <th className="py-3 px-4">Weighted Precision</th>
                <th className="py-3 px-4">Weighted Recall</th>
                <th className="py-3 px-4 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {models.map((m, idx) => {
                const isChampion = m.model_name === bestModelName;
                return (
                  <tr key={m.model_name} className={`hover:bg-slate-50/80 transition-colors ${isChampion ? 'bg-blue-50/40' : ''}`}>
                    <td className="py-3.5 px-4 font-medium text-slate-900 flex items-center gap-2.5">
                      <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                        idx === 0 ? 'bg-amber-100 text-amber-800' : 'bg-slate-100 text-slate-600'
                      }`}>
                        {idx + 1}
                      </span>
                      {m.model_name}
                    </td>
                    <td className="py-3.5 px-4 font-semibold text-slate-900">{(m.accuracy * 100).toFixed(2)}%</td>
                    <td className="py-3.5 px-4 text-emerald-700 font-semibold">{(m.f1_weighted * 100).toFixed(2)}%</td>
                    <td className="py-3.5 px-4 text-slate-600">{(m.precision_weighted * 100).toFixed(2)}%</td>
                    <td className="py-3.5 px-4 text-slate-600">{(m.recall_weighted * 100).toFixed(2)}%</td>
                    <td className="py-3.5 px-4 text-right">
                      {isChampion ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-blue-100 border border-blue-200 text-blue-800 text-xs font-semibold">
                          ★ Best Model
                        </span>
                      ) : (
                        <span className="text-xs text-slate-400 font-mono">Evaluated</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
