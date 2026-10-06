import React, { useState } from 'react';
import { 
  PieChart as PieIcon, BarChart3, TrendingUp, Globe, Film, 
  Tag, Compass, Grid, Award
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  LineChart, Line, Legend 
} from 'recharts';

const Analytics = ({ analyticsData }) => {
  const [activeSubTab, setActiveSubTab] = useState('distributions');
  const dataset = analyticsData?.dataset || {};
  const featureImportances = analyticsData?.feature_importances || [];
  const bestModelMetrics = analyticsData?.best_model_metrics || {};
  const classes = analyticsData?.classes || [];

  const ratingByType = dataset?.rating_by_type || [];
  const releaseTrends = dataset?.release_trends || [];
  const topCountries = dataset?.top_countries || [];
  const topGenres = dataset?.top_genres || [];
  const durationStats = dataset?.duration_stats || {};
  const confusionMatrix = bestModelMetrics?.confusion_matrix || [];

  return (
    <div className="space-y-8 animate-fade-in pb-12">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-200 text-blue-700 text-xs font-semibold uppercase tracking-wider mb-2">
          <Compass className="w-3.5 h-3.5" /> Exploratory Data Analysis & Diagnostics
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900">Catalog & Diagnostic Analytics</h1>
        <p className="text-slate-500 text-sm mt-1">
          Detailed analysis of Netflix rating distributions, release patterns, genre dynamics, feature importances, and confusion matrix diagnostics.
        </p>
      </div>

      {/* Sub-tab Navigation */}
      <div className="flex border-b border-slate-200 space-x-6">
        {[
          { id: 'distributions', label: 'Ratings & Formats', icon: BarChart3 },
          { id: 'trends', label: 'Release Trends & Duration', icon: TrendingUp },
          { id: 'regions', label: 'Countries & Genres', icon: Globe },
          { id: 'diagnostics', label: 'Confusion Matrix & Features', icon: Grid }
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeSubTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id)}
              className={`flex items-center gap-2 pb-3 text-sm font-semibold border-b-2 transition-all duration-150 ${
                isActive
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-slate-500 hover:text-slate-900'
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* TAB 1: Ratings & Formats */}
      {activeSubTab === 'distributions' && (
        <div className="space-y-6">
          {/* Stacked Bar Chart: Ratings by Content Type */}
          <div className="dashboard-card p-6 sm:p-8">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
              <div>
                <h3 className="text-lg font-bold text-slate-900">Audience Ratings by Content Format</h3>
                <p className="text-xs sm:text-sm text-slate-500 mt-0.5">
                  Distribution of Movies vs TV Shows across each audience rating category.
                </p>
              </div>
              <div className="flex items-center gap-4 text-xs font-medium">
                <span className="flex items-center gap-1.5 text-slate-700">
                  <span className="w-3 h-3 rounded bg-blue-600" /> Movies
                </span>
                <span className="flex items-center gap-1.5 text-slate-700">
                  <span className="w-3 h-3 rounded bg-sky-500" /> TV Shows
                </span>
              </div>
            </div>

            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={ratingByType} margin={{ top: 20, right: 20, left: -10, bottom: 20 }}>
                  <XAxis dataKey="rating" stroke="#94a3b8" fontSize={11} angle={-30} textAnchor="end" />
                  <YAxis stroke="#94a3b8" fontSize={11} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                  />
                  <Bar dataKey="Movie" stackId="a" fill="#2563EB" />
                  <Bar dataKey="TV Show" stackId="a" fill="#0EA5E9" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: Release Trends & Duration */}
      {activeSubTab === 'trends' && (
        <div className="space-y-6">
          {/* Release Trends Line Chart */}
          <div className="dashboard-card p-6 sm:p-8">
            <div className="mb-6">
              <h3 className="text-lg font-bold text-slate-900">Netflix Content Release Volume by Year</h3>
              <p className="text-xs sm:text-sm text-slate-500 mt-0.5">
                Annual catalog volume trends for productions released between 2000 and 2021.
              </p>
            </div>

            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={releaseTrends} margin={{ top: 10, right: 20, left: -10, bottom: 10 }}>
                  <XAxis dataKey="year" stroke="#94a3b8" fontSize={11} />
                  <YAxis stroke="#94a3b8" fontSize={11} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="count" 
                    stroke="#2563EB" 
                    strokeWidth={2.5} 
                    dot={{ fill: '#2563EB', r: 3 }} 
                    activeDot={{ r: 5 }} 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Duration Metrics Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="dashboard-card p-5 text-center">
              <span className="text-xs text-slate-500 block mb-1 font-medium">Movie Average Duration</span>
              <span className="text-2xl font-bold text-slate-900">{durationStats.movie_avg_duration_min} min</span>
            </div>
            <div className="dashboard-card p-5 text-center">
              <span className="text-xs text-slate-500 block mb-1 font-medium">Movie Median Duration</span>
              <span className="text-2xl font-bold text-slate-900">{durationStats.movie_median_duration_min} min</span>
            </div>
            <div className="dashboard-card p-5 text-center">
              <span className="text-xs text-slate-500 block mb-1 font-medium">TV Show Avg Seasons</span>
              <span className="text-2xl font-bold text-blue-600">{durationStats.tv_avg_seasons} Seasons</span>
            </div>
            <div className="dashboard-card p-5 text-center">
              <span className="text-xs text-slate-500 block mb-1 font-medium">Max TV Series Seasons</span>
              <span className="text-2xl font-bold text-indigo-600">{durationStats.tv_max_seasons} Seasons</span>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: Countries & Genres */}
      {activeSubTab === 'regions' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Top Countries */}
          <div className="dashboard-card p-6">
            <h3 className="text-base font-bold text-slate-900 mb-0.5">Top 10 Producing Countries</h3>
            <p className="text-xs text-slate-500 mb-4">Total titles originating from top global territories</p>
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topCountries} layout="vertical" margin={{ top: 10, right: 20, left: 30, bottom: 10 }}>
                  <XAxis type="number" stroke="#94a3b8" fontSize={10} />
                  <YAxis type="category" dataKey="country" stroke="#94a3b8" fontSize={11} width={80} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                  />
                  <Bar dataKey="count" fill="#2563EB" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Top Genres */}
          <div className="dashboard-card p-6">
            <h3 className="text-base font-bold text-slate-900 mb-0.5">Top 15 Netflix Genres</h3>
            <p className="text-xs text-slate-500 mb-4">Frequency of multi-label category occurrences</p>
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topGenres} layout="vertical" margin={{ top: 10, right: 20, left: 40, bottom: 10 }}>
                  <XAxis type="number" stroke="#94a3b8" fontSize={10} />
                  <YAxis type="category" dataKey="genre" stroke="#94a3b8" fontSize={10} width={110} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                  />
                  <Bar dataKey="count" fill="#10B981" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: Diagnostics (Confusion Matrix & Feature Importances) */}
      {activeSubTab === 'diagnostics' && (
        <div className="space-y-6">
          {/* Top Feature Importances */}
          <div className="dashboard-card p-6 sm:p-8">
            <div className="mb-4">
              <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                <Award className="w-5 h-5 text-amber-500" /> Random Forest Feature Importance (Top 15)
              </h3>
              <p className="text-xs sm:text-sm text-slate-500 mt-0.5">
                Derived using Mean Decrease in Impurity across 184 engineered features.
              </p>
            </div>

            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart 
                  data={featureImportances.slice(0, 15)} 
                  layout="vertical"
                  margin={{ top: 10, right: 20, left: 80, bottom: 10 }}
                >
                  <XAxis type="number" stroke="#94a3b8" fontSize={10} domain={[0, 'auto']} />
                  <YAxis type="category" dataKey="feature" stroke="#94a3b8" fontSize={11} width={130} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                    formatter={(val) => [(val * 100).toFixed(2) + '%', 'Importance']}
                  />
                  <Bar dataKey="importance" fill="#2563EB" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Confusion Matrix Heatmap */}
          <div className="dashboard-card p-6 sm:p-8">
            <div className="mb-4">
              <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                <Grid className="w-5 h-5 text-blue-600" /> 14-Class Confusion Matrix (Tuned Random Forest)
              </h3>
              <p className="text-xs sm:text-sm text-slate-500 mt-0.5">
                Actual test set distribution (1,758 predictions). Diagonal elements represent accurate predictions.
              </p>
            </div>

            <div className="overflow-x-auto pt-2">
              <div className="min-w-[650px]">
                <table className="w-full text-center text-xs border-collapse">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-200">
                      <th className="p-1.5 text-left text-slate-500 font-mono text-[10px]">Actual \ Pred</th>
                      {classes.map(cls => (
                        <th key={cls} className="p-1.5 text-slate-700 font-mono text-[10px] w-9">
                          {cls}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {confusionMatrix.map((row, rowIdx) => {
                      const actualClass = classes[rowIdx];
                      return (
                        <tr key={actualClass} className="hover:bg-slate-50 border-b border-slate-100">
                          <td className="p-1.5 text-left font-mono font-bold text-slate-800 text-[10px]">
                            {actualClass}
                          </td>
                          {row.map((val, colIdx) => {
                            const isDiagonal = rowIdx === colIdx;
                            const isHigh = val > 50;
                            return (
                              <td 
                                key={colIdx} 
                                className={`p-1.5 font-mono text-[11px] rounded ${
                                  isDiagonal 
                                    ? isHigh ? 'bg-blue-600 text-white font-bold' : 'bg-blue-100 text-blue-900 font-semibold' 
                                    : val > 0 ? 'bg-slate-100/70 text-slate-700' : 'text-slate-300'
                                }`}
                              >
                                {val}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Analytics;
