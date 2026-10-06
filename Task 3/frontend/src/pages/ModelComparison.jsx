import React from 'react';
import { 
  BarChart3, Award, TrendingUp, Sliders, CheckCircle2, 
  Layers, Check, ArrowUpRight
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  Legend 
} from 'recharts';

const ModelComparison = ({ analyticsData }) => {
  const models = analyticsData?.model_comparison || [];
  const tuningComparison = analyticsData?.tuning_comparison || [];
  const tuningParams = analyticsData?.tuning_parameters || {};
  const bestModelName = analyticsData?.best_model_name || 'Random Forest (Tuned)';

  const chartData = models.map(m => ({
    name: m.model_name.replace(' (Baseline)', ' (Base)').replace(' (Tuned)', ' (Opt)'),
    Accuracy: parseFloat((m.accuracy * 100).toFixed(2)),
    'Weighted F1': parseFloat((m.f1_weighted * 100).toFixed(2)),
    'Macro F1': parseFloat((m.f1_macro * 100).toFixed(2)),
  }));

  return (
    <div className="space-y-8 animate-fade-in pb-12">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-200 text-blue-700 text-xs font-semibold uppercase tracking-wider mb-2">
          <BarChart3 className="w-3.5 h-3.5" /> Benchmarks & Validation
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900">Machine Learning Model Comparison</h1>
        <p className="text-slate-500 text-sm mt-1">
          Evaluating Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and a Voting Ensemble using 5-fold cross-validation and a 20% stratified test split.
        </p>
      </div>

      {/* Pre-Tuning vs Post-Tuning Optimization Impact Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {tuningComparison.map((tc) => (
          <div key={tc.model} className="dashboard-card p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-lg bg-blue-50 flex items-center justify-center text-blue-600">
                  <Sliders className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-slate-900 text-base">{tc.model} Optimization</h3>
                  <span className="text-xs text-slate-500">5-Fold Cross Validation Tuning</span>
                </div>
              </div>
              <div className="px-2.5 py-1 rounded-md bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-semibold flex items-center gap-1">
                <TrendingUp className="w-3.5 h-3.5" /> +{tc.improvement_acc}% Acc
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 pt-3 border-t border-slate-100">
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/80">
                <span className="text-xs text-slate-500 block mb-1">Baseline Accuracy</span>
                <span className="text-xl font-bold text-slate-800">{(tc.baseline_acc * 100).toFixed(2)}%</span>
                <span className="text-[11px] text-slate-500 block mt-0.5">F1: {(tc.baseline_f1 * 100).toFixed(2)}%</span>
              </div>
              <div className="p-3 rounded-lg bg-blue-50/60 border border-blue-100">
                <span className="text-xs text-blue-700 block mb-1 font-semibold">Tuned Accuracy</span>
                <span className="text-xl font-bold text-slate-900">{(tc.tuned_acc * 100).toFixed(2)}%</span>
                <span className="text-[11px] text-blue-600 block mt-0.5">F1: {(tc.tuned_f1 * 100).toFixed(2)}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Model Performance Comparison Bar Chart */}
      <div className="dashboard-card p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-lg font-bold text-slate-900">Comparative Accuracy & F1-Score Metrics</h2>
            <p className="text-xs sm:text-sm text-slate-500 mt-0.5">
              Side-by-side evaluation across baseline and optimized models on the held-out test split (1,758 samples).
            </p>
          </div>
          <span className="text-xs font-mono text-slate-600 px-2.5 py-1 rounded bg-slate-100 border border-slate-200 self-start sm:self-auto">
            Test Size: 20% (Stratified)
          </span>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 20, left: -10, bottom: 20 }}>
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} angle={-15} textAnchor="end" />
              <YAxis stroke="#94a3b8" fontSize={11} domain={[0, 70]} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                formatter={(val) => [`${val}%`, '']}
              />
              <Legend verticalAlign="top" height={36} />
              <Bar dataKey="Accuracy" fill="#2563EB" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Weighted F1" fill="#10B981" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Macro F1" fill="#6366F1" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Full Comprehensive Metrics Table */}
      <div className="dashboard-card p-6 sm:p-8">
        <div className="mb-6">
          <h2 className="text-lg font-bold text-slate-900">Detailed Evaluation Metrics Table</h2>
          <p className="text-xs sm:text-sm text-slate-500 mt-0.5">
            Precision, Recall, and F1 metrics calculated using weighted averages to account for class imbalance.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-xs font-semibold uppercase text-slate-500 bg-slate-50">
                <th className="py-3 px-4">Model Architecture</th>
                <th className="py-3 px-4">Accuracy</th>
                <th className="py-3 px-4">Weighted F1</th>
                <th className="py-3 px-4">Weighted Precision</th>
                <th className="py-3 px-4">Weighted Recall</th>
                <th className="py-3 px-4">Macro F1</th>
                <th className="py-3 px-4 text-right">Selection</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {models.map((m) => {
                const isChampion = m.model_name === bestModelName;
                return (
                  <tr key={m.model_name} className={`hover:bg-slate-50/80 transition-colors ${isChampion ? 'bg-blue-50/40' : ''}`}>
                    <td className="py-3.5 px-4 font-medium text-slate-900 flex items-center gap-2">
                      {isChampion && <Award className="w-4 h-4 text-amber-500 flex-shrink-0" />}
                      {m.model_name}
                    </td>
                    <td className="py-3.5 px-4 font-semibold text-slate-900">{(m.accuracy * 100).toFixed(2)}%</td>
                    <td className="py-3.5 px-4 font-semibold text-emerald-700">{(m.f1_weighted * 100).toFixed(2)}%</td>
                    <td className="py-3.5 px-4 text-slate-600">{(m.precision_weighted * 100).toFixed(2)}%</td>
                    <td className="py-3.5 px-4 text-slate-600">{(m.recall_weighted * 100).toFixed(2)}%</td>
                    <td className="py-3.5 px-4 text-indigo-700 font-mono">{(m.f1_macro * 100).toFixed(2)}%</td>
                    <td className="py-3.5 px-4 text-right">
                      {isChampion ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-blue-100 border border-blue-200 text-blue-800 text-xs font-semibold">
                          ✓ Best Production Model
                        </span>
                      ) : (
                        <span className="text-xs text-slate-400 font-mono">Baseline</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Hyperparameter Tuning Details Card */}
      <div className="dashboard-card p-6 sm:p-8">
        <h2 className="text-lg font-bold text-slate-900 mb-1 flex items-center gap-2">
          <Sliders className="w-5 h-5 text-blue-600" /> Hyperparameter Tuning Configurations (Task 3 Specs)
        </h2>
        <p className="text-xs sm:text-sm text-slate-500 mb-6">
          Rigorous 5-fold cross-validation parameter exploration for Decision Tree and Random Forest models.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Decision Tree Tuning Spec */}
          <div className="p-5 rounded-lg bg-slate-50 border border-slate-200 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-slate-900 text-sm">Decision Tree (GridSearchCV)</h3>
              <span className="text-xs px-2 py-0.5 rounded bg-blue-100 text-blue-700 font-mono font-medium">
                450 Folds Evaluated
              </span>
            </div>
            <p className="text-xs text-slate-500">
              Tested max_depth, min_samples_split, min_samples_leaf, criterion on 5 stratified folds.
            </p>
            <div className="p-3 rounded-lg bg-white border border-slate-200 text-xs font-mono text-slate-700 space-y-1">
              <div><span className="text-slate-400">criterion:</span> "{tuningParams.decision_tree?.criterion || 'gini'}"</div>
              <div><span className="text-slate-400">max_depth:</span> {tuningParams.decision_tree?.max_depth || 12}</div>
              <div><span className="text-slate-400">min_samples_split:</span> {tuningParams.decision_tree?.min_samples_split || 5}</div>
              <div><span className="text-slate-400">min_samples_leaf:</span> {tuningParams.decision_tree?.min_samples_leaf || 1}</div>
            </div>
          </div>

          {/* Random Forest Tuning Spec */}
          <div className="p-5 rounded-lg bg-slate-50 border border-slate-200 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-slate-900 text-sm">Random Forest (RandomizedSearchCV)</h3>
              <span className="text-xs px-2 py-0.5 rounded bg-emerald-100 text-emerald-700 font-mono font-medium">
                75 Folds Evaluated
              </span>
            </div>
            <p className="text-xs text-slate-500">
              Explored n_estimators, max_depth, min_samples_split, min_samples_leaf, and max_features.
            </p>
            <div className="p-3 rounded-lg bg-white border border-slate-200 text-xs font-mono text-slate-700 space-y-1">
              <div><span className="text-slate-400">n_estimators:</span> {tuningParams.random_forest?.n_estimators || 100}</div>
              <div><span className="text-slate-400">max_depth:</span> {tuningParams.random_forest?.max_depth || 30}</div>
              <div><span className="text-slate-400">min_samples_split:</span> {tuningParams.random_forest?.min_samples_split || 5}</div>
              <div><span className="text-slate-400">min_samples_leaf:</span> {tuningParams.random_forest?.min_samples_leaf || 1}</div>
              <div><span className="text-slate-400">max_features:</span> "{tuningParams.random_forest?.max_features || 'sqrt'}"</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelComparison;
