import React from 'react';
import { 
  Info, ShieldCheck, Database, Sliders, CheckCircle2, 
  Cpu, Layers, AlertCircle, FileText
} from 'lucide-react';

const About = () => {
  return (
    <div className="space-y-8 animate-fade-in pb-12 max-w-5xl mx-auto">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-200 text-blue-700 text-xs font-semibold uppercase tracking-wider mb-2">
          <Info className="w-3.5 h-3.5" /> Project Overview & Methodology
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900">About Audience Rating Classifier</h1>
        <p className="text-slate-500 text-sm mt-1">
          Netflix Audience Rating Classifier — Comprehensive Documentation & Architecture.
        </p>
      </div>

      {/* Section 1: Executive Summary & Objective */}
      <div className="dashboard-card p-6 sm:p-8 space-y-4">
        <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-blue-600" /> Problem Statement & Objective
        </h2>
        <p className="text-slate-600 text-sm leading-relaxed">
          Task 3 is the Medium-level <strong>“Netflix Audience Rating Classification”</strong> task assigned during the Auspify Technologies Machine Learning Internship. The central objective is to formulate, implement, and tune multi-class classification machine learning models to accurately predict the parental audience rating category of streaming content (e.g., <code>TV-MA</code>, <code>TV-14</code>, <code>R</code>, <code>PG-13</code>, <code>TV-PG</code>, <code>TV-Y7</code>, <code>TV-Y</code>) using exclusively intrinsic content attributes.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
          <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
            <span className="text-xs text-slate-500 block mb-0.5 font-medium">Target Variable</span>
            <span className="text-base font-bold text-slate-900 font-mono">rating</span>
            <span className="text-[11px] text-slate-500 block mt-0.5">14 Discrete Classes</span>
          </div>
          <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
            <span className="text-xs text-slate-500 block mb-0.5 font-medium">Dataset Source</span>
            <span className="text-base font-bold text-slate-900">Dataset.csv (Netflix)</span>
            <span className="text-[11px] text-emerald-700 font-medium block mt-0.5">8,790 Records (0 Missing)</span>
          </div>
          <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
            <span className="text-xs text-slate-500 block mb-0.5 font-medium">Evaluation Focus</span>
            <span className="text-base font-bold text-slate-900">Weighted F1 & Accuracy</span>
            <span className="text-[11px] text-blue-700 font-medium block mt-0.5">5-Fold Cross Validation</span>
          </div>
        </div>
      </div>

      {/* Section 2: Dataset & Feature Engineering */}
      <div className="dashboard-card p-6 sm:p-8 space-y-4">
        <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
          <Database className="w-5 h-5 text-blue-600" /> Feature Engineering Architecture
        </h2>
        <p className="text-slate-600 text-sm leading-relaxed">
          In strict adherence to the assessment plan, we avoid memorization and ensure high generalization capacity by applying reproducible Scikit-Learn transformers:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
            <h4 className="text-sm font-bold text-slate-800 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" /> Excluded Attributes
            </h4>
            <ul className="text-xs text-slate-600 space-y-1 list-disc list-inside">
              <li><code className="text-rose-600">show_id</code>: Excluded as arbitrary identifier.</li>
              <li><code className="text-rose-600">title</code>: Excluded to prevent model from memorizing specific titles and maximize feature transferability.</li>
            </ul>
          </div>

          <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
            <h4 className="text-sm font-bold text-slate-800 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" /> Derived Temporal & Duration Features
            </h4>
            <ul className="text-xs text-slate-600 space-y-1 list-disc list-inside">
              <li><code className="text-blue-600">date_added</code>: Parsed into <code>added_year</code>, <code>added_month</code>, <code>added_day</code>.</li>
              <li><code className="text-blue-600">duration</code>: Decomposed into numeric <code>duration_value</code> and <code>duration_type</code> ('min' vs 'Season').</li>
              <li><code className="text-blue-600">content_age</code>: Derived as <code>added_year - release_year</code>.</li>
            </ul>
          </div>

          <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
            <h4 className="text-sm font-bold text-slate-800 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" /> Categorical & Genre Vectorization
            </h4>
            <ul className="text-xs text-slate-600 space-y-1 list-disc list-inside">
              <li><code className="text-indigo-600">listed_in</code>: Engineered into <code>genre_count</code> plus 42 individual binary genre indicators.</li>
              <li><code className="text-indigo-600">country</code>: Decomposed into <code>country_count</code> and top global production territories.</li>
              <li><code className="text-indigo-600">director</code>: Encoded for top directors, normalized presence, and safe 'Not Given' fallback.</li>
            </ul>
          </div>

          <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
            <h4 className="text-sm font-bold text-slate-800 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" /> Preprocessing Safety
            </h4>
            <ul className="text-xs text-slate-600 space-y-1 list-disc list-inside">
              <li><code>StandardScaler</code> for numerical features.</li>
              <li><code>OneHotEncoder(handle_unknown='ignore')</code> for unseen categorical inputs during inference.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Section 3: Modeling & Hyperparameter Optimization */}
      <div className="dashboard-card p-6 sm:p-8 space-y-4">
        <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
          <Cpu className="w-5 h-5 text-emerald-600" /> Modeling & Cross-Validation Strategy
        </h2>
        <p className="text-slate-600 text-sm leading-relaxed">
          Five machine learning architectures were trained, tuned, and combined into a champion ensemble:
        </p>

        <div className="space-y-3">
          <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
            <div className="flex justify-between items-center mb-1">
              <h4 className="text-sm font-bold text-slate-800">1. Logistic Regression (Baseline)</h4>
              <span className="text-xs font-mono text-slate-600 font-medium">Accuracy ~54% | F1 ~52%</span>
            </div>
            <p className="text-xs text-slate-500">
              Multinomial L-BFGS linear classifier providing a fast, interpretable benchmark. Uses balanced class weights to handle imbalanced rating distribution.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
            <div className="flex justify-between items-center mb-1">
              <h4 className="text-sm font-bold text-slate-800">2. Decision Tree (Baseline & Tuned)</h4>
              <span className="text-xs font-mono text-emerald-700 font-medium">Tuned via RandomizedSearchCV (20 iterations)</span>
            </div>
            <p className="text-xs text-slate-500">
              Tuned using 5-fold cross-validation across criterion, max_depth, min_samples_split, min_samples_leaf, and max_features. Pruning prevents overfitting on 184+ feature matrix.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
            <div className="flex justify-between items-center mb-1">
              <h4 className="text-sm font-bold text-slate-800">3. Random Forest (Tuned)</h4>
              <span className="text-xs font-mono text-slate-600 font-medium">300-400 estimators | balanced weights</span>
            </div>
            <p className="text-xs text-slate-500">
              Ensemble of decorrelated decision trees with randomized feature subsets. Tuned via RandomizedSearchCV across n_estimators, max_depth, max_features, and class_weight.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
            <div className="flex justify-between items-center mb-1">
              <h4 className="text-sm font-bold text-slate-800">4. Gradient Boosting (Tuned)</h4>
              <span className="text-xs font-mono text-purple-700 font-medium">Boosted trees | learning_rate tuned</span>
            </div>
            <p className="text-xs text-slate-500">
              Sequential boosting model tuned over learning_rate, n_estimators, max_depth, subsample, and max_features. Excels at capturing complex decision boundaries in the imbalanced rating space.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-blue-50/60 border border-blue-200">
            <div className="flex justify-between items-center mb-1">
              <h4 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                ★ 5. Voting Ensemble — RF + Gradient Boosting + ExtraTrees (Champion)
              </h4>
              <span className="text-xs font-mono text-blue-700 font-bold">Soft Voting | Best Accuracy</span>
            </div>
            <p className="text-xs text-slate-600">
              Soft-voting ensemble combining the tuned Random Forest, Gradient Boosting, and ExtraTrees classifiers. Averages class probability distributions to reduce variance and achieve the highest generalization across 14 rating categories.
            </p>
          </div>
        </div>
      </div>

      {/* Section 4: Real-world Limitations & Ethical Considerations */}
      <div className="dashboard-card p-6 sm:p-8 space-y-3">
        <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-amber-500" /> Limitations & Practical Considerations
        </h2>
        <div className="text-xs sm:text-sm text-slate-600 space-y-2 leading-relaxed">
          <p>
            • <strong>Class Imbalance:</strong> Two classes (<code>TV-MA</code> and <code>TV-14</code>) account for over 61% of all records in the catalog, while rare classes such as <code>UR</code>, <code>NC-17</code>, and <code>TV-Y7-FV</code> have fewer than 10 instances each. Stratified validation and weighted F1-metrics were strictly utilized.
          </p>
          <p>
            • <strong>Regional Discrepancies:</strong> Parental guidelines vary internationally. The catalog harmonizes these ratings into dominant North American streaming categories.
          </p>
          <p>
            • <strong>Text Feature Scope:</strong> TF-IDF embeddings on content descriptions and cast are included to enrich semantic signal while remaining computationally tractable.
          </p>
        </div>
      </div>
    </div>
  );
};

export default About;
