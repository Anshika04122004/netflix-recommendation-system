import React, { useState } from 'react';
import { 
  Sparkles, Film, Tv, Clock, Globe, Calendar, 
  User, Check, AlertCircle, ArrowUpRight, ShieldCheck, RefreshCw
} from 'lucide-react';
import { predictRating } from '../services/api';

const PRESETS = [
  {
    name: 'Sci-Fi & Horror Series',
    type: 'TV Show',
    title: 'Stranger Dimensions',
    director: 'The Duffer Brothers',
    country: 'United States',
    date_added: '2021-07-15',
    release_year: 2016,
    duration_val: '4',
    duration_unit: 'Seasons',
    genres: ['TV Sci-Fi & Fantasy', 'TV Horror', 'TV Dramas']
  },
  {
    name: 'Crime Epic Movie',
    type: 'Movie',
    title: 'The Syndicated Godfather',
    director: 'Martin Scorsese',
    country: 'United States',
    date_added: '2019-11-27',
    release_year: 2019,
    duration_val: '209',
    duration_unit: 'min',
    genres: ['Dramas', 'Crime TV Shows']
  },
  {
    name: 'Family Animated Adventure',
    type: 'Movie',
    title: 'Kingdom of Wonder',
    director: 'Rajiv Chilaka',
    country: 'India',
    date_added: '2020-05-10',
    release_year: 2019,
    duration_val: '86',
    duration_unit: 'min',
    genres: ['Children & Family Movies', 'Comedies']
  },
  {
    name: 'Stand-Up Comedy Special',
    type: 'Movie',
    title: 'Raw & Unfiltered Live',
    director: 'Stan Lathan',
    country: 'United States',
    date_added: '2021-10-05',
    release_year: 2021,
    duration_val: '67',
    duration_unit: 'min',
    genres: ['Stand-Up Comedy']
  }
];

const ALL_GENRES = [
  'Action & Adventure', 'Anime Features', 'Anime Series', 'British TV Shows',
  'Children & Family Movies', 'Classic & Cult TV', 'Classic Movies', 'Comedies',
  'Crime TV Shows', 'Cult Movies', 'Documentaries', 'Docuseries', 'Dramas',
  'Faith & Spirituality', 'Horror Movies', 'Independent Movies', 'International Movies',
  'International TV Shows', "Kids' TV", 'Korean TV Shows', 'LGBTQ Movies',
  'Movies', 'Music & Musicals', 'Reality TV', 'Romantic Movies', 'Romantic TV Shows',
  'Sci-Fi & Fantasy', 'Science & Nature TV', 'Spanish-Language TV Shows',
  'Sports Movies', 'Stand-Up Comedy', 'Stand-Up Comedy & Talk Shows',
  'TV Action & Adventure', 'TV Comedies', 'TV Dramas', 'TV Horror',
  'TV Mysteries', 'TV Sci-Fi & Fantasy', 'TV Shows', 'TV Thrillers',
  'Teen TV Shows', 'Thrillers'
];

const TOP_COUNTRIES = [
  'United States', 'India', 'United Kingdom', 'Pakistan', 'Canada',
  'Japan', 'South Korea', 'France', 'Spain', 'Mexico',
  'Egypt', 'Australia', 'Turkey', 'Nigeria', 'Germany', 'Not Given'
];

const RatingPrediction = () => {
  const [formData, setFormData] = useState({
    type: 'Movie',
    title: 'The Dark Horizon',
    director: 'Christopher Nolan',
    cast: 'Leonardo DiCaprio, Cillian Murphy',
    country: 'United States',
    date_added: '2021-09-25',
    release_year: 2020,
    duration_val: '148',
    duration_unit: 'min',
    genres: ['Action & Adventure', 'Sci-Fi & Fantasy'],
    description: 'An action-packed sci-fi thriller about heroes fighting interdimensional threats.'
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const applyPreset = (preset) => {
    setFormData({
      type: preset.type,
      title: preset.title,
      director: preset.director,
      country: preset.country,
      date_added: preset.date_added,
      release_year: preset.release_year,
      duration_val: preset.duration_val,
      duration_unit: preset.duration_unit,
      genres: [...preset.genres]
    });
  };

  const toggleGenre = (genre) => {
    setFormData(prev => {
      const exists = prev.genres.includes(genre);
      if (exists) {
        return { ...prev, genres: prev.genres.filter(g => g !== genre) };
      } else {
        return { ...prev, genres: [...prev.genres, genre] };
      }
    });
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const payload = {
        type: formData.type,
        title: formData.title,
        director: formData.director || 'Not Given',
        cast: formData.cast || '',
        country: formData.country || 'United States',
        date_added: formData.date_added || '2021-09-25',
        release_year: parseInt(formData.release_year) || 2020,
        duration: `${formData.duration_val} ${formData.duration_unit}`,
        listed_in: formData.genres.length > 0 ? formData.genres.join(', ') : 'Documentaries',
        description: formData.description || ''
      };

      const res = await predictRating(payload);
      setResult(res);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'Prediction failed. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  React.useEffect(() => {
    if (!result) {
      handleSubmit();
    }
  }, []);

  return (
    <div className="space-y-8 animate-fade-in pb-12">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-200 text-blue-700 text-xs font-semibold uppercase tracking-wider mb-2">
          <Sparkles className="w-3.5 h-3.5" /> Interactive ML Inference Engine
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900">Audience Rating Prediction</h1>
        <p className="text-slate-500 text-sm mt-1">
          Specify content metadata attributes to predict its parental classification category and probability distribution.
        </p>
      </div>

      {/* Preset Quick Buttons */}
      <div className="dashboard-card p-4">
        <span className="text-xs uppercase font-semibold text-slate-500 tracking-wider block mb-2.5">
          ⚡ Quick Test Presets:
        </span>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.name}
              type="button"
              onClick={() => applyPreset(p)}
              className="px-3 py-1.5 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-xs font-medium text-slate-700 hover:text-slate-900 transition-all flex items-center gap-1.5"
            >
              <Sparkles className="w-3 h-3 text-blue-600" />
              {p.name}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Form & Output */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Form Inputs (7 Cols) */}
        <div className="lg:col-span-7 dashboard-card p-6 sm:p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Content Type Toggle */}
            <div>
              <label className="text-xs uppercase font-semibold text-slate-600 tracking-wider block mb-2">
                Content Format *
              </label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setFormData({ ...formData, type: 'Movie', duration_unit: 'min', duration_val: '105' })}
                  className={`flex items-center justify-center gap-2 py-2.5 rounded-lg border text-sm font-semibold transition-all ${
                    formData.type === 'Movie'
                      ? 'bg-blue-50 border-blue-600 text-blue-700 shadow-sm'
                      : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  <Film className="w-4 h-4 text-blue-600" /> Movie (Feature Film)
                </button>
                <button
                  type="button"
                  onClick={() => setFormData({ ...formData, type: 'TV Show', duration_unit: 'Seasons', duration_val: '2' })}
                  className={`flex items-center justify-center gap-2 py-2.5 rounded-lg border text-sm font-semibold transition-all ${
                    formData.type === 'TV Show'
                      ? 'bg-blue-50 border-blue-600 text-blue-700 shadow-sm'
                      : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  <Tv className="w-4 h-4 text-blue-600" /> TV Show (Series)
                </button>
              </div>
            </div>

            {/* Title & Director Row */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-semibold text-slate-600 uppercase tracking-wider block mb-1.5">
                  Content Title
                </label>
                <input
                  type="text"
                  value={formData.title}
                  onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                  placeholder="e.g. Inception"
                  className="w-full bg-white border border-slate-300 rounded-lg px-3.5 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm"
                />
                <span className="text-[10px] text-slate-400 mt-1 block">
                  * Excluded from ML features to avoid overfitting
                </span>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-600 uppercase tracking-wider block mb-1.5">
                  Director Name(s)
                </label>
                <input
                  type="text"
                  value={formData.director}
                  onChange={(e) => setFormData({ ...formData, director: e.target.value })}
                  placeholder="e.g. Christopher Nolan, Rajiv Chilaka"
                  className="w-full bg-white border border-slate-300 rounded-lg px-3.5 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm"
                />
              </div>
            </div>

            {/* Country & Date Added */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-semibold text-slate-600 uppercase tracking-wider block mb-1.5">
                  Country of Origin
                </label>
                <select
                  value={formData.country}
                  onChange={(e) => setFormData({ ...formData, country: e.target.value })}
                  className="w-full bg-white border border-slate-300 rounded-lg px-3.5 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm"
                >
                  {TOP_COUNTRIES.map(c => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-600 uppercase tracking-wider block mb-1.5">
                  Date Added to Netflix
                </label>
                <input
                  type="date"
                  value={formData.date_added}
                  onChange={(e) => setFormData({ ...formData, date_added: e.target.value })}
                  className="w-full bg-white border border-slate-300 rounded-lg px-3.5 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm"
                />
              </div>
            </div>

            {/* Release Year & Duration */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-semibold text-slate-600 uppercase tracking-wider block mb-1.5">
                  Release Year: <span className="text-slate-900 font-bold">{formData.release_year}</span>
                </label>
                <input
                  type="range"
                  min="1950"
                  max="2025"
                  value={formData.release_year}
                  onChange={(e) => setFormData({ ...formData, release_year: e.target.value })}
                  className="w-full accent-blue-600 mt-2"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-600 uppercase tracking-wider block mb-1.5">
                  Duration
                </label>
                <div className="flex gap-2">
                  <input
                    type="number"
                    min="1"
                    value={formData.duration_val}
                    onChange={(e) => setFormData({ ...formData, duration_val: e.target.value })}
                    className="w-2/3 bg-white border border-slate-300 rounded-lg px-3.5 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm"
                  />
                  <select
                    value={formData.duration_unit}
                    onChange={(e) => setFormData({ ...formData, duration_unit: e.target.value })}
                    className="w-1/3 bg-white border border-slate-300 rounded-lg px-2.5 py-2 text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm"
                  >
                    <option value="min">min</option>
                    <option value="Season">Season</option>
                    <option value="Seasons">Seasons</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Genre Multi-Select */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-semibold text-slate-600 uppercase tracking-wider">
                  Genres & Categories ({formData.genres.length} Selected) *
                </label>
                <span className="text-[11px] text-slate-400">Select all that apply</span>
              </div>
              <div className="max-h-48 overflow-y-auto p-3 rounded-lg bg-slate-50 border border-slate-200 flex flex-wrap gap-1.5">
                {ALL_GENRES.map(g => {
                  const isSelected = formData.genres.includes(g);
                  return (
                    <button
                      key={g}
                      type="button"
                      onClick={() => toggleGenre(g)}
                      className={`text-xs px-2.5 py-1 rounded-md transition-all flex items-center gap-1 ${
                        isSelected
                          ? 'bg-blue-600 text-white font-medium shadow-sm'
                          : 'bg-white text-slate-600 hover:text-slate-900 hover:bg-slate-100 border border-slate-200'
                      }`}
                    >
                      {isSelected && <Check className="w-3 h-3" />}
                      {g}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-semibold text-sm shadow-sm transition-all flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" /> Classifying Content...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" /> Predict Audience Rating
                </>
              )}
            </button>
          </form>
        </div>

        {/* Prediction Results Display (5 Cols) */}
        <div className="lg:col-span-5 space-y-6">
          {error && (
            <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
              {error}
            </div>
          )}

          {result && (
            <div className="dashboard-card p-6 sm:p-8 space-y-6">
              <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                <span className="text-xs uppercase font-semibold text-slate-500 tracking-wider">
                  Classification Output
                </span>
                <span className="text-xs text-blue-700 font-mono font-medium flex items-center gap-1 bg-blue-50 px-2 py-0.5 rounded border border-blue-100">
                  <ShieldCheck className="w-3.5 h-3.5" /> {result.model_used}
                </span>
              </div>

              {/* Rating Badge Display */}
              <div className="text-center py-5 bg-slate-50 rounded-xl border border-slate-200/80">
                <span className="inline-block px-5 py-2 rounded-lg text-3xl font-extrabold text-white bg-blue-600 shadow-sm tracking-wide">
                  {result.predicted_rating}
                </span>
                <div className="mt-2 text-sm font-bold text-slate-900">
                  {result.rating_category_info?.name}
                </div>
                <div className="inline-block mt-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-white border border-slate-200 text-slate-700">
                  Target Audience: {result.rating_category_info?.level}
                </div>
              </div>

              {/* Description */}
              <p className="text-xs text-slate-600 leading-relaxed bg-white p-3.5 rounded-lg border border-slate-200 shadow-sm">
                {result.rating_category_info?.description}
              </p>

              {/* Confidence Bar */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500 font-medium">Model Confidence</span>
                  <span className="font-bold text-slate-900">{(result.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                  <div 
                    className="h-full bg-blue-600 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(result.confidence * 100 * 1.5, 100)}%` }}
                  />
                </div>
              </div>

              {/* Top Probability Distribution */}
              <div>
                <h4 className="text-xs font-semibold text-slate-600 uppercase tracking-wider mb-2.5">
                  Category Probability Distribution
                </h4>
                <div className="space-y-2">
                  {Object.entries(result.probabilities || {})
                    .slice(0, 5)
                    .map(([cls, prob]) => (
                      <div key={cls} className="flex items-center gap-2 text-xs">
                        <span className="w-14 font-mono font-semibold text-slate-700">{cls}</span>
                        <div className="flex-1 bg-slate-100 rounded-full h-2 overflow-hidden">
                          <div 
                            className="h-full bg-blue-600 rounded-full"
                            style={{ width: `${(prob * 100).toFixed(1)}%` }}
                          />
                        </div>
                        <span className="w-12 text-right font-mono text-slate-500 font-medium">
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                </div>
              </div>

              {/* Active Feature Drivers */}
              <div className="pt-2 border-t border-slate-100">
                <span className="text-[11px] text-slate-500 uppercase tracking-wider block mb-2 font-semibold">
                  Key Feature Drivers:
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {result.top_contributing_features?.map((f, i) => (
                    <span key={i} className="text-[11px] px-2 py-0.5 rounded bg-slate-100 border border-slate-200 text-slate-600">
                      {f.feature}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RatingPrediction;
