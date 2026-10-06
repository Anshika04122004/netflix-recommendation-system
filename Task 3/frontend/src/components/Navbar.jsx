import React from 'react';
import { Film, LayoutDashboard, Sparkles, BarChart3, PieChart, Info } from 'lucide-react';

const Navbar = ({ activeTab, setActiveTab, isBackendHealthy }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'predict', label: 'Rating Prediction', icon: Sparkles },
    { id: 'models', label: 'Model Comparison', icon: BarChart3 },
    { id: 'analytics', label: 'Analytics & Insights', icon: PieChart },
    { id: 'about', label: 'About', icon: Info },
  ];

  return (
    <header className="sticky top-0 z-50 bg-white border-b border-slate-200 shadow-sm transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo */}
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => setActiveTab('dashboard')}>
            <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center shadow-sm">
              <Film className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold tracking-tight text-slate-900 font-sans">
                  Netflix Audience Classifier
                </span>
              </div>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all duration-150 ${
                    isActive
                      ? 'bg-blue-50 text-blue-700 shadow-sm font-semibold'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-blue-600' : 'text-slate-500'}`} />
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Status Indicator */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-50 border border-emerald-200 text-xs">
              <span className={`w-2 h-2 rounded-full ${isBackendHealthy ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`} />
              <span className="text-emerald-800 font-medium">
                {isBackendHealthy ? 'API Active • 8,790 Records' : 'Connecting API...'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Nav */}
      <div className="md:hidden flex overflow-x-auto px-4 py-2 border-t border-slate-200 space-x-2 bg-slate-50">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs whitespace-nowrap font-medium transition-all ${
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-600 hover:text-slate-900 bg-white border border-slate-200'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {item.label}
            </button>
          );
        })}
      </div>
    </header>
  );
};

export default Navbar;
