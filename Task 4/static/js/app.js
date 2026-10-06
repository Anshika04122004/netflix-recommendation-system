/**
 * Netflix Content Segmentation - Front-End Controller
 */

// Global State
let globalOverview = null;
let globalClusters = {};
let scatterPoints = [];
let activeClusterFilter = 'all';
let activeTypeFilter = 'All';
let currentCatalogPage = 1;
let currentCatalogSearch = '';
let currentCatalogCluster = null;
let optimizationChartInstance = null;

// Cluster Color Schemes
const CLUSTER_COLORS = {
    0: '#f43f5e', // Rose Ruby (Mainstream Comedy & Family)
    1: '#818cf8', // Indigo Violet (Global Serialized Dramas)
    2: '#f59e0b', // Warm Amber (Heritage Cinema Archives)
    3: '#10b981', // Emerald Teal (Real-World Docs & Stand-Up)
    4: '#38bdf8', // Sky Cyan (Youth Animation & TV)
    5: '#a855f7'  // Violet Amethyst (Global Indie Cinema)
};

// Preset Content Examples for Simulator
const SIMULATOR_PRESETS = {
    kdrama: {
        title: "Crash Landing on Destiny",
        type: "TV Show",
        year: 2022,
        duration: "2 Seasons",
        rating: "TV-MA",
        country: "South Korea",
        genres: "International TV Shows, Romantic TV Shows, TV Dramas, TV Mysteries"
    },
    standup: {
        title: "Midnight Uncensored Live",
        type: "Movie",
        year: 2021,
        duration: "68 min",
        rating: "TV-MA",
        country: "United States",
        genres: "Stand-Up Comedy, Comedies"
    },
    retro: {
        title: "Thunderbolt 1985",
        type: "Movie",
        year: 1985,
        duration: "108 min",
        rating: "PG-13",
        country: "United States",
        genres: "Action & Adventure, Classic Movies, Cult Movies"
    },
    kids: {
        title: "Magic Forest Explorers",
        type: "TV Show",
        year: 2023,
        duration: "3 Seasons",
        rating: "TV-Y7",
        country: "United Kingdom",
        genres: "Kids' TV, TV Comedies, British TV Shows"
    }
};

// Initialize Application on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

async function initApp() {
    try {
        await fetchOverviewData();
        await fetchClustersData();
        await initScatterCanvas();
        initOptimizationChart('elbow');
        populateBenchmarkTable();
        renderPersonaCards();
        initPredictor();
        initCatalogExplorer();
        initEventListeners();
    } catch (err) {
        console.error("Initialization error:", err);
    }
}

// --------------------------------------------------------------------------
// 1. Data Fetching
// --------------------------------------------------------------------------
async function fetchOverviewData() {
    const res = await fetch("/api/overview");
    globalOverview = await res.json();
    
    // Populate KPI values
    if (globalOverview.summary) {
        document.getElementById("kpi-total-titles").innerText = globalOverview.summary.total_records?.toLocaleString() || "8,790";
        document.getElementById("kpi-movies-count").innerText = `${globalOverview.summary.movie_count?.toLocaleString()} Movies`;
        document.getElementById("kpi-tv-count").innerText = `${globalOverview.summary.tv_show_count?.toLocaleString()} Shows`;
        document.getElementById("kpi-clusters-count").innerText = `${globalOverview.cluster_count || 6} Clusters`;
    }
}

async function fetchClustersData() {
    const res = await fetch("/api/clusters");
    globalClusters = await res.json();
    renderClusterLegend();
}

// --------------------------------------------------------------------------
// 2. Interactive Scatter Latent Canvas
// --------------------------------------------------------------------------
async function initScatterCanvas() {
    const res = await fetch("/api/scatter-points?limit=4000");
    scatterPoints = await res.json();
    renderScatterPlot();
}

function renderClusterLegend() {
    const container = document.getElementById("cluster-legend-items");
    if (!container) return;
    
    container.innerHTML = "";
    Object.values(globalClusters).forEach(c => {
        const item = document.createElement("div");
        item.className = "legend-item";
        item.dataset.cluster = c.cluster_id;
        item.innerHTML = `
            <div class="legend-left">
                <span class="legend-dot" style="background: ${c.color};"></span>
                <span class="legend-name">C${c.cluster_id}: ${c.name.split('&')[0]}</span>
            </div>
            <span class="legend-count">${c.total_titles}</span>
        `;
        item.addEventListener("click", () => {
            const filterSelect = document.getElementById("cluster-filter");
            filterSelect.value = c.cluster_id.toString();
            activeClusterFilter = c.cluster_id.toString();
            renderScatterPlot();
        });
        container.appendChild(item);
    });
}

function renderScatterPlot() {
    const canvas = document.getElementById("clusterCanvas");
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);

    // Filter points
    const filteredPoints = scatterPoints.filter(p => {
        const matchCluster = activeClusterFilter === 'all' || p.cluster.toString() === activeClusterFilter;
        const matchType = activeTypeFilter === 'All' || p.type === activeTypeFilter;
        return matchCluster && matchType;
    });

    if (filteredPoints.length === 0) return;

    // Determine coordinate bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    scatterPoints.forEach(p => {
        if (p.pca_x < minX) minX = p.pca_x;
        if (p.pca_x > maxX) maxX = p.pca_x;
        if (p.pca_y < minY) minY = p.pca_y;
        if (p.pca_y > maxY) maxY = p.pca_y;
    });

    const pad = 40;
    const scaleX = (width - pad * 2) / (maxX - minX || 1);
    const scaleY = (height - pad * 2) / (maxY - minY || 1);

    // Draw grid background
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 1;
    for (let x = pad; x < width - pad; x += 60) {
        ctx.beginPath();
        ctx.moveTo(x, pad);
        ctx.lineTo(x, height - pad);
        ctx.stroke();
    }
    for (let y = pad; y < height - pad; y += 60) {
        ctx.beginPath();
        ctx.moveTo(pad, y);
        ctx.lineTo(width - pad, y);
        ctx.stroke();
    }

    // Render data points
    filteredPoints.forEach(p => {
        const cx = pad + (p.pca_x - minX) * scaleX;
        const cy = height - pad - (p.pca_y - minY) * scaleY;
        
        // Save screen coords for collision detection
        p.screen_x = cx;
        p.screen_y = cy;

        ctx.beginPath();
        ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = CLUSTER_COLORS[p.cluster] || '#94a3b8';
        ctx.globalAlpha = 0.72;
        ctx.fill();
    });

    ctx.globalAlpha = 1.0;
}

// Tooltip collision detection on canvas
const canvasEl = document.getElementById("clusterCanvas");
const tooltipEl = document.getElementById("scatter-tooltip");

if (canvasEl && tooltipEl) {
    canvasEl.addEventListener("mousemove", (e) => {
        const rect = canvasEl.getBoundingClientRect();
        const mouseX = (e.clientX - rect.left) * (canvasEl.width / rect.width);
        const mouseY = (e.clientY - rect.top) * (canvasEl.height / rect.height);

        // Find closest point within threshold
        let closest = null;
        let minDist = 12;

        scatterPoints.forEach(p => {
            if (p.screen_x && p.screen_y) {
                const dist = Math.hypot(p.screen_x - mouseX, p.screen_y - mouseY);
                if (dist < minDist) {
                    minDist = dist;
                    closest = p;
                }
            }
        });

        if (closest) {
            const clusterMeta = globalClusters[closest.cluster] || { name: `Cluster ${closest.cluster}`, color: '#f43f5e' };
            tooltipEl.innerHTML = `
                <div class="tooltip-title">${closest.title}</div>
                <span class="tooltip-badge" style="background: ${clusterMeta.color}">${clusterMeta.badge || `C${closest.cluster}`}</span>
                <div class="tooltip-row"><strong>Format:</strong> ${closest.type} (${closest.duration})</div>
                <div class="tooltip-row"><strong>Year:</strong> ${closest.release_year} | <strong>Rating:</strong> ${closest.rating}</div>
                <div class="tooltip-row"><strong>Origin:</strong> ${closest.country}</div>
                <div class="tooltip-row" style="font-size: 11px; margin-top: 4px; color: #cbd5e1;">${closest.listed_in}</div>
            `;
            tooltipEl.style.left = `${e.clientX - rect.left + 15}px`;
            tooltipEl.style.top = `${e.clientY - rect.top + 15}px`;
            tooltipEl.classList.remove("hidden");
        } else {
            tooltipEl.classList.add("hidden");
        }
    });

    canvasEl.addEventListener("mouseleave", () => {
        tooltipEl.classList.add("hidden");
    });
}

// --------------------------------------------------------------------------
// 3. Benchmarks & Optimization Chart (Chart.js)
// --------------------------------------------------------------------------
function initOptimizationChart(metricType = 'elbow') {
    if (!globalOverview || !globalOverview.k_optimization) return;
    
    const kData = globalOverview.k_optimization;
    const labels = kData.map(d => `K=${d.k}`);
    const ctx = document.getElementById("optimizationChart");
    if (!ctx) return;

    if (optimizationChartInstance) {
        optimizationChartInstance.destroy();
    }

    let datasetLabel = '';
    let datasetValues = [];
    let datasetColor = '#38bdf8';
    let title = '';

    if (metricType === 'elbow') {
        datasetLabel = 'Inertia (WCSS)';
        datasetValues = kData.map(d => d.inertia);
        datasetColor = '#38bdf8';
        title = 'Elbow Method: Within-Cluster Sum of Squares vs K';
    } else if (metricType === 'silhouette') {
        datasetLabel = 'Silhouette Score';
        datasetValues = kData.map(d => d.silhouette_score);
        datasetColor = '#f43f5e';
        title = 'Silhouette Score Analysis across K Clusters';
    } else if (metricType === 'davies') {
        datasetLabel = 'Davies-Bouldin Index (Lower is Better)';
        datasetValues = kData.map(d => d.davies_bouldin_score);
        datasetColor = '#f59e0b';
        title = 'Davies-Bouldin Index vs K Clusters';
    }

    document.getElementById("optimizationChart-title")?.innerText = title;

    optimizationChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: datasetLabel,
                data: datasetValues,
                borderColor: datasetColor,
                backgroundColor: 'rgba(56, 189, 248, 0.08)',
                fill: true,
                tension: 0.35,
                borderWidth: 2.5,
                pointBackgroundColor: datasetColor,
                pointBorderColor: '#0f172a',
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 7
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 12 } }
                },
                tooltip: {
                    backgroundColor: '#0f172a',
                    titleColor: '#ffffff',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    padding: 10
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans' } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans' } }
                }
            }
        }
    });
}

function populateBenchmarkTable() {
    const tbody = document.getElementById("benchmark-tbody");
    if (!tbody || !globalOverview || !globalOverview.algorithm_benchmark) return;

    tbody.innerHTML = "";
    Object.entries(globalOverview.algorithm_benchmark).forEach(([name, d]) => {
        const isSelected = d.status === "Production Selected";
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong>${name}</strong></td>
            <td>${d.num_clusters}</td>
            <td>${d.silhouette_score}</td>
            <td>${d.davies_bouldin_score}</td>
            <td>${d.calinski_harabasz_score}</td>
            <td>
                <span class="status-badge ${isSelected ? 'selected' : 'benchmarked'}">
                    ${isSelected ? '<i class="fa-solid fa-check"></i> Selected' : 'Benchmarked'}
                </span>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// --------------------------------------------------------------------------
// 4. Persona Universes Cards
// --------------------------------------------------------------------------
function renderPersonaCards() {
    const container = document.getElementById("personas-container");
    if (!container) return;

    container.innerHTML = "";
    Object.values(globalClusters).forEach(c => {
        const card = document.createElement("div");
        card.className = "persona-card";
        card.style.setProperty("--card-accent", c.color);

        const genresHtml = c.top_genres.slice(0, 4).map(g => `
            <span class="genre-tag">${g.genre} (${g.percent}%)</span>
        `).join("");

        const exemplarHtml = c.exemplar_titles.slice(0, 3).map(e => `
            <div class="exemplar-item">
                <span>${e.title}</span>
                <span>${e.release_year} • ${e.rating}</span>
            </div>
        `).join("");

        card.innerHTML = `
            <div class="persona-header">
                <div class="persona-icon"><i class="fa-solid fa-${c.icon || 'shapes'}"></i></div>
                <div class="persona-title-group">
                    <span class="persona-badge" style="color: ${c.color}">Cluster ${c.cluster_id}</span>
                    <h3 class="persona-name">${c.name}</h3>
                    <p class="persona-tagline">${c.tagline}</p>
                </div>
            </div>

            <div class="persona-stats-row">
                <div>
                    <div class="stat-box-val">${c.total_titles}</div>
                    <div class="stat-box-lbl">${c.percentage_of_catalog}% Share</div>
                </div>
                <div>
                    <div class="stat-box-val">${c.movie_percent}% / ${c.tv_show_percent}%</div>
                    <div class="stat-box-lbl">Movie / TV</div>
                </div>
                <div>
                    <div class="stat-box-val">${c.mean_release_year}</div>
                    <div class="stat-box-lbl">Avg Year</div>
                </div>
            </div>

            <div>
                <div class="persona-section-title">Top Categorical Genres</div>
                <div class="genre-tags">${genresHtml}</div>
            </div>

            <div>
                <div class="persona-section-title">Catalog Exemplars</div>
                <div class="exemplar-list">${exemplarHtml}</div>
            </div>

            <div style="font-size: 12px; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 10px;">
                <strong style="color: ${c.color}">Demographic:</strong> ${c.target_demographic}
            </div>
        `;
        container.appendChild(card);
    });
}

// --------------------------------------------------------------------------
// 5. Live Simulator & Segment Predictor
// --------------------------------------------------------------------------
function initPredictor() {
    const yearSlider = document.getElementById("sim-year");
    const yearDisplay = document.getElementById("year-display");
    if (yearSlider && yearDisplay) {
        yearSlider.addEventListener("input", (e) => {
            yearDisplay.innerText = e.target.value;
        });
    }

    // Preset buttons
    document.querySelectorAll(".preset-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const key = btn.dataset.preset;
            const data = SIMULATOR_PRESETS[key];
            if (data) {
                document.getElementById("sim-title").value = data.title;
                document.getElementById("sim-type").value = data.type;
                document.getElementById("sim-year").value = data.year;
                document.getElementById("year-display").innerText = data.year;
                document.getElementById("sim-duration").value = data.duration;
                document.getElementById("sim-rating").value = data.rating;
                document.getElementById("sim-country").value = data.country;
                document.getElementById("sim-genres").value = data.genres;
                
                // Auto trigger prediction
                document.getElementById("predict-form").dispatchEvent(new Event("submit"));
            }
        });
    });

    // Form submit
    const form = document.getElementById("predict-form");
    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            await executeLivePrediction();
        });
    }
}

async function executeLivePrediction() {
    const payload = {
        title: document.getElementById("sim-title").value,
        type: document.getElementById("sim-type").value,
        release_year: parseInt(document.getElementById("sim-year").value, 10),
        duration: document.getElementById("sim-duration").value,
        rating: document.getElementById("sim-rating").value,
        country: document.getElementById("sim-country").value,
        listed_in: document.getElementById("sim-genres").value
    };

    const outputContainer = document.getElementById("prediction-output");
    outputContainer.innerHTML = `
        <div style="margin: auto; text-align: center; color: #94a3b8;">
            <i class="fa-solid fa-spinner fa-spin" style="font-size: 28px; color: #f43f5e; margin-bottom: 10px;"></i>
            <div>Computing latent vector projection and segment distances...</div>
        </div>
    `;

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const result = await res.json();
        renderPredictionResults(result);
    } catch (err) {
        outputContainer.innerHTML = `<div style="color: #f43f5e; padding: 20px;">Prediction error: ${err.message}</div>`;
    }
}

function renderPredictionResults(result) {
    const outputContainer = document.getElementById("prediction-output");
    const clusterMeta = globalClusters[result.predicted_cluster_id] || { name: `Cluster ${result.predicted_cluster_id}`, color: '#f43f5e' };

    const sisterHtml = result.nearest_neighbors.slice(0, 4).map(n => `
        <div class="sister-title-card">
            <div>
                <strong>${n.title}</strong>
                <div style="font-size: 11px; color: #94a3b8;">${n.type} • ${n.release_year} • ${n.country}</div>
            </div>
            <span class="table-badge" style="background: ${CLUSTER_COLORS[n.cluster] || '#f43f5e'}">C${n.cluster}</span>
        </div>
    `).join("");

    outputContainer.innerHTML = `
        <div class="pred-result-container">
            <div class="pred-match-banner" style="--banner-accent: ${clusterMeta.color}">
                <div style="font-size: 11px; font-weight: 700; color: ${clusterMeta.color}; text-transform: uppercase;">
                    <i class="fa-solid fa-bullseye"></i> Assigned Content Archetype
                </div>
                <div class="pred-match-title">Cluster ${result.predicted_cluster_id}: ${clusterMeta.name}</div>
                <p style="font-size: 12.5px; color: #cbd5e1; margin-top: 4px;">${clusterMeta.tagline}</p>
                <div class="pred-coords-tag">
                    <strong>PCA Latent Coordinates:</strong> [X: ${result.pca_coords.x}, Y: ${result.pca_coords.y}, Z: ${result.pca_coords.z}]
                </div>
            </div>

            <div>
                <div class="persona-section-title">Catalog Nearest Neighbor Matches</div>
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    ${sisterHtml}
                </div>
            </div>

            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 12px; font-size: 12px; color: #94a3b8;">
                <strong style="color: #ffffff;">Strategic Target:</strong> ${clusterMeta.business_strategy}
            </div>
        </div>
    `;
}

// --------------------------------------------------------------------------
// 6. Filterable Catalog Explorer
// --------------------------------------------------------------------------
function initCatalogExplorer() {
    loadCatalogPage(1);

    // Search input
    let searchDebounceTimer;
    const searchInput = document.getElementById("catalog-search-input");
    if (searchInput) {
        searchInput.addEventListener("input", (e) => {
            clearTimeout(searchDebounceTimer);
            searchDebounceTimer = setTimeout(() => {
                currentCatalogSearch = e.target.value;
                currentCatalogPage = 1;
                loadCatalogPage(1);
            }, 300);
        });
    }

    // Cluster filter buttons
    document.querySelectorAll(".cat-filter-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".cat-filter-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            const cVal = btn.dataset.cluster;
            currentCatalogCluster = cVal === "all" ? null : parseInt(cVal, 10);
            currentCatalogPage = 1;
            loadCatalogPage(1);
        });
    });

    // Pagination buttons
    document.getElementById("prev-page-btn")?.addEventListener("click", () => {
        if (currentCatalogPage > 1) {
            loadCatalogPage(currentCatalogPage - 1);
        }
    });

    document.getElementById("next-page-btn")?.addEventListener("click", () => {
        loadCatalogPage(currentCatalogPage + 1);
    });
}

async function loadCatalogPage(page = 1) {
    currentCatalogPage = page;
    let url = `/api/catalog?page=${page}&page_size=20`;
    if (currentCatalogSearch) url += `&search=${encodeURIComponent(currentCatalogSearch)}`;
    if (currentCatalogCluster !== null) url += `&cluster=${currentCatalogCluster}`;

    try {
        const res = await fetch(url);
        const data = await res.json();
        renderCatalogTable(data);
    } catch (err) {
        console.error("Catalog load error:", err);
    }
}

function renderCatalogTable(data) {
    const tbody = document.getElementById("catalog-table-body");
    if (!tbody) return;

    tbody.innerHTML = "";
    if (!data.items || data.items.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: #94a3b8; padding: 24px;">No matching titles found in catalog.</td></tr>`;
        return;
    }

    data.items.forEach(item => {
        const clusterMeta = globalClusters[item.cluster] || { name: `C${item.cluster}`, color: '#f43f5e' };
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong class="table-title">${item.title}</strong></td>
            <td><span style="color: ${item.type === 'Movie' ? '#f43f5e' : '#38bdf8'}">${item.type}</span></td>
            <td>${item.release_year}</td>
            <td>${item.rating}</td>
            <td>${item.duration}</td>
            <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis;">${item.listed_in}</td>
            <td>${item.country || 'Global'}</td>
            <td>
                <span class="table-badge" style="background: ${clusterMeta.color}">
                    C${item.cluster}: ${clusterMeta.badge || clusterMeta.name.split('&')[0]}
                </span>
            </td>
        `;
        tbody.appendChild(tr);
    });

    // Update counter & pagination
    document.getElementById("catalog-results-count").innerText = `Showing page ${data.page} of ${data.total_pages} (${data.total.toLocaleString()} total titles)`;
    document.getElementById("page-indicator").innerText = `Page ${data.page} of ${data.total_pages}`;
    
    document.getElementById("prev-page-btn").disabled = data.page <= 1;
    document.getElementById("next-page-btn").disabled = data.page >= data.total_pages;
}

// --------------------------------------------------------------------------
// 7. Event Listeners & Tab Controls
// --------------------------------------------------------------------------
function initEventListeners() {
    // Optimization metric tabs
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            const chartType = btn.dataset.chart;
            initOptimizationChart(chartType);
        });
    });

    // Scatter filters
    document.getElementById("cluster-filter")?.addEventListener("change", (e) => {
        activeClusterFilter = e.target.value;
        renderScatterPlot();
    });

    document.getElementById("type-filter")?.addEventListener("change", (e) => {
        activeTypeFilter = e.target.value;
        renderScatterPlot();
    });

    document.getElementById("reset-view-btn")?.addEventListener("click", () => {
        activeClusterFilter = 'all';
        activeTypeFilter = 'All';
        document.getElementById("cluster-filter").value = 'all';
        document.getElementById("type-filter").value = 'All';
        renderScatterPlot();
    });

    // Smooth scroll for nav items
    document.querySelectorAll(".nav-item").forEach(link => {
        link.addEventListener("click", () => {
            document.querySelectorAll(".nav-item").forEach(l => l.classList.remove("active"));
            link.classList.add("active");
        });
    });
}
