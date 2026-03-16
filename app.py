"""
╔══════════════════════════════════════════════════════════╗
║           ONCO·SCAN  Cancer Detection System             ║
║           Built with Streamlit + scikit-learn            ║
╚══════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, time, warnings
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, precision_recall_curve)

warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="OnchoScan · Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── THEME / CSS ────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg:       #080c14;
    --surface:  #0e1420;
    --card:     #111827;
    --border:   #1f2d40;
    --accent:   #00d4ff;
    --accent2:  #ff4b6e;
    --accent3:  #39d353;
    --muted:    #6b7a8d;
    --text:     #e2e8f0;
    --mono:     'Space Mono', monospace;
    --sans:     'DM Sans', sans-serif;
  }

  html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg);
    color: var(--text);
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
  }
  section[data-testid="stSidebar"] * { color: var(--text) !important; }

  /* Main area */
  .main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }

  /* Cards */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
  }
  .card-accent { border-left: 3px solid var(--accent); }
  .card-danger  { border-left: 3px solid var(--accent2); }
  .card-ok      { border-left: 3px solid var(--accent3); }

  /* Stat tiles */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.9rem;
    margin: 1rem 0;
  }
  .stat-tile {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem;
    text-align: center;
  }
  .stat-tile .val {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
  }
  .stat-tile .lbl {
    font-size: 0.72rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: .35rem;
  }
  .stat-tile.cyan  .val { color: var(--accent);  }
  .stat-tile.red   .val { color: var(--accent2); }
  .stat-tile.green .val { color: var(--accent3); }
  .stat-tile.white .val { color: var(--text);    }

  /* Big diagnosis badge */
  .diag-badge {
    display: inline-block;
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 700;
    padding: .5rem 1.4rem;
    border-radius: 8px;
    letter-spacing: .06em;
  }
  .diag-M { background: rgba(255,75,110,.15); color: #ff4b6e; border: 1px solid #ff4b6e; }
  .diag-B { background: rgba(57,211,83,.12);  color: #39d353; border: 1px solid #39d353; }

  /* Risk bar */
  .risk-bar-wrap { height: 10px; background: #1a2030; border-radius: 5px; overflow: hidden; }
  .risk-bar-fill  { height: 100%; border-radius: 5px;
                    background: linear-gradient(90deg, #39d353, #ffd553, #ff4b6e); }

  /* Headings */
  h1, h2, h3 { font-family: var(--sans); color: var(--text); }
  .page-title {
    font-family: var(--mono);
    font-size: 2.2rem;
    color: var(--accent);
    letter-spacing: .04em;
    margin-bottom: 0;
  }
  .page-sub {
    color: var(--muted);
    font-size: .9rem;
    margin-bottom: 1.8rem;
    font-family: var(--mono);
  }

  /* Tab styling */
  button[data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: .8rem !important;
    letter-spacing: .05em !important;
    color: var(--muted) !important;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
  }

  /* Metric overrides */
  [data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
  }
  [data-testid="metric-container"] label { color: var(--muted) !important; font-family: var(--mono) !important; font-size: .7rem !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: var(--mono) !important; }

  /* Upload zone */
  [data-testid="stFileUploader"] { background: var(--card); border-radius: 10px; border: 1px dashed var(--border); }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* Plotly chart backgrounds */
  .js-plotly-plot .plotly { background: transparent !important; }
  
  /* Divider */
  hr { border-color: var(--border); }
  
  /* Select / slider overrides */
  .stSelectbox > div > div { background: var(--card) !important; border-color: var(--border) !important; color: var(--text) !important; }
  .stSlider > div { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY DARK TEMPLATE ───────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='#111827', plot_bgcolor='#0e1420',
    font=dict(color='#e2e8f0', family='DM Sans'),
    xaxis=dict(gridcolor='#1f2d40', linecolor='#1f2d40'),
    yaxis=dict(gridcolor='#1f2d40', linecolor='#1f2d40'),
    margin=dict(l=40, r=20, t=40, b=40),
)
C_BENIGN    = '#39d353'
C_MALIGNANT = '#ff4b6e'
C_ACCENT    = '#00d4ff'

# ─── FEATURE NAMES ──────────────────────────────────────────
FEATURE_GROUPS = ['radius','texture','perimeter','area','smoothness',
                  'compactness','concavity','concave_points','symmetry','fractal_dimension']
SUFFIXES = ['mean','se','worst']
FEATURE_COLS = [f"{f}_{s}" for s in SUFFIXES for f in FEATURE_GROUPS]
META_COLS = ['patient_id','age','hospital','diagnosis']

# ─── CACHING ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_resource(show_spinner=False)
def train_pipeline(csv_bytes: bytes):
    df = pd.read_csv(io.BytesIO(csv_bytes))
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values
    y = (df['diagnosis'] == 'M').astype(int).values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    pca_full  = PCA(n_components=min(20, X_sc.shape[1]))
    X_pf      = pca_full.fit_transform(X_sc)
    cumvar    = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp    = max(2, int(np.argmax(cumvar >= 0.95)) + 1)

    pca   = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    pca2  = PCA(n_components=2, random_state=42)
    X_2d  = pca2.fit_transform(X_sc)

    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    km.fit(X_pca)

    db = DBSCAN(eps=1.8, min_samples=8)
    db_labels = db.fit_predict(X_pca)

    nn = NearestNeighbors(n_neighbors=11, metric='cosine')
    nn.fit(X_sc)

    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=.2, random_state=42, stratify=y)
    models = {
        'Random Forest':      RandomForestClassifier(n_estimators=200, random_state=42),
        'Logistic Regression':LogisticRegression(max_iter=1000, random_state=42),
        'Gradient Boosting':  GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    results = {}
    for name, clf in models.items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_sc, y, cv=cv, scoring='roc_auc')
        clf.fit(X_tr, y_tr)
        yp = clf.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, yp)
        results[name] = dict(clf=clf, y_prob=yp, y_pred=clf.predict(X_te),
                             fpr=fpr, tpr=tpr, auc=auc(fpr, tpr),
                             cv_mean=cv_scores.mean(), cv_std=cv_scores.std(),
                             report=classification_report(y_te, clf.predict(X_te), output_dict=True))

    best_name = max(results, key=lambda n: results[n]['auc'])
    best_clf  = results[best_name]['clf']
    imp = pd.Series(
        best_clf.feature_importances_ if hasattr(best_clf, 'feature_importances_') else
        np.abs(best_clf.coef_[0]), index=feature_cols
    ).sort_values(ascending=False)

    return dict(df=df, feature_cols=feature_cols, X_sc=X_sc, y=y,
                scaler=scaler, pca_full=pca_full, pca=pca, pca2=pca2,
                X_pca=X_pca, X_2d=X_2d, cumvar=cumvar, n_comp=n_comp,
                km=km, km_labels=km.predict(X_pca),
                db_labels=db_labels, nn=nn,
                results=results, best_name=best_name, best_clf=best_clf,
                importances=imp, X_te=X_te, y_te=y_te)

# ─── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 OnchoScan")
    st.markdown("<span style='color:#6b7a8d;font-size:.8rem;font-family:monospace'>v1.0 · Cancer Detection System</span>", unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader("Upload CSV Dataset", type=['csv'],
                                help="Needs 30 cell nucleus features + 'diagnosis' column (B/M)")
    use_sample = st.button("▶ Use Built-in Sample Dataset", use_container_width=True)

    st.divider()
    st.markdown("**Navigation**")
    page = st.radio("", ["📊 Overview", "🧬 Clustering & PCA",
                         "🤖 Model Performance", "🔍 Patient Lookup",
                         "🧪 Run New Prediction"],
                    label_visibility='collapsed')
    st.divider()
    st.markdown("<span style='color:#6b7a8d;font-size:.75rem'>Built with scikit-learn · Plotly · Streamlit</span>", unsafe_allow_html=True)

# ─── LOAD DATA ──────────────────────────────────────────────
csv_bytes = None
if uploaded:
    csv_bytes = uploaded.read()
elif use_sample or ('csv_bytes' in st.session_state):
    if use_sample:
        with open("cancer_samples.csv", "rb") as f:
            csv_bytes = f.read()
        st.session_state['csv_bytes'] = csv_bytes
    else:
        csv_bytes = st.session_state.get('csv_bytes')

if csv_bytes is None:
    # Landing screen
    st.markdown('<p class="page-title">ONCO·SCAN</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Cancer Detection & Analysis System</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "🧬", "RAG Retrieval", "Find the most similar historical cases for any new patient using cosine-similarity vector search"),
        (c2, "🔮", "Ensemble Models", "Random Forest, Logistic Regression & Gradient Boosting trained and compared automatically"),
        (c3, "📐", "PCA + Clustering", "Dimensionality reduction + KMeans & DBSCAN unsupervised clustering"),
    ]:
        col.markdown(f"""
        <div class="card card-accent">
          <div style="font-size:2rem">{icon}</div>
          <div style="font-weight:600;margin:.5rem 0 .3rem">{title}</div>
          <div style="color:#6b7a8d;font-size:.85rem">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.info("👈 Upload a CSV or click **Use Built-in Sample Dataset** to get started.")
    st.stop()

# ─── TRAIN ──────────────────────────────────────────────────
with st.spinner("Training pipeline… this takes ~10 seconds on first load"):
    M = train_pipeline(csv_bytes)

df           = M['df']
feature_cols = M['feature_cols']
X_sc         = M['X_sc']
y            = M['y']
X_2d         = M['X_2d']
km_labels    = M['km_labels']
db_labels    = M['db_labels']
results      = M['results']
best_name    = M['best_name']
importances  = M['importances']

n_total = len(df)
n_benign    = (df.diagnosis == 'B').sum()
n_malignant = (df.diagnosis == 'M').sum()
best_auc    = results[best_name]['auc']

# ════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<p class="page-title">OVERVIEW</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-sub">Dataset · {n_total} patients · {len(feature_cols)} features</p>', unsafe_allow_html=True)

    # Stat tiles
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-tile cyan">
        <div class="val">{n_total}</div>
        <div class="lbl">Total Patients</div>
      </div>
      <div class="stat-tile green">
        <div class="val">{n_benign}</div>
        <div class="lbl">Benign Cases</div>
      </div>
      <div class="stat-tile red">
        <div class="val">{n_malignant}</div>
        <div class="lbl">Malignant Cases</div>
      </div>
      <div class="stat-tile white">
        <div class="val">{len(feature_cols)}</div>
        <div class="lbl">Features</div>
      </div>
      <div class="stat-tile cyan">
        <div class="val">{best_auc:.3f}</div>
        <div class="lbl">Best AUC</div>
      </div>
      <div class="stat-tile green">
        <div class="val">{results[best_name]['report']['accuracy']:.1%}</div>
        <div class="lbl">Accuracy</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        # Donut
        fig = go.Figure(go.Pie(
            labels=['Benign', 'Malignant'],
            values=[n_benign, n_malignant],
            hole=.65,
            marker_colors=[C_BENIGN, C_MALIGNANT],
            textinfo='percent+label',
            textfont=dict(size=13, color='white'),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title='Diagnosis Distribution',
                          showlegend=False, height=300)
        fig.add_annotation(text=f"{n_malignant/n_total:.0%}<br>Malignant",
                           font=dict(size=16, color=C_MALIGNANT, family='Space Mono'),
                           showarrow=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Age distribution by diagnosis
        fig = go.Figure()
        for diag, col, name in [('B', C_BENIGN, 'Benign'), ('M', C_MALIGNANT, 'Malignant')]:
            ages = df[df.diagnosis == diag]['age'] if 'age' in df.columns else pd.Series([])
            if len(ages):
                fig.add_trace(go.Histogram(x=ages, name=name, marker_color=col,
                                           opacity=.7, nbinsx=25))
        fig.update_layout(**PLOTLY_LAYOUT, title='Age Distribution by Diagnosis',
                          barmode='overlay', height=300,
                          xaxis_title='Age', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)

    # Feature distributions - top 6
    st.markdown("#### Top Feature Distributions")
    top6 = importances[:6].index.tolist()
    cols = st.columns(3)
    for i, feat in enumerate(top6):
        with cols[i % 3]:
            fig = go.Figure()
            for diag, col, name in [('B', C_BENIGN, 'Benign'), ('M', C_MALIGNANT, 'Malignant')]:
                vals = df[df.diagnosis == diag][feat]
                fig.add_trace(go.Violin(y=vals, name=name, line_color=col,
                                        fillcolor=col.replace(')', ', .15)').replace('rgb', 'rgba'),
                                        box_visible=True, meanline_visible=True,
                                        points='outliers'))
            fig.update_layout(**PLOTLY_LAYOUT, title=feat.replace('_', ' ').title(),
                              height=250, showlegend=i == 0, margin=dict(l=20, r=10, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # Data preview
    st.markdown("#### Dataset Preview")
    show_cols = META_COLS + feature_cols[:6]
    st.dataframe(df[[c for c in show_cols if c in df.columns]].head(20),
                 use_container_width=True, height=280)

# ════════════════════════════════════════════════
#  PAGE: CLUSTERING & PCA
# ════════════════════════════════════════════════
elif page == "🧬 Clustering & PCA":
    st.markdown('<p class="page-title">CLUSTERING & PCA</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Unsupervised structure discovery in the feature space</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["PCA Embedding", "KMeans", "DBSCAN"])

    with tab1:
        c1, c2 = st.columns([3, 1])
        with c1:
            diag_map  = {'B': C_BENIGN, 'M': C_MALIGNANT}
            label_map = {'B': 'Benign', 'M': 'Malignant'}
            plot_df = pd.DataFrame({'x': X_2d[:, 0], 'y': X_2d[:, 1],
                                    'diagnosis': df.diagnosis.values,
                                    'label': df.diagnosis.map(label_map).values})
            if 'patient_id' in df.columns:
                plot_df['patient_id'] = df.patient_id.values
            fig = px.scatter(plot_df, x='x', y='y', color='label',
                             color_discrete_map={'Benign': C_BENIGN, 'Malignant': C_MALIGNANT},
                             hover_data=['patient_id'] if 'patient_id' in plot_df else [],
                             opacity=.7, size_max=6)
            fig.update_traces(marker=dict(size=6))
            fig.update_layout(**PLOTLY_LAYOUT, title='PCA 2D Projection',
                              xaxis_title='PC1', yaxis_title='PC2', height=450)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            n_comp = M['n_comp']
            cumvar = M['cumvar']
            st.markdown(f"""
            <div class="card card-accent">
              <div style="font-size:.75rem;color:#6b7a8d;font-family:monospace;text-transform:uppercase;letter-spacing:.08em">PCA Stats</div>
              <div style="margin-top:.8rem">
                <div style="color:#00d4ff;font-family:monospace;font-size:1.6rem;font-weight:700">{n_comp}</div>
                <div style="color:#6b7a8d;font-size:.75rem">Components (95% var)</div>
              </div>
              <div style="margin-top:.8rem">
                <div style="color:#e2e8f0;font-family:monospace;font-size:1.2rem">{cumvar[n_comp-1]*100:.1f}%</div>
                <div style="color:#6b7a8d;font-size:.75rem">Variance Explained</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Scree plot
        pca_full = M['pca_full']
        n_show = min(20, len(pca_full.explained_variance_ratio_))
        fig2 = go.Figure()
        fig2.add_bar(x=list(range(1, n_show+1)),
                     y=pca_full.explained_variance_ratio_[:n_show]*100,
                     marker_color=C_ACCENT, name='Individual')
        fig2.add_scatter(x=list(range(1, n_show+1)),
                         y=cumvar[:n_show]*100,
                         mode='lines+markers', name='Cumulative',
                         line=dict(color='white', width=2, dash='dot'),
                         marker=dict(size=5))
        fig2.add_hline(y=95, line_dash='dash', line_color='#ffd553',
                       annotation_text='95%', annotation_font_color='#ffd553')
        fig2.update_layout(**PLOTLY_LAYOUT, title='Explained Variance per Component',
                           xaxis_title='Principal Component', yaxis_title='Variance (%)', height=300)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        km_plot = pd.DataFrame({'x': X_2d[:, 0], 'y': X_2d[:, 1],
                                'cluster': [f"Cluster {l}" for l in km_labels],
                                'true': df.diagnosis.map({'B':'Benign','M':'Malignant'}).values})
        fig = px.scatter(km_plot, x='x', y='y', color='cluster',
                         symbol='true',
                         color_discrete_sequence=['#00d4ff', '#ff4b6e', '#ffd553', '#39d353'],
                         opacity=.7)
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(**PLOTLY_LAYOUT, title='KMeans Clusters vs True Diagnosis',
                          xaxis_title='PC1', yaxis_title='PC2', height=480)
        st.plotly_chart(fig, use_container_width=True)

        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(y, km_labels)
        st.markdown(f"""
        <div class="card card-accent" style="display:inline-block;padding:.8rem 1.4rem">
          <span style="color:#6b7a8d;font-family:monospace;font-size:.75rem">Adjusted Rand Index: </span>
          <span style="color:#00d4ff;font-family:monospace;font-size:1.2rem;font-weight:700">{ari:.4f}</span>
          <span style="color:#6b7a8d;font-size:.75rem"> (1.0 = perfect cluster alignment with true labels)</span>
        </div>""", unsafe_allow_html=True)

    with tab3:
        n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        noise_n    = (db_labels == -1).sum()
        db_plot    = pd.DataFrame({'x': X_2d[:, 0], 'y': X_2d[:, 1],
                                   'cluster': [('Noise' if l == -1 else f'Cluster {l}') for l in db_labels]})
        fig = px.scatter(db_plot, x='x', y='y', color='cluster',
                         color_discrete_sequence=['gray','#00d4ff','#ff4b6e','#39d353','#ffd553','#ab47bc'],
                         opacity=.7)
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(**PLOTLY_LAYOUT, title=f'DBSCAN Clusters (eps=1.8, min_samples=8)',
                          xaxis_title='PC1', yaxis_title='PC2', height=480)
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        col1.metric("Clusters Found", n_clusters)
        col2.metric("Noise Points", f"{noise_n} ({noise_n/len(df):.1%})")

# ════════════════════════════════════════════════
#  PAGE: MODEL PERFORMANCE
# ════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown('<p class="page-title">MODEL PERFORMANCE</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Ensemble evaluation across 3 classifiers</p>', unsafe_allow_html=True)

    # Model card summaries
    cols = st.columns(3)
    icons = {'Random Forest': '🌲', 'Logistic Regression': '📈', 'Gradient Boosting': '🚀'}
    colors_m = {'Random Forest': 'cyan', 'Logistic Regression': 'green', 'Gradient Boosting': 'red'}
    for col, (name, res) in zip(cols, results.items()):
        rep = res['report']
        col.markdown(f"""
        <div class="card card-accent">
          <div style="font-size:1.5rem">{icons[name]}</div>
          <div style="font-weight:600;margin:.4rem 0">{name}</div>
          <div style="font-family:monospace;font-size:1.4rem;color:{'#00d4ff' if name==best_name else '#e2e8f0'}">
            AUC {res['auc']:.3f} {'⭐' if name==best_name else ''}
          </div>
          <div style="color:#6b7a8d;font-size:.78rem;margin-top:.4rem">
            CV: {res['cv_mean']:.3f} ± {res['cv_std']:.3f}<br>
            Acc: {rep['accuracy']:.3f} &nbsp; F1(M): {rep['1']['f1-score']:.3f}
          </div>
        </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # ROC curves
        fig = go.Figure()
        roc_cols = ['#00d4ff', '#ff4b6e', '#ab47bc']
        for (name, res), col in zip(results.items(), roc_cols):
            fig.add_scatter(x=res['fpr'], y=res['tpr'], mode='lines', name=f"{name} ({res['auc']:.3f})",
                            line=dict(color=col, width=2.5))
        fig.add_scatter(x=[0,1], y=[0,1], mode='lines', name='Random',
                        line=dict(color='gray', dash='dot', width=1))
        fig.update_layout(**PLOTLY_LAYOUT, title='ROC Curves', height=380,
                          xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Confusion matrix (best)
        cm = confusion_matrix(M['y_te'], results[best_name]['y_pred'])
        fig = go.Figure(go.Heatmap(
            z=cm, x=['Predicted Benign', 'Predicted Malignant'],
            y=['Actual Benign', 'Actual Malignant'],
            colorscale=[[0,'#0e1420'],[0.5,'#0f3460'],[1,'#00d4ff']],
            text=cm, texttemplate='<b>%{text}</b>', textfont=dict(size=22),
            showscale=False
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title=f'Confusion Matrix · {best_name}', height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Precision-Recall + Feature Importance
    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        for (name, res), col in zip(results.items(), roc_cols):
            prec, rec, _ = precision_recall_curve(M['y_te'], res['y_prob'])
            fig.add_scatter(x=rec, y=prec, mode='lines', name=name,
                            line=dict(color=col, width=2))
        fig.update_layout(**PLOTLY_LAYOUT, title='Precision-Recall Curves', height=350,
                          xaxis_title='Recall', yaxis_title='Precision')
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        top15 = importances[:15]
        bar_colors = ['#ff4b6e' if 'worst' in f else ('#ffd553' if 'mean' in f else '#00d4ff')
                      for f in top15.index]
        fig = go.Figure(go.Bar(
            x=top15.values, y=top15.index,
            orientation='h', marker_color=bar_colors,
            text=[f'{v:.4f}' for v in top15.values],
            textposition='outside', textfont=dict(size=10, color='#e2e8f0')
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title='Feature Importances (Top 15)',
                          height=350, xaxis_title='Importance',
                          yaxis=dict(autorange='reversed', gridcolor='#1f2d40'))
        st.plotly_chart(fig, use_container_width=True)

    # Per-class metrics table
    st.markdown("#### Detailed Classification Report")
    rep = results[best_name]['report']
    report_df = pd.DataFrame({
        'Class':     ['Benign (0)', 'Malignant (1)', 'Macro Avg', 'Weighted Avg'],
        'Precision': [rep['0']['precision'], rep['1']['precision'], rep['macro avg']['precision'], rep['weighted avg']['precision']],
        'Recall':    [rep['0']['recall'],    rep['1']['recall'],    rep['macro avg']['recall'],    rep['weighted avg']['recall']],
        'F1-Score':  [rep['0']['f1-score'],  rep['1']['f1-score'],  rep['macro avg']['f1-score'],  rep['weighted avg']['f1-score']],
        'Support':   [rep['0']['support'],   rep['1']['support'],   rep['macro avg']['support'],   rep['weighted avg']['support']],
    })
    st.dataframe(report_df.style.format({'Precision': '{:.4f}', 'Recall': '{:.4f}',
                                          'F1-Score': '{:.4f}', 'Support': '{:.0f}'}),
                 use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════
#  PAGE: PATIENT LOOKUP + RAG
# ════════════════════════════════════════════════
elif page == "🔍 Patient Lookup":
    st.markdown('<p class="page-title">PATIENT LOOKUP</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">RAG-style retrieval · find the most similar historical cases</p>', unsafe_allow_html=True)

    if 'patient_id' not in df.columns:
        st.warning("No 'patient_id' column found in dataset.")
        st.stop()

    pid_options = df['patient_id'].tolist()
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_pid = st.selectbox("Select Patient ID", pid_options)
    with col2:
        k_neighbors = st.slider("Similar cases (k)", 3, 15, 5)

    idx = df[df.patient_id == selected_pid].index[0]
    patient = df.iloc[idx]
    patient_features = X_sc[idx]

    # Retrieve similar
    nn = M['nn']
    dists, idxs = nn.kneighbors(patient_features.reshape(1, -1), n_neighbors=k_neighbors+1)
    idxs  = idxs[0][1:]
    dists = dists[0][1:]
    similar_df = df.iloc[idxs].copy()
    similar_df['similarity'] = np.round(1 - dists, 4)
    similar_df['pca_x'] = X_2d[idxs, 0]
    similar_df['pca_y'] = X_2d[idxs, 1]

    # Model prediction for this patient
    best_clf = M['best_clf']
    prob = best_clf.predict_proba(patient_features.reshape(1, -1))[0, 1]
    pred = 'M' if prob >= 0.5 else 'B'
    true_diag = patient['diagnosis']
    correct = pred == true_diag
    maj_vote = similar_df['diagnosis'].mode()[0]

    # Header cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="card {'card-danger' if true_diag=='M' else 'card-ok'}">
          <div style="color:#6b7a8d;font-family:monospace;font-size:.72rem;text-transform:uppercase">True Diagnosis</div>
          <div class="diag-badge diag-{true_diag}" style="margin-top:.5rem">
            {'⬤ MALIGNANT' if true_diag=='M' else '⬤ BENIGN'}
          </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="card {'card-danger' if pred=='M' else 'card-ok'}">
          <div style="color:#6b7a8d;font-family:monospace;font-size:.72rem;text-transform:uppercase">Model Prediction</div>
          <div class="diag-badge diag-{pred}" style="margin-top:.5rem">
            {'⬤ MALIGNANT' if pred=='M' else '⬤ BENIGN'}
          </div>
          <div style="margin-top:.5rem;font-size:.8rem;color:#6b7a8d">Confidence: {prob if pred=='M' else 1-prob:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="card card-accent">
          <div style="color:#6b7a8d;font-family:monospace;font-size:.72rem;text-transform:uppercase">RAG Majority Vote</div>
          <div class="diag-badge diag-{maj_vote}" style="margin-top:.5rem">
            {'⬤ MALIGNANT' if maj_vote=='M' else '⬤ BENIGN'}
          </div>
          <div style="margin-top:.5rem;font-size:.8rem;color:#6b7a8d">
            From {k_neighbors} similar cases · {'✓ Match' if maj_vote==true_diag else '✗ Mismatch'}
          </div>
        </div>""", unsafe_allow_html=True)

    # Risk bar
    st.markdown(f"""
    <div style="margin:1rem 0 .4rem">
      <span style="font-family:monospace;font-size:.75rem;color:#6b7a8d;text-transform:uppercase;letter-spacing:.08em">Malignancy Probability</span>
      <span style="font-family:monospace;font-size:.85rem;color:#ff4b6e;margin-left:1rem;font-weight:700">{prob:.1%}</span>
    </div>
    <div class="risk-bar-wrap">
      <div class="risk-bar-fill" style="width:{prob*100:.1f}%"></div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    cola, colb = st.columns([2, 1])

    with cola:
        # PCA scatter with patient highlighted
        plot_df2 = pd.DataFrame({'x': X_2d[:, 0], 'y': X_2d[:, 1],
                                 'type': df.diagnosis.map({'B': 'Benign', 'M': 'Malignant'}).values})
        fig = px.scatter(plot_df2, x='x', y='y', color='type',
                         color_discrete_map={'Benign': C_BENIGN, 'Malignant': C_MALIGNANT},
                         opacity=.4)
        fig.update_traces(marker=dict(size=5))
        # Highlight similar cases
        fig.add_scatter(x=X_2d[idxs, 0], y=X_2d[idxs, 1], mode='markers',
                        name='Similar Cases', marker=dict(color='#ffd553', size=10, symbol='diamond'))
        # Highlight patient
        fig.add_scatter(x=[X_2d[idx, 0]], y=[X_2d[idx, 1]], mode='markers',
                        name=f'{selected_pid} (query)',
                        marker=dict(color='white', size=14, symbol='star',
                                    line=dict(color='#00d4ff', width=2)))
        fig.update_layout(**PLOTLY_LAYOUT, title='Patient in Feature Space',
                          height=380, xaxis_title='PC1', yaxis_title='PC2')
        st.plotly_chart(fig, use_container_width=True)

    with colb:
        st.markdown("**Top Similar Cases**")
        show_cols = ['patient_id', 'diagnosis', 'similarity']
        if 'age' in similar_df.columns:
            show_cols.insert(2, 'age')
        st.dataframe(similar_df[show_cols].reset_index(drop=True).style.applymap(
            lambda v: f'color: {C_MALIGNANT}' if v == 'M' else f'color: {C_BENIGN}',
            subset=['diagnosis']
        ), use_container_width=True, height=340)

    # Radar chart - patient vs population means
    top8 = importances[:8].index.tolist()
    patient_vals = [patient[f] for f in top8]
    benign_means = [df[df.diagnosis=='B'][f].mean() for f in top8]
    malig_means  = [df[df.diagnosis=='M'][f].mean() for f in top8]

    fig_radar = go.Figure()
    for vals, name, col in [(benign_means, 'Benign Mean', C_BENIGN),
                             (malig_means,  'Malignant Mean', C_MALIGNANT),
                             (patient_vals, f'{selected_pid}', 'white')]:
        # Normalize to 0-1 per feature
        feat_min = df[top8].min().values
        feat_max = df[top8].max().values
        norm = [(v - lo) / (hi - lo + 1e-9) for v, lo, hi in zip(vals, feat_min, feat_max)]
        fig_radar.add_trace(go.Scatterpolar(r=norm + [norm[0]],
                                            theta=[f.replace('_', '\n') for f in top8] + [top8[0].replace('_','\n')],
                                            name=name, line=dict(color=col, width=2),
                                            fill='toself', fillcolor=col.replace('#', 'rgba(') if col != 'white' else 'rgba(255,255,255,.05)',
                                            opacity=.3 if name != f'{selected_pid}' else .8))
    fig_radar.update_layout(**PLOTLY_LAYOUT, polar=dict(
        bgcolor='#0e1420',
        radialaxis=dict(visible=True, range=[0, 1], gridcolor='#1f2d40', tickfont=dict(size=8)),
        angularaxis=dict(gridcolor='#1f2d40')
    ), title='Patient Feature Fingerprint (vs Population)', height=380)
    st.plotly_chart(fig_radar, use_container_width=True)

# ════════════════════════════════════════════════
#  PAGE: NEW PREDICTION
# ════════════════════════════════════════════════
elif page == "🧪 Run New Prediction":
    st.markdown('<p class="page-title">NEW PREDICTION</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Enter cell nucleus measurements to get an instant prediction</p>', unsafe_allow_html=True)

    st.info("💡 Tip: Use the **Auto-fill** buttons below to load a sample benign or malignant case.")

    col_b, col_m, col_r = st.columns([1, 1, 4])
    auto_fill = None
    if col_b.button("🟢 Fill Benign Example"):
        auto_fill = 'B'
    if col_m.button("🔴 Fill Malignant Example"):
        auto_fill = 'M'

    # Pick reference row
    if auto_fill:
        ref_row = df[df.diagnosis == auto_fill].iloc[3]
        st.session_state['autofill_vals'] = {f: float(ref_row[f]) for f in feature_cols if f in df.columns}
        st.session_state['autofill_diag']  = auto_fill

    # Build input form
    vals = {}
    groups_rendered = 0
    for suffix in SUFFIXES:
        suffix_feats = [f for f in feature_cols if f.endswith(f'_{suffix}')]
        with st.expander(f"📐 {suffix.upper()} features", expanded=(suffix == 'mean')):
            cols = st.columns(5)
            for ci, feat in enumerate(suffix_feats):
                af_key = f'autofill_{feat}'
                af_val = None
                if 'autofill_vals' in st.session_state:
                    af_val = st.session_state['autofill_vals'].get(feat)

                col_range = df[feat] if feat in df.columns else pd.Series([0.0, 1.0])
                mn, mx = float(col_range.min()), float(col_range.max())
                default = af_val if af_val is not None else float(col_range.mean())
                vals[feat] = cols[ci % 5].number_input(
                    feat, min_value=mn*0.5, max_value=mx*2.0,
                    value=round(default, 4), format="%.4f",
                    key=f"inp_{feat}"
                )

    if st.button("🔬 RUN ANALYSIS", use_container_width=True):
        input_vec = np.array([[vals[f] for f in feature_cols]])
        input_sc  = M['scaler'].transform(input_vec)

        # Model predictions
        st.divider()
        st.markdown("### Results")

        model_results = {}
        for name, res in results.items():
            clf = res['clf']
            prob = clf.predict_proba(input_sc)[0, 1]
            pred = 'M' if prob >= 0.5 else 'B'
            model_results[name] = {'prob': prob, 'pred': pred}

        # Ensemble average
        ens_prob = np.mean([r['prob'] for r in model_results.values()])
        ens_pred = 'M' if ens_prob >= 0.5 else 'B'

        # Display
        c1, c2, c3, c4 = st.columns(4)
        for col, (name, mr) in zip([c1, c2, c3], model_results.items()):
            short = {'Random Forest':'RF','Logistic Regression':'LR','Gradient Boosting':'GB'}[name]
            col.markdown(f"""
            <div class="card {'card-danger' if mr['pred']=='M' else 'card-ok'}">
              <div style="color:#6b7a8d;font-family:monospace;font-size:.7rem">{short}</div>
              <div class="diag-badge diag-{mr['pred']}" style="font-size:1rem;padding:.3rem .8rem;margin-top:.4rem">
                {'MALIGNANT' if mr['pred']=='M' else 'BENIGN'}
              </div>
              <div style="color:#6b7a8d;font-size:.78rem;margin-top:.4rem">p = {mr['prob']:.3f}</div>
            </div>""", unsafe_allow_html=True)
        c4.markdown(f"""
        <div class="card card-accent">
          <div style="color:#6b7a8d;font-family:monospace;font-size:.7rem">ENSEMBLE</div>
          <div class="diag-badge diag-{ens_pred}" style="font-size:1rem;padding:.3rem .8rem;margin-top:.4rem">
            {'MALIGNANT' if ens_pred=='M' else 'BENIGN'}
          </div>
          <div style="color:#6b7a8d;font-size:.78rem;margin-top:.4rem">avg p = {ens_prob:.3f}</div>
        </div>""", unsafe_allow_html=True)

        # Risk bar
        st.markdown(f"""
        <div style="margin:1.2rem 0 .4rem">
          <span style="font-family:monospace;font-size:.75rem;color:#6b7a8d;text-transform:uppercase">Ensemble Malignancy Risk</span>
          <span style="font-family:monospace;font-size:1rem;color:{'#ff4b6e' if ens_pred=='M' else '#39d353'};margin-left:1rem;font-weight:700">{ens_prob:.1%}</span>
        </div>
        <div class="risk-bar-wrap" style="height:14px">
          <div class="risk-bar-fill" style="width:{ens_prob*100:.1f}%"></div>
        </div>""", unsafe_allow_html=True)

        # RAG retrieval
        st.markdown("#### Most Similar Historical Cases")
        dists, idxs = M['nn'].kneighbors(input_sc, n_neighbors=6)
        sim_df = df.iloc[idxs[0]].copy()
        sim_df['similarity'] = np.round(1 - dists[0], 4)
        show = ['patient_id','age','diagnosis','similarity'] if 'patient_id' in sim_df.columns else ['diagnosis','similarity']
        st.dataframe(sim_df[[c for c in show if c in sim_df.columns]].reset_index(drop=True)
                     .style.applymap(lambda v: f'color:{C_MALIGNANT}' if v=='M' else f'color:{C_BENIGN}',
                                     subset=['diagnosis']),
                     use_container_width=True)

        # Position in PCA space
        input_2d = M['pca2'].transform(input_sc)
        plot_df3  = pd.DataFrame({'x': X_2d[:, 0], 'y': X_2d[:, 1],
                                  'diag': df.diagnosis.map({'B':'Benign','M':'Malignant'}).values})
        fig = px.scatter(plot_df3, x='x', y='y', color='diag',
                         color_discrete_map={'Benign':C_BENIGN,'Malignant':C_MALIGNANT}, opacity=.35)
        fig.update_traces(marker=dict(size=5))
        fig.add_scatter(x=X_2d[idxs[0], 0], y=X_2d[idxs[0], 1], mode='markers',
                        name='Similar Cases', marker=dict(color='#ffd553', size=10, symbol='diamond'))
        fig.add_scatter(x=[input_2d[0, 0]], y=[input_2d[0, 1]], mode='markers',
                        name='New Patient',
                        marker=dict(color='white', size=16, symbol='star',
                                    line=dict(color='#00d4ff', width=2)))
        fig.update_layout(**PLOTLY_LAYOUT, title='Position in Feature Space', height=380,
                          xaxis_title='PC1', yaxis_title='PC2')
        st.plotly_chart(fig, use_container_width=True)

        # Downloadable result
        result_row = {f: vals[f] for f in feature_cols}
        result_row.update({'ensemble_prediction': ens_pred,
                           'malignancy_probability': round(ens_prob, 4),
                           **{f'{n.replace(" ","_")}_prob': round(r["prob"],4) for n, r in model_results.items()}})
        result_csv = pd.DataFrame([result_row]).to_csv(index=False)
        st.download_button("⬇ Download Prediction as CSV", result_csv,
                           file_name="oncoscan_prediction.csv", mime="text/csv")
