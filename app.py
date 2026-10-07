from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data import FEATURES, make_business_data
from data_assistant import answer_dataset_question
from decision_log import LOG_PATH, append_decision, read_decisions
from ml import (classification_results, cluster_data, feature_importance,
                local_classification_explanation, numeric_features, prepare_features,
                q_learning_demo, regression_results)

st.set_page_config(page_title="Explainable BI", page_icon="◈", layout="wide")
st.markdown("""
<style>
html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', sans-serif; }
.stApp { background: #f3f6f5; color: #172a2b; }
[data-testid="stAppViewContainer"] .main .block-container { max-width: 1420px; padding: 1.15rem 2rem 2.5rem; }
[data-testid="stSidebar"] { background: #102326; }
[data-testid="stSidebar"] * { color: #f4fbf8 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#c6d7d2 !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] *, [data-baseweb="popover"] * { color:#173133 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background:#ffffff !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] span { color:#243638; }
[data-testid="stAlert"] { color:#172a2b; }
.stCaption, [data-testid="stCaptionContainer"] { color:#506263 !important; }
.hero { background:linear-gradient(115deg,#12383b,#176b60); padding:19px 25px; border-radius:13px; color:#f3fbf8; margin:0 0 17px; box-shadow:0 5px 18px rgba(22,77,66,.10); }
.hero h1 { color:#fff; margin:3px 0 4px; font-size:1.65rem; letter-spacing:-.04em; }
.hero p { color:#d1e5de; margin:0; font-size:.94rem; }
.eyebrow { text-transform:uppercase; letter-spacing:.13em; font-size:.66rem; font-weight:700; color:#a8dfc9; }
.section-tag { font-size:.67rem; font-weight:700; color:#17604f !important; letter-spacing:.1em; }
div[data-testid="stMetric"] { background:white; border:1px solid #dce6e2; padding:11px 14px; border-radius:11px; box-shadow:0 2px 8px rgba(27,55,47,.035); }
div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"] { color:#405557 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#142e2e !important; }
.stDataFrame, [data-testid="stTable"] { color:#183033; }
[data-testid="stPlotlyChart"] { background:#fff; border:1px solid #e1e9e6; border-radius:11px; padding:5px; }
[data-testid="stExpander"] { background:#fff; border:1px solid #e1e9e6; border-radius:10px; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] { margin-bottom:.35rem; }
@media (max-width: 1100px) {
  [data-testid="stAppViewContainer"] .main .block-container { padding:1rem 1.1rem 2rem; }
  .hero { padding:16px 19px; }
  .hero h1 { font-size:1.45rem; }
}
@media (max-width: 760px) {
  [data-testid="stAppViewContainer"] .main .block-container { padding:.8rem .75rem 1.5rem; }
  .hero { padding:15px; }
  .hero h1 { font-size:1.25rem; }
}
</style>
""", unsafe_allow_html=True)


def show_chart(figure, height: int = 320):
    """Render compact responsive charts with a consistent visual style."""
    figure.update_layout(height=height, font_family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif")
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False, "responsive": True})

with st.sidebar:
    st.markdown("## ◈  Explainable BI")
    st.caption("Business decision studio")
    st.divider()
    page_names = ["Overview", "Project guide", "Revenue analysis", "Risk decisions", "Customer segments", "Decision simulator", "Ask your data", "What-if & history"]
    active_page = st.selectbox("GO TO PAGE", page_names, index=0, label_visibility="visible")
    st.divider()
    source = st.selectbox("DATA SOURCE", ["Built-in business data", "Upload a personal dataset"], label_visibility="visible")
    uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"]) if source == "Upload a personal dataset" else None
    if source == "Upload a personal dataset" and uploaded:
        try:
            data = pd.read_excel(uploaded) if uploaded.name.lower().endswith(".xlsx") else pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read this dataset: {exc}")
            st.stop()
        dataset_label = uploaded.name
    else:
        data = make_business_data()
        dataset_label = "Built-in synthetic sample"
        if source == "Upload a personal dataset":
            st.info("Upload a CSV or XLSX file to use your data. Showing the built-in sample meanwhile.")
    st.divider()
    st.caption(f"{len(data):,} rows · {len(data.columns)} columns")
    st.caption("Synthetic fixed-seed sample." if uploaded is None else f"Using personal file: {uploaded.name}")
    st.caption("Data assistant runs locally · decision log saved locally")

page_headlines = {
    "Overview": "See the shape of your business",
    "Project guide": "Understand the methods and their limits",
    "Revenue analysis": "See what moves the outcome",
    "Risk decisions": "Understand a model's risk decision",
    "Customer segments": "Explore customer groups and outliers",
    "Decision simulator": "Compare actions in a simple simulator",
    "Ask your data": "Ask questions in plain language",
    "What-if & history": "Compare a decision before and after",
}
st.markdown(f'<div class="hero"><div class="eyebrow">Explainable business intelligence · {active_page}</div><h1>{page_headlines[active_page]}</h1><p>Explore business outcomes, understand model reasoning, and compare decisions with an auditable workflow.</p></div>', unsafe_allow_html=True)

if len(data) < 12:
    st.error("Use at least 12 rows for the built-in train/test workflows. More records are usually needed for a useful evaluation.")
    st.stop()

numeric_cols = [column for column in data.select_dtypes(include="number").columns if data[column].notna().any()]
base_features = [c for c in FEATURES if c in numeric_cols]
if len(base_features) < 2:
    base_features = [c for c in numeric_cols if c not in {"revenue", "high_risk", "churned"}][:8]
if len(base_features) < 2:
    st.error("Your CSV needs at least two numeric feature columns.")
    st.stop()

if active_page == "Overview":
    st.markdown('<span class="section-tag">BUSINESS OVERVIEW</span>', unsafe_allow_html=True)
    st.subheader("Your business at a glance")
    with st.expander("Start here · a beginner's guide"):
        st.markdown("1. Choose a built-in or uploaded dataset in the left panel.\n2. Select a page from **Go to page**.\n3. Use the overview filters to change the metric and records shown.\n4. Inspect one record in **Risk decisions**, or compare changed inputs in **What-if & history**.\n5. Expand scorecards to understand validation performance and its limits.")
    overview_filters = st.columns([1, 2])
    with overview_filters[0]:
        target_revenue = st.selectbox("Outcome metric", numeric_cols, index=numeric_cols.index("revenue") if "revenue" in numeric_cols else 0, key="overview_metric")
    overview_data = data
    if "high_risk" in data:
        risk_options = sorted(data["high_risk"].dropna().unique().tolist())
        if len(risk_options) <= 12:
            with overview_filters[1]:
                selected_risks = st.multiselect("Include risk groups", risk_options, default=risk_options, key="overview_risk_filter")
            overview_data = data[data["high_risk"].isin(selected_risks)]
    if overview_data.empty:
        st.info("No records match these filters. Select at least one risk group to continue.")
        st.stop()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Records in view", f"{len(overview_data):,}")
    k2.metric(f"Average {target_revenue.replace('_', ' ')}", f"{overview_data[target_revenue].mean():,.1f}", help="Average of the selected metric in the filtered records.")
    k3.metric(f"Total {target_revenue.replace('_', ' ')}", f"{overview_data[target_revenue].sum():,.1f}")
    k4.metric("High risk share", f"{overview_data['high_risk'].mean():.1%}" if "high_risk" in overview_data and len(overview_data) else "—", help="Share of records labeled high risk within the current filter.")
    left, right = st.columns([1.5, 1])
    with left:
        st.markdown("#### Revenue distribution")
        st.caption("How many records fall into each revenue range?")
        show_chart(px.histogram(overview_data, x=target_revenue, nbins=28, color_discrete_sequence=["#16816b"], template="plotly_white"), 290)
    with right:
        if "high_risk" in data:
            st.markdown("#### Risk mix")
            risk_mix = overview_data["high_risk"].map({0: "Standard risk", 1: "High risk"}).fillna(overview_data["high_risk"].astype(str)).value_counts().rename_axis("Group").reset_index(name="Records")
            show_chart(px.pie(risk_mix, names="Group", values="Records", hole=.6, color_discrete_sequence=["#16816b", "#e59a42"], template="plotly_white"), 290)
        else:
            st.markdown("#### Revenue by business driver")
            show_chart(px.box(overview_data, y=target_revenue, color_discrete_sequence=["#16816b"], template="plotly_white"), 290)
    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.markdown("#### Explore a relationship")
        st.caption("Change the axes or color to explore different patterns. Each dot is a row in your dataset.")
        numeric_for_scatter = [c for c in numeric_cols if c != "high_risk" and overview_data[c].nunique(dropna=True) > 1] or [target_revenue]
        scatter_controls = st.columns(3)
        default_x = "marketing_spend" if "marketing_spend" in numeric_for_scatter else numeric_for_scatter[0]
        with scatter_controls[0]: scatter_x = st.selectbox("Horizontal axis", numeric_for_scatter, index=numeric_for_scatter.index(default_x), key="overview_scatter_x")
        with scatter_controls[1]: scatter_y = st.selectbox("Vertical axis", numeric_for_scatter, index=numeric_for_scatter.index(target_revenue) if target_revenue in numeric_for_scatter else 0, key="overview_scatter_y")
        low_cardinality = [c for c in overview_data.columns if c not in {scatter_x, scatter_y} and overview_data[c].nunique(dropna=True) <= 12]
        with scatter_controls[2]: scatter_color = st.selectbox("Color by", ["None"] + low_cardinality, key="overview_scatter_color")
        scatter_data = overview_data.sample(min(600, len(overview_data)), random_state=42).copy()
        color_column = None if scatter_color == "None" else scatter_color
        if color_column and pd.api.types.is_numeric_dtype(scatter_data[color_column]):
            scatter_data[color_column] = scatter_data[color_column].astype("string")
        scatter_figure = px.scatter(scatter_data, x=scatter_x, y=scatter_y, color=color_column, color_discrete_sequence=["#16816b", "#e59a42", "#5874a6", "#bd6471"], opacity=.76, template="plotly_white", height=340)
        scatter_figure.update_layout(margin=dict(l=15, r=10, t=25, b=15), legend_title_text="")
        show_chart(scatter_figure, 330)
    with chart_b:
        st.markdown("#### Which measures move together?")
        st.caption("Correlation shows how numeric measures vary together; it does not prove one causes another.")
        corr_cols = numeric_cols[:10]
        corr = overview_data[corr_cols].corr(numeric_only=True)
        heatmap = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", template="plotly_white", height=340)
        heatmap.update_layout(margin=dict(l=15, r=10, t=25, b=15), coloraxis_colorbar_title="Correlation")
        show_chart(heatmap, 330)
    with st.expander("Data quality details"):
        quality = pd.DataFrame({"Column": data.columns, "Missing": data.isna().sum().values, "Missing %": (data.isna().mean().values * 100).round(1)})
        st.dataframe(quality, hide_index=True, use_container_width=True)
    with st.expander("Preview first 8 records"):
        st.dataframe(overview_data.head(8), hide_index=True, use_container_width=True)

if active_page == "Project guide":
    st.markdown('<span class="section-tag">PROJECT GUIDE / OBJECTIVE · METHODS · BACKEND</span>', unsafe_allow_html=True)
    st.subheader("One objective, four ML perspectives")
    st.markdown("""
    **Project objective:** Build an explainable business decision-support tool that uses historical business and customer data to estimate revenue and customer risk, discover customer groups, and compare potential actions. Each result includes an evaluation or a clear description of the assumptions behind it.

    The data included with the project is generated synthetic data. It makes every workflow runnable, but it does not establish evidence about a real company. Upload a suitable business CSV or XLSX file to explore your own data.
    """)
    st.markdown("#### How the backend processes a request")
    st.code("""CSV or built-in records
        ↓
Select target column and input features in the dashboard
        ↓
Convert inputs to numeric values · median-impute missing features
        ↓
Split into training (75%) and held-out evaluation (25%)
        ↓
Fit scikit-learn model(s) on training data only
        ↓
Score on held-out data · calculate feature-importance view
        ↓
Render charts, metrics, segments, or learned action policy""", language="text")
    st.markdown("#### What each coursework objective does")
    st.markdown("""
    **CO1 — Estimate a numeric business outcome.** The Revenue lab predicts a selected numeric target (revenue by default) from selected business drivers. It compares multiple linear regression, degree-2 polynomial regression, Ridge, Lasso, and random forest regression. R², RMSE, and MAE are computed on records excluded from fitting. A separate PCA view projects standardized inputs to two components for exploration; PCA is not used as a secret substitute target or as the regression score.

    **CO2 — Classify records by risk.** The Risk models page predicts a selected numeric class label (built-in `high_risk` by default) using regularized logistic regression, an RBF SVM, and random forest. It reports accuracy, precision, recall, F1, and ROC AUC on a stratified holdout set. Positive class means label value 1 for the supplied binary labels.

    **CO3 — Find customer groups and outliers.** K-Means assigns each record to one of a selected number of groups. DBSCAN instead groups nearby records and marks isolated points as noise (`-1`). Inputs are standardized first; PCA maps the result to two dimensions for display. Group profiles show the average selected business measures.

    **CO4 — Compare predictive signals and learn a toy policy.** A random forest ensemble summarizes risk prediction across many decision trees. Tabular Q-learning separately learns which action has the highest estimated long-term reward for a declining, steady, or growing sales state in a small simulated environment. The transition and reward values are explicit assumptions, not measured company outcomes.
    """)
    st.markdown("#### How to read explanations")
    st.info("Permutation importance shuffles one feature and measures the resulting change in held-out model performance. It shows model reliance, not cause and effect. Segment names are numeric cluster IDs; business interpretation needs domain review. Q-learning recommendations are only as realistic as the simulator assumptions.")
    with st.expander("Why these methods and settings were chosen"):
        st.markdown("""
        - **A fixed random seed** makes the sample and train/test split repeatable, so reruns are comparable.
        - **A 75/25 train/test split** leaves most rows for learning while reserving unseen rows for a basic generalization check. The test split is not used to fit the models.
        - **Median imputation** fills missing numeric inputs with a value that is less affected by extreme observations than the mean. Missingness can still contain information, so this simple baseline may not be best for every dataset.
        - **Standardization** puts differently scaled features on a comparable scale for PCA, logistic regression, SVM and K-Means. Tree models do not need it, so their pipelines keep their natural scale.
        - **Linear and polynomial regression** show simple trends and curved relationships; Ridge and Lasso show regularization, which can reduce sensitivity to noisy or redundant inputs. Random Forest gives a nonlinear ensemble comparison.
        - **Precision, recall and F1** are included because a risk system must balance false alarms against missed risks. ROC AUC checks ranking over multiple thresholds; it does not select a deployment threshold.
        - **K-Means** is an approachable fixed-number segmentation baseline. **DBSCAN** is included because it can mark outliers as noise without requiring a segment count. PCA is only a 2D visualization aid.
        - **Q-learning** makes action selection visible as state, action, reward and transition updates. Its rewards and transitions are made-up teaching assumptions and are not inferred from your uploaded file.
        """)
    with st.expander("What does confidence mean here?"):
        st.markdown("""
        **Case estimate:** the selected record's predicted class probability is a model output. Unless calibrated and independently checked, it is not a guarantee that the outcome is true with exactly that probability.

        **Validation evidence:** accuracy, precision, recall, F1 and ROC AUC summarize predictions over a held-out sample. They describe expected model behavior for similar data, not certainty about one case. Small samples, class imbalance, data drift, mislabeled targets and leakage can make these scores misleading.

        **Decision confidence:** treat a decision as more supportable when held-out metrics are strong, both classes have enough examples, the model performs similarly across relevant groups and time periods, and a domain expert agrees with the explanation. This application does not certify model safety or fairness.
        """)
    st.markdown("#### Personal dataset checklist")
    st.markdown("Use one row per business entity or observation, numeric predictors for the current ML workflows, and a numeric outcome column. For classification, use a binary target coded 0/1. Do not include identifiers as predictive features. Check dates, currency units, duplicate rows, missingness, and whether the target could leak future information into the inputs.")
    st.markdown("#### Backend source map")
    st.markdown("- `data.py` creates the reproducible synthetic sample.\n- `ml.py` contains feature preparation, sklearn training/evaluation, PCA clustering, feature importance, and the Q-learning simulation.\n- `app.py` collects selections, calls those workflows, and displays outputs.")
    with st.expander("Show the actual ML backend code"):
        import inspect
        from ml import (classification_results as backend_classification,
                        cluster_data as backend_clustering,
                        prepare_features as backend_preparation,
                        q_learning_demo as backend_q_learning,
                        regression_results as backend_regression)
        backend_view = st.selectbox("Backend workflow", ["Feature preparation", "Regression training", "Classification training", "Clustering", "Q-learning simulator"])
        backend_functions = {
            "Feature preparation": backend_preparation,
            "Regression training": backend_regression,
            "Classification training": backend_classification,
            "Clustering": backend_clustering,
            "Q-learning simulator": backend_q_learning,
        }
        st.code(inspect.getsource(backend_functions[backend_view]), language="python")

if active_page == "Revenue analysis":
    st.markdown('<span class="section-tag">CO1 / FEATURE ENGINEERING · PCA · MULTIPLE LINEAR REGRESSION</span>', unsafe_allow_html=True)
    st.subheader("What moves revenue?")
    st.caption("Compare linear, polynomial, regularized and ensemble regressors on a held-out test split.")
    targets = [c for c in numeric_cols if c not in {"churned", "high_risk"}]
    default_target = "revenue" if "revenue" in targets else targets[-1]
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1: target = st.selectbox("Outcome", targets, index=targets.index(default_target), key="reg_target")
    with c2: reg_features = st.multiselect("Business drivers", [c for c in numeric_cols if c not in {target, "churned", "high_risk"}], default=[c for c in base_features if c != target][:6], key="reg_features")
    with c3: alpha = st.slider("Regularization α", .01, 10.0, 1.0, .01)
    if len(reg_features) >= 1:
        scores, fitted, holdout = regression_results(data, reg_features, target, alpha)
        winner = scores.sort_values("RMSE").iloc[0]
        a, b, c = st.columns(3)
        a.metric("Best held-out RMSE", f"{winner['RMSE']:,.2f}")
        b.metric("Best model", winner["Model"])
        c.metric("Best R²", f"{winner['R²']:.3f}")
        show_chart(px.bar(scores.sort_values("RMSE"), x="RMSE", y="Model", orientation="h", color="R²", color_continuous_scale="teal", template="plotly_white", title="Held-out regression comparison · 25% test data"), 320)
        st.dataframe(scores.style.format({"R²": "{:.3f}", "RMSE": "{:,.2f}", "MAE": "{:,.2f}"}), hide_index=True, use_container_width=True)
        st.markdown("#### Local explanation · strongest model")
        x_test, y_test = holdout
        importance = feature_importance(fitted[winner["Model"]], reg_features, x_test, y_test)
        show_chart(px.bar(importance, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="teal", template="plotly_white", title="Permutation importance on held-out records"), 310)
        st.caption("Permutation importance measures the performance drop when one feature is shuffled. It describes model reliance, not causation.")
        with st.expander("PCA view of the selected business drivers"):
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            from ml import prepare_features
            z = StandardScaler().fit_transform(prepare_features(data, reg_features))
            pca = PCA(n_components=2).fit_transform(z)
            show_chart(px.scatter(x=pca[:, 0], y=pca[:, 1], color=data[target], labels={"x": "Principal component 1", "y": "Principal component 2", "color": target}, template="plotly_white", title="Standardized records projected into two principal components"), 320)
    else:
        st.info("Select at least one business driver.")

if active_page == "Risk decisions":
    st.markdown('<span class="section-tag">CO2 / POLYNOMIAL · REGULARIZATION · LOGISTIC REGRESSION</span>', unsafe_allow_html=True)
    st.subheader("Find the records that need attention")
    candidates = [c for c in numeric_cols if c not in {"revenue"}]
    if not candidates:
        st.info("Add a numeric risk label to your CSV, or use the built-in data.")
    else:
        default_cls = "high_risk" if "high_risk" in candidates else candidates[-1]
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1: cls_target = st.selectbox("Risk label", candidates, index=candidates.index(default_cls), key="cls_target")
        with c2: cls_features = st.multiselect("Risk indicators", [c for c in numeric_cols if c not in {cls_target, "churned", "high_risk", "revenue"}], default=[c for c in base_features if c != cls_target][:6], key="cls_features")
        with c3: c_value = st.slider("Logistic / SVM C", .05, 10.0, 1.0, .05)
        if len(cls_features) >= 1:
            try:
                scores, fitted, holdout = classification_results(data, cls_features, cls_target, c_value)
                winner = scores.sort_values("F1", ascending=False).iloc[0]
                a, b, c = st.columns(3)
                a.metric("Best F1", f"{winner['F1']:.3f}")
                b.metric("Best model", winner["Model"])
                c.metric("ROC AUC", f"{winner['ROC AUC']:.3f}")
                x_test, y_test = holdout
                importance = feature_importance(fitted[winner["Model"]], cls_features, x_test, y_test, classification=True)
                with st.expander("Compare models · validation metrics and global signals"):
                    show_chart(px.bar(scores.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1"], var_name="Metric", value_name="Score"), x="Model", y="Score", color="Metric", barmode="group", range_y=[0, 1], template="plotly_white", title="Held-out classification metrics"), 320)
                    st.dataframe(scores.style.format({k: "{:.3f}" for k in ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]}), hide_index=True, use_container_width=True)
                    st.markdown("#### Which signals influence the model overall?")
                    show_chart(px.bar(importance, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="teal", template="plotly_white"), 300)
                    st.caption("These feature scores explain overall model behavior. The selected-record explanation below is specific to one input row.")

                st.markdown("---")
                st.markdown("#### Inspect one decision")
                st.write("Choose a record and model. The app will show the predicted class, its estimated risk probability, and which input values most shift that probability in this model.")
                decision_col, model_col = st.columns([1, 1])
                with decision_col:
                    row_number = st.slider("Record to inspect", min_value=1, max_value=len(data), value=1, key="risk_record")
                with model_col:
                    decision_model_name = st.selectbox("Model used for this decision", list(fitted.keys()), index=list(fitted.keys()).index(winner["Model"]), key="decision_model")
                selected_model = fitted[decision_model_name]
                positive_class = 1 if 1 in list(selected_model.classes_) else max(selected_model.classes_)
                selected_row = numeric_features(data.iloc[[row_number - 1]], cls_features)
                prediction = selected_model.predict(selected_row)[0]
                training_median = pd.Series(selected_model.named_steps["simpleimputer"].statistics_, index=cls_features)
                positive_probability, local_effects = local_classification_explanation(
                    selected_model, selected_row, training_median,
                    cls_features, positive_class=positive_class,
                )
                predicted_probability = float(selected_model.predict_proba(selected_row)[0, list(selected_model.classes_).index(prediction)])
                decision_label = "Higher risk" if prediction == positive_class else "Lower risk"
                decision_color = st.warning if prediction == positive_class else st.success
                decision_color(f"**Model decision for record {row_number}: {decision_label}.** Estimated probability of the positive class ({positive_class}): **{positive_probability:.1%}**. Predicted class probability: **{predicted_probability:.1%}**.")
                st.caption("A probability is the model's estimate for this record, not a guarantee or a calibrated measure of certainty. Check the validation scorecard before relying on it.")

                impact_chart = local_effects.copy()
                impact_chart["Direction"] = impact_chart["Change from typical (percentage points)"].map(lambda v: "Raises model risk estimate" if v > 0 else "Lowers model risk estimate")
                show_chart(px.bar(impact_chart.sort_values("Change from typical (percentage points)"), x="Change from typical (percentage points)", y="Feature", color="Direction", orientation="h", color_discrete_map={"Raises model risk estimate": "#c75b45", "Lowers model risk estimate": "#16816b"}, template="plotly_white", title="Why did the model reach this estimate? · one input changed at a time"), 320)
                st.caption("For each input, the app replaces only that value with its dataset median and recalculates risk. This local sensitivity view is approximate and is not causal proof.")

                with st.expander("▸ Model trust scorecard · validation results"):
                    selected_metrics = scores.loc[scores["Model"] == decision_model_name].iloc[0]
                    trust_cards = st.columns(5)
                    for card, metric_name in zip(trust_cards, ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]):
                        card.metric(metric_name, f"{selected_metrics[metric_name]:.1%}" if pd.notna(selected_metrics[metric_name]) else "N/A")
                    st.caption(f"Measured on {len(y_test):,} held-out records (25% of the dataset). These metrics describe the model overall, not this one record. Precision = share of flagged records that were correct; recall = share of actual positive records found; F1 balances precision and recall; ROC AUC measures ranking separation.")
                    from sklearn.metrics import confusion_matrix
                    matrix = confusion_matrix(y_test, selected_model.predict(x_test), labels=selected_model.classes_)
                    matrix_df = pd.DataFrame(matrix, index=[f"Actual {label}" for label in selected_model.classes_], columns=[f"Predicted {label}" for label in selected_model.classes_])
                    st.markdown("**Validation record counts**")
                    st.dataframe(matrix_df, use_container_width=True)
                    st.caption("A row is the actual class and a column is the model's predicted class. The diagonal cells are correct predictions.")
            except ValueError as exc:
                st.error(f"Could not train the selected label: {exc}. Choose a binary label with examples of both classes.")
        else:
            st.info("Select at least one risk indicator.")

if active_page == "Customer segments":
    st.markdown('<span class="section-tag">CO3 / SUPPORT VECTOR MACHINE · K-MEANS · DBSCAN</span>', unsafe_allow_html=True)
    st.subheader("Discover customer groups and unusual records")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1: method = st.selectbox("Segmentation", ["K-Means", "DBSCAN"])
    with c2: cluster_features = st.multiselect("Signals for segmentation", base_features, default=base_features[:5], key="cluster_features")
    with c3: clusters = st.slider("K-Means segments", 2, 8, 4)
    if len(cluster_features) >= 2:
        chart, labels = cluster_data(data, cluster_features, method, clusters)
        chart["Revenue"] = data.loc[chart.index, "revenue"] if "revenue" in data else 0
        show_chart(px.scatter(chart, x="PC 1", y="PC 2", color="Segment", size="Revenue" if "revenue" in data else None, hover_data=["Revenue"] if "revenue" in data else None, template="plotly_white", title=f"{method} segments in a 2D PCA projection"), 360)
        summary = data.assign(Segment=labels).groupby("Segment")[cluster_features + (["revenue"] if "revenue" in data else [])].mean().round(2)
        st.markdown("#### Segment profile · average values")
        st.dataframe(summary, use_container_width=True)
        st.caption("DBSCAN marks noise points as segment −1. The plot is a PCA projection; it preserves broad structure, not every high-dimensional distance.")
    else:
        st.info("Choose at least two signals to create a PCA segment map.")

if active_page == "Decision simulator":
    st.markdown('<span class="section-tag">CO4 / ENSEMBLE LEARNING · Q-LEARNING</span>', unsafe_allow_html=True)
    st.subheader("Compare learned actions under changing business conditions")
    st.caption("A random forest estimates risk; tabular Q-learning learns a simple policy in a transparent simulated sales environment.")
    q, actions, reward_matrix = q_learning_demo()
    state_labels = ["Declining sales", "Steady sales", "Growing sales"]
    c1, c2 = st.columns([1, 1.2])
    with c1:
        state = st.selectbox("Current business state", state_labels)
        state_idx = state_labels.index(state)
        learned = actions[int(q[state_idx].argmax())]
        st.markdown("#### Q-learning policy")
        st.success(f"Suggested action: **{learned}**")
        st.write(f"**Why?** In the **{state.lower()}** state, this action has the highest learned Q-value (**{q[state_idx].max():.2f}**) among the four choices. The simulator's immediate reward for it is **{reward_matrix[state_idx, int(q[state_idx].argmax())]:.1f}** units. Q-values also include estimated future rewards from simulated state changes.")
        st.caption("Policy learned from 500 episodes; rewards are illustrative synthetic units.")
        show_chart(px.bar(x=actions, y=q[state_idx], labels={"x": "Action", "y": "Learned Q value"}, color=q[state_idx], color_continuous_scale="teal", template="plotly_white", title=f"Action values · {state.lower()}"), 290)
    with c2:
        st.markdown("#### Random forest risk ensemble")
        if "high_risk" in data and len(base_features) > 0 and set(pd.to_numeric(data["high_risk"], errors="coerce").dropna().unique()) == {0, 1}:
            _, fitted, _ = classification_results(data, base_features, "high_risk", 1.0)
            model = fitted["Random Forest"]
            forest = model.named_steps["randomforestclassifier"]
            importance = pd.DataFrame({"Feature": base_features, "Importance": forest.feature_importances_}).sort_values("Importance", ascending=True)
            show_chart(px.bar(importance, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="teal", template="plotly_white", title="Ensemble feature importance"), 290)
            st.caption("Random Forest combines many decision trees. Importance summarizes split improvements across the ensemble.")
        else:
            st.info("The ensemble view needs a numeric high_risk label coded 0/1 with examples of both classes. The built-in dataset meets this requirement.")
    with st.expander("How the simulator works"):
        st.write("States are declining, steady, or growing sales. Actions have explicit synthetic reward values and transition probabilities. Q-learning updates action values over repeated episodes. This illustrates the method; it is not a real-world financial forecast.")

if active_page == "Ask your data":
    st.markdown('<span class="section-tag">DATA ASSISTANT / ASK QUESTIONS IN PLAIN LANGUAGE</span>', unsafe_allow_html=True)
    st.subheader("Ask questions about the loaded dataset")
    st.write("This local assistant uses transparent pandas calculations on the data currently loaded in the sidebar. It does not call an external AI service or send your file anywhere.")
    st.info("Try: “What is the average revenue?”, “Which columns have missing values?”, “What correlates with customer retention?”, or “Show the distribution of high risk.”")
    if "dataset_chat_history" not in st.session_state:
        st.session_state.dataset_chat_history = []
    for message in st.session_state.dataset_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("table") is not None:
                st.dataframe(message["table"], hide_index=True, use_container_width=True)
    question = st.chat_input("Ask about averages, totals, columns, missing values, distributions, or correlations…", key="dataset_question")
    if question:
        st.session_state.dataset_chat_history.append({"role": "user", "content": question})
        answer, result_table = answer_dataset_question(data, question)
        st.session_state.dataset_chat_history.append({"role": "assistant", "content": answer, "table": result_table})
        st.rerun()
    if st.button("Clear conversation", key="clear_dataset_chat"):
        st.session_state.dataset_chat_history = []
        st.rerun()

if active_page == "What-if & history":
    st.markdown('<span class="section-tag">SCENARIO ANALYSIS / AUDITABLE CHANGE LOG</span>', unsafe_allow_html=True)
    st.subheader("Change a business input and compare model outcomes")
    st.write("This what-if tool compares one selected record before and after you edit its numeric inputs. It keeps the model fixed and records both estimates, the changes you made, and validation metrics. It estimates what the fitted model associates with the changed inputs; it cannot prove what would happen in the real world.")
    target_options = [column for column in numeric_cols if column not in {"churned", "high_risk"}]
    if not target_options:
        st.info("Upload a CSV with a numeric outcome column to run a revenue or outcome scenario.")
    else:
        with st.expander("How to use this page"):
            st.markdown("1. Select a numeric outcome and the inputs that the model may use.\n2. Pick a row to use as the starting situation.\n3. Change one or more input values to define a scenario.\n4. Compare baseline and scenario estimates, inspect model metrics, then save the scenario to the local Excel log.\n5. Change one assumption at a time when possible; this makes the comparison easier to explain.")
        scenario_target = st.selectbox("Outcome to estimate", target_options, index=target_options.index("revenue") if "revenue" in target_options else 0, key="scenario_target")
        scenario_features = st.multiselect("Inputs that influence the estimate", [c for c in numeric_cols if c not in {scenario_target, "churned", "high_risk"}], default=[c for c in base_features if c != scenario_target][:6], key="scenario_features")
        if not scenario_features:
            st.warning("Choose at least one input feature.")
        else:
            scenario_row_number = st.slider("Starting record", 1, len(data), 1, key="scenario_record")
            starting_values = prepare_features(data.iloc[[scenario_row_number - 1]], scenario_features)
            st.markdown("##### Define the scenario inputs")
            st.caption("The initial value for each input is taken from the selected row. Keep the current value to hold that input constant.")
            scenario_values = {}
            input_columns = st.columns(min(3, len(scenario_features)))
            for number, feature in enumerate(scenario_features):
                with input_columns[number % len(input_columns)]:
                    scenario_values[feature] = st.number_input(feature.replace("_", " ").title(), value=float(starting_values.iloc[0][feature]), key=f"scenario_value_{feature}", help="Proposed value for this scenario.")
            action_name = st.selectbox("Decision being considered", ["Increase marketing", "Change discount", "Improve retention", "Adjust another input", "Other"], key="scenario_action")
            decision_notes = st.text_input("Decision notes (optional)", placeholder="Why are you considering this change?", key="scenario_notes")
            try:
                regression_scores, regression_models, regression_holdout = regression_results(data, scenario_features, scenario_target, alpha=1.0)
                selected_regressor_name = st.selectbox("Regression model", list(regression_models), index=int(regression_scores["RMSE"].argmin()), key="scenario_regressor")
                selected_regressor = regression_models[selected_regressor_name]
                baseline_input = prepare_features(data.iloc[[scenario_row_number - 1]], scenario_features)
                edited_input = baseline_input.copy()
                for feature, value in scenario_values.items():
                    edited_input.loc[:, feature] = value
                baseline_outcome = float(selected_regressor.predict(baseline_input)[0])
                scenario_outcome = float(selected_regressor.predict(edited_input)[0])
                outcome_change = scenario_outcome - baseline_outcome
                base_col, change_col, new_col = st.columns(3)
                base_col.metric("Baseline model estimate", f"{baseline_outcome:,.2f}")
                change_col.metric("Estimated change", f"{outcome_change:+,.2f}", help="Scenario estimate minus baseline estimate.")
                new_col.metric("Scenario model estimate", f"{scenario_outcome:,.2f}")
                st.caption(f"Why did the estimate change? The same {selected_regressor_name} model saw the edited input values; it predicts {outcome_change:+,.2f} change in {scenario_target}. This is a model-based what-if comparison, not a causal effect or guaranteed business outcome.")
                changed_features = [feature for feature in scenario_features if float(baseline_input.iloc[0][feature]) != float(edited_input.iloc[0][feature])]
                if changed_features:
                    one_at_a_time = []
                    for feature in changed_features:
                        single_change = baseline_input.copy()
                        single_change.loc[:, feature] = edited_input.iloc[0][feature]
                        one_at_a_time.append({"Changed input": feature, "Estimated outcome shift": float(selected_regressor.predict(single_change)[0] - baseline_outcome)})
                    show_chart(px.bar(pd.DataFrame(one_at_a_time).sort_values("Estimated outcome shift"), x="Estimated outcome shift", y="Changed input", color="Estimated outcome shift", color_continuous_scale="RdYlGn", orientation="h", template="plotly_white", title="Which changed inputs shifted the estimate? · one at a time"), 290)
                    st.caption("Each bar changes one input while holding the others at baseline. Effects can interact, so these bars may not add up exactly to the total scenario change.")
                reg_metric_row = regression_scores.loc[regression_scores["Model"] == selected_regressor_name].iloc[0]
                with st.expander("▸ Confidence and model validation for this estimate"):
                    trust_cols = st.columns(3)
                    trust_cols[0].metric("Held-out R²", f"{reg_metric_row['R²']:.3f}")
                    trust_cols[1].metric("Held-out RMSE", f"{reg_metric_row['RMSE']:,.2f}")
                    trust_cols[2].metric("Held-out MAE", f"{reg_metric_row['MAE']:,.2f}")
                    st.caption(f"Scores come from {len(regression_holdout[1]):,} held-out records. RMSE and MAE describe typical prediction error in the target's units; R² describes variance explained on this split. They are overall model checks, not confidence intervals for this specific scenario.")
                    st.dataframe(regression_scores, hide_index=True, use_container_width=True)

                risk_record = {"Target": scenario_target, "Target model": selected_regressor_name,
                               "Baseline estimate": baseline_outcome, "Scenario estimate": scenario_outcome,
                               "Estimated change": outcome_change, "Decision": action_name,
                               "Record number": scenario_row_number,
                               "Input changes": {feature: {"before": float(baseline_input.iloc[0][feature]), "after": float(edited_input.iloc[0][feature])} for feature in scenario_features if float(baseline_input.iloc[0][feature]) != float(edited_input.iloc[0][feature])},
                               "Notes": decision_notes,
                               "Regression R2": float(reg_metric_row["R²"]), "Regression RMSE": float(reg_metric_row["RMSE"]),
                               "Regression MAE": float(reg_metric_row["MAE"])}
                risk_columns = [column for column in base_features if column not in {scenario_target}][:8]
                classification_summary = None
                if "high_risk" in data and len(risk_columns) > 0:
                    try:
                        risk_scores, risk_models, risk_holdout = classification_results(data, risk_columns[:8], "high_risk", c=1.0)
                        selected_risk_model = risk_models["Random Forest"]
                        positive_class = 1 if 1 in list(selected_risk_model.classes_) else max(selected_risk_model.classes_)
                        base_risk_input = prepare_features(data.iloc[[scenario_row_number - 1]], risk_columns[:8])
                        edited_risk_input = base_risk_input.copy()
                        for feature, value in scenario_values.items():
                            if feature in edited_risk_input:
                                edited_risk_input.loc[:, feature] = value
                        positive_index = list(selected_risk_model.classes_).index(positive_class)
                        baseline_risk = float(selected_risk_model.predict_proba(base_risk_input)[0, positive_index])
                        scenario_risk = float(selected_risk_model.predict_proba(edited_risk_input)[0, positive_index])
                        risk_record.update({"Risk model": "Random Forest", "Baseline risk probability": baseline_risk,
                                            "Scenario risk probability": scenario_risk,
                                            "Risk probability change": scenario_risk - baseline_risk,
                                            "Risk precision": float(risk_scores.loc[risk_scores["Model"] == "Random Forest", "Precision"].iloc[0]),
                                            "Risk recall": float(risk_scores.loc[risk_scores["Model"] == "Random Forest", "Recall"].iloc[0]),
                                            "Risk F1": float(risk_scores.loc[risk_scores["Model"] == "Random Forest", "F1"].iloc[0]),
                                            "Risk ROC AUC": float(risk_scores.loc[risk_scores["Model"] == "Random Forest", "ROC AUC"].iloc[0])})
                        classification_summary = (baseline_risk, scenario_risk, risk_scores, risk_holdout)
                    except ValueError as exc:
                        st.warning(f"Risk comparison is unavailable for this dataset: {exc}")
                if classification_summary:
                    baseline_risk, scenario_risk, risk_scores, risk_holdout = classification_summary
                    risk_a, risk_b, risk_c = st.columns(3)
                    risk_a.metric("Baseline risk estimate", f"{baseline_risk:.1%}")
                    risk_b.metric("Risk probability change", f"{scenario_risk - baseline_risk:+.1%}")
                    risk_c.metric("Scenario risk estimate", f"{scenario_risk:.1%}")
                    st.caption("This risk estimate comes from a separately trained Random Forest classifier. It may move differently from revenue because it predicts a different target.")
                    with st.expander("▸ Risk model validation scorecard"):
                        forest_metrics = risk_scores.loc[risk_scores["Model"] == "Random Forest"].iloc[0]
                        metric_cols = st.columns(5)
                        for card, metric in zip(metric_cols, ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]):
                            card.metric(metric, f"{forest_metrics[metric]:.1%}" if pd.notna(forest_metrics[metric]) else "N/A")
                        st.caption(f"Calculated on {len(risk_holdout[1]):,} held-out records. These indicate overall risk model performance; they are not confidence bounds on the scenario's risk estimate.")
                if st.button("Save this scenario to the Excel change log", type="primary", key="save_scenario"):
                    risk_record["Dataset"] = dataset_label
                    updated_log = append_decision(risk_record)
                    st.success(f"Scenario saved. The workbook now contains {len(updated_log):,} decision record(s).")
                if LOG_PATH.exists():
                    st.download_button("Download Excel decision log", data=LOG_PATH.read_bytes(), file_name="decision_log.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_decisions")
            except (ValueError, KeyError) as exc:
                st.error(f"The selected regression workflow could not run: {exc}. Check that the outcome and input columns contain usable numeric data.")
    with st.expander("Review saved decision history"):
        saved_log = read_decisions()
        if saved_log.empty:
            st.caption("No scenarios have been saved yet. Save a scenario above to create the Excel audit log.")
        else:
            st.dataframe(saved_log.tail(100), hide_index=True, use_container_width=True)

st.markdown("---")
st.caption("Explainable BI · Educational decision support · Predictions are associations learned from data, not causal claims.")
