import alibi
import altair as alt
import io
import joblib
import pandas as pd
import requests
import streamlit as st

INCOME_CLASSIFIER_URI = 'https://storage.googleapis.com/seldon-models/sklearn/income/model/model.joblib'
INCOME_EXPLAINER_URI = 'https://storage.googleapis.com/seldon-models/sklearn/income/explainer/explainer.dill'


@st.cache(ignore_hash=True)
def load_data():
    return alibi.datasets.fetch_adult()


@st.cache(ignore_hash=True)
def load_model():
    resp = requests.get(INCOME_CLASSIFIER_URI)
    model = joblib.load(io.BytesIO(resp.content))
    return model


@st.cache(ignore_hash=True)
def load_explainer():
    resp = requests.get(INCOME_EXPLAINER_URI)
    explainer = joblib.load(io.BytesIO(resp.content))
    return explainer


# load assets
data = load_data()
model = load_model()
explainer = load_explainer()
explainer.predict_fn = model.predict

# prepare data for display
cmap = dict.fromkeys(data.category_map.keys())
for key, val in data.category_map.items():
    cmap[key] = {i: v for i, v in enumerate(val)}

raw_data = pd.DataFrame(data.data).replace(cmap)
raw_data.columns = data.feature_names
raw_data['Income'] = data.target
raw_data['Income'] = raw_data['Income'].replace(dict(enumerate(data.target_names)))

# sidebar header
st.sidebar.title('Alibi Explanations Demo')
st.sidebar.markdown(
    """
    Pick a dataset and a model, and explore black-box model explanations powered by
    [Alibi](https://github.com/SeldonIO/alibi).
    """
)

# sidebar controls
threshold = st.sidebar.slider(label='Precision threshold:', min_value=0.0, max_value=1.0, value=0.9)

# dataset
st.title('Adult income dataset')
st.dataframe(raw_data, height=200)

# select instance
instance = st.text_input(label='Select an instance to make a prediction:', value=1)

# prediction
x = data.data[int(instance)].reshape(1, -1)
pred = model.predict_proba(x)

# prediction bar chart
st.write('Model prediction:')
p = pd.DataFrame(pred.T, index=data.target_names).reset_index()
p.columns = ['Income', 'Probability']
chart = alt.Chart(p).mark_bar(size=30).encode(
    y='Probability:Q',
    x='Income:O',
    opacity=alt.condition(
        alt.datum.Probability > 0.5,
        alt.value(1),
        alt.value(0.5)
    )).properties(width=200).configure_axisX(labelAngle=0)
st.altair_chart(chart)

# explanations
explain = st.button('Explain prediction!')
if explain:
    explanation = explainer.explain(x, threshold=threshold)
    precision = explanation['precision']
    if precision < threshold:
        st.warning('Could not find an anchor satisfying the {:.2f} precision threshold'.format(threshold))
    st.write('Anchor explanation with precision {:.2f}:'.format(precision))
    st.write(explanation['names'])
