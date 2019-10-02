import alibi
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

# dataset
st.title('Adult income dataset')
st.dataframe(raw_data, height=200)

# select instance
instance = st.text_input(label='Select an instance to make a prediction:', value=1)

# prediction
x = data.data[int(instance)].reshape(1, -1)
pred = model.predict_proba(x)
st.write('Model prediction:')
st.write(pred)

# explanations
explain = st.button('Explain prediction!')
if explain:
    explanation = explainer.explain(x)
    st.write('Anchor explanation:')
    st.write(explanation['names'])
