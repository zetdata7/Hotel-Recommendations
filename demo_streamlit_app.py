import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pydeck as pdk
import umap
import seaborn as sns
import altair as alt
from altair import datum

st.set_page_config("Hotel2vec",layout="wide")

@st.cache
def load_embeddings(hotel_ids,emb_file):
    emb = np.load(emb_file)
    hotelids = np.load(hotel_ids)
    return hotelids,emb

@st.cache
def load_properties_info(path_to_property):
    properties_df = pd.read_csv(path_to_property,dtype={"star_rating":str})
    properties_df = properties_df[~properties_df["market_id"].isna()]
    properties_df.rename(columns={"latitude":"lat","longitude":"long", "unit_count":"room_count", "hotel_id":"expe_property_id"},inplace=True)
    properties_df = properties_df.dropna(subset=["lat","long","star_rating"])
    print(properties_df)
    properties_df["market_name_l"] = properties_df["market_name"].str.lower()
    return properties_df

hotel_ids,embeddings = load_embeddings("")
property_info = load_properties_info("")

property_info = property_info.set_index("expe_property_id")
property_info = property_info.loc[property_info.index.intersection(hotel_ids)].reindex(hotel_ids).reset_index()


col_slider, col_table = st.beta_columns([4,14])

with col_slider:
    st.header('Search')
    #destination_selected = st.text_input("Destination", "Paris")
    hotel_id_selected = st.number_input("Hotel id",value=31419890,min_value=1)
    hotel_selected = property_info[property_info.expe_property_id==hotel_id_selected]

    st.text(hotel_selected["property_name"].values[0])
    st.text("Star rating: {}".format(hotel_selected["star_rating"].values[0]))
    st.text("Guest rating: {:4.1f}".format(hotel_selected["guest_review_rating"].values[0]))
    st.text("ADR: {0:4.0f}".format(hotel_selected["adr_lm"].values[0]))




dest_id_selected = property_info[property_info.expe_property_id==hotel_id_selected]["market_id"].values[0]
print(dest_id_selected)

mask_market = property_info["market_id"]==dest_id_selected
market_vectors = embeddings[mask_market]
print(market_vectors.shape)
cos_sims = cosine_similarity(market_vectors)
most_similar_indices = np.argsort(cos_sims,axis=1)[:,::-1]
market_info_df = property_info[mask_market].copy().reset_index()
print(market_info_df.shape)
index_to_slice = market_info_df[market_info_df.expe_property_id==hotel_id_selected].index[0]

cols_selected = ["property_name","star_rating","room_count","long","lat","guest_review_rating","adr_lm","type_category"]
df_for_plot = market_info_df.iloc[most_similar_indices[index_to_slice]].iloc[1:12,:][cols_selected]
center_hotel = market_info_df.iloc[most_similar_indices[index_to_slice]].iloc[0:1,:][cols_selected]
print(center_hotel)

with col_table:
    st.header("Most similar properties")
    st.dataframe(df_for_plot[["property_name","guest_review_rating","adr_lm","star_rating","room_count"]])






vecs_projected = umap.UMAP(n_neighbors=5).fit_transform(market_vectors)
#axes = sns.scatterplot(x=vecs_projected[0],y=vecs_projected[1])

st.write("")

st.header("Projection of embedding space")
st.text("Projection in a 2-d space of the embeddings using UMAP.")

#vr_filter = st.checkbox("Filter VRs")

proj1, proj2, proj3 = st.beta_columns([3,3,3])

#if vr_filter:

price_max = st.slider("Max price",min_value=50,max_value=1000,value=300,step=10)

with proj1:
    c = alt.Chart(pd.DataFrame({"dim1":vecs_projected[:,0],"dim2":vecs_projected[:,1],"star":market_info_df.star_rating.values})).mark_point().encode(
    x='dim1',
    y='dim2',
    color="star")

    st.write(c)

with proj2:
    c = alt.Chart(pd.DataFrame({"dim1":vecs_projected[:,0],"dim2":vecs_projected[:,1],"guest_rating":market_info_df.guest_review_rating.values})).mark_point().encode(
    x='dim1',
    y='dim2',
    color=alt.Color('guest_rating', scale=alt.Scale(scheme='purples'))
    )

    st.write(c)

with proj3:
    c = alt.Chart(pd.DataFrame({"dim1":vecs_projected[:,0],"dim2":vecs_projected[:,1],"ADR":market_info_df.adr_lm.values})).mark_point().encode(
    x='dim1',
    y='dim2',
    color=alt.Color('ADR', scale=alt.Scale(scheme='purples'))
    ).transform_filter(
        (datum.ADR<price_max)
        )


    st.write(c)

vr_mask = ~ ((market_info_df["type_category"].isin([2,3,-1])))
vecs_projected = vecs_projected[vr_mask]
market_info_df = market_info_df[vr_mask]


with st.beta_container():
    st.write("Filtering out VRs")

    st.write("")

    proj1_f, proj2_f, proj3_f = st.beta_columns([3,3,3])

    with proj1_f:
        c = alt.Chart(pd.DataFrame({"dim1":vecs_projected[:,0],"dim2":vecs_projected[:,1],"star":market_info_df.star_rating.values})).mark_point().encode(
        x='dim1',
        y='dim2',
        color="star")

        st.write(c)

    with proj2_f:
        c = alt.Chart(pd.DataFrame({"dim1":vecs_projected[:,0],"dim2":vecs_projected[:,1],"guest_rating":market_info_df.guest_review_rating.values})).mark_point().encode(
        x='dim1',
        y='dim2',
        color=alt.Color('guest_rating', scale=alt.Scale(scheme='purples'))
        )


        st.write(c)

    with proj3_f:
        c = alt.Chart(pd.DataFrame({"dim1":vecs_projected[:,0],"dim2":vecs_projected[:,1],"ADR":market_info_df.adr_lm.values})).mark_point().encode(
        x='dim1',
        y='dim2',
        color=alt.Color('ADR', scale=alt.Scale(scheme='purples'))
        ).transform_filter(
        (datum.ADR<price_max)
        )


        st.write(c)


st.write("")

