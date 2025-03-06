import streamlit as st

st.set_page_config(
    page_title="Brazilian E-Commerce Seller Churn Prediction",
    page_icon="https://seeklogo.com/images/O/olist-logo-9DCE4443F8-seeklogo.com.png",
    layout="wide"
)

# Success message below the navigation
st.sidebar.success("Select a page above")

# Create layout
with st.container():
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(
            "https://d3hw41hpah8tvx.cloudfront.net/images/ilustracao_carrossel_36d060f00a.svg",
            use_container_width=True
        )

    with col2:
        st.markdown(
            """
            <div style="display: flex; flex-direction: column; justify-content: center; height: 32vh;">
                <h1 style="font-size: 40px;">Welcome to Olist's Seller Churn Prediction Tool</h1>
            </div>
            """,
            unsafe_allow_html=True
    )


st.write("")
st.markdown("<br><br>", unsafe_allow_html=True)
st.info("👈 To use this tool, go to the left and select a page.")

# Three Key Sections Below
col3, col4, col5 = st.columns([2, 2, 2])

with col3:
    st.markdown("#### 🔍 Seller Churn Analysis")
    st.markdown("---")
    st.write("Identify patterns in seller activity and predict whether a seller is likely to churn in the next quarter.")

with col4:
    st.markdown("#### 🤖 Powered by XGBoost")
    st.markdown("---")
    st.write("Using XGBoost, a powerful machine learning model, to accurately predict seller churn based on historical data.")

with col5:
    st.markdown("#### 📈 Strategic Insights")
    st.markdown("---")
    st.write("Enable proactive decision-making by understanding seller behavior and reducing churn risk.")

# Disclaimer and Additional Information
st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #7F8C8D;">
    <p>Disclaimer: This tool provides insights based on historical data and may require updates for accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('''
    For details on how the model works, visit: [Click Here](https://github.com/PurwadhikaDev/AlphaGroup_JC_DS_FT_BDG_05_FinalProject)
    
    Created by: Team Alpha
''')
