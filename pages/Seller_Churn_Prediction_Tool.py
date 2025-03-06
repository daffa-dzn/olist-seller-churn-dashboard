import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import io
import re

# Load Churn Model
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Load Churn Data
@st.cache_data
def load_churn_data(path):
    try:
        churn_df = pd.read_csv(path)
        churn_df['seller_active_quarter'] = pd.PeriodIndex(churn_df['seller_active_quarter'], freq='Q-DEC')
        churn_df = churn_df.set_index('seller_active_quarter').sort_index()
        return churn_df
    except FileNotFoundError:
        st.error(f"The file at {path} was not found.")
        return None
    except Exception as e:
        st.error(f"Error loading churn data: {str(e)}")
        return None

# Filter Profitable Sellers
def filter_profitable_sellers(df, monthly_subscription_cost, sales_commission_rate, profitability='profitable'):
    net_sales = lambda x: x["sales"].sum() * (1 - sales_commission_rate)
    total_subscription_cost = lambda x: x["n_months_active_quarter"].sum() * monthly_subscription_cost

    if profitability == 'profitable':
        return df.groupby("seller_id").filter(lambda x: net_sales(x) >= total_subscription_cost(x))["seller_id"]
    elif profitability == 'unprofitable':
        return df.groupby("seller_id").filter(lambda x: net_sales(x) < total_subscription_cost(x))["seller_id"]
    else:
        st.error("Invalid profitability filter. Use 'profitable' or 'unprofitable'.")
        return pd.Series()

# Get Profitable Seller Sales for 1 Year
def get_profitable_seller_sales_year(df, monthly_subscription_cost, sales_commission_rate, period_start, period_end):
    df_filtered_period = df.query("index >= @period_start and index <= @period_end")
    profitable_sellers_id = filter_profitable_sellers(df_filtered_period, monthly_subscription_cost, sales_commission_rate)
    df_filtered_period = df_filtered_period[df_filtered_period.seller_id.isin(profitable_sellers_id)]

    seller_sales = df_filtered_period.groupby("seller_id").agg(
        median_sales=("sales", "median"),
        net_sales=("sales", "sum")
    ).reset_index()
    
    return seller_sales

# Check Priority Sellers
def check_priority_sellers(test_data, seller_sales_year, segment='top_20_pct'):
    top_sellers_list = seller_sales_year[seller_sales_year.net_sales >= seller_sales_year.net_sales.quantile(0.80)]["seller_id"]
    upper_20_bound = test_data['sales'].quantile(0.80)

    if segment == 'top_20_pct':
        mask = (test_data['seller_id'].isin(top_sellers_list)) | (test_data['sales'] >= upper_20_bound)
    elif segment == 'regular':
        mask = (~test_data['seller_id'].isin(top_sellers_list)) & (test_data['sales'] < upper_20_bound)
    else:
        st.error("Invalid segment. Use 'top_20_pct' or 'regular'.")
        return None

    return mask

# Process Churn Data
def process_churn_data(churn_df):
    try:
        processed_df = churn_df.copy()
        columns_to_drop = ['n_delivered_customers', 'n_approved_orders', 'n_orders']
        existing_columns = [col for col in columns_to_drop if col in processed_df.columns]
        if existing_columns:
            processed_df.drop(columns=existing_columns, inplace=True)

        if 'n_delivered_carrier' in processed_df.columns and 'n_months_active_quarter' in processed_df.columns:
            processed_df.insert(
                loc=2, 
                column='n_delivered_carrier_per_active_month', 
                value=processed_df['n_delivered_carrier'] / processed_df['n_months_active_quarter']
            )
            processed_df.drop(columns=['n_delivered_carrier'], inplace=True)

        return processed_df
        
    except Exception as e:
        st.error(f"Error processing churn data: {str(e)}")
        return None
    
# Function to extract numbers from seller IDs and sort them correctly
def natural_sort(seller_list):
    return sorted(seller_list, key=lambda x: int(re.search(r'\d+', x).group()))

# Function to get the next quarter
def get_next_quarter(current_quarter):
    return (pd.Period(current_quarter, freq='Q') + 1).strftime('%YQ%q')

st.set_page_config(
    page_title="Brazilian E-Commerce Seller Churn Prediction",
    page_icon="https://seeklogo.com/images/O/olist-logo-9DCE4443F8-seeklogo.com.png",
    layout="wide"
)

# Load Model
model_path = 'seller_churn_XGB.sav'
model = load_model(model_path)

# Load Full Historical Data (Planted in the App)
@st.cache_data
def load_historical_data():
    return load_churn_data("churn_dataset.csv")  # Load planted data using existing function

# Load Prediction Data from User Upload
@st.cache_data
def load_uploaded_data(uploaded_file):
    if uploaded_file is not None:
        return load_churn_data(uploaded_file)
    return None

# Load historical data directly
churn_historical = load_historical_data()

# Streamlit UI
st.title("📊 Seller Churn Prediction App")

# Sidebar Upload
st.sidebar.header("Upload Data")
predict_file = st.sidebar.file_uploader("Upload Data to Predict (CSV)", type=['csv'])

if predict_file:
    churn_predict = load_uploaded_data(predict_file)

    if churn_predict is not None:
        # Apply Processing to Historical & Predict Data
        churn_historical_processed = process_churn_data(churn_historical)
        churn_predict_processed = process_churn_data(churn_predict)

        monthly_subscription_cost = st.sidebar.number_input("Monthly Subscription Cost", value=39.00)
        sales_commission_rate = st.sidebar.slider("Sales Commission Rate", 0.0, 1.0, 0.18)

        # Filter Profitable Sellers
        profitable_sellers_historical = filter_profitable_sellers(churn_historical_processed, monthly_subscription_cost, sales_commission_rate)
        profitable_sellers_predict = filter_profitable_sellers(churn_predict_processed, monthly_subscription_cost, sales_commission_rate)

        # Get Top Sellers (Last Year)
        seller_sales_year = get_profitable_seller_sales_year(churn_historical_processed, monthly_subscription_cost, sales_commission_rate, '2017Q1', '2017Q4')
        top_sellers_last_year = seller_sales_year[seller_sales_year["net_sales"] >= seller_sales_year["net_sales"].quantile(0.80)]

        # Ensure Top Sellers Predict has the same structure as Top Sellers Last Year
        # Correct: Only show sellers whose sales are in the top 20% of the prediction dataset
        top_sellers_predict = churn_predict_processed[churn_predict_processed["sales"] >= churn_predict_processed["sales"].quantile(0.80)]

        # Get seller IDs that exist in the Predict Data
        sellers_in_predict_data = set(churn_predict_processed["seller_id"])

        # Combine Top Sellers (Only those who are also in Predict Data)
        combined_top_sellers = sellers_in_predict_data.intersection(
            set(top_sellers_last_year["seller_id"]).union(set(top_sellers_predict["seller_id"]))
        )

        # Identify Reguler Sellers (Sellers in Predict Data but NOT in Top Sellers)
        non_top_sellers = sellers_in_predict_data - combined_top_sellers

        # Select Sellers
        st.sidebar.subheader("Select Active Sellers")
        selected_segment = st.sidebar.radio("Seller Type", ["Priority Sellers", "Standard Sellers"])  # ✅ Renamed options

        selected_seller = None  # ✅ Ensure selected_seller is always defined

        if selected_segment == "Priority Sellers":
            selected_seller = st.sidebar.selectbox("Select a Seller", natural_sort(list(combined_top_sellers)))  # ✅ Fixed sorting
        else:
            selected_seller = st.sidebar.selectbox("Select a Seller", natural_sort(list(non_top_sellers)))  # ✅ Fixed sorting

        # Display Data
        st.subheader("📂 Preview of Data (Data to Predict)")
        st.dataframe(churn_predict_processed.head())

        # Ensure seller_active_quarter is included in top_sellers_predict
        if "seller_active_quarter" not in top_sellers_predict.columns:
            top_sellers_predict = top_sellers_predict.reset_index()  # Fix if it was removed as an index

        # Display Priority Sellers Tables
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 Priority Sellers (Past)")  # Short & balanced
            st.write(f"Total Seller : {top_sellers_last_year.shape[0]}")
            top_sellers_last_year["net_sales"] = top_sellers_last_year["net_sales"].round(0)  # Round to 2 decimal places
            st.dataframe(top_sellers_last_year.sort_values(by='net_sales',ascending=False))

        with col2:
            st.subheader("📌 Priority Sellers (Now)")  # Short & balanced
            st.write(f"Total Seller : {top_sellers_predict.shape[0]}")
            top_sellers_predict["sales"] = top_sellers_predict["sales"].round(0)  # Round to 2 decimal places
            st.dataframe(top_sellers_predict[["seller_active_quarter", "seller_id", "sales"]].sort_values(by='sales',ascending=False))

        # ✅ Title for Single Seller Prediction
        st.subheader("🎯 Predict Churn for Selected Seller")

        # Predict Button (Single Seller)
        if st.button("🚀 Predict Churn for Selected Seller"):
            if selected_seller is None:
                st.warning("⚠️ Please select a seller before predicting.")
            else:
                selected_data = churn_predict_processed[churn_predict_processed["seller_id"].isin([selected_seller])]

                if selected_data.empty:
                    st.warning(f"⚠️ No prediction data available for seller {selected_seller}.")
                else:
                    selected_data = selected_data.reset_index()

                    X_selected = selected_data.drop(columns=[col for col in ["seller_id", "churn", "is_churn"] if col in selected_data.columns])

                    missing_features = set(model.feature_names_in_) - set(X_selected.columns)
                    if missing_features:
                        st.error(f"Missing required features for prediction: {missing_features}")
                    else:
                        predictions = model.predict(X_selected)

                        st.subheader("📢 Prediction Result")
                        for idx, row in selected_data.iterrows():
                            current_quarter = row["seller_active_quarter"]
                            next_quarter = get_next_quarter(current_quarter)  # Get the next quarter
                            churn_status = predictions[idx]

                            if churn_status == 1:
                                st.markdown(
                                    f"""
                                    <div style="background-color:#FFDDDD; padding:10px; border-radius:5px; 
                                    border-left: 5px solid red; color:#000000; font-weight:bold;">
                                    On {next_quarter}, seller {selected_seller} is <span style="color:red;">going to churn</span>.
                                    </div>
                                    """, unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f"""
                                    <div style="background-color:#DDFFDD; padding:10px; border-radius:5px; 
                                    border-left: 5px solid green; color:#000000; font-weight:bold;">
                                    On {next_quarter}, seller {selected_seller} is <span style="color:green;">not going to churn</span>.
                                    </div>
                                    """, unsafe_allow_html=True
                                )
        
        # ✅ Bulk Prediction for All Sellers
        st.subheader("🎯 Predict Churn for All Sellers")
        
        if st.button("🚀 Predict Churn for All Sellers"):
            if churn_predict_processed is None or churn_predict_processed.empty:
                st.warning("⚠️ Please upload a valid prediction dataset first.")
            else:
                full_data = churn_predict_processed.reset_index()
        
                X_full = full_data.drop(columns=[col for col in ["seller_id", "churn", "is_churn"] if col in full_data.columns])
        
                missing_features = set(model.feature_names_in_) - set(X_full.columns)
                if missing_features:
                    st.error(f"Missing required features for prediction: {missing_features}")
                else:
                    # Predict for all sellers and rename column
                    full_data["is_churn"] = model.predict(X_full)  # ✅ Renamed column
        
                    # ✅ Split Priority & Standard Sellers who will churn
                    priority_churn_sellers = full_data[
                        (full_data["seller_id"].isin(combined_top_sellers)) & (full_data["is_churn"] == 1)
                    ]
                    standard_churn_sellers = full_data[
                        (full_data["seller_id"].isin(non_top_sellers)) & (full_data["is_churn"] == 1)
                    ]
        
                    # Create Three Columns for Display
                    col1, col2, col3 = st.columns(3)
        
                    # 📊 Column 1 - All Predicted Sellers
                    with col1:
                        st.subheader("📊 All Predicted Sellers")
                        st.dataframe(full_data[["seller_active_quarter", "seller_id", "sales", "is_churn"]].head())  # ✅ Renamed column
        
                        output_csv = io.BytesIO()
                        full_data.to_csv(output_csv, index=False)
                        output_csv.seek(0)
        
                        st.download_button(
                            label="📥 Download Full Predictions",
                            data=output_csv,
                            file_name="churn_predictions_all.csv",
                            mime="text/csv"
                        )
        
                    # 🚨 Column 2 - Priority Sellers Who Will Churn
                    with col2:
                        st.subheader("🚨 Priority Sellers (Churn)")
                        st.dataframe(priority_churn_sellers[["seller_active_quarter", "seller_id", "sales", "is_churn"]].head())  # ✅ Renamed column
        
                        output_csv_priority = io.BytesIO()
                        priority_churn_sellers.to_csv(output_csv_priority, index=False)
                        output_csv_priority.seek(0)
        
                        st.download_button(
                            label="📥 Download Priority Churners",
                            data=output_csv_priority,
                            file_name="priority_sellers_churn.csv",
                            mime="text/csv"
                        )
        
                    # ⚠️ Column 3 - Standard Sellers Who Will Churn
                    with col3:
                        st.subheader("⚠️ Standard Sellers (Churn)")
                        st.dataframe(standard_churn_sellers[["seller_active_quarter", "seller_id", "sales", "is_churn"]].head())  # ✅ Renamed column
        
                        output_csv_standard = io.BytesIO()
                        standard_churn_sellers.to_csv(output_csv_standard, index=False)
                        output_csv_standard.seek(0)
        
                        st.download_button(
                            label="📥 Download Standard Churners",
                            data=output_csv_standard,
                            file_name="standard_sellers_churn.csv",
                            mime="text/csv"
                        )

       # Seller Performance Visualization
        st.subheader("📊 Seller Performance Over Time")

        if selected_seller:
            # Filter Last Year Data for the selected seller
            seller_data = churn_historical[churn_historical["seller_id"] == selected_seller]

            if not seller_data.empty:
                # Create two columns for side-by-side plots
                col1, col2 = st.columns(2)

                # with col1:
                #     st.subheader("Total Sales per Quarter")
                #     fig, ax = plt.subplots(figsize=(6, 3))  # Smaller figure size
                #     ax.plot(seller_data.index.astype(str), seller_data["sales"], marker="o", linestyle="-", color="b", label="Total Sales")
                #     ax.set_xlabel("Quarter")
                #     ax.set_ylabel("Sales")
                #     ax.set_title(f"Sales Trend - {selected_seller}")
                #     ax.legend()
                #     ax.grid(True)
                #     st.pyplot(fig)

                # with col2:
                #     if "n_orders" in seller_data.columns:
                #         st.subheader("Total Orders per Quarter")
                #         fig, ax = plt.subplots(figsize=(6, 3))  # Same size as sales graph
                #         ax.plot(seller_data.index.astype(str), seller_data["n_orders"], marker="o", linestyle="-", color="g", label="Total Orders")
                #         ax.set_xlabel("Quarter")
                #         ax.set_ylabel("Number of Orders")
                #         ax.set_title(f"Order Trend - {selected_seller}")
                #         ax.legend()
                #         ax.grid(True)
                #         st.pyplot(fig)
                #     else:
                #         st.warning(f"⚠️ 'n_orders' column is missing in the dataset for seller {selected_seller}.")

                with col1:
                    import plotly.express as px

                    st.subheader("Total Sales per Quarter")

                    # Ensure 'seller_active_quarter' is a string for categorical display
                    seller_data_reset = seller_data.reset_index()
                    seller_data_reset["seller_active_quarter"] = seller_data_reset["seller_active_quarter"].astype(str)

                    # Create interactive Plotly chart with rotated x-axis labels
                    fig = px.line(
                        seller_data_reset,
                        x="seller_active_quarter",
                        y="sales",
                        markers=True
                    )

                    # Rotate x-axis labels for better readability
                    fig.update_layout(
                        xaxis_title="Quarter",
                        title={
                            "text": f"Sales Trend - {selected_seller}",
                            "x": 0.5,  # Centers the title
                            "xanchor": "center"  # Ensures proper alignment
                        },
                        yaxis_title="Sales",
                        xaxis_tickangle=-45  # Rotates labels by 45 degrees
                    )

                    # Display in Streamlit
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    if "n_orders" in seller_data.columns:
                        st.subheader("Total Orders per Quarter")

                        # Ensure 'seller_active_quarter' is a string for categorical display
                        # Create interactive Plotly chart with rotated x-axis labels
                        fig = px.line(
                            seller_data_reset,
                            x="seller_active_quarter",
                            y="n_orders",
                            markers=True
                        )

                        # Rotate x-axis labels for better readability
                        fig.update_layout(
                            xaxis_title="Quarter",
                            title={
                            "text": f"Order Trend - {selected_seller}",
                            "x": 0.5,  # Centers the title
                            "xanchor": "center"  # Ensures proper alignment
                        },
                            yaxis_title="Number of Order",
                            xaxis_tickangle=-45  # Rotates labels by 45 degrees
                        )

                        # Display in Streamlit
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"⚠️ No historical data available for seller {selected_seller}.")

else:
    st.warning("⚠️ Please upload a CSV file for prediction.")
