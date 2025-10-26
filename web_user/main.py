import streamlit as st #Use for make UI
import pandas as pd    #Use for data manipulation and analysis
import joblib           #Use for saving/loading ML models

# Load the trained model
model = joblib.load('./churn_model.pkl')

st.set_page_config(page_title="Churn Prediction", page_icon="💳") #set page icon and title of webpage
st.title("💳 Credit Card User Churn Prediction System")           #title of Project
st.markdown("### 🧠 Predict if a customer will leave or stay with the bank")
st.markdown("---") #make horizontal line 

# --- Get input from user ---
with st.container(): # Creates a separate block/section
    st.markdown("### 🧾 Enter Customer Details")
    col1, col2 = st.columns(2)
    with col1:
       age = st.number_input("👤Customer Age", 18, 100)
       credit_limit = st.number_input("💰Credit Limit", 100, 50000)
       total_trans_amt = st.number_input("💳 Total Transaction Amount", 0, 100000)

    with col2:
       income = st.selectbox("🏦 Income Category", ['Less than $40K', '$40K-$60K', '$60K-$80K', 'Above $80K'])
       months_on_book = st.number_input("📅 Months on Book", 1, 300)

# --- Create new customer DataFrame ---
new_customer = pd.DataFrame({  #connect input from user (pd.Dataframe)
    'Customer_Age':[age],
    'Income_Category':[income],
    'Credit_Limit':[credit_limit],
    'Months_on_book':[months_on_book],
    'Total_Trans_Amt':[total_trans_amt]
})

# --- Preprocess categorical columns like in training ---
categorical_cols = ['Income_Category']  # Add more if needed
#convert Income_Category into Numerical column using one-hot encoding
new_customer = pd.get_dummies(new_customer, columns=categorical_cols, drop_first=True)

# --- Add missing columns (set to 0) ---
train_columns = model.feature_names_in_  # All features model expects
for col in train_columns:
    if col not in new_customer.columns:
        new_customer[col] = 0

# --- Reorder columns to match training ---
new_customer = new_customer[train_columns]

# --- Make prediction ---
if st.button("🔍 Predict Churn"):
    prediction = model.predict(new_customer)
    if prediction[0] == 1:
        st.error("⚠️ This customer is likely to **CHURN**!")
    else:
        st.success("✅ This customer is likely to **STAY**!")
        
        
# Optional: sidebar info
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2920/2920215.png", width=120 )
st.sidebar.header("📘 Project Info")
st.sidebar.write("""
- **Project:** Credit Card User Churn Prediction  
- **Tech Stack:** Python, Scikit-learn, Streamlit, JupyterNotebook  
- **Developer:** Nirankar Singh & Team  
""")
st.sidebar.markdown("📅 *Developed as a BTech ML Project*")
st.markdown("---")
st.markdown("💡 *Tip: Use this dashboard to identify high-risk customers and plan retention strategies.*")

