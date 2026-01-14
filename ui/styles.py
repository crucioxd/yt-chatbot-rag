import streamlit as st


def inject_css():
    st.markdown("""
    <style>
        .answer-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
            color: #262730;
            line-height: 1.6;
            font-size: 16px;
        }
    </style>
    """, unsafe_allow_html=True)
