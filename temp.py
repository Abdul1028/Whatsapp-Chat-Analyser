import streamlit as st

st.write("")

a = st.number_input("Number 1: ")
b = st.number_input("Number 2: ")

c = st.button("Total")

d = a+b

if c:
    st.write(d)



