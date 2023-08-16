import streamlit as st

def main():
    _, col1, _ = st.columns(3)
    with col1:
        st.image("images/function.png")
    
    st.markdown("<h2 style='text-align: center;'>Playmath</h2>", unsafe_allow_html=True)
    st.markdown('''
        Welcome to "**Playmath**", a place where you can fiddle with simple yet powerful tools to gain 
        better intuition in various maths domains!

        They are developed mainly out of my desire to explore particular math concepts, but still, 
        I'd like to believe that you'll be benefitted from _playing_ around with them even though you, yourself, didn't do _programming_ part.

        Currently, there're only two playgrounds. Click the sidebar to visit and have fun!
    ''')

    col2, col3 = st.columns(2)
    with col2:
        st.subheader("📈 Linear Regression")
    with col3:
        st.subheader("🌌 ODE")

    st.markdown("<p style='text-align: center;'>Icons are provided by <a href='https://www.flaticon.com/'>Flaticon</a>.</p>", unsafe_allow_html=True)

if __name__=="__main__":
    main()