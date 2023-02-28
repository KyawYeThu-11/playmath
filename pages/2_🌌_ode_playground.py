import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from latex2sympy2 import latex2sympy
from util import *


def get_dy(x, y, equation_code):
    return eval(equation_code)

def get_y(x_0, y_0, delta_t, t, equation_code):
    coords=list()
    x = x_0
    y = y_0
    for time in np.arange(0, t, delta_t):
        dy = get_dy(x, y, equation_code)
        # In first-order DE, x == t
        x += delta_t
        y += dy * delta_t
        coords.append((x, y))
    return coords, y

def main():
    ## Title and Intro
    _, col1, _ = st.columns(3)
    with col1:
        st.image("images/calculus.png")

    st.markdown("<h2 style='text-align: center;'>First-Order Differential Equation Playground</h2>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Functionality", "Background", "Side note"])
    with tab1: 
        st.markdown('''
            Given a first-order differential equation and necessary parameters, this program will output 
            1. Solution curve, a particular solution satisfying the initial condition `y(x_0)=y_0`
            2. Slope field, a geometric picture of the possible solutions to the equation.  
        ''')
    with tab2: 
        st.markdown('''
            Many differential equations cannot be solved by obtaining an explicit formula for
            the solution. However, we can often find numerical approximations to solutions, one of which 
            is [Euler’s method](https://brilliant.org/wiki/differential-equations-eulers-method-small-step/), 
            which is also the one used here. Be approximations accurate enough, a pretty reasonable **solution curve** 
            can be plotted by joining the estimated points.

            This is exactly what this program does, and probably what most other tools online do for plotting a solution curve. 
            If you are looking for one that can output a solution formula, if any, you may want to check this tool from 
            [WolframAlpha](https://www.wolframalpha.com/examples/mathematics/differential-equations).

            When it comes to constructing a **slope field**, we can just place a grid of points on the space. Then, draw a vector for
            each point in the direction, governed by x's direction and y's direction, which can be derived from the equation.  
        ''')
    with tab3:
        st.markdown('''
            I am sad to let you know that this program cannot work for all sorts of first-order differential equations as of right now.
            For example,
            - variables in the equation must be only `x` and `y`
            - the equation cannot contain a constant in an English letter
            - the program allows the equation to be comprised of only a set of common functions like `sin`, `e`, `ln`
            - other corner cases that I haven't discovered
            
            I do have plans to upgrade this program after this spring semester \:D so that a wider range of equations becomes
            acceptable. Hence, I'm welcoming your inputs if you find any corner cases which crash this little naive program. Have fun playing!  
        ''')

    st.subheader("Playground")
    ## Inputs for variables
    st.write("Specify the initial coordinate.")
    col2, col3 = st.columns(2)
    with col2:
        x_0 = st.number_input("x_0", value=1.0, min_value=-10.0, max_value=10.0, step=0.5)
    with col3:
        y_0 = st.number_input("y_0", value=1.0, min_value=-10.0, max_value=10.0, step=0.5)
    
    st.write("Specify the step size and input variable.")
    col4, col5 = st.columns(2)
    with col4:
        delta_t = st.number_input("delta_t (step size)", value=0.01, help="The smaller the step size the more accurate the estimation of the curve.")
    with col5:
        input_t = st.number_input("t (input variable)", value=5)
    
    ## Equation
    equation = st.text_input(
        "Write a first-order differential equation in latex. (y'=)",
        help="Since it is assumed that your equation begins with 'y'=', only the right hand side of the equation needs to be written.")    
    if not equation:
        st.stop()

    st.latex("y'="+equation)
    equation_code = str(latex2sympy(equation))
    # print(equation_code)
    equation_code = replace_functions(equation_code)
    # print(equation_code)
    
    ## Processing 
    coords, solution = get_y(x_0, y_0, delta_t, input_t, equation_code)
    df = pd.DataFrame(data=coords, columns=['x', 'y'])

    fig, ax = plt.subplots(1, 2, figsize=(22, 8))
    ## Plotting solution curve
    x = df['x']
    y = df['y']
    ax[0].plot(x, y)
    ax[0].set_title(f'Solution Curve for y\'={equation}')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')

    ## Plotting slope field
    max_x, max_y = df['x'].max(), df['y'].max()
    space_x, space_y = max_x/10, max_y/10
    x1 = np.arange(-max_x, max_x, space_x)
    y1 = np.arange(-max_y, max_y, space_y) if max_y!=0 else x1
    X1, Y1 = np.meshgrid(x1, y1) # rectangular grid with points
    dy = get_dy(X1, Y1, equation_code)
    dx = 1

    ### normalization
    n = np.sqrt(dy**2+dx**2)
    dy, dx = dy/n, dx/n

    ### defining color
    color_n = -2
    color = np.sqrt(((dx-color_n)/2)*2 + ((dy-color_n)/2)*2)

    ### plot
    ax[1].set_title(f'Slope Field for y\'={equation}')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].quiver(X1, Y1, dx, dy, color)

    st.pyplot(fig)

if __name__=="__main__":
    main()